"""pour_fabric_mimic — 양팔 잡기→들기→붓기, **저차원(언더액추·PhysX mimic) 손 전용**.

★`pour_fabric/pour_fabric_env.py` 사본(09.14, 사용자 결정 "원본 냅두고 저차원 핸드 전용 신설").
  다른 점: (1) 양팔 26 DOF fabric 두 인스턴스가 서로의 슬라이스를 매 스텝 동기화(`sync_other`)
  (2) 부팅에서 언더액추 계약 검사 — URDF `<mimic>` 표 ↔ USD `physxMimicJoint` gearing 대조,
      종속관절 한계 확장, open≠grip·grip∈한계 (grasp_fj_rh 09.07 이식)
  (3) `ctrl/mimic_err_max` 등 mimic 건전성 지표.
  보상·관측·성공 판정 구조는 원본과 같다(RewardContext 동일 → 같은 t2r 프롬프트 골격).

(아래는 원본 설명)

★09.13 재작성. 보상은 여기 없다 — `RewardContext` 를 채워 `cfg.reward_code_path` 의
  생성 코드(`modules/t2r`)에 넘긴다. 성공 판정(fill·spill·xy)은 env 가 계산하고 ctx 에
  넣는다: 보상이 바꿀 수 없는 **기준 지표**다.

에피소드: reset(테이블 위 컵 2 + 소스 컵 안 비드) → hold(비드 정착, 팔 고정) → 정책.
제어: 팔 = palm 6D(앵커+델타) → Fabrics → 관절 PD. 손 = 시너지 폐쇄도(접촉 동결).
"""

from __future__ import annotations

import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material, find_matching_prim_paths
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from openarm.agnostic.modules.adr import TaskADR
from openarm.agnostic.modules.object_wrench import WrenchDR
from openarm.agnostic.modules.perception_delay import noisy_pose
from .t2r.context import RewardContext
from .t2r.loader import call_reward_fn, load_reward_fn
from openarm.common.bead_assets import bead_offsets_in_cup

from . import bimanual as _bm
from . import pour_fabric_env_cfg as _cfg
from .bead_flags import BeadGeometry, _local as _bead_local, compute_bead_flags
from .start_bank import StartBank
from .pour_fabric_env_cfg import PHYSICS_ADR_TERMINAL, PourFabricMimicEnvCfg
from .side_rig import SideRig, palm_vmax_vec

# 에피소드 단위 로그 키(로그 전용 — 보상·관측과 무관).
_EP_KEYS = ("tilt_max", "toward_max", "dir_err_at_max", "lift_max", "lipdz_at_max",
            "in_target_max", "in_source_min", "spill_end")

_FABRICS_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "FABRICS", "src"))
if os.path.isdir(_FABRICS_SRC) and _FABRICS_SRC not in sys.path:
    sys.path.insert(0, _FABRICS_SRC)

import fabrics_sim.fabrics.openarm_rh56f1_pose_fabric as _fab_rh   # noqa: E402
import fabrics_sim.fabrics.openarm_tesollo_pose_fabric as _fab_tes  # noqa: E402
from fabrics_sim.integrator.integrators import DisplacementIntegrator  # noqa: E402
from fabrics_sim.utils.utils import initialize_warp                   # noqa: E402
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel      # noqa: E402


def _fabric_class(name: str):
    for mod in (_fab_tes, _fab_rh):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f"Fabrics 클래스 '{name}' 를 찾을 수 없다")


#: PhysX mimic 제약이 종속관절 prim 에 남기는 속성. 규약 `q_dep + gearing·q_ref + offset = 0`
#: → **gearing = −multiplier**(임포터 정상 동작).
_MIMIC_GEARING_ATTR = "physxMimicJoint:rotZ:gearing"


class PourFabricMimicEnv(DirectRLEnv):
    cfg: PourFabricMimicEnvCfg

    # ==================================================================
    def __init__(self, cfg: PourFabricMimicEnvCfg, render_mode: str | None = None, **kw):
        _cfg.resolve_cfg(cfg)
        self.pair = _bm.get_pair(cfg.pair_name)
        self._grav_comp = float(cfg.gravity_compensation) if cfg.enable_gravity else 0.0
        sp = cfg.robot_cfg.spawn
        if (bool(sp.rigid_props.disable_gravity) == bool(cfg.enable_gravity)
                or bool(sp.articulation_props.enabled_self_collisions) != bool(cfg.enable_self_collisions)):
            raise RuntimeError("물리 스위치가 파생 cfg 에 반영되지 않았다 — resolve_cfg 경로 확인")
        self._reward_fn, self._reward_src = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kw)

        N, dev = self.num_envs, self.device
        # 소스=우(붓기)·리시버=좌. 대향 관절 미러 부호: thumb_2 축 Z → 좌측 −1.
        self.src = SideRig(self, self.pair.source, role="src",
                           sensors=self._sensor_store["src"], palm_sensor=self._palm_store["src"],
                           delta_lo=cfg.src_palm_delta_lo, delta_hi=cfg.src_palm_delta_hi,
                           oppose_sign=1.0)
        self.rcv = SideRig(self, self.pair.receiver, role="rcv",
                           sensors=self._sensor_store["rcv"], palm_sensor=self._palm_store["rcv"],
                           delta_lo=cfg.rcv_palm_delta_lo, delta_hi=cfg.rcv_palm_delta_hi,
                           oppose_sign=-1.0)
        self.rigs = (self.src, self.rcv)
        w_src, w_rcv = self.src.hand_action_width, self.rcv.hand_action_width
        self._arm_w = self.pair.source.num_arm_joints if str(cfg.arm_control) == "joint" else 6
        if self._arm_w + w_src != cfg.num_actions_per_side or self._arm_w + w_rcv != cfg.num_actions_per_side:
            raise RuntimeError(f"손 액션 폭 불일치: cfg {cfg.num_actions_per_side - self._arm_w} vs "
                               f"src {w_src} / rcv {w_rcv}")

        # ---- 시작 자세 전체(두 팔 + 두 손 + head) ---------------------------------------------
        self._reset_q = self.robot.data.default_joint_pos[0].clone()
        for rig in self.rigs:
            self._reset_q[rig.arm_t] = rig.reset_q[rig.arm_t]
            self._reset_q[rig.syn_t] = rig.reset_q[rig.syn_t]
        for rig in self.rigs:
            rig.reset_q = self._reset_q.clone()

        # ---- Fabrics ×2 --------------------------------------------------------------------
        initialize_warp(str(dev)[-1])
        self._world = WorldMeshesModel(
            batch_size=N, device=dev, max_objects_per_env=int(cfg.fabrics_max_objects_per_env),
            world_dict=self._build_fabric_world())
        self._world_ids, self._world_indicator = self._world.get_object_ids()
        for rig in self.rigs:
            arm_sl, hand_sl = _bm.fabric_slots(rig.profile)
            rig.setup_fabric(_fabric_class(rig.profile.fabric_class), DisplacementIntegrator, arm_sl, hand_sl)
        # ---- 언더액추 계약(부팅 게이트) ----
        self._mimic_pairs = self._load_mimic_pairs()
        self._assert_mimic_constraints_present()
        self._assert_hand_pose_usable()
        self._widen_dependent_joint_limits()

        # ---- 시작 자세 실측(앵커·fabric 게이트) — 리셋 직후 버퍼는 stale, 물리 2스텝 뒤 읽는다 ----
        q0 = self._reset_q.unsqueeze(0).expand(N, -1).contiguous()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        self.robot.set_joint_position_target(q0)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        for rig in self.rigs:
            rig.init_anchor()
            rig.init_real_control()        # ★09.30 실기 명령 지연·손 속도 상한(기본 끔)

        A = cfg.action_space
        self._half = cfg.num_actions_per_side
        self.actions = torch.zeros(N, A, device=dev)
        self.prev_actions = torch.zeros(N, A, device=dev)
        self._policy_dt = float(cfg.sim.dt) * int(cfg.decimation)
        # ★09.28 palm 지령 방식: "absolute"(a = 앵커+델타 자세) | "increment"(a = 목표 속도, 누적) — cfg 주석 참조
        mode = str(cfg.palm_action_mode)
        if mode not in ("absolute", "increment", "workspace"):
            raise ValueError(f"palm_action_mode 는 absolute|increment|workspace: {mode}")
        self._palm_increment = mode == "increment"
        self._palm_workspace = mode == "workspace"
        if str(cfg.hand_control) not in ("synergy", "direct"):
            raise ValueError(f"hand_control 는 synergy|direct: {cfg.hand_control}")
        if str(cfg.hand_direct_range) not in ("limits", "grip"):
            raise ValueError(f"hand_direct_range 는 limits|grip: {cfg.hand_direct_range}")
        if str(cfg.hand_control) == "direct":
            print(f"[pour_fabric_mimic] 손 full-joint 지령 · 관절별 절대 목표(범위 {cfg.hand_direct_range}) · EMA α {cfg.hand_ema} · "
                  f"전 범위 {cfg.hand_full_range_s} s 속도 상한 · 결합·게이트 없음 · 힘 멈춤 "
                  f"{(str(cfg.contact_freeze_threshold) + ' N') if cfg.hand_direct_contact_freeze else '없음'}", flush=True)
        if str(cfg.arm_control) not in ("fabric", "joint"):
            raise ValueError(f"arm_control 는 fabric|joint: {cfg.arm_control}")
        if str(cfg.arm_joint_mode) not in ("increment", "absolute"):
            raise ValueError(f"arm_joint_mode 는 increment|absolute: {cfg.arm_joint_mode}")
        if str(cfg.arm_control) == "joint":
            if bool(cfg.obs_fabric_state) or bool(cfg.fabric_adr):
                raise ValueError("arm_control=joint 에서는 obs_fabric_state·fabric_adr 를 끈다(fabric 을 적분하지 않는다)")
            for rig in self.rigs:
                rig.init_joint_arm()
        if self._palm_workspace:
            for rig in self.rigs:
                print(f"[pour_fabric_mimic:{rig.role}] palm 작업공간 절대 지령(DEXTRAH) · 위치 박스 "
                      f"{[round(float(v), 3) for v in rig.box_lo]}~{[round(float(v), 3) for v in rig.box_hi]} · "
                      f"fabric 상태 관측 {bool(cfg.obs_fabric_state)}", flush=True)
        self._palm_cmd = torch.zeros(N, 2, 6, device=dev)     # 증분 방식의 EMA 상태 (src, rcv)
        self._palm_vmax = tuple(palm_vmax_vec(v[0], v[1]).to(dev)
                                for v in (cfg.src_palm_vel_max, cfg.rcv_palm_vel_max))
        self._palm_leash = (tuple(cfg.src_palm_leash), tuple(cfg.rcv_palm_leash))
        if str(cfg.hand_action_mode) not in ("absolute", "increment"):
            raise ValueError(f"hand_action_mode 는 absolute|increment: {cfg.hand_action_mode}")
        if str(cfg.hand_action_mode) == "increment":
            print(f"[pour_fabric_mimic] 손 증분 지령 · 최대 닫힘 속도 {cfg.synergy_close_speed}/스텝", flush=True)
        if self._palm_increment:
            for rig, v, lsh in zip(self.rigs, (cfg.src_palm_vel_max, cfg.rcv_palm_vel_max), self._palm_leash):
                print(f"[pour_fabric_mimic:{rig.role}] palm 증분 지령 · 최대 {v[0]} m/s·{v[1]} deg/s · "
                      f"간극 상한 {lsh[0]} m·{lsh[1]} deg · EMA α {cfg.palm_action_ema_alpha}", flush=True)

        # ---- 비드 판정 상태 ------------------------------------------------------------------
        k = int(cfg.bead_count)
        self._geom = BeadGeometry(
            inner_radius=float(cfg.cup_inner_radius), inside_z_min=float(cfg.cup_inside_z_min),
            inside_z_max=float(cfg.cup_inside_z_max), mouth_z=float(cfg.cup_mouth_z))
        self._prev_tgt_z = torch.full((N, k), -1e6, device=dev)
        self._crossed = torch.zeros(N, k, dtype=torch.bool, device=dev)
        self._prev_in_tgt = torch.zeros(N, device=dev)
        self._prev_in_src = torch.ones(N, device=dev)
        self._prev_spill = torch.zeros(N, device=dev)
        self._flags_fresh = torch.ones(N, dtype=torch.bool, device=dev)
        # xy 검증 배치 유지 · z 는 바닥 기준 재적층(shaker 0.65: 원본 z 는 림 위로 나간다)
        _offs = torch.tensor(bead_offsets_in_cup(k), device=dev)
        _layer = torch.arange(k, device=dev) // 5
        _offs[:, 2] = -float(cfg.object_origin_offset_z) + float(cfg.bead_z_from_bottom) + float(cfg.bead_layer_dz) * _layer
        self._bead_offs = _offs
        if float(_offs[:, 2].max()) > float(cfg.cup_mouth_z) - 0.01:
            raise RuntimeError(f"[pour_fabric_mimic] 비드 최상층 z {float(_offs[:, 2].max()):.3f} 이 림 {cfg.cup_mouth_z:.3f} 에 닿는다")

        self._prev_arm_qd = {rig.role: torch.zeros_like(self.robot.data.joint_vel[:, rig.arm_t])
                             for rig in (self.src, self.rcv)}
        # ★09.25 에피소드 단위 지표(사용자 지시: "학습 로그와 결정론 로그 차이가 너무 커서 분석이 어렵다").
        #   task/src_tilt_deg 는 **전 env·전 스텝 평균**이라 접근·파지 구간(기울기 0°)과 리셋이 섞여 실제 붓는 각을 가린다
        #   (i18: 곡선 34° vs 결정론 env 별 최대 중앙 87°). 에피소드마다 최대 기울기·방향·높이·배달을 따로 모아
        #   **끝난 에피소드의 값**을 로그한다. 보상·관측에는 쓰지 않는다(로그 전용).
        self._ep_cur = {k: torch.zeros(N, device=dev) for k in _EP_KEYS}
        self._ep_cur["in_source_min"].fill_(1.0)
        self._ep_last = {k: torch.full((N,), float("nan"), device=dev) for k in _EP_KEYS}
        self._success_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self._success_streak = torch.zeros(N, dtype=torch.long, device=dev)
        self._dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._mim_qd_streak = torch.zeros(N, dtype=torch.long, device=dev)
        self._phys_f_streak = torch.zeros(N, dtype=torch.long, device=dev)
        self._src_spawn = torch.zeros(N, 3, device=dev)
        self._rcv_spawn = torch.zeros(N, 3, device=dev)
        self._src_grasped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._rcv_grasped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._shape_prev_g = {"src": torch.zeros(N, dtype=torch.bool, device=dev),
                              "rcv": torch.zeros(N, dtype=torch.bool, device=dev)}   # 파지 놓침 로그(09.29)
        self._shape_tick = 0
        self._cups_nested = torch.zeros(N, dtype=torch.bool, device=dev)
        self._cups_center_dist = torch.zeros(N, device=dev)
        self._last_terms: dict = {}
        self._setup_dr()
        self._log_tick = 0
        self._bank = StartBank(self)          # ★09.30 쥔 상태 시작 뱅크(기본 끔)

        print(f"[pour_fabric_mimic] pair={self.pair.name} usd={self.pair.usd_relpath} "
              f"src={self.src.profile.name} rcv={self.rcv.profile.name} "
              f"action={A} obs={cfg.observation_space} critic={cfg.state_space} "
              f"beads={k} reward={self._reward_src}", flush=True)

    # ==================================================================
    # s2r DR (09.14): 지각 지연·노이즈 · 물리 DR(EventTerm, ADR 확장) · 들린 컵 외란
    # ==================================================================
    def _setup_dr(self) -> None:
        cfg, N, dev = self.cfg, self.num_envs, self.device
        em = getattr(self, "event_manager", None)
        phys = {k: v for k, v in PHYSICS_ADR_TERMINAL.items()} if em is not None else {}
        self.adr = TaskADR(
            {"obs": {"object_xyz": (float(cfg.obs_noise_object_xyz), float(cfg.adr_obs_noise_object_xyz_max)),
                     "object_rot_deg": (float(cfg.obs_noise_object_rot_deg), float(cfg.adr_obs_noise_object_rot_max_deg)),
                     "delay_steps": (float(cfg.perception_delay_base_steps), float(cfg.perception_delay_max_steps))},
             "wrench": {"force_scale": (0.0, float(cfg.wrench_force_scale_max)),
                        "torque_scale": (0.0, float(cfg.wrench_torque_scale_max))},
             "fabric": {"damping": tuple(float(v) for v in cfg.adr_fabric_damping_range),
                        "velocity_ff": tuple(float(v) for v in cfg.adr_fabric_velocity_ff_range)}},
            num_increments=int(cfg.adr_num_increments), increment_interval=int(cfg.adr_increment_interval),
            trigger_threshold=float(cfg.adr_trigger_threshold), enabled=bool(cfg.enable_adr),
            event_manager=em, physics_cfg=phys)
        # 지각 링버퍼 (N, L, 7): pos3 + quat4. 지연 상한은 ADR 진행도로 0 → max.
        L = int(cfg.perception_delay_max_steps) + 1
        self._perc_buf = {k: torch.zeros(N, L, 7, device=dev) for k in ("src", "rcv")}
        self._perc_flush = torch.ones(N, dtype=torch.bool, device=dev)
        self._perc_arange = torch.arange(N, device=dev)
        self._wrench = {k: WrenchDR(N, dev, force_scale=1.0, torque_scale=1.0,
                                    prob_range=tuple(cfg.wrench_prob_range)) for k in ("src", "rcv")}
        self._cup_mass = {"src": torch.full((N,), _cfg.POUR_CUP_MASS, device=dev),
                          "rcv": torch.full((N,), _cfg.POUR_CUP_MASS, device=dev)}
        print(f"[pour_fabric_mimic] s2r DR: events={'on' if em is not None else 'OFF'} adr={bool(cfg.enable_adr)} "
              f"obs_noise(q {cfg.obs_noise_qpos} qd {cfg.obs_noise_qvel} body {cfg.obs_noise_body}) "
              f"perception delay 0→{cfg.perception_delay_max_steps} step · object noise → "
              f"{cfg.adr_obs_noise_object_xyz_max} m/{cfg.adr_obs_noise_object_rot_max_deg}° · "
              f"wrench → {cfg.wrench_force_scale_max} N/kg · cup friction {cfg.cup_friction_range}", flush=True)

    def _read_cup_mass(self, cup: RigidObject) -> torch.Tensor:
        """실제 질량 (N,) — 질량 DR 뒤 공칭으로 정규화하면 외란 비가 흔들린다(grasp_s2r 09.10)."""
        view = getattr(cup, "root_physx_view", None)
        m = None
        if view is not None:
            try:
                m = view.get_masses()
            except Exception:       # 뷰 미준비 → 공칭
                m = None
        if m is None or m.ndim != 2 or m.shape[0] != self.num_envs:
            return torch.full((self.num_envs,), _cfg.POUR_CUP_MASS, device=self.device)
        return m[:, 0].to(self.device, dtype=torch.float32)

    def _perceive(self, key: str, cup: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
        """컵 pose 지각: 지연(0..d_max 균등, env 별 매 스텝 재추첨) + 코히런트 노이즈. (pos env-local, quat)."""
        pos = self._local(cup.data.root_pos_w)
        quat = cup.data.root_quat_w
        cur = torch.cat([pos, quat], dim=1)
        buf = self._perc_buf[key]
        fl = self._perc_flush
        buf[fl] = cur[fl].unsqueeze(1)
        buf = torch.roll(buf, shifts=1, dims=1)
        buf[:, 0] = cur
        self._perc_buf[key] = buf
        d_max = int(round(self.adr.get_param("obs", "delay_steps")))
        idx = torch.randint(0, d_max + 1, (self.num_envs,), device=self.device)
        got = buf[self._perc_arange, idx]
        p, q = noisy_pose(got[:, :3], got[:, 3:], self.adr.get_param("obs", "object_xyz"),
                          self.adr.get_param("obs", "object_rot_deg"))
        return p, q

    def _apply_wrench(self) -> None:
        """들린 컵에 질량정규화 외란(매 스텝 재추첨, decay 0). ADR 진행도로 스케일 0 → 종점."""
        fs = self.adr.get_param("wrench", "force_scale")
        ts = self.adr.get_param("wrench", "torque_scale")
        lift_min = float(self.cfg.wrench_lift_min_m)
        fire = 0.0
        for key, cup, spawn in (("src", self.source_cup, self._src_spawn), ("rcv", self.receiver_cup, self._rcv_spawn)):
            w = self._wrench[key]
            w.force_scale, w.torque_scale = fs, ts
            lifted = (self._local(cup.data.root_pos_w)[:, 2] - spawn[:, 2]) > lift_min
            forces, torques = w.step(self._cup_mass[key], lifted)
            cup.set_external_force_and_torque(forces, torques, is_global=True)
            fire += float((forces.view(self.num_envs, -1).norm(dim=-1) > 0.0).float().mean())
        self.extras["dr/wrench_fire_frac"] = fire / 2.0

    # ==================================================================
    # 언더액추 계약 (grasp_fj_rh 이식 — 양손)
    # ==================================================================
    def _load_mimic_pairs(self) -> dict[str, tuple[str, float]]:
        """자산 **폴더**의 URDF `<mimic>` 표에서 종속→(리더, 배율)을 읽는다(양손). 코드에 표를 안 적는다."""
        _dir = os.path.dirname(os.path.join(_cfg._ASSETS_DIR, self.pair.usd_relpath))
        _cands = sorted(glob.glob(os.path.join(_dir, "*.urdf")))
        if len(_cands) != 1:
            raise RuntimeError(f"[pour_fabric_mimic] 자산 URDF 를 특정 못했다({_dir}): {_cands}")
        names = set(self.robot.data.joint_names)
        sides = tuple(rig.profile.hand_joint_names[0].split("_hj_")[0] for rig in self.rigs)
        pairs: dict[str, tuple[str, float]] = {}
        for j in ET.parse(_cands[0]).getroot().iter("joint"):
            m = j.find("mimic")
            dep = j.get("name") or ""
            if m is None or not any(dep.startswith(f"{s}_hj_") for s in sides) or dep not in names:
                continue
            pairs[dep] = (m.get("joint"), float(m.get("multiplier", 1.0)))
        if not pairs:
            raise RuntimeError(f"[pour_fabric_mimic] URDF 에 손 mimic 이 없다 ({_cands[0]}) — 이 트랙은 언더액추 손 전용")
        return pairs

    def _assert_mimic_constraints_present(self) -> None:
        """USD 에 PhysX mimic 제약이 gearing 값까지 맞게 붙어 있는지(09.02: headless 빌드가 12개를 잃었다)."""
        import isaacsim.core.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        root = str(self.cfg.robot_cfg.prim_path).replace("env_.*", "env_0")
        found: dict[str, float] = {}
        for p in stage.Traverse():
            if not str(p.GetPath()).startswith(root):
                continue
            a = p.GetAttribute(_MIMIC_GEARING_ATTR)
            if a and a.Get() is not None:
                found[p.GetName()] = float(a.Get())
        bad = []
        for dep, (_lead, mult) in self._mimic_pairs.items():
            got = found.get(dep)
            if got is None:
                bad.append(f"{dep}: mimic 제약 없음")
            elif abs(got + mult) > 1e-3:
                bad.append(f"{dep}: gearing {got:.4f} ≠ −배율 {-mult:.4f}")
        if bad:
            raise RuntimeError("[pour_fabric_mimic] USD 언더액추 결합이 깨졌다 — " + " · ".join(bad))
        jn = self.robot.data.joint_names
        self._mim_dep_t = torch.tensor([jn.index(d) for d in self._mimic_pairs], device=self.device, dtype=torch.long)
        self._mim_lead_t = torch.tensor([jn.index(v[0]) for v in self._mimic_pairs.values()], device=self.device, dtype=torch.long)
        self._mim_mult = torch.tensor([v[1] for v in self._mimic_pairs.values()], device=self.device)
        print(f"[pour_fabric_mimic] 언더액추 확인 — 구동 {sum(len(r.syn_ids) for r in self.rigs)} · "
              f"종속 {len(self._mimic_pairs)}(PhysX mimic, gearing 대조 통과)", flush=True)

    def _assert_hand_pose_usable(self) -> None:
        """grip 이 한계 밖이거나 open==grip 이면 그 손가락은 지표에 흔적 없이 죽는다(09.02)."""
        for rig in self.rigs:
            p = rig.profile
            _c = rig.syn_grip.clamp(rig.syn_lo, rig.syn_hi)
            bad = [f"{n}: grip {float(rig.syn_grip[i]):.3f} → clamp {float(_c[i]):.3f}"
                   for i, n in enumerate(p.hand_joint_names) if abs(float(rig.syn_grip[i] - _c[i])) > 1e-6]
            if bad:
                raise RuntimeError(f"[pour_fabric_mimic:{rig.role}] grip 자세가 관절 한계 밖: " + " · ".join(bad))
            dead = [n for i, n in enumerate(p.hand_joint_names) if not bool(rig.syn_movable[i])]
            if dead:
                raise RuntimeError(f"[pour_fabric_mimic:{rig.role}] open == grip 인 손 관절 {dead}")

    def _widen_dependent_joint_limits(self) -> None:
        """종속관절 한계 확장 — mimic 제약과 한계가 동시에 만족 불가가 되는 것을 막는다(09.07 실측)."""
        m = float(self.cfg.mimic_dep_limit_margin_rad)
        if m <= 0.0:
            print("[pour_fabric_mimic] 종속관절 한계 확장 OFF (자산 값 그대로)", flush=True)
            return
        lim = self.robot.data.joint_pos_limits[:, self._mim_dep_t, :].clone()
        lim[..., 0] -= m
        lim[..., 1] += m
        self.robot.write_joint_limits_to_sim(lim, joint_ids=self._mim_dep_t.tolist())
        print(f"[pour_fabric_mimic] 종속 {int(self._mim_dep_t.numel())}관절 한계를 ±{m} rad 확장", flush=True)

    # ==================================================================
    def _build_fabric_world(self) -> dict | None:
        """fabric 장애물 = 테이블 박스 1개(두 프로필 palm 박스 합집합에서 파생)."""
        cfg = self.cfg
        if not bool(cfg.fabric_table_obstacle):
            print("[pour_fabric_mimic] ⚠fabric 테이블 장애물 OFF", flush=True)
            return None
        los = [p.palm_box_min for p in (self.pair.source, self.pair.receiver)]
        his = [p.palm_box_max for p in (self.pair.source, self.pair.receiver)]
        lo = [min(v[i] for v in los) for i in range(2)]
        hi = [max(v[i] for v in his) for i in range(2)]
        m = float(cfg.fabric_table_margin_xy)
        sx, sy = (hi[0] - lo[0]) + 2 * m, (hi[1] - lo[1]) + 2 * m
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        th = float(cfg.fabric_table_thickness)
        cz = float(cfg.table_surface_z) - 0.5 * th
        return {"table": {"env_index": "all", "type": "box", "scaling": f"{sx} {sy} {th}",
                          "transform": f"{cx} {cy} {cz} 0. 0. 0. 1."}}

    # ==================================================================
    def _setup_scene(self) -> None:
        cfg = self.cfg
        self.robot = Articulation(cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # 작업면: env_0 에 정적 프림 → clone 이 복제(grasp_s2r 규약). 마찰 재질 바인딩.
        cfg.table_spawn.func("/World/envs/env_0/Table", cfg.table_spawn, translation=(0.0, 0.0, 0.0))
        mu = float(cfg.surface_friction)
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu, restitution=0.0)
        mat.func("/World/Materials/taskSurface", mat)
        tables = find_matching_prim_paths(_cfg.TABLE_PRIM)
        if not tables:
            raise RuntimeError(f"[pour_fabric_mimic] 테이블 프림이 없다: {_cfg.TABLE_PRIM}")
        for tp in tables:
            bind_physics_material(tp, "/World/Materials/taskSurface")

        # 손가락 마디별 접촉 센서 — body 하나당 하나, **자기 컵만** 필터.
        self._sensor_store: dict = {}
        self._palm_store: dict = {}
        for role, prof, flt in (("src", self.pair.source, list(cfg.source_contact_filter)),
                                ("rcv", self.pair.receiver, list(cfg.receiver_contact_filter))):
            store: dict = {}
            for finger, bodies in prof.finger_sensor_bodies.items():
                ss = []
                for body in bodies:
                    s = ContactSensor(ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body}",
                        filter_prim_paths_expr=flt, history_length=1, track_air_time=False))
                    ss.append(s)
                    self.scene.sensors[f"contact_{role}_{finger}_{body}"] = s
                store[finger] = ss
            self._sensor_store[role] = store
            ps = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{prof.palm_body}",
                filter_prim_paths_expr=flt, history_length=1, track_air_time=False))
            self.scene.sensors[f"contact_{role}_palm"] = ps
            self._palm_store[role] = ps

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, float(cfg.ground_plane_z)))
        light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

        self.scene.clone_environments(copy_from_source=True)
        self.source_cup = RigidObject(cfg.source_cup_cfg)
        self.scene.rigid_objects["source_cup"] = self.source_cup
        self.receiver_cup = RigidObject(cfg.receiver_cup_cfg)
        self.scene.rigid_objects["receiver_cup"] = self.receiver_cup
        # 컵↔컵 접촉(09.14 s2r 충돌 신호): 소스 컵 센서를 리시버 컵으로 필터.
        self._cup_cup_sensor = ContactSensor(ContactSensorCfg(
            prim_path=_cfg.SOURCE_CUP_BODY, filter_prim_paths_expr=[_cfg.RECEIVER_CUP_BODY],
            history_length=1, track_air_time=False))
        # 부팅 검사: 강체 prim 경로가 env 수만큼 매칭돼야 한다(0 이면 센서·필터가 전부 조용히 0 이 된다).
        for expr in (_cfg.SOURCE_CUP_BODY, _cfg.RECEIVER_CUP_BODY):
            n_hit = len(find_matching_prim_paths(expr))
            if n_hit != self.num_envs:
                raise RuntimeError(f"[pour_fabric_mimic] 물체 강체 prim {expr} 매칭 {n_hit} ≠ env {self.num_envs} — "
                                   f"POUR_CUP_BODY_NAME 이 USD 와 다르다")
        self.scene.sensors["contact_cups"] = self._cup_cup_sensor
        self.beads = RigidObjectCollection(cfg.beads_cfg)
        self.scene.rigid_object_collections["beads"] = self.beads
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ==================================================================
    def _hold_mask(self) -> torch.Tensor:
        return self.episode_length_buf < int(self.cfg.hold_steps)

    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _cup_up(self, cup: RigidObject) -> torch.Tensor:
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        return quat_apply(cup.data.root_quat_w, z)

    def _mouth(self, cup: RigidObject) -> torch.Tensor:
        off = torch.zeros(self.num_envs, 3, device=self.device)
        off[:, 2] = float(self.cfg.cup_mouth_z)
        return self._local(cup.data.root_pos_w + quat_apply(cup.data.root_quat_w, off))

    def _close_gate(self, rig: SideRig, cup: RigidObject, grasped: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if not bool(cfg.close_gate_enabled):
            return torch.ones(self.num_envs, device=self.device)
        d = (rig.palm_pos() - self._local(cup.data.root_pos_w)).norm(dim=-1)
        r = float(cfg.close_gate_radius)
        ramp = max(float(cfg.close_gate_ramp) * r, 1e-6)
        g = ((r - d) / ramp).clamp(0.0, 1.0)
        return torch.where(grasped, torch.ones_like(g), g)    # 파지 후 해제(들고 갈 때 잠기지 않게)

    # ==================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        active = ~self._hold_mask()
        h = self._half
        for i, (rig, cup, grasped) in enumerate(
                ((self.src, self.source_cup, self._src_grasped),
                 (self.rcv, self.receiver_cup, self._rcv_grasped))):
            a = self.actions[:, i * h:(i + 1) * h]
            w = self._arm_w
            if rig.joint_arm:
                rig.step_joint_arm(a[:, :w], active)
            elif self._palm_increment:
                alpha = float(self.cfg.palm_action_ema_alpha)
                self._palm_cmd[:, i] = alpha * a[:, :6] + (1.0 - alpha) * self._palm_cmd[:, i]
                rig.step_palm_target(self._palm_cmd[:, i], self._palm_vmax[i], self._policy_dt,
                                     self._palm_leash[i], active)
            elif self._palm_workspace:
                rig.workspace_palm_target(a[:, :6], active)
            else:
                rig.compose_palm_target(a[:, :6], active)
            prev = rig.syn_target
            gate = self._close_gate(rig, cup, grasped) * active.float()
            rig.syn_target = rig.synergy_targets(a[:, w:], gate)
            rig.syn_vel = (rig.syn_target - prev) / self._policy_dt
            rig.sync_fabric_hand()
            rig.push_command()                 # ★09.30 지연된 목표를 물리로(끄면 그대로)
        # 양팔 26 DOF 를 두 인스턴스가 나눠 가진다 — 적분 전에 반대팔 슬라이스를 상대 상태로 맞춘다.
        self.src.sync_other(self.rcv)
        self.rcv.sync_other(self.src)
        if not self.src.joint_arm:                       # joint 모드는 fabric 을 적분하지 않는다
            self._apply_fabric_adr()
            for rig in self.rigs:
                q_pin = rig.fabric_q
                rig.step_fabric(self._world_ids, self._world_indicator)
                rig.pin_fabric(~active, q_pin)
        self._apply_wrench()

    def _apply_fabric_adr(self) -> None:
        """★09.28 DEXTRAH ADR custom: fabric 감쇠·PD 속도 목표 계수를 ADR 진행도로 옮긴다(cfg `fabric_adr`)."""
        if not bool(self.cfg.fabric_adr):
            return
        gain = self.adr.get_param("fabric", "damping")
        ff = self.adr.get_param("fabric", "velocity_ff")
        for rig in self.rigs:
            rig.fabric_damping.fill_(gain)
            rig.velocity_ff_scale = ff

    def _apply_action(self) -> None:
        for rig in self.rigs:
            rig.apply_action()
        if self._grav_comp > 0.0:
            tau = self.robot.root_physx_view.get_gravity_compensation_forces()
            self.robot.set_joint_effort_target(self._grav_comp * tau[:, : self.robot.num_joints])

    # ==================================================================
    def _tactile(self, rig: SideRig, noisy: bool) -> torch.Tensor:
        """손끝 촉각 5칸 [N] — 실기 TouchData1.finger_forces 대응(사용자 결정 09.14). 실기 포화 10.24 N 에 맞춰 자른다."""
        f = rig.tip_tactile()
        if noisy:
            f = f + torch.randn_like(f) * float(self.cfg.tactile_obs_noise_n)
        f = f.clamp(0.0, float(self.cfg.tactile_obs_clip_n))
        k = float(self.cfg.tactile_obs_tanh_n)
        return torch.tanh(f / k) if k > 0.0 else f

    def _side_obs(self, rig: SideRig, cup_p: torch.Tensor, cup_q: torch.Tensor,
                  noisy: bool) -> list[torch.Tensor]:
        """한 팔의 actor 관측. cup_p/cup_q 는 **지각된**(perceived: 지연+노이즈) 컵 pose 다.

        ★09.14 s2r: hand_qd 없음(실기 드라이버 velocity ≠ 관절속도). noisy=True 면 관절·FK 에
          상시 노이즈. critic 은 noisy=False + 참 pose 로 같은 함수를 부른다.
        """
        cfg = self.cfg
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        palm = rig.palm_pos()
        R = rig.palm_R()
        tips = rig.tips_pos()
        arm_q, arm_qd, hand_q = q[:, rig.arm_t], qd[:, rig.arm_t], q[:, rig.hand_t]
        if noisy:
            arm_q = arm_q + torch.randn_like(arm_q) * float(cfg.obs_noise_qpos)
            arm_qd = arm_qd + torch.randn_like(arm_qd) * float(cfg.obs_noise_qvel)
            hand_q = hand_q + torch.randn_like(hand_q) * float(cfg.obs_noise_qpos)
            palm = palm + torch.randn_like(palm) * float(cfg.obs_noise_body)
            tips = tips + torch.randn_like(tips) * float(cfg.obs_noise_body)
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        cup_up = quat_apply(cup_q, z)
        return [
            arm_q, arm_qd, hand_q,
            palm, torch.cat([R[:, :, 0], R[:, :, 1]], dim=1),
        ] + ([(tips - palm.unsqueeze(1)).reshape(self.num_envs, -1)] if bool(cfg.obs_tips_rel_palm) else []) + [
            cup_p - palm,
            (tips - cup_p.unsqueeze(1)).reshape(self.num_envs, -1),
            rig.joint_err(),
            cup_up,
        ] + ([rig.arm_q_target] if rig.joint_arm
             else [rig.fabric_arm_q()] if bool(cfg.obs_fabric_state) else [])

    def _mouth_from(self, cup_p: torch.Tensor, cup_q: torch.Tensor) -> torch.Tensor:
        off = torch.zeros(self.num_envs, 3, device=self.device)
        off[:, 2] = float(self.cfg.cup_mouth_z)
        return cup_p + quat_apply(cup_q, off)

    def _get_observations(self) -> dict:
        # ---- actor: 지각된 컵 pose(지연+노이즈) + 노이즈 관절/FK, hand_qd 없음 ----------------
        sp, sq = self._perceive("src", self.source_cup)
        rp, rq = self._perceive("rcv", self.receiver_cup)
        self._perc_flush[:] = False
        parts = self._side_obs(self.src, sp, sq, noisy=True) + self._side_obs(self.rcv, rp, rq, noisy=True)
        parts += [rp - sp] + ([self._mouth_from(rp, rq) - self._mouth_from(sp, sq)] if bool(self.cfg.obs_mouth_diff) else []) + [
                  self._tactile(self.src, noisy=True), self._tactile(self.rcv, noisy=True), self.prev_actions]
        obs = torch.cat(parts, dim=1)

        # ---- critic: 참값(clean) + hand_qd + 비드 GT + 속도 + 접촉력 -------------------------
        tp_s, tq_s = self._local(self.source_cup.data.root_pos_w), self.source_cup.data.root_quat_w
        tp_r, tq_r = self._local(self.receiver_cup.data.root_pos_w), self.receiver_cup.data.root_quat_w
        clean = self._side_obs(self.src, tp_s, tq_s, noisy=False) + self._side_obs(self.rcv, tp_r, tq_r, noisy=False)
        clean += [tp_r - tp_s] + ([self._mouth(self.receiver_cup) - self._mouth(self.source_cup)] if bool(self.cfg.obs_mouth_diff) else []) + [
                  self._tactile(self.src, noisy=False), self._tactile(self.rcv, noisy=False), self.prev_actions]
        qd = self.robot.data.joint_vel
        bead_fracs = torch.stack([self._prev_in_src, self._prev_in_tgt, self._prev_spill,
                                  self._crossed.float().mean(dim=-1)], dim=1)
        centroid_rel = self.beads.data.object_pos_w.mean(dim=1) - self.receiver_cup.data.root_pos_w
        centroid_local = quat_apply_inverse(self.receiver_cup.data.root_quat_w, centroid_rel)
        state = torch.cat(clean + [
            qd[:, self.src.hand_t], qd[:, self.rcv.hand_t],
            bead_fracs, centroid_local,
            self.source_cup.data.root_lin_vel_w, self.source_cup.data.root_ang_vel_w,
            self.receiver_cup.data.root_lin_vel_w, self.receiver_cup.data.root_ang_vel_w,
            (self.episode_length_buf.float() / float(self.max_episode_length)).unsqueeze(1),
            self.src.finger_forces().clamp(max=float(self.cfg.contact_obs_clip)),
            self.rcv.finger_forces().clamp(max=float(self.cfg.contact_obs_clip)),
        ], dim=1)
        return {"policy": torch.nan_to_num(obs), "critic": torch.nan_to_num(state)}

    # ==================================================================
    def _ctx_palm_axes(self, rig: SideRig) -> torch.Tensor:
        """보상 입력 palm_axes (N,6) = [손바닥 법선, 두 번째 축]. 계약상 앞 3칸이 법선이다(09.15, cfg 주석 참조)."""
        R = rig.palm_R()
        return torch.cat([R[:, :, int(self.cfg.ctx_palm_normal_col)], R[:, :, int(self.cfg.ctx_palm_second_col)]], dim=1)

    def _build_context(self, flags, d_in_target, d_spill) -> RewardContext:
        cfg = self.cfg
        src_f, rcv_f = self.src.finger_forces(), self.rcv.finger_forces()
        self._src_grasped = self.src.grasped(src_f)
        self._rcv_grasped = self.rcv.grasped(rcv_f)
        scup, rcup = self.source_cup, self.receiver_cup
        s_up, r_up = self._cup_up(scup), self._cup_up(rcup)

        def _tilt(up):
            return torch.acos(up[:, 2].clamp(-1.0, 1.0))
        return RewardContext(
            table_z=float(cfg.table_surface_z), cup_radius=float(cfg.cup_inner_radius),
            cup_mouth_z=float(cfg.cup_mouth_z), cup_bottom_z=-float(cfg.object_origin_offset_z),
            num_beads=int(cfg.bead_count),
            src_palm_pos=self.src.palm_pos(), src_palm_lin_vel=self.src.palm_lin_vel(),
            src_palm_axes=self._ctx_palm_axes(self.src),
            src_tips_pos=self.src.tips_pos(), src_hand_closure=self.src.closure(),
            src_finger_force=src_f, src_palm_force=self.src.palm_force(),
            src_grasped=self._src_grasped, src_arm_qd=self.robot.data.joint_vel[:, self.src.arm_t],
            src_arm_limit_margin=self._arm_limit_margin(self.src), src_arm_qdd=self._arm_qdd(self.src),
            rcv_palm_pos=self.rcv.palm_pos(), rcv_palm_lin_vel=self.rcv.palm_lin_vel(),
            rcv_palm_axes=self._ctx_palm_axes(self.rcv),
            rcv_tips_pos=self.rcv.tips_pos(), rcv_hand_closure=self.rcv.closure(),
            rcv_finger_force=rcv_f, rcv_palm_force=self.rcv.palm_force(),
            rcv_grasped=self._rcv_grasped, rcv_arm_qd=self.robot.data.joint_vel[:, self.rcv.arm_t],
            rcv_arm_limit_margin=self._arm_limit_margin(self.rcv), rcv_arm_qdd=self._arm_qdd(self.rcv),
            src_cup_pos=self._local(scup.data.root_pos_w), src_cup_quat=scup.data.root_quat_w,
            src_cup_up=s_up, src_cup_tilt=_tilt(s_up), src_cup_mouth_pos=self._mouth(scup),
            src_cup_lin_vel=scup.data.root_lin_vel_w, src_cup_ang_vel=scup.data.root_ang_vel_w,
            src_cup_spawn_pos=self._src_spawn,
            rcv_cup_pos=self._local(rcup.data.root_pos_w), rcv_cup_quat=rcup.data.root_quat_w,
            rcv_cup_up=r_up, rcv_cup_tilt=_tilt(r_up), rcv_cup_mouth_pos=self._mouth(rcup),
            rcv_cup_lin_vel=rcup.data.root_lin_vel_w, rcv_cup_ang_vel=rcup.data.root_ang_vel_w,
            rcv_cup_spawn_pos=self._rcv_spawn,
            bead_in_source_frac=flags.in_source_frac, bead_in_target_frac=flags.in_target_frac,
            bead_spill_frac=flags.spill_frac, bead_centroid=self._local(flags.centroid_w),
            d_in_target=d_in_target, d_spill=d_spill,
            cup_cup_force=self._cup_cup_sensor.data.force_matrix_w.view(self.num_envs, -1, 3).sum(dim=1).norm(dim=-1),
            src_hand_foreign_force=self.src.foreign_force(),
            rcv_hand_foreign_force=self.rcv.foreign_force(),
            cups_nested=self._cups_nested, success=self._success_now,
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self.prev_actions,
        )

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        flags = compute_bead_flags(
            bead_pos_w=self.beads.data.object_pos_w,
            source_pos_w=self.source_cup.data.root_pos_w, source_quat_w=self.source_cup.data.root_quat_w,
            target_pos_w=self.receiver_cup.data.root_pos_w, target_quat_w=self.receiver_cup.data.root_quat_w,
            geom_source=self._geom, geom_target=self._geom,
            prev_target_local_z=self._prev_tgt_z, crossed_mask=self._crossed)
        fresh = self._flags_fresh.float()
        d_in_target = (1.0 - fresh) * (flags.in_target_frac - self._prev_in_tgt)
        d_spill = (1.0 - fresh) * (flags.spill_frac - self._prev_spill)
        self._prev_in_tgt, self._prev_in_src, self._prev_spill = (
            flags.in_target_frac, flags.in_source_frac, flags.spill_frac)
        self._prev_tgt_z, self._crossed = flags.target_local_z, flags.crossed_mask
        self._flags_fresh[:] = False

        xy = (self.source_cup.data.root_pos_w[:, :2] - self.receiver_cup.data.root_pos_w[:, :2]).norm(dim=-1)
        # ★09.13 hacking 차단: 소스 컵을 리시버에 끼워 넣으면(원점 거리 < 9 cm) 성공이 아니다.
        center_d = (self.source_cup.data.root_pos_w - self.receiver_cup.data.root_pos_w).norm(dim=-1)
        self._cups_nested = center_d < float(cfg.cups_nested_dist)
        self._cups_center_dist = center_d
        self._success_now = ((flags.in_target_frac >= float(cfg.success_fill_ratio))
                             & (flags.spill_frac <= float(cfg.success_spill_max))
                             & (xy < float(cfg.success_xy_thresh))
                             & (~self._cups_nested))
        self._success_streak = torch.where(self._success_now, self._success_streak + 1,
                                           torch.zeros_like(self._success_streak))

        ctx = self._build_context(flags, d_in_target, d_spill)
        if self._bank.capturing:
            self._bank.maybe_capture(flags.in_source_frac)
        total, terms = call_reward_fn(self._reward_fn, ctx)
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        # hold 중엔 보상 0(팔이 고정된 구간의 보상은 정책과 무관하다).
        total = total * (~self._hold_mask()).float()
        self._last_terms = terms

        self.extras["action/step_delta"] = (self.actions - self.prev_actions).abs().mean()
        self.prev_actions.copy_(self.actions)
        # ADR: 순간 성공률 트리거(누적 평균은 관성으로 안 오른다 — adr.py 규약)
        if self.adr.maybe_increment(float(self._success_now.float().mean())):
            print(f"[pour_fabric_mimic][ADR] 증분 {self.adr.increment_counter}/{self.adr.num_increments} "
                  f"(progress {self.adr.progress:.2f})", flush=True)
        self._log(total, terms, flags, ctx)
        return total

    # ★09.23 사용자 지시 "관절을 극단적으로 꺾고 팔이 흔들린다" — 보상이 직접 볼 수 있게 두 신호를 낸다.
    #   여유는 soft limit 기준 정규화(1 = 범위 중앙, 0 = 한계). 가속도는 관절속도 차분/스텝(팔 떨림).
    #   실기에서도 관절 각도·속도는 읽히므로 s2r 계약을 깨지 않는다(관측 obs 는 그대로).
    def _arm_limit_margin(self, rig) -> torch.Tensor:
        lim = self.robot.data.soft_joint_pos_limits[:, rig.arm_t]      # (N,A,2)
        q = self.robot.data.joint_pos[:, rig.arm_t]
        half = 0.5 * (lim[..., 1] - lim[..., 0]).clamp(min=1e-6)
        mid = 0.5 * (lim[..., 1] + lim[..., 0])
        return (1.0 - (q - mid).abs() / half).clamp(0.0, 1.0)

    def _arm_qdd(self, rig) -> torch.Tensor:
        qd = self.robot.data.joint_vel[:, rig.arm_t]
        prev = self._prev_arm_qd[rig.role]
        out = (qd - prev) / float(self.step_dt)
        prev.copy_(qd)
        return out

    def trace_snapshot(self) -> dict:
        """play `--trace_steps` 용 스텝 스냅샷(전 env, CPU float32). 09.18 "손이 왜 안 오므리나" 진단.

        읽기 전용 — 학습 경로는 부르지 않는다. 손가락별 힘·오므림 지령·손끝-컵 상대 위치·보상 항을 남긴다.
        """
        def _np(t: torch.Tensor):
            return t.detach().float().cpu().numpy()

        snap = {"actions": _np(self.actions), "hold": _np(self._hold_mask().float())}
        for tag, rig, cup, grasped in (("src", self.src, self.source_cup, self._src_grasped),
                                       ("rcv", self.rcv, self.receiver_cup, self._rcv_grasped)):
            cup_l = self._local(cup.data.root_pos_w)          # 손끝·손바닥이 env-local 이라 컵도 local
            snap.update({
                f"{tag}_f": _np(rig.finger_forces()), f"{tag}_f_palm": _np(rig.palm_force()),
                f"{tag}_foreign": _np(rig.foreign_force()),
                f"{tag}_syn_close": _np(rig.syn_close), f"{tag}_closure": _np(rig.closure()),
                f"{tag}_grasped": _np(grasped.float()),
                f"{tag}_close_gate": _np(self._close_gate(rig, cup, grasped)),
                f"{tag}_tips_rel": _np(rig.tips_pos() - cup_l.unsqueeze(1)),
                f"{tag}_palm_rel": _np(rig.palm_pos() - cup_l),
                f"{tag}_arm_qd": _np(self.robot.data.joint_vel[:, rig.arm_t]),
                f"{tag}_cup_pos": _np(cup_l),
                f"{tag}_cup_up": _np(self._cup_up(cup)),
                # 09.22 "팔이 관절 한계까지 밀려 있고 흔들린다"(사용자 영상) 진단: 관절 각도·명령 palm·실제 palm·컵 각속도
                f"{tag}_arm_q": _np(self.robot.data.joint_pos[:, rig.arm_t]),
                f"{tag}_palm_cmd": _np(torch.cat([rig.palm_targets[:, :3] + rig.fab_to_env,
                                                  rig.palm_targets[:, 3:]], dim=1)),
                f"{tag}_palm_pose": _np(rig.palm_pose_6d()),
                f"{tag}_cup_angvel": _np(cup.data.root_ang_vel_w),
            })
        # 09.22 "비드가 컵을 통과하는 것처럼 보인다"(사용자 영상) 진단: 비드 위치를 두 컵 로컬 프레임으로.
        #   소스를 떠나는 스텝에 로컬 z 가 입구보다 아래이고 xy 가 벽 밖이면 벽으로 나간 것이다.
        bead_w = self.beads.data.object_pos_w
        snap["bead_src_local"] = _np(_bead_local(bead_w, self.source_cup.data.root_pos_w,
                                                 self.source_cup.data.root_quat_w))
        snap["bead_tgt_local"] = _np(_bead_local(bead_w, self.receiver_cup.data.root_pos_w,
                                                 self.receiver_cup.data.root_quat_w))
        for k, v in self._last_terms.items():
            snap[f"term_{k}"] = _np(v)
        # ★09.28 "리셋 순간 수백 N — 손가락 관통인가" 진단: mimic 결합 오차·종속 속도, 종료 사유, 손가락 마디별 컵 힘,
        #   손가락 센서 바디의 컵 로컬 위치(관통 판정: 컵 축 거리 < 컵 반지름 & 높이가 컵 안).
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        snap["mim_err"] = _np((q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs())
        snap["mim_dep_qd"] = _np(qd[:, self._mim_dep_t].abs())
        for k, v in getattr(self, "_last_done", {}).items():
            snap[f"done_{k}"] = _np(v.float())
        for tag, rig, cup in (("src", self.src, self.source_cup), ("rcv", self.rcv, self.receiver_cup)):
            mid, dist, tip = rig.finger_link_forces()
            snap[f"{tag}_link_mid"], snap[f"{tag}_link_tip"] = _np(mid), _np(tip)
            ids = getattr(rig, "_diag_body_ids", None)
            if ids is None:
                names = [b for f in rig.fingers for b in rig.profile.finger_sensor_bodies[f]]
                rig._diag_body_ids = ids = torch.tensor([rig._one_body(b) for b in names], device=self.device)
                rig._diag_body_names = names
            bp = self.robot.data.body_pos_w[:, ids] - cup.data.root_pos_w.unsqueeze(1)
            cq = cup.data.root_quat_w.unsqueeze(1).expand(-1, bp.shape[1], -1).reshape(-1, 4)
            snap[f"{tag}_link_pos_cup"] = _np(quat_apply_inverse(cq, bp.reshape(-1, 3)).reshape(bp.shape))
        return snap

    def trace_meta(self) -> dict:
        """play `--trace_steps` 가 `<trace>_meta.json` 으로 저장한다. 스텝마다 변하지 않는 기준값."""
        lim = self.robot.data.soft_joint_pos_limits[0]
        out = {"bead_geom": {"inner_radius": self._geom.inner_radius, "inside_z_min": self._geom.inside_z_min,
                             "inside_z_max": self._geom.inside_z_max, "mouth_z": self._geom.mouth_z},
               "dt_env": float(self.step_dt)}
        for tag, rig in (("src", self.src), ("rcv", self.rcv)):
            out[f"{tag}_arm_joint_names"] = [self.robot.joint_names[i] for i in rig.arm_ids]
            # ★09.30 s2r(SIME2REAL 세션): 배포 계약의 손 관측 순서 확인용 — actor 관측 hand_q 순서(PhysX)와 손 액션 → 관절 대응
            out[f"{tag}_hand_obs_joint_names"] = [self.robot.joint_names[i] for i in rig.hand_ids]
            out[f"{tag}_hand_target_joint_names"] = [self.robot.joint_names[i] for i in rig.syn_ids]
            out[f"{tag}_hand_action_slot"] = [int(v) for v in rig.syn_slot.tolist()]
            out[f"{tag}_arm_lim"] = lim[rig.arm_t].detach().float().cpu().tolist()
            out[f"{tag}_palm_box_lo"] = [float(v) for v in rig.box_lo]
            out[f"{tag}_palm_box_hi"] = [float(v) for v in rig.box_hi]
        out["mim_dep_names"] = list(self._mimic_pairs.keys())
        out["mim_lead_names"] = [v[0] for v in self._mimic_pairs.values()]
        for tag, rig in (("src", self.src), ("rcv", self.rcv)):
            out[f"{tag}_link_body_names"] = [b for f in rig.fingers for b in rig.profile.finger_sensor_bodies[f]]
        return out

    def _log_grasp_shape(self, ctx: RewardContext) -> None:
        """★09.29 파지 모양·접근 방향 로그(rh_aglt shape_metrics 사본) — 쥔 스텝·안 쥔 스텝 따로 평균. 키 shape/{src,rcv}_*.
        중심선 쪽 법선 부호: 소스(오른팔, 컵 y<0)는 +y, 리시버(왼팔)는 −y."""
        from .shape_metrics import grasp_shape
        thr = float(self.cfg.contact_force_threshold)
        # 놓침(쥠→안 쥠) 비율은 매 스텝 누적해야 맞다 — 가벼운 비교 두 번
        for side, grasped in (("src", self._src_grasped), ("rcv", self._rcv_grasped)):
            prev = self._shape_prev_g[side]
            self.extras[f"shape/{side}_release_rate"] = (prev & ~grasped).float().mean()
            self._shape_prev_g[side] = grasped.clone()
        # ★09.29 사용자 "영상 없이 로그로, 병목 없이": 모양 지표는 shape_log_every 스텝마다만 계산(나머지 스텝은 키 없음)
        self._shape_tick += 1
        if self._shape_tick % max(int(self.cfg.shape_log_every), 1) != 0:
            return
        for side, rig, grasped, sign in (("src", self.src, self._src_grasped, 1.0), ("rcv", self.rcv, self._rcv_grasped, -1.0)):
            mids, _dists, _tips = rig.finger_link_forces()
            vals = grasp_shape(grasped=grasped, palm_pos=getattr(ctx, f"{side}_palm_pos"),
                               palm_normal=getattr(ctx, f"{side}_palm_axes")[:, 0:3],
                               finger_dir=getattr(ctx, f"{side}_palm_axes")[:, 3:6],
                               tips_pos=getattr(ctx, f"{side}_tips_pos"), finger_force=getattr(ctx, f"{side}_finger_force"),
                               prox_force=mids, cup_pos=getattr(ctx, f"{side}_cup_pos"), cup_up=getattr(ctx, f"{side}_cup_up"),
                               cup_radius=float(self.cfg.cup_inner_radius), cup_mouth_z=float(self.cfg.cup_mouth_z),
                               closure=getattr(ctx, f"{side}_hand_closure"), thr=thr, mid_sign=sign)
            for k, v in vals.items():
                self.extras["shape/" + side + "_" + k.split("/", 1)[1]] = v
            # 컵 좌표 파지 자세(기울여도 같은 파지면 같은 값) — 쥔 스텝 전체와 컵 20° 이상 기운 쥔 스텝을 따로
            up = getattr(ctx, f"{side}_cup_up")
            ax = getattr(ctx, f"{side}_palm_axes")
            n_dot = (ax[:, 0:3] * up).sum(-1).abs()
            f_dot = (ax[:, 3:6] * up).sum(-1).abs()
            n_touch = (getattr(ctx, f"{side}_finger_force") > thr).float().sum(-1)
            tilted = grasped & (getattr(ctx, f"{side}_cup_tilt") > math.radians(20.0))
            for tag, msk in (("", grasped), ("tilt_", tilted)):
                if bool(msk.any()):
                    w = msk.float(); k_ = w.sum()
                    self.extras[f"shape/{side}_{tag}n_dot_cupaxis"] = (n_dot * w).sum() / k_
                    self.extras[f"shape/{side}_{tag}f_dot_cupaxis"] = (f_dot * w).sum() / k_
                    self.extras[f"shape/{side}_{tag}fingers5"] = (n_touch * w).sum() / k_
            self.extras[f"shape/{side}_tilted_grasp_frac"] = tilted.float().mean()

    def _log_thumb_rim(self, side: str, palm, tips, cup_pos, cup_up) -> None:
        """엄지 입구 걸림 접근 계측 — 보상·관측과 무관, env 가 직접 잰다(09.15 사용자 "지표로깅으로 확인 가능하게").

        near = palm↔컵 원점 < `thumb_rim_near_m`. 걸림 = 엄지 끝(tips 0번)의 컵 축방향 높이 ≥ 입구 − `thumb_rim_band_m`
        이고 반경 < 컵 벽(내경 + 4 mm) + `thumb_rim_radial_margin_m`(= 입구 위). iter_03 `rim_hook` 과 같은 기하.
        """
        cfg = self.cfg
        rel = tips[:, 0, :] - cup_pos
        axial = (rel * cup_up).sum(dim=-1)
        radial = (rel - axial.unsqueeze(-1) * cup_up).norm(dim=-1)
        near = (palm - cup_pos).norm(dim=-1) < float(cfg.thumb_rim_near_m)
        hook = near & (axial >= float(cfg.cup_mouth_z) - float(cfg.thumb_rim_band_m)) \
            & (radial < float(cfg.cup_inner_radius) + 0.004 + float(cfg.thumb_rim_radial_margin_m))
        n_near = near.float().sum().clamp(min=1.0)
        self.extras[f"task/{side}_near_rate"] = near.float().mean()
        self.extras[f"task/{side}_thumb_over_rim_near"] = hook.float().sum() / n_near
        self.extras[f"task/{side}_thumb_above_rim_mm_near"] = \
            1000.0 * torch.where(near, axial - float(cfg.cup_mouth_z), torch.zeros_like(axial)).sum() / n_near

    def _log_grasp_pocket(self, side: str, palm, tips, cup_pos, cup_up) -> None:
        """파지 포켓 계측 — 보상·관측과 무관, env 가 직접 잰다(09.16 사용자 "대향 여부를 지표로").

        라운드 7 영상에서 배치가 **검지-엄지-컵** 이었다(엄지가 컵 반대편이 아니라 검지 쪽).
        그 자세는 손가락을 굽혀도 컵을 잡을 수 없는데 보상의 `pinch_geo` 는 0.19~0.38 을 지불한다.
          oppose     = 엄지 끝 반경방향과 검지/중지 평균 반경방향이 반대인 정도(1 = 정확히 맞은편)
          tip_gap    = 엄지 끝 ↔ 가까운 4지 끝 거리(mm) — 컵 지름과 비교한다
          in_pocket  = 컵 축이 두 끝을 잇는 선분에서 `cup_inner_radius + pocket_axis_margin_m`
                       안이고, 두 끝이 모두 내경 밖(= 컵이 실제로 사이에 있다)
        near 는 palm↔컵 원점 < `pocket_near_m` 인 env 로 한정한다.
        """
        cfg = self.cfg

        def _axis_seg(a, b):
            """컵 축(반경평면 원점)에서 선분 ab 까지의 거리."""
            ab = b - a
            t = torch.clamp(-(a * ab).sum(dim=-1) / (ab.pow(2).sum(dim=-1) + 1e-9), 0.0, 1.0)
            return (a + t.unsqueeze(-1) * ab).norm(dim=-1)

        rel = tips - cup_pos.unsqueeze(1)
        axial = (rel * cup_up.unsqueeze(1)).sum(dim=-1)
        rv = rel - axial.unsqueeze(-1) * cup_up.unsqueeze(1)
        r = rv.norm(dim=-1)
        d = rv / (r.unsqueeze(-1) + 1e-6)
        fdir = d[:, 1:3, :].mean(dim=1)
        fdir = fdir / (fdir.norm(dim=-1, keepdim=True) + 1e-6)
        oppose = torch.clamp(-(d[:, 0, :] * fdir).sum(dim=-1), 0.0, 1.0)
        gap = torch.minimum((tips[:, 0] - tips[:, 1]).norm(dim=-1),
                            (tips[:, 0] - tips[:, 2]).norm(dim=-1))
        seg = torch.minimum(_axis_seg(rv[:, 0, :], rv[:, 1, :]), _axis_seg(rv[:, 0, :], rv[:, 2, :]))
        r_in = float(cfg.cup_inner_radius)
        outside = (r[:, 0] > r_in) & (r[:, 1:3].min(dim=-1).values > r_in)
        in_pocket = outside & (seg < r_in + float(cfg.pocket_axis_margin_m))
        near = (palm - cup_pos).norm(dim=-1) < float(cfg.pocket_near_m)
        n_near = near.float().sum().clamp(min=1.0)
        self.extras[f"task/{side}_thumb_oppose_near"] = (oppose * near).sum() / n_near
        self.extras[f"task/{side}_tip_gap_mm_near"] = 1000.0 * (gap * near).sum() / n_near
        self.extras[f"task/{side}_cup_in_pocket_near"] = (in_pocket & near).float().sum() / n_near

    def _log(self, total, terms, flags, ctx: RewardContext) -> None:
        cfg = self.cfg
        for k, v in terms.items():
            self.extras[f"reward/{k}"] = v.mean()
        self.extras["reward/total"] = total.mean()
        self.extras["task/success_now"] = self._success_now.float().mean()
        self.extras["task/episode_success"] = (
            self._success_streak >= int(cfg.success_hold_steps)).float().mean()
        self.extras["task/src_grasped"] = self._src_grasped.float().mean()
        self.extras["task/rcv_grasped"] = self._rcv_grasped.float().mean()
        self.extras["task/src_closure"] = ctx.src_hand_closure.mean()
        self.extras["task/rcv_closure"] = ctx.rcv_hand_closure.mean()
        self.extras["task/src_cup_lift"] = (ctx.src_cup_pos[:, 2] - self._src_spawn[:, 2]).mean()
        self.extras["task/rcv_cup_lift"] = (ctx.rcv_cup_pos[:, 2] - self._rcv_spawn[:, 2]).mean()
        self.extras["task/src_tilt_deg"] = torch.rad2deg(ctx.src_cup_tilt).mean()
        self.extras["task/rcv_tilt_deg"] = torch.rad2deg(ctx.rcv_cup_tilt).mean()
        self.extras["task/aim_dist"] = (ctx.src_cup_mouth_pos - ctx.rcv_cup_mouth_pos).norm(dim=-1).mean()
        self.extras["task/cups_center_dist"] = self._cups_center_dist.mean()
        self.extras["task/nested_rate"] = self._cups_nested.float().mean()
        thr = float(cfg.collision_force_threshold)
        self.extras["task/cup_collision_rate"] = (ctx.cup_cup_force > thr).float().mean()
        self.extras["task/src_hand_foreign_rate"] = (ctx.src_hand_foreign_force > thr).float().mean()
        self.extras["task/rcv_hand_foreign_rate"] = (ctx.rcv_hand_foreign_force > thr).float().mean()
        self.extras["contact/cup_cup_max"] = ctx.cup_cup_force.max()
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        _err = (q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs()
        # 09.07 기준선: 자유 폐쇄 ≤0.05 rad · 파지 ≤0.04 · 10 rad 넘는 epoch 반복이면 종속 한계 부족.
        self.extras["ctrl/mimic_err_max"] = _err.max()
        self.extras["ctrl/hand_dep_qd_max"] = qd[:, self._mim_dep_t].abs().max()
        self.extras.update(self.adr.log_dict())
        self.extras["dr/obs_object_xyz"] = self.adr.get_param("obs", "object_xyz")
        self.extras["dr/delay_steps"] = self.adr.get_param("obs", "delay_steps")
        self.extras["dr/wrench_force_scale"] = self.adr.get_param("wrench", "force_scale")
        self.extras["dr/src_mass_mean"] = self._cup_mass["src"].mean()
        self.extras["task/src_palm_to_cup"] = (ctx.src_palm_pos - ctx.src_cup_pos).norm(dim=-1).mean()
        self.extras["task/rcv_palm_to_cup"] = (ctx.rcv_palm_pos - ctx.rcv_cup_pos).norm(dim=-1).mean()
        self.extras["contact/src_max"] = ctx.src_finger_force.max(dim=1).values.mean()
        self.extras["contact/rcv_max"] = ctx.rcv_finger_force.max(dim=1).values.mean()
        self._log_thumb_rim("src", ctx.src_palm_pos, ctx.src_tips_pos, ctx.src_cup_pos, ctx.src_cup_up)
        self._log_thumb_rim("rcv", ctx.rcv_palm_pos, ctx.rcv_tips_pos, ctx.rcv_cup_pos, ctx.rcv_cup_up)
        self._log_grasp_pocket("src", ctx.src_palm_pos, ctx.src_tips_pos, ctx.src_cup_pos, ctx.src_cup_up)
        self._log_grasp_pocket("rcv", ctx.rcv_palm_pos, ctx.rcv_tips_pos, ctx.rcv_cup_pos, ctx.rcv_cup_up)
        self._log_grasp_shape(ctx)
        self.extras["bead/in_source"] = flags.in_source_frac.mean()
        self.extras["bead/in_target"] = flags.in_target_frac.mean()
        self.extras["bead/spill"] = flags.spill_frac.mean()
        self.extras["bead/crossed"] = flags.crossed_frac.mean()
        self._ep_update(ctx, flags)
        self.extras["done/drop"] = self._dropped.float().mean()
        for rig, tag in ((self.src, "src"), (self.rcv, "rcv")):
            if rig.joint_arm:
                # ★09.28 full-joint 로그: 팔 목표가 관절 한계 2 % 안에 붙은 비율, 팔·손 목표−실측 간극
                span = (rig.arm_hi - rig.arm_lo).clamp(min=1e-6)
                near_lim = ((rig.arm_q_target - rig.arm_lo) < 0.02 * span) | ((rig.arm_hi - rig.arm_q_target) < 0.02 * span)
                self.extras[f"joint/{tag}_arm_limit_frac"] = near_lim.float().mean()
                q = self.robot.data.joint_pos
                self.extras[f"joint/{tag}_arm_track_err_deg"] = torch.rad2deg(
                    (rig.arm_q_target - q[:, rig.arm_t]).abs().max(dim=1).values).mean()
                self.extras[f"joint/{tag}_hand_track_err_deg"] = torch.rad2deg(
                    (rig.syn_target - q[:, rig.syn_t]).abs().max(dim=1).values).mean()
                continue
            perr = (rig.palm_targets[:, :3] + rig.fab_to_env - rig.palm_pos()).norm(dim=-1)
            self.extras[f"fabric/{tag}_palm_err"] = perr.mean()
            # 회전 추종(각 슬롯 wrap 후 최대 절대 오차) — 붓기 tilt 지령이 실제로 도달하는지의 근거
            rerr = rig.palm_targets[:, 3:] - rig.palm_pose_6d()[:, 3:]
            rerr = torch.atan2(torch.sin(rerr), torch.cos(rerr))   # 09.17: ±π 경계 wrap(없으면 360° 가짜 오차)
            rerr = torch.remainder(rerr + math.pi, 2 * math.pi) - math.pi
            self.extras[f"fabric/{tag}_rot_err_deg"] = torch.rad2deg(rerr.abs().max(dim=1).values).mean()
        self._log_tick += 1
        every = int(cfg.console_log_interval)
        if every > 0 and self._log_tick % every == 0:
            print(f"[METRICS] step={self._log_tick:>8d} rew={total.mean():+.3f} "
                  f"gS={self._src_grasped.float().mean():.2f} gR={self._rcv_grasped.float().mean():.2f} "
                  f"liftS={float(self.extras['task/src_cup_lift']):+.3f} "
                  f"tiltS={float(self.extras['task/src_tilt_deg']):.1f} "
                  f"aim={float(self.extras['task/aim_dist']):.3f} "
                  f"inT={flags.in_target_frac.mean():.3f} inS={flags.in_source_frac.mean():.3f} "
                  f"spill={flags.spill_frac.mean():.3f} succ={self._success_now.float().mean():.3f}",
                  flush=True)
            _e = {k: float(self.extras[f"ep/{k}"]) for k in _EP_KEYS if f"ep/{k}" in self.extras}
            if _e:
                print("[EPISODE] n=%d " % int(self.extras["ep/n_done"]) + " ".join(f"{k}={v:.3f}" for k, v in _e.items()),
                      flush=True)

    # ==================================================================
    def _ep_update(self, ctx: RewardContext, flags) -> None:
        """에피소드 누적(스텝마다) + 끝난 에피소드 평균을 ep/* 로 로그. 로그 전용."""
        up = ctx.src_cup_up
        tilt = torch.rad2deg(ctx.src_cup_tilt)
        v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
        d_hat = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        u_d = (up[:, :2] * d_hat).sum(dim=-1)
        toward = torch.rad2deg(torch.atan2(u_d, up[:, 2]))          # 받는 컵 쪽 부호 있는 기울기
        h = up[:, :2]
        hn = h.norm(dim=-1).clamp(min=1e-6)
        dir_err = torch.rad2deg(torch.acos(((h / hn.unsqueeze(-1)) * d_hat).sum(-1).clamp(-1.0, 1.0)))
        lift = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
        lipdz = ctx.src_cup_mouth_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]
        c = self._ep_cur
        new_max = tilt > c["tilt_max"]
        c["tilt_max"] = torch.where(new_max, tilt, c["tilt_max"])
        c["dir_err_at_max"] = torch.where(new_max, dir_err, c["dir_err_at_max"])
        c["lipdz_at_max"] = torch.where(new_max, lipdz, c["lipdz_at_max"])
        c["toward_max"] = torch.maximum(c["toward_max"], toward)
        c["lift_max"] = torch.maximum(c["lift_max"], lift)
        c["in_target_max"] = torch.maximum(c["in_target_max"], flags.in_target_frac)
        c["in_source_min"] = torch.minimum(c["in_source_min"], flags.in_source_frac)
        c["spill_end"] = flags.spill_frac
        for k in _EP_KEYS:
            last = self._ep_last[k]
            ok = ~torch.isnan(last)
            if bool(ok.any()):
                self.extras[f"ep/{k}"] = last[ok].mean()
        self.extras["ep/n_done"] = (~torch.isnan(self._ep_last["tilt_max"])).float().sum()
        # 붓기 구간 조건부 현재 기울기: 두 컵 파지 + 소스 3 cm 이상 들었을 때만 평균(전 스텝 평균의 희석을 피한다)
        carry = self._src_grasped & self._rcv_grasped & (lift > 0.03)
        if bool(carry.any()):
            self.extras["task/src_tilt_carry_deg"] = tilt[carry].mean()
            self.extras["task/src_toward_carry_deg"] = toward[carry].mean()

    def _ep_close(self, env_ids) -> None:
        """리셋 직전: 충분히 진행된 에피소드의 누적값을 '마지막 에피소드'로 옮기고 누적을 비운다."""
        ran = self.episode_length_buf[env_ids] > int(self.cfg.hold_steps)
        ids = env_ids[ran] if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)[ran]
        for k in _EP_KEYS:
            if len(ids):
                self._ep_last[k][ids] = self._ep_cur[k][ids]
            self._ep_cur[k][env_ids] = 1.0 if k == "in_source_min" else 0.0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        floor = float(cfg.table_surface_z) + float(cfg.object_origin_offset_z) - float(cfg.drop_below_table_m)
        s_z = self._local(self.source_cup.data.root_pos_w)[:, 2]
        r_z = self._local(self.receiver_cup.data.root_pos_w)[:, 2]
        self._dropped = ((s_z < floor) | (r_z < floor)) & (~self._hold_mask())
        qd = torch.cat([self.robot.data.joint_vel[:, self.src.arm_t],
                        self.robot.data.joint_vel[:, self.rcv.arm_t]], dim=1)
        runaway = (qd.abs() > float(cfg.runaway_joint_vel)).any(dim=-1)
        # 언더액추 폭주: 종속관절 속도(mimic 제약이 깨진 서명) — 09.14 사용자 결정, 깨진 env 는 즉시 리셋.
        dep_qd = self.robot.data.joint_vel[:, self._mim_dep_t].abs().max(dim=-1).values
        # 09.14 라운드 1: 속도 기준을 빠져나간 느린 폭주 → 결합 오차 자체도 종료(사용자 결정 "2 추가").
        q = self.robot.data.joint_pos
        mim_err = (q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs().max(dim=-1).values
        # 09.17 사용자 결정: 속도 튐은 연속 K 스텝 지속일 때만 종료(충돌 순간 한 번은 통과) · 오차·비유한 상태는 즉시.
        qd_spike = dep_qd > float(cfg.mimic_runaway_dep_qd)
        self._mim_qd_streak = torch.where(qd_spike, self._mim_qd_streak + 1, torch.zeros_like(self._mim_qd_streak))
        nonfinite = ~(torch.isfinite(self.robot.data.joint_pos).all(dim=-1) & torch.isfinite(self.robot.data.joint_vel).all(dim=-1))
        mimic_runaway = ((self._mim_qd_streak >= int(cfg.mimic_runaway_qd_steps))
                         | (mim_err > float(cfg.mimic_runaway_err_rad)) | nonfinite)
        phys = self._phys_anomaly()
        terminated = runaway | mimic_runaway | self._dropped | phys["any"]
        # ★09.28 trace 진단용 종료 사유(읽기 전용 — 학습 경로는 이 값을 쓰지 않는다)
        self._last_done = {"runaway": runaway, "mimic_qd": self._mim_qd_streak >= int(cfg.mimic_runaway_qd_steps),
                           "mimic_err": mim_err > float(cfg.mimic_runaway_err_rad), "nonfinite": nonfinite,
                           "dropped": self._dropped, "phys_cup_speed": phys["cup_speed"],
                           "phys_contact": phys["contact"], "cup_far": phys["cup_far"]}
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["task/runaway_rate"] = runaway.float().mean()
        self.extras["done/mimic_runaway"] = mimic_runaway.float().mean()
        self.extras["done/mimic_err_runaway"] = (mim_err > float(cfg.mimic_runaway_err_rad)).float().mean()
        self.extras["done/mimic_qd_spike"] = qd_spike.float().mean()
        self.extras["done/phys_cup_speed"] = phys["cup_speed"].float().mean()
        self.extras["done/phys_contact"] = phys["contact"].float().mean()
        self.extras["done/cup_far"] = phys["cup_far"].float().mean()
        return terminated, truncated

    def _phys_anomaly(self) -> dict:
        """★09.29 물리 이상 종료 사유 (N,) bool — 컵 튕김·손 접촉 폭발·컵 이탈. cfg 값 0 이면 그 항은 끈다."""
        cfg = self.cfg
        off = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        live = ~self._hold_mask()
        cup_speed, cup_far, contact = off.clone(), off.clone(), off.clone()
        for cup, spawn in ((self.source_cup, self._src_spawn), (self.receiver_cup, self._rcv_spawn)):
            if float(cfg.phys_cup_speed_max) > 0.0:
                cup_speed |= cup.data.root_lin_vel_w.norm(dim=-1) > float(cfg.phys_cup_speed_max)
            if float(cfg.phys_cup_angvel_max) > 0.0:
                cup_speed |= cup.data.root_ang_vel_w.norm(dim=-1) > float(cfg.phys_cup_angvel_max)
            if float(cfg.cup_leash_m) > 0.0:
                d = (self._local(cup.data.root_pos_w)[:, :2] - spawn[:, :2]).norm(dim=-1)
                cup_far |= d > float(cfg.cup_leash_m)
        if float(cfg.phys_contact_max_n) > 0.0:
            thr = float(cfg.phys_contact_max_n)
            hot = off.clone()
            for rig in self.rigs:
                f = torch.stack([rig.finger_forces().max(dim=-1).values, rig.palm_force(), rig.foreign_force()], dim=-1)
                hot |= (f > thr).any(dim=-1)
            self._phys_f_streak = torch.where(hot, self._phys_f_streak + 1, torch.zeros_like(self._phys_f_streak))
            contact = self._phys_f_streak >= int(cfg.phys_contact_steps)
        cup_speed, cup_far, contact = cup_speed & live, cup_far & live, contact & live
        return {"cup_speed": cup_speed, "contact": contact, "cup_far": cup_far, "any": cup_speed | contact | cup_far}

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self, "_ep_cur"):
            self._ep_close(env_ids)
        super()._reset_idx(env_ids)
        n, dev, cfg = len(env_ids), self.device, self.cfg
        self._mim_qd_streak[env_ids] = 0
        for _g in self._shape_prev_g.values():
            _g[env_ids] = False
        self._phys_f_streak[env_ids] = 0

        q0 = self._reset_q.unsqueeze(0).expand(n, -1).contiguous()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        for rig in self.rigs:
            rig.reset(env_ids)
            if hasattr(rig, "_real_hand_cap"):
                rig.reset_real_control(env_ids)
        if hasattr(self, "_palm_cmd"):
            self._palm_cmd[env_ids] = 0.0                      # 증분 방식 EMA 상태 (목표는 rig.reset 이 앵커로)

        rest_z = float(cfg.table_surface_z) + float(cfg.object_origin_offset_z)
        rng = float(cfg.object_spawn_range)
        for rig, cup, store in ((self.src, self.source_cup, self._src_spawn),
                                (self.rcv, self.receiver_cup, self._rcv_spawn)):
            offs = (torch.rand(n, 2, device=dev) - 0.5) * 2.0 * rng
            pos = torch.zeros(n, 3, device=dev)
            pos[:, 0] = rig.profile.object_spawn_center[0] + float(cfg.object_spawn_dx) + offs[:, 0]
            pos[:, 1] = rig.profile.object_spawn_center[1] + offs[:, 1]
            pos[:, 2] = rest_z
            store[env_ids] = pos
            root = torch.zeros(n, 13, device=dev)
            root[:, :3] = pos + self.scene.env_origins[env_ids]
            root[:, 2] += float(cfg.object_spawn_pad)
            root[:, 3] = 1.0
            cup.write_root_state_to_sim(root, env_ids=env_ids)

        k = int(cfg.bead_count)
        bead = torch.zeros(n, k, 13, device=dev)
        bead[:, :, :3] = (self._src_spawn[env_ids].unsqueeze(1) + self._bead_offs.unsqueeze(0)
                          + self.scene.env_origins[env_ids].unsqueeze(1))
        bead[:, :, 2] += float(cfg.object_spawn_pad)
        bead[:, :, 3] = 1.0
        self.beads.write_object_state_to_sim(bead, env_ids=env_ids)

        self._perc_flush[env_ids] = True
        for w in self._wrench.values():
            w.reset(env_ids)
        self._cup_mass["src"] = self._read_cup_mass(self.source_cup)
        self._cup_mass["rcv"] = self._read_cup_mass(self.receiver_cup)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        for _buf in self._prev_arm_qd.values():
            _buf[env_ids] = 0.0
        self._success_now[env_ids] = False
        self._success_streak[env_ids] = 0
        self._dropped[env_ids] = False
        self._src_grasped[env_ids] = False
        self._rcv_grasped[env_ids] = False
        self._crossed[env_ids] = False
        self._prev_tgt_z[env_ids] = -1e6
        self._prev_in_tgt[env_ids] = 0.0
        self._prev_in_src[env_ids] = 1.0
        self._prev_spill[env_ids] = 0.0
        self._flags_fresh[env_ids] = True
        if hasattr(self, "_bank"):
            self._bank.on_reset(env_ids)
            _bank_ids = self._bank.restore(env_ids)   # 뱅크 env 는 쥔 상태·hold 건너뜀으로 덮는다
            if len(_bank_ids) > 0:
                for rig in self.rigs:
                    if hasattr(rig, "_real_hand_cap"):
                        rig.reset_real_control(_bank_ids)   # 지연 기록도 뱅크 목표로
