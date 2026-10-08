"""rh_aglt(좌·우 공통 본문) — RH56F1 한 팔: 먼 출발 → 접근 → 잡기 → 들기 → 이송(키포인트 목표열).

에피소드: 리셋(먼 출발 자세 · 편 손 · 테이블 위 컵 1개) → hold(컵 정착, 팔·손 고정) → 정책.
제어: 팔 7 = 관절 증분 + EMA(`arm_action`, cup_grasp 법칙), 손 6 = pour_bi_rh 와 같은 grip 구간 절대 목표 + 힘 멈춤(`hand_action`).
실기 반응(`real_response`, 09.30): 명령한 목표는 팔 9~12·손 3~5 스텝 늦게 PD 에 들어가고, 손은 펌웨어 힘 멈춤·편 손 하한을 거친다.
쓰지 않는 반대팔은 프로필 휴식 자세에서 PD 가 잡는다(양팔 자산 유지, 사용자 09.29 "b").
보상은 여기 없다 — `reward_context.RewardContext` 를 채워 `cfg.reward_code_path` 의 생성 코드에 넘긴다.
성공(목표 달성)·종료는 env 가 정한다: 보상이 바꿀 수 없는 기준 지표다.
"""

from __future__ import annotations

import dataclasses
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material, find_matching_prim_paths
from isaaclab.utils.math import matrix_from_quat, quat_apply

from openarm.agnostic.modules import keypoint_goal as kg
from openarm.agnostic.modules.adr import TaskADR
from openarm.agnostic.modules.object_wrench import WrenchDR
from openarm.agnostic.modules.perception_delay import noisy_pose
from openarm.common.bead_assets import bead_offsets_in_cup, make_beads_cfg

from . import mimic as _mimic
from . import rh_aglt_env_cfg as _cfg
from .arm_action import step_arm_increment
from .goals import GoalParams, GoalState
from .grasp_capture import GraspCapture
from .hand_action import closure, linear_grip_raw, step_hand_target
from .layout import actor_layout, critic_extra_layout
from .real_response import DelayLine, firmware_stop, open_floor, sample_delay, sample_open_floor
from .reward_context import RewardContext, call_reward_fn, load_reward_fn
from .shape_metrics import grasp_shape
from .spawn import sample_cup_xy

_TAG = "[rh_aglt]"


class RhAgltEnv(DirectRLEnv):
    cfg: _cfg.RhAgltEnvCfg

    # ==================================================================
    def __init__(self, cfg: _cfg.RhAgltEnvCfg, render_mode: str | None = None, **kw):
        _cfg.resolve_cfg(cfg)
        self.profile = _cfg.PROFILES[cfg.profile_name]
        self._grav_comp = float(cfg.gravity_compensation) if cfg.enable_gravity else 0.0
        self._reward_fn, self._reward_src = load_reward_fn(cfg.reward_code_path)
        super().__init__(cfg, render_mode, **kw)
        N, dev, p = self.num_envs, self.device, self.profile

        # ---- 관절·바디 인덱스 (전부 프로필에서) ----
        self.arm_ids, _ = self.robot.find_joints(p.arm_joint_regex)
        jn = self.robot.data.joint_names
        self.hand_ids = [jn.index(n) for n in p.hand_joint_names]
        if len(self.arm_ids) != p.num_arm_joints or len(self.hand_ids) != p.num_hand_joints:
            raise RuntimeError(f"{_TAG} 관절 수 불일치: arm {len(self.arm_ids)} hand {len(self.hand_ids)}")
        self.arm_t = torch.tensor(self.arm_ids, device=dev, dtype=torch.long)
        self.hand_t = torch.tensor(self.hand_ids, device=dev, dtype=torch.long)
        self.palm_idx = self._one_body(p.palm_body)
        self.fingers = list(p.finger_sensor_bodies.keys())
        self.tip_t = torch.tensor([self._one_body(b) for b in p.fingertip_bodies], device=dev, dtype=torch.long)
        link_bodies = [b for f in self.fingers for b in p.finger_sensor_bodies[f]]
        self.link_t = torch.tensor([self._one_body(b) for b in link_bodies], device=dev, dtype=torch.long)
        self.grp_a = torch.tensor([self.fingers.index(f) for f in p.contact_group_a], device=dev, dtype=torch.long)
        self.grp_b = torch.tensor([self.fingers.index(f) for f in p.contact_group_b], device=dev, dtype=torch.long)
        # 손 관절 → 손가락(힘 멈춤 스코프: 손가락당 구동관절, pour_fabric_mimic finger 스코프와 같다)
        self.hand_finger = torch.tensor(
            [next(i for i, f in enumerate(self.fingers) if f"_{f}_" in n) for n in p.hand_joint_names],
            device=dev, dtype=torch.long)

        # ---- 시작 자세·한계 ----
        self._reset_q = self.robot.data.default_joint_pos[0].clone()
        lim = self.robot.data.soft_joint_pos_limits[0]
        self.arm_lo, self.arm_hi = lim[self.arm_t, 0].clone(), lim[self.arm_t, 1].clone()
        self.q_home = self._reset_q[self.arm_t].clone()
        if bool(((self.q_home < self.arm_lo) | (self.q_home > self.arm_hi)).any()):
            raise RuntimeError(f"{_TAG} 먼 출발 팔 자세가 관절 한계 밖: {self.q_home.tolist()}")
        self.q_open = torch.tensor(p.hand_open_pose, device=dev)
        self.q_grip = torch.tensor(p.hand_grip_pose, device=dev)
        h_lo, h_hi = lim[self.hand_t, 0], lim[self.hand_t, 1]
        if bool(((torch.minimum(self.q_open, self.q_grip) < h_lo - 1e-6)
                 | (torch.maximum(self.q_open, self.q_grip) > h_hi + 1e-6)).any()):
            raise RuntimeError(f"{_TAG} 손 open/grip 자세가 관절 한계 밖")
        self.hand_cap = (h_hi - h_lo) * float(self.step_dt) / float(cfg.hand_full_range_s)   # 전 범위를 full_range_s 에
        self.arm_q_target = self.q_home.unsqueeze(0).repeat(N, 1)
        self.hand_target = self.q_open.unsqueeze(0).repeat(N, 1)

        # ---- 언더액추 계약 ----
        side = p.hand_joint_names[0].split("_hj_")[0]
        usd_abs = os.path.join(_cfg._ASSETS_DIR, p.usd_relpath)
        self._mimic_pairs = {d: v for d, v in _mimic.parse_mimic_pairs(_mimic.find_asset_urdf(usd_abs), side).items()
                             if d in jn}
        if len(self._mimic_pairs) != p.num_hand_joints:
            raise RuntimeError(f"{_TAG} 관절 목록에 있는 종속 mimic {len(self._mimic_pairs)} ≠ 구동 수 {p.num_hand_joints}")
        _mimic.assert_usd_gearing(str(cfg.robot_cfg.prim_path).replace("env_.*", "env_0"), self._mimic_pairs)
        self._mim_dep_t = torch.tensor([jn.index(d) for d in self._mimic_pairs], device=dev, dtype=torch.long)
        self._mim_lead_t = torch.tensor([jn.index(v[0]) for v in self._mimic_pairs.values()], device=dev, dtype=torch.long)
        self._mim_mult = torch.tensor([v[1] for v in self._mimic_pairs.values()], device=dev)
        self._widen_dependent_limits()

        # ---- 목표 ----
        ctr, hw = cfg.spawn_center, cfg.spawn_half
        z0 = float(cfg.table_surface_z) + float(cfg.cup_origin_offset_z)
        m = float(cfg.goal_box_xy_margin)
        seq = kg.GoalSeqCfg(
            first_xy_range=float(cfg.goal_first_xy_range), first_z_range=tuple(float(v) for v in cfg.goal_first_z_range),
            delta_distance=float(cfg.goal_delta_distance),
            box_min=(ctr[0] - hw[0] - m, ctr[1] - hw[1] - m, z0 + float(cfg.goal_box_z_range[0])),
            box_max=(ctr[0] + hw[0] + m, ctr[1] + hw[1] + m, z0 + float(cfg.goal_box_z_range[1])),
            success_steps=int(cfg.goal_success_steps), max_goals=int(cfg.goal_max))
        self.goals = GoalState(N, len(self.fingers), GoalParams(
            half_height=float(cfg.cup_half_height), seq=seq, tol_start=float(cfg.tol_start),
            tol_floor=float(cfg.tol_floor), tol_factor=float(cfg.tol_factor), tol_interval=int(cfg.tol_interval),
            tol_success_threshold=float(cfg.tol_success_threshold), tol_eval=float(cfg.tol_eval),
            lift_latch_m=float(cfg.lift_latch_m)), dev)

        # ---- 버퍼 ----
        A = cfg.action_space
        self.actions = torch.zeros(N, A, device=dev)
        self.prev_actions = torch.zeros(N, A, device=dev)
        # 09.30 보상 궤적 원장: RH_AGLT_TRACE=1 이면 매 스텝 RewardContext 를 남겨 trace_snapshot() 으로 내보낸다
        self._keep_ctx = os.environ.get("RH_AGLT_TRACE", "0") == "1"
        self._last_ctx = None
        self._spawn = torch.zeros(N, 3, device=dev)
        self._grasped = torch.zeros(N, dtype=torch.bool, device=dev)
        self._goal_out = {"kp_dist": torch.zeros(N, device=dev), "near_goal": torch.zeros(N, dtype=torch.bool, device=dev),
                          "success": torch.zeros(N, dtype=torch.bool, device=dev)}
        self._mim_qd_streak = torch.zeros(N, dtype=torch.long, device=dev)
        self._phys_f_streak = torch.zeros(N, dtype=torch.long, device=dev)
        self._last_done: dict = {}
        # 중심선(반대 손) 방향 부호: 시작 손바닥이 y<0(우팔)이면 +y 가 중심 쪽 — 로그 지표 approach_palm_to_mid 용
        self._mid_sign = 1.0 if float(self._reset_palm_y()) < 0.0 else -1.0
        self._log_tick = 0
        self._setup_dr()
        self._boot_checks()
        # 10.01 붓기 인계 수집(기본 끔): 비드 배치·쥔 상태 수집기
        self._capture = GraspCapture(self) if str(cfg.grasp_capture_path) else None
        print(f"{_TAG} profile={p.name} usd={p.usd_relpath} action={A} obs={cfg.observation_space} "
              f"critic={cfg.state_space} 종속 mimic {len(self._mimic_pairs)} · reward={self._reward_src}", flush=True)

    # ==================================================================
    def _one_body(self, name: str) -> int:
        ids, _ = self.robot.find_bodies(name)
        if len(ids) != 1:
            raise RuntimeError(f"{_TAG} body '{name}' 해석 실패: {ids}")
        return ids[0]

    def _reset_palm_y(self) -> float:
        """리셋 자세 FK 없이 부호만 — 프로필 소환 중심 y 가 같은 쪽이다(우 −, 좌 +)."""
        return float(self.cfg.spawn_center[1])

    def _widen_dependent_limits(self) -> None:
        m = float(self.cfg.mimic_dep_limit_margin_rad)
        if m <= 0.0:
            return
        lim = self.robot.data.joint_pos_limits[:, self._mim_dep_t, :].clone()
        lim[..., 0] -= m
        lim[..., 1] += m
        self.robot.write_joint_limits_to_sim(lim, joint_ids=self._mim_dep_t.tolist())

    def _boot_checks(self) -> None:
        """먼 출발이 정말 먼지: 시작 자세 손바닥 실측 ↔ 소환 표본 컵 원점 거리의 평균이 대역 안인가."""
        cfg, N = self.cfg, self.num_envs
        q0 = self._reset_q.unsqueeze(0).expand(N, -1).contiguous()
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        self.robot.set_joint_position_target(q0)
        self.scene.write_data_to_sim()
        for _ in range(2):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        palm = self._palm_pos()[0]
        xy, _, _ = sample_cup_xy(4096, cfg.spawn_center, cfg.spawn_half, cfg.spawn_holes, float(cfg.cup_radius),
                                 float(cfg.spawn_hole_margin), self.device)
        z = torch.full((xy.shape[0], 1), float(cfg.table_surface_z) + float(cfg.cup_origin_offset_z), device=self.device)
        d = (torch.cat([xy, z], dim=1) - palm).norm(dim=-1)
        lo, hi = (float(v) for v in cfg.start_palm_cup_dist_band)
        print(f"{_TAG} 시작 palm {[round(float(v), 4) for v in palm]} · 손바닥↔컵 거리 평균 {float(d.mean()):.3f} "
              f"(min {float(d.min()):.3f} max {float(d.max()):.3f}) · 대역 {cfg.start_palm_cup_dist_band}", flush=True)
        if not lo <= float(d.mean()) <= hi:
            raise RuntimeError(f"{_TAG} 먼 출발 거리 평균 {float(d.mean()):.3f} 가 대역 밖 — arm_start_q·spawn 을 다시 재라")
        hand_z = self._hand_min_z()[0]
        if float(hand_z) < float(cfg.table_surface_z) + float(cfg.cup_origin_offset_z) + float(cfg.cup_mouth_z):
            print(f"{_TAG} ⚠시작 손 최저 z {float(hand_z):.3f} 이 컵 윗면보다 낮다 — 손 그림자 소환 회피가 필요하다", flush=True)

    def _setup_dr(self) -> None:
        cfg, N, dev = self.cfg, self.num_envs, self.device
        em = getattr(self, "event_manager", None)
        self.adr = TaskADR(
            {"obs": {"object_xyz": (0.0, float(cfg.adr_obs_noise_object_xyz_max)),
                     "object_rot_deg": (0.0, float(cfg.adr_obs_noise_object_rot_max_deg)),
                     "delay_steps": (0.0, float(cfg.perception_delay_max_steps))},
             "wrench": {"force_scale": (0.0, float(cfg.wrench_force_scale_max)),
                        "torque_scale": (0.0, float(cfg.wrench_torque_scale_max))}},
            num_increments=int(cfg.adr_num_increments), increment_interval=int(cfg.adr_increment_interval),
            trigger_threshold=float(cfg.adr_trigger_threshold), enabled=bool(cfg.enable_adr),
            event_manager=em, physics_cfg=dict(_cfg.PHYSICS_ADR_TERMINAL) if em is not None else {})
        self._perc_buf = torch.zeros(N, int(cfg.perception_delay_max_steps) + 1, 7, device=dev)
        self._perc_flush = torch.ones(N, dtype=torch.bool, device=dev)
        self._arange = torch.arange(N, device=dev)
        self._wrench = WrenchDR(N, dev, force_scale=0.0, torque_scale=0.0, prob_range=tuple(cfg.wrench_prob_range))
        self._cup_mass = torch.full((N,), _cfg.CUP_MASS, device=dev)
        # 실기 반응 — 정책·관측은 명령한 목표(arm_q_target·hand_target)를, PD 는 지연·펌웨어를 거친 목표를 쓴다
        a_d, h_d = tuple(cfg.arm_cmd_delay_steps), tuple(cfg.hand_cmd_delay_steps)
        self._arm_line = DelayLine(N, len(self.arm_ids), int(a_d[1]), dev)
        self._hand_line = DelayLine(N, len(self.hand_ids), int(h_d[1]), dev)
        self._arm_delay = sample_delay(N, a_d, dev)
        self._hand_delay = sample_delay(N, h_d, dev)
        self._thumb_abd_i = _cfg.thumb_abduction_index(self.profile)
        self._hand_floor = sample_open_floor(N, tuple(cfg.hand_open_floor_deg_lo), tuple(cfg.hand_open_floor_deg_hi), dev)
        self._arm_line.fill(self._arange, self.q_home)
        self._hand_line.fill(self._arange, self.q_open)
        self.arm_q_applied = self.q_home.unsqueeze(0).repeat(N, 1)
        self.hand_applied = open_floor(self.q_open.unsqueeze(0).repeat(N, 1), self.q_open, self.q_grip, self._hand_floor)

    # ==================================================================
    def _setup_scene(self) -> None:
        cfg = self.cfg
        p = _cfg.PROFILES[cfg.profile_name]
        self.robot = Articulation(cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        cfg.table_spawn.func("/World/envs/env_0/Table", cfg.table_spawn, translation=(0.0, 0.0, 0.0))
        mu = float(cfg.surface_friction)
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=mu, dynamic_friction=mu, restitution=0.0)
        mat.func("/World/Materials/taskSurface", mat)
        tables = find_matching_prim_paths(_cfg.TABLE_PRIM)
        if not tables:
            raise RuntimeError(f"{_TAG} 테이블 프림이 없다: {_cfg.TABLE_PRIM}")
        for tp in tables:
            bind_physics_material(tp, "/World/Materials/taskSurface")
        # 손가락 링크별 센서(body 하나당 하나 — 다중 body 단일 센서는 force_matrix 가 0), 자기 컵만 필터.
        self._finger_sensors: dict[str, list] = {}
        for finger, bodies in p.finger_sensor_bodies.items():
            ss = []
            for body in bodies:
                s = ContactSensor(ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{body}",
                                                   filter_prim_paths_expr=[_cfg.CUP_BODY], history_length=1))
                self.scene.sensors[f"contact_{finger}_{body}"] = s
                ss.append(s)
            self._finger_sensors[finger] = ss
        self._palm_sensor = ContactSensor(ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{p.palm_body}",
                                                           filter_prim_paths_expr=[_cfg.CUP_BODY], history_length=1))
        self.scene.sensors["contact_palm"] = self._palm_sensor
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(),
                           translation=(0.0, 0.0, float(cfg.ground_plane_z)))
        light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        self.scene.clone_environments(copy_from_source=True)
        self.cup = RigidObject(cfg.cup_cfg)
        self.scene.rigid_objects["cup"] = self.cup
        n_hit = len(find_matching_prim_paths(_cfg.CUP_BODY))
        if n_hit != self.scene.num_envs:
            raise RuntimeError(f"{_TAG} 컵 강체 prim {_cfg.CUP_BODY} 매칭 {n_hit} ≠ env {self.scene.num_envs}")
        # 10.01 붓기 인계 수집(기본 끔): pour_fabric_mimic 과 같은 비드 자산·개수
        self.beads = None
        if int(cfg.bead_count) > 0:
            self.beads = RigidObjectCollection(make_beads_cfg(_cfg._ASSETS_DIR, n=int(cfg.bead_count)))
            self.scene.rigid_object_collections["beads"] = self.beads
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _place_beads(self, env_ids: torch.Tensor, cup_pos_local: torch.Tensor) -> None:
        """비드를 컵 안에 쌓는다 — xy 는 검증 배치(bead_offsets_in_cup), z 는 바닥 기준 층(pour_fabric_mimic 과 같은 식)."""
        if getattr(self, "beads", None) is None:
            return
        cfg, dev, k = self.cfg, self.device, int(self.cfg.bead_count)
        if getattr(self, "_bead_offs", None) is None:
            offs = torch.tensor(bead_offsets_in_cup(k), device=dev)
            layer = torch.arange(k, device=dev) // 5
            offs[:, 2] = -float(cfg.cup_origin_offset_z) + float(cfg.bead_z_from_bottom) + float(cfg.bead_layer_dz) * layer
            if float(offs[:, 2].max()) > float(cfg.cup_mouth_z) - 0.01:
                raise RuntimeError(f"{_TAG} 비드 {k} 개 맨 위 층이 입구에 닿는다: {float(offs[:, 2].max()):.4f}")
            self._bead_offs = offs
        b = torch.zeros(len(env_ids), k, 13, device=dev)
        b[:, :, :3] = cup_pos_local.unsqueeze(1) + self._bead_offs.unsqueeze(0) + self.scene.env_origins[env_ids].unsqueeze(1)
        b[:, :, 2] += float(cfg.object_spawn_pad)
        b[:, :, 3] = 1.0
        self.beads.write_object_state_to_sim(b, env_ids=env_ids)

    # ==================================================================
    # 좌표·접촉
    # ==================================================================
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self.scene.env_origins

    def _palm_pos(self) -> torch.Tensor:
        return self._local(self.robot.data.body_pos_w[:, self.palm_idx])

    def _palm_R(self) -> torch.Tensor:
        return matrix_from_quat(self.robot.data.body_quat_w[:, self.palm_idx])

    def _tips_pos(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.tip_t] - self.scene.env_origins[:, None, :]

    def _hand_min_z(self) -> torch.Tensor:
        z = torch.cat([self.robot.data.body_pos_w[:, self.tip_t, 2], self.robot.data.body_pos_w[:, self.link_t, 2],
                       self.robot.data.body_pos_w[:, self.palm_idx, 2].unsqueeze(1)], dim=1)
        return z.min(dim=1).values - self.scene.env_origins[:, 2]

    def _cup_pos(self) -> torch.Tensor:
        return self._local(self.cup.data.root_pos_w)

    def _cup_up(self) -> torch.Tensor:
        z = torch.zeros(self.num_envs, 3, device=self.device)
        z[:, 2] = 1.0
        return quat_apply(self.cup.data.root_quat_w, z)

    @staticmethod
    def _mag(sensor) -> torch.Tensor:
        return sensor.data.force_matrix_w.view(sensor.data.force_matrix_w.shape[0], -1, 3).sum(dim=1).norm(dim=-1)

    @staticmethod
    def _net(sensor) -> torch.Tensor:
        return sensor.data.net_forces_w.view(sensor.data.net_forces_w.shape[0], -1, 3).sum(dim=1).norm(dim=-1)

    def _link_forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(첫마디, 손끝) 자기 컵 접촉력 (N,F). 프로필 규약: body 튜플 = (첫마디, 센서)."""
        prox = torch.stack([self._mag(self._finger_sensors[f][0]) for f in self.fingers], dim=1)
        tip = torch.stack([self._mag(self._finger_sensors[f][-1]) for f in self.fingers], dim=1)
        return prox, tip

    def _tactile_raw(self) -> torch.Tensor:
        """손끝 촉각 (N,F) [N] — 실기 TouchData1 은 대상과 무관하게 재므로 net 힘."""
        return torch.stack([self._net(self._finger_sensors[f][-1]) for f in self.fingers], dim=1)

    def _foreign_force(self) -> torch.Tensor:
        tot = torch.zeros(self.num_envs, device=self.device)
        for s in [s for f in self.fingers for s in self._finger_sensors[f]] + [self._palm_sensor]:
            tot = tot + (self._net(s) - self._mag(s)).clamp(min=0.0)
        return tot

    def _grasp_flag(self, finger_f: torch.Tensor) -> torch.Tensor:
        thr = float(self.cfg.contact_force_threshold)
        return (finger_f[:, self.grp_a] > thr).any(dim=1) & (finger_f[:, self.grp_b] > thr).any(dim=1)

    def _hold_mask(self) -> torch.Tensor:
        return self.episode_length_buf < int(self.cfg.hold_steps)

    # ==================================================================
    # 제어
    # ==================================================================
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        self._raw_actions = actions.detach()            # 09.30 로그: 클램프 전 정책 출력(평균이 ±1 밖인지)
        self.actions = actions.clamp(-1.0, 1.0)
        active = (~self._hold_mask()).unsqueeze(1)
        a_arm, a_hand = self.actions[:, : len(self.arm_ids)], self.actions[:, len(self.arm_ids):]
        q_new = step_arm_increment(self.arm_q_target, a_arm, k=float(cfg.k_arm), alpha=float(cfg.arm_ema),
                                   q_lo=self.arm_lo, q_hi=self.arm_hi)
        self.arm_q_target = torch.where(active, q_new, self.q_home.unsqueeze(0).expand_as(q_new))
        prox, tip = self._link_forces()
        thr = float(cfg.contact_freeze_threshold)
        touch = ((prox > thr) | (tip > thr))[:, self.hand_finger]
        h_new = step_hand_target(self.hand_target, linear_grip_raw(a_hand, self.q_open, self.q_grip),
                                 alpha=float(cfg.hand_ema), cap=self.hand_cap, freeze=touch,
                                 q_open=self.q_open, q_grip=self.q_grip)
        self.hand_target = torch.where(active, h_new, self.q_open.unsqueeze(0).expand_as(h_new))
        self._arm_line.push(self.arm_q_target)
        self._hand_line.push(self.hand_target)
        self.arm_q_applied = self._arm_line.read(self._arm_delay)
        h_app = firmware_stop(self.hand_applied, self._hand_line.read(self._hand_delay), touch, self.q_open, self.q_grip)
        self.hand_applied = open_floor(h_app, self.q_open, self.q_grip, self._hand_floor)
        self._apply_wrench()

    def _apply_action(self) -> None:
        # 위치 목표만 — 속도 목표 0(실기 JTC·RH56F1 위치 서보와 같은 PD)
        # PD 에는 지연·펌웨어를 거친 목표(real_response) — 관측의 명령 목표와 다르다
        self.robot.set_joint_position_target(self.arm_q_applied, joint_ids=self.arm_ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.arm_q_applied), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(self.hand_applied, joint_ids=self.hand_ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(self.hand_applied), joint_ids=self.hand_ids)
        if self._grav_comp > 0.0:
            tau = self.robot.root_physx_view.get_gravity_compensation_forces()
            self.robot.set_joint_effort_target(self._grav_comp * tau[:, : self.robot.num_joints])

    def _apply_wrench(self) -> None:
        """들린 컵에 질량정규화 외란(ADR 로 0 → 종점). 래치가 섰고 지금도 들려 있을 때만."""
        self._wrench.force_scale = self.adr.get_param("wrench", "force_scale")
        self._wrench.torque_scale = self.adr.get_param("wrench", "torque_scale")
        up = (self._cup_pos()[:, 2] - self._spawn[:, 2]) > float(self.cfg.wrench_lift_min_m)
        f, t = self._wrench.step(self._cup_mass, up & self.goals.lifted)
        self.cup.set_external_force_and_torque(f, t, is_global=True)

    # ==================================================================
    # 관측
    # ==================================================================
    def _perceive(self) -> tuple[torch.Tensor, torch.Tensor]:
        """컵 pose 지각: 지연(0..d_max 균등, env 별 매 스텝) + 코히런트 잡음. d_max·잡음은 ADR 진행도."""
        cur = torch.cat([self._cup_pos(), self.cup.data.root_quat_w], dim=1)
        buf = self._perc_buf
        buf[self._perc_flush] = cur[self._perc_flush].unsqueeze(1)
        buf = torch.roll(buf, shifts=1, dims=1)
        buf[:, 0] = cur
        self._perc_buf = buf
        self._perc_flush[:] = False
        d_max = int(round(self.adr.get_param("obs", "delay_steps")))
        got = buf[self._arange, torch.randint(0, d_max + 1, (self.num_envs,), device=self.device)]
        return noisy_pose(got[:, :3], got[:, 3:], self.adr.get_param("obs", "object_xyz"),
                          self.adr.get_param("obs", "object_rot_deg"))

    def _tactile(self, noisy: bool) -> torch.Tensor:
        f = self._tactile_raw()
        if noisy:
            f = f + torch.randn_like(f) * float(self.cfg.tactile_obs_noise_n)
        f = f.clamp(0.0, float(self.cfg.tactile_obs_clip_n))
        return torch.tanh(f / float(self.cfg.tactile_obs_tanh_n))

    def _actor_parts(self, cup_p: torch.Tensor, cup_q: torch.Tensor, noisy: bool) -> list[torch.Tensor]:
        cfg, N = self.cfg, self.num_envs
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        arm_q, arm_qd, hand_q = q[:, self.arm_t], qd[:, self.arm_t], q[:, self.hand_t]
        palm, tips = self._palm_pos(), self._tips_pos()
        if noisy:
            arm_q = arm_q + torch.randn_like(arm_q) * float(cfg.obs_noise_qpos)
            arm_qd = arm_qd + torch.randn_like(arm_qd) * float(cfg.obs_noise_qvel)
            hand_q = hand_q + torch.randn_like(hand_q) * float(cfg.obs_noise_qpos)
            palm = palm + torch.randn_like(palm) * float(cfg.obs_noise_body)
            tips = tips + torch.randn_like(tips) * float(cfg.obs_noise_body)
        R = self._palm_R()
        z = torch.zeros(N, 3, device=self.device)
        z[:, 2] = 1.0
        kp_rel_goal = self.goals.cup_keypoints(cup_p, cup_q) - self.goals.goal_keypoints()
        return [
            arm_q, arm_qd, self.arm_q_target,
            hand_q, ((self.hand_target - q[:, self.hand_t]) / float(cfg.joint_err_norm)).clamp(-1.0, 1.0),
            palm, torch.cat([R[:, :, 0], R[:, :, 1]], dim=1),
            cup_p - palm, quat_apply(cup_q, z), (tips - cup_p.unsqueeze(1)).reshape(N, -1),
            kp_rel_goal.reshape(N, -1), self.goals.goal_pos - palm,
            self._tactile(noisy),
            self.prev_actions,
        ]

    def trace_snapshot(self) -> dict:
        """play `--trace_steps` 용: 직전 스텝 RewardContext 전 필드(전 env, CPU). RH_AGLT_TRACE=1 로 켠다.

        오프라인에서 후보 보상을 실제 정책 궤적 위에서 다시 계산하려는 것(t2r_aglt.py trace). 읽기 전용.
        """
        if self._last_ctx is None:
            raise RuntimeError(f"{_TAG} trace_snapshot: RH_AGLT_TRACE=1 로 기동해야 한다")
        snap = {}
        for fld in dataclasses.fields(self._last_ctx):
            v = getattr(self._last_ctx, fld.name)
            snap[f"ctx.{fld.name}"] = (v.detach().cpu().numpy() if torch.is_tensor(v)
                                       else __import__("numpy").asarray(v))
        return snap

    @staticmethod
    def _check_layout(parts: list[torch.Tensor], layout) -> None:
        for t, (name, w) in zip(parts, layout):
            if t.shape[-1] != w:
                raise RuntimeError(f"{_TAG} 관측 '{name}' 폭 {t.shape[-1]} ≠ 배치 {w}")
        if len(parts) != len(layout):
            raise RuntimeError(f"{_TAG} 관측 칸 수 {len(parts)} ≠ 배치 {len(layout)}")

    def _get_observations(self) -> dict:
        p = self.profile
        a, h, f = p.num_arm_joints, p.num_hand_joints, len(self.fingers)
        cp, cq = self._perceive()
        actor = self._actor_parts(cp, cq, noisy=True)
        clean = self._actor_parts(self._cup_pos(), self.cup.data.root_quat_w, noisy=False)
        if self._log_tick == 0:
            self._check_layout(actor, actor_layout(a, h, f))
        k = float(self.cfg.critic_contact_tanh_n)
        prox, tip = self._link_forces()
        extra = [
            self.robot.data.joint_vel[:, self.hand_t],
            self.cup.data.root_lin_vel_w, self.cup.data.root_ang_vel_w,
            self.robot.data.body_lin_vel_w[:, self.palm_idx],
            torch.tanh(prox / k), torch.tanh(tip / k), torch.tanh(self._mag(self._palm_sensor) / k).unsqueeze(1),
            self.goals.lifted.float().unsqueeze(1), self._grasped.float().unsqueeze(1),
            (self.episode_length_buf.float() / float(self.max_episode_length)).unsqueeze(1),
            (self.goals.trk.successes.float() / float(self.cfg.goal_max)).unsqueeze(1),
            self.goals.kp_dist.clamp(max=1.0).unsqueeze(1),
            (self.goals.trk.near_goal_steps.float() / float(self.cfg.goal_success_steps)).clamp(max=2.0).unsqueeze(1),
        ]
        if self._log_tick == 0:
            self._check_layout(extra, critic_extra_layout(a, h, f))
        return {"policy": torch.nan_to_num(torch.cat(actor, dim=1)),
                "critic": torch.nan_to_num(torch.cat(clean + extra, dim=1))}

    # ==================================================================
    # 보상 · 목표
    # ==================================================================
    def _arm_limit_margin(self) -> torch.Tensor:
        q = self.robot.data.joint_pos[:, self.arm_t]
        half = (0.5 * (self.arm_hi - self.arm_lo)).clamp(min=1e-6)
        mid = 0.5 * (self.arm_hi + self.arm_lo)
        return (1.0 - (q - mid).abs() / half).clamp(0.0, 1.0)

    def _build_context(self, prox, tip, goal_pos: torch.Tensor) -> RewardContext:
        cfg = self.cfg
        R = self._palm_R()
        up = self._cup_up()
        q = self.robot.data.joint_pos
        return RewardContext(
            table_z=float(cfg.table_surface_z), cup_radius=float(cfg.cup_radius),
            cup_mouth_z=float(cfg.cup_mouth_z), cup_bottom_z=float(cfg.cup_bottom_z),
            lift_latch_m=float(cfg.lift_latch_m), goal_tol=float(self.goals.tol), goal_max=int(cfg.goal_max),
            palm_pos=self._palm_pos(), palm_lin_vel=self.robot.data.body_lin_vel_w[:, self.palm_idx],
            palm_axes=torch.cat([R[:, :, 2], R[:, :, 1]], dim=1),
            tips_pos=self._tips_pos(),
            hand_closure=closure(q[:, self.hand_t], self.q_open, self.q_grip).mean(dim=1),
            hand_q_target_closure=closure(self.hand_target, self.q_open, self.q_grip),
            finger_force=prox + tip, finger_prox_force=prox, finger_tip_force=tip,
            palm_force=self._mag(self._palm_sensor), hand_foreign_force=self._foreign_force(),
            grasped=self._grasped, hand_min_z=self._hand_min_z(),
            arm_q=q[:, self.arm_t], arm_qd=self.robot.data.joint_vel[:, self.arm_t],
            arm_limit_margin=self._arm_limit_margin(),
            cup_pos=self._cup_pos(), cup_quat=self.cup.data.root_quat_w, cup_up=up,
            cup_tilt=torch.acos(up[:, 2].clamp(-1.0, 1.0)),
            cup_lin_vel=self.cup.data.root_lin_vel_w, cup_ang_vel=self.cup.data.root_ang_vel_w,
            cup_spawn_pos=self._spawn, lifted=self.goals.lifted,
            goal_pos=goal_pos, goal_kp_dist=self._goal_out["kp_dist"],
            near_goal=self._goal_out["near_goal"], goal_success=self._goal_out["success"],
            goal_successes=self.goals.trk.successes.float(),
            episode_progress=self.episode_length_buf.float() / float(self.max_episode_length),
            actions=self.actions, prev_actions=self.prev_actions,
        )

    def _get_rewards(self) -> torch.Tensor:
        active = ~self._hold_mask()
        prox, tip = self._link_forces()
        self._grasped = self._grasp_flag(prox + tip)
        # 목표 전진 전에 ctx 가 쓸 목표 위치를 남긴다(성공한 스텝의 보상은 달성한 목표 기준).
        goal_pos = self.goals.goal_pos.clone()
        self._goal_out = self.goals.step(self._cup_pos(), self.cup.data.root_quat_w, self._grasped, active)
        if self._capture is not None:
            self._capture.step(self._grasped, active)
        ctx = self._build_context(prox, tip, goal_pos)
        if self._keep_ctx:                                # play --trace_steps 전용(RH_AGLT_TRACE=1) — 학습 경로는 건너뛴다
            self._last_ctx = dataclasses.replace(ctx, prev_actions=ctx.prev_actions.clone(), actions=ctx.actions.clone())
        total, terms = call_reward_fn(self._reward_fn, ctx)
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0) * active.float()
        if self.goals.update_curriculum():
            print(f"{_TAG}[tol] → {self.goals.tol:.4f}", flush=True)
        metric = float(self.goals.trk.prev_episode_successes.float().mean()) / float(self.cfg.goal_max)
        if self.adr.maybe_increment(metric):
            print(f"{_TAG}[ADR] {self.adr.increment_counter}/{self.adr.num_increments}", flush=True)
        self._log(total, terms, ctx)
        self.prev_actions.copy_(self.actions)
        return total

    # ==================================================================
    # 종료
    # ==================================================================
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        live = ~self._hold_mask()
        cup = self._cup_pos()
        floor = float(cfg.table_surface_z) + float(cfg.cup_origin_offset_z) - float(cfg.drop_below_table_m)
        dropped = cup[:, 2] < floor
        out = ((cup[:, 0] < cfg.cup_out_x[0]) | (cup[:, 0] > cfg.cup_out_x[1])
               | (cup[:, 1] < cfg.cup_out_y[0]) | (cup[:, 1] > cfg.cup_out_y[1]))
        tipped = self._cup_up()[:, 2] < math.cos(math.radians(float(cfg.cup_tilt_max_deg)))
        runaway = (self.robot.data.joint_vel[:, self.arm_t].abs() > float(cfg.runaway_joint_vel)).any(dim=-1)
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        mim_err = (q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs().max(dim=-1).values
        spike = qd[:, self._mim_dep_t].abs().max(dim=-1).values > float(cfg.mimic_runaway_dep_qd)
        self._mim_qd_streak = torch.where(spike, self._mim_qd_streak + 1, torch.zeros_like(self._mim_qd_streak))
        nonfinite = ~(torch.isfinite(q).all(dim=-1) & torch.isfinite(qd).all(dim=-1))
        mimic = (self._mim_qd_streak >= int(cfg.mimic_runaway_qd_steps)) | (mim_err > float(cfg.mimic_runaway_err_rad)) | nonfinite
        floor_hit = self._hand_min_z() < float(cfg.table_surface_z) - float(cfg.hand_floor_depth_m)
        phys = self._phys_anomaly()
        reasons = {"dropped": dropped & live, "out": out & live, "tipped": tipped & live, "runaway": runaway,
                   "mimic": mimic, "floor": floor_hit & live, "phys_cup": phys["cup"], "phys_contact": phys["contact"]}
        terminated = torch.zeros_like(dropped)
        for v in reasons.values():
            terminated |= v
        # 직전 스텝까지의 성공수(cup_grasp 규약) — 5번째 성공 다음 스텝에 truncate 된다(한 스텝 늦음, 의도).
        goals_done = self.goals.trk.successes >= int(cfg.goal_max)
        truncated = (self.episode_length_buf >= self.max_episode_length - 1) | goals_done
        self._last_done = reasons
        for k, v in reasons.items():
            self.extras[f"done/{k}"] = v.float().mean()
        self.extras["done/goals_done"] = goals_done.float().mean()
        return terminated, truncated

    def _phys_anomaly(self) -> dict:
        """컵 튕김(선·각속도)·손 접촉 폭발(연속 스텝) — pour_fabric_mimic 09.29 기준값."""
        cfg = self.cfg
        live = ~self._hold_mask()
        v = self.cup.data.root_lin_vel_w.norm(dim=-1) > float(cfg.phys_cup_speed_max)
        w = self.cup.data.root_ang_vel_w.norm(dim=-1) > float(cfg.phys_cup_angvel_max)
        prox, tip = self._link_forces()
        f = torch.stack([(prox + tip).max(dim=-1).values, self._mag(self._palm_sensor), self._foreign_force()], dim=-1)
        hot = (f > float(cfg.phys_contact_max_n)).any(dim=-1)
        self._phys_f_streak = torch.where(hot, self._phys_f_streak + 1, torch.zeros_like(self._phys_f_streak))
        return {"cup": (v | w) & live, "contact": (self._phys_f_streak >= int(cfg.phys_contact_steps)) & live}

    # ==================================================================
    def _reset_idx(self, env_ids) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self, "goals"):
            ended = env_ids[self.episode_length_buf[env_ids] > int(self.cfg.hold_steps)]
            if len(ended):
                self.extras["ep/successes"] = self.goals.trk.successes[ended].float().mean()
                self.extras["ep/lifted"] = self.goals.lifted[ended].float().mean()
        super()._reset_idx(env_ids)
        cfg, n, dev = self.cfg, len(env_ids), self.device
        q0 = self._reset_q.unsqueeze(0).expand(n, -1).contiguous()
        if not hasattr(self, "goals"):
            self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
            self.robot.set_joint_position_target(q0, env_ids=env_ids)
            init = torch.tensor(self.cfg.cup_cfg.init_state.pos, device=self.device).expand(len(env_ids), 3)
            self._place_beads(env_ids, init)          # 비드가 cfg 기본 위치 한 점에 겹쳐 있지 않게
            return                                    # 부모 __init__ 중의 첫 리셋 — 버퍼가 아직 없다
        # 실기 반응: 에피소드마다 지연·편 손 하한을 새로 뽑고, 손은 실기 쉬는 자세(엄지 외전 1.40~1.57 · 손가락 하한)에서 시작
        self._arm_delay[env_ids] = sample_delay(n, tuple(cfg.arm_cmd_delay_steps), dev)
        self._hand_delay[env_ids] = sample_delay(n, tuple(cfg.hand_cmd_delay_steps), dev)
        self._hand_floor[env_ids] = sample_open_floor(n, tuple(cfg.hand_open_floor_deg_lo), tuple(cfg.hand_open_floor_deg_hi), dev)
        h0 = open_floor(self.q_open.unsqueeze(0).expand(n, -1), self.q_open, self.q_grip, self._hand_floor[env_ids])
        t_lo, t_hi = (float(v) for v in cfg.thumb1_reset_range)
        h0[:, self._thumb_abd_i] = t_lo + torch.rand(n, device=dev) * (t_hi - t_lo)
        q0[:, self.hand_t] = h0
        for _ in range(2):                            # 종속의 종속(thumb_4 ← thumb_3)까지 mimic 비로 맞춘다
            q0[:, self._mim_dep_t] = self._mim_mult * q0[:, self._mim_lead_t]
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
        self.robot.set_joint_position_target(q0, env_ids=env_ids)
        self.arm_q_target[env_ids] = self.q_home
        self.hand_target[env_ids] = self.q_open
        self._arm_line.fill(env_ids, self.q_home)
        self._hand_line.fill(env_ids, self.q_open)
        self.arm_q_applied[env_ids] = self.q_home
        self.hand_applied[env_ids] = h0
        self._mim_qd_streak[env_ids] = 0
        self._phys_f_streak[env_ids] = 0

        xy, n_re, n_fail = sample_cup_xy(n, cfg.spawn_center, cfg.spawn_half, cfg.spawn_holes,
                                         float(cfg.cup_radius), float(cfg.spawn_hole_margin), dev)
        pos = torch.cat([xy, torch.full((n, 1), float(cfg.table_surface_z) + float(cfg.cup_origin_offset_z), device=dev)], dim=1)
        self._spawn[env_ids] = pos
        root = torch.zeros(n, 13, device=dev)
        root[:, :3] = pos + self.scene.env_origins[env_ids]
        root[:, 2] += float(cfg.object_spawn_pad)
        root[:, 3] = 1.0
        self.cup.write_root_state_to_sim(root, env_ids=env_ids)
        self._place_beads(env_ids, pos)
        if getattr(self, "_capture", None) is not None:
            self._capture.on_reset(env_ids)
        quat = torch.zeros(n, 4, device=dev)
        quat[:, 0] = 1.0
        self.goals.reset(env_ids, pos, quat)
        self.extras["diag/spawn_resampled_frac"] = n_re / max(n, 1)
        self.extras["diag/spawn_reject_failed"] = float(n_fail)

        self._perc_flush[env_ids] = True
        self._wrench.reset(env_ids)
        self._cup_mass = self._read_cup_mass()
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._grasped[env_ids] = False

    def _read_cup_mass(self) -> torch.Tensor:
        """실제 질량(질량 DR 뒤) — 공칭으로 정규화하면 외란 비가 흔들린다(grasp_s2r 09.10)."""
        try:
            m = self.cup.root_physx_view.get_masses()
        except Exception:                             # 뷰 미준비 → 공칭
            return torch.full((self.num_envs,), _cfg.CUP_MASS, device=self.device)
        if m is None or m.ndim != 2 or m.shape[0] != self.num_envs:
            return torch.full((self.num_envs,), _cfg.CUP_MASS, device=self.device)
        return m[:, 0].to(self.device, dtype=torch.float32)

    # ==================================================================
    def _log(self, total: torch.Tensor, terms: dict, ctx: RewardContext) -> None:
        ex = self.extras
        for k, v in terms.items():
            ex[f"reward/{k}"] = v.mean()
        ex["reward/total"] = total.mean()
        d_palm = (ctx.palm_pos - ctx.cup_pos).norm(dim=-1)
        ex["task/palm_cup_dist"] = d_palm.mean()
        ex["task/near_10cm"] = (d_palm < 0.10).float().mean()
        ex["task/grasped"] = ctx.grasped.float().mean()
        ex["task/lifted"] = ctx.lifted.float().mean()
        ex["task/cup_lift_m"] = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).mean()
        ex["task/kp_dist"] = ctx.goal_kp_dist.mean()
        ex["task/near_goal"] = ctx.near_goal.float().mean()
        ex["task/goal_success"] = ctx.goal_success.float().mean()
        ex["task/successes"] = ctx.goal_successes.mean()
        ex["task/tol"] = float(self.goals.tol)
        ex["task/closure"] = ctx.hand_closure.mean()
        ex["task/cup_tilt_deg"] = torch.rad2deg(ctx.cup_tilt).mean()
        # 09.30 팔 한계·액션 포화 — 우 i05 결정론 재생 목표 성공 0.10(확률적 학습 ~1.0) 원인 가르기
        n_arm = len(self.arm_ids)
        raw = getattr(self, "_raw_actions", None)
        if raw is not None:
            ex["act/arm_abs_raw"] = raw[:, :n_arm].abs().mean()
            ex["act/arm_sat_frac"] = (raw[:, :n_arm].abs() > 1.0).float().mean()
            ex["act/hand_sat_frac"] = (raw[:, n_arm:].abs() > 1.0).float().mean()
        margin = ctx.arm_limit_margin
        ex["arm/min_margin"] = margin.min(dim=1).values.mean()
        ex["arm/at_limit_frac"] = (margin < 0.05).float().mean()
        for j in range(margin.shape[1]):
            ex[f"arm/j{j + 1}_at_limit"] = (margin[:, j] < 0.05).float().mean()
        touching = ctx.finger_force > float(self.cfg.contact_force_threshold)
        ex["contact/n_fingers"] = touching.float().sum(dim=1).mean()
        ex["contact/prox_frac"] = (ctx.finger_prox_force > float(self.cfg.contact_force_threshold)).float().mean()
        ex["contact/palm_frac"] = (ctx.palm_force > float(self.cfg.contact_force_threshold)).float().mean()
        ex["contact/foreign_frac"] = (ctx.hand_foreign_force > float(self.cfg.contact_force_threshold)).float().mean()
        # 파지 모양(쥔 스텝만, 로그 전용 — 사용자 09.29 "로그로 확인되게")
        ex.update(grasp_shape(
            grasped=ctx.grasped, palm_pos=ctx.palm_pos, palm_normal=ctx.palm_axes[:, 0:3],
            finger_dir=ctx.palm_axes[:, 3:6], tips_pos=ctx.tips_pos, finger_force=ctx.finger_force,
            prox_force=ctx.finger_prox_force, cup_pos=ctx.cup_pos, cup_up=ctx.cup_up, cup_radius=ctx.cup_radius,
            cup_mouth_z=ctx.cup_mouth_z, closure=ctx.hand_closure, thr=float(self.cfg.contact_force_threshold),
            mid_sign=self._mid_sign))
        span = (self.arm_hi - self.arm_lo).clamp(min=1e-6)
        near_lim = ((self.arm_q_target - self.arm_lo) < 0.02 * span) | ((self.arm_hi - self.arm_q_target) < 0.02 * span)
        ex["joint/arm_limit_frac"] = near_lim.float().mean()
        ex["joint/arm_track_err_deg"] = torch.rad2deg((self.arm_q_target - ctx.arm_q).abs().max(dim=1).values).mean()
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        ex["ctrl/mimic_err_max"] = (q[:, self._mim_dep_t] - self._mim_mult * q[:, self._mim_lead_t]).abs().max()
        ex["ctrl/hand_dep_qd_max"] = qd[:, self._mim_dep_t].abs().max()
        ex.update(self.adr.log_dict())
        ex["dr/wrench_force_scale"] = self.adr.get_param("wrench", "force_scale")
        ex["dr/arm_delay_ms"] = self._arm_delay.float().mean() * float(self.step_dt) * 1000.0
        ex["dr/hand_delay_ms"] = self._hand_delay.float().mean() * float(self.step_dt) * 1000.0
        ex["dr/hand_open_floor_deg"] = torch.rad2deg(self._hand_floor[:, 2:]).mean()
        ex["joint/arm_cmd_lag_deg"] = torch.rad2deg((self.arm_q_target - self.arm_q_applied).abs().max(dim=1).values).mean()
        self._log_tick += 1
        every = int(self.cfg.console_log_interval)
        if every > 0 and self._log_tick % every == 0:
            print(f"[METRICS] step={self._log_tick:>8d} rew={float(total.mean()):+.3f} "
                  f"palm_cup={float(ex['task/palm_cup_dist']):.3f} grasp={float(ex['task/grasped']):.2f} "
                  f"lifted={float(ex['task/lifted']):.2f} kp={float(ex['task/kp_dist']):.3f} "
                  f"succ={float(ex['task/successes']):.2f} tol={self.goals.tol:.3f}", flush=True)
