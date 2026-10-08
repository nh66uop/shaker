"""rh_aglt_r 환경 설정 — RH56F1 **오른팔** 한 팔: 먼 출발 → 접근 → 잡기 → 들기 → 이송(키포인트 목표).

사용자 결정(09.29): 새 디렉토리·새 env(`tasks/rh_aglt_{r,l}`), 손마다 컵 1개, 무작위 keypoint 목표,
좌우 따로 학습, 양팔 자산 유지(쓰지 않는 팔은 홈 고정). 09.29 2차: 컵 x 0.10–0.40·y 폭 0.20, 팔 관절 증분(cup_grasp),
손 = pour_bi_rh 와 같은 사상, 관측 잡음·지연 켬(컵 회전 잡음 제외 — 축대칭), 컵 질량·마찰 다양화(마찰 1.5–2.5). 기존 트랙(grasp_fj·grasp_fj_rh·
manip_stages·pour_fabric_mimic)은 import 하지 않는다 — 중립 `modules/` 만 쓴다.

값의 출처(전부 실측·기존 결정):
  · 자산·게인·손 자세·접촉 링크: `modules/robot_profiles.RH56F1_RIGHT`(09.15 손 30/0.3·종속 0/0)
  · 컵: pour_bi_rh 와 같은 `shaker_closed_thick_rl` × 0.65(지름 57 mm, 0.134 kg). 손·컵 마찰 U(1.5, 2.5)
    (09.17 실측: 0.97 에서 들기 1/73, 2.0 에서 25/86 — 사용자 09.29 "마찰 다양화", 범위 1.5–2.5)
  · 먼 출발·소환 범위: 이 트랙 CLAUDE.md 의 URDF IK 격자(손 방향은 RH 캘리브 홈과 같다)
  · 목표·tol 커리큘럼·성공: cup_grasp(DG5F-M-S) 구조, RH 컵 크기에 맞춘 값
  · 손 지령·힘 멈춤·mimic 폭주 종료·물리 이상 종료: pour_fabric_mimic full-joint 가족 값
"""

from __future__ import annotations

import os
import re

import isaaclab.envs.mdp as _mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from openarm.agnostic.modules import object_bank as _ob
from openarm.agnostic.modules import robot_profiles as _rp

from .layout import actor_layout, critic_extra_layout, layout_width

_ASSETS_DIR = _ob.ASSETS_DIR

#: 이 패키지가 쓰는 프로필(오른팔). 좌 패키지는 자기 프로필을 둔다.
PROFILES: dict[str, _rp.RobotProfile] = {_rp.RH56F1_RIGHT.name: _rp.RH56F1_RIGHT}
DEFAULT_PROFILE = _rp.RH56F1_RIGHT.name

# =============================================================================
# 컵 — pour_bi_rh 와 같은 자산. 강체는 루트가 아니라 `baseLink` 하위 prim 이다(접촉 필터는 강체 prim 을 가리켜야 한다).
# 기하(스케일 1, object_bank 09.01 pxr 실측): 바닥 −0.0921 · 림 +0.0829 · 외경 r 0.0440.
# =============================================================================
CUP_USD = os.path.join(_ASSETS_DIR, "cup", "shaker_closed_thick_rl.usd")
CUP_MASS = 0.134                       # 스케일과 무관한 실물 질량(0.65³ 로 줄이면 29 g 이라 스치기만 해도 넘어진다)
CUP_UNIT = {"bottom_z": -0.0921, "rim_z": 0.0829, "outer_r": 0.0440}
CUP_PRIM = "/World/envs/env_.*/Object"
CUP_BODY = CUP_PRIM + "/baseLink"
TABLE_PRIM = "/World/envs/env_.*/Table"
TABLE_SPAWN = sim_utils.UsdFileCfg(usd_path=os.path.join(_ASSETS_DIR, "simulation_setting/env_v1/usd/env_v1.usda"))
#: env_v1 상판 구멍 (cx, cy, r) — grasp_fj_t2r 09.17 collision 메시 실측(카운터보어 다섯 개 중 오른쪽 셋).
TABLE_HOLES_RIGHT = ((0.135, -0.30, 0.0305), (0.27, -0.30, 0.0305), (0.135, 0.0, 0.0305))


def build_cup_cfg(scale: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=CUP_PRIM,
        spawn=sim_utils.UsdFileCfg(
            usd_path=CUP_USD, scale=(scale, scale, scale), activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=CUP_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                max_angular_velocity=100.0, max_linear_velocity=100.0,
                max_depenetration_velocity=1.0, disable_gravity=False)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.16, 0.30)),
    )


def arm_joint_keys(profile: _rp.RobotProfile) -> list[str]:
    """프로필 init 에서 활성 팔 관절 이름 7개(끝 번호 순)."""
    pat = re.compile(f"^{profile.arm_joint_regex}$")
    keys = sorted((k for k in profile.init_joint_pos if pat.match(k)), key=lambda k: int(k.rsplit("_", 1)[1]))
    if len(keys) != profile.num_arm_joints:
        raise RuntimeError(f"[{profile.name}] init 에서 팔 관절 {profile.num_arm_joints} 개를 못 찾았다: {keys}")
    return keys


#: 좌우 거울 부호(y 반전) — 팔 (−1,−1,−1,+1,−1,−1,−1), 손 6+6 전부 +1(09.14·09.29 URDF FK). 자기 역이라 양방향에 같다.
ARM_MIRROR_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)


def start_joint_pos(profile: _rp.RobotProfile, arm_start_q: tuple) -> dict:
    """리셋 자세 = 활성 팔 먼 출발 자세 + **쉬는 팔은 그 거울상**(사용자 09.29 "왼팔 오른팔 모두 대칭적으로").

    쉬는 손은 활성 손의 init(편 손)과 같은 값이다(손 거울 부호 +1). 프로필 init 의 옛 휴식 자세는 쓰는 팔이
    작업면 가운데(y ≈ 0)로 뻗어 반대쪽 컵 소환 범위와 겹쳤다(09.29 정면 영상).
    """
    if len(arm_start_q) != profile.num_arm_joints:
        raise ValueError(f"arm_start_q 길이 {len(arm_start_q)} ≠ {profile.num_arm_joints}")
    out = dict(profile.init_joint_pos)
    keys = arm_joint_keys(profile)
    me = keys[0].split("_", 1)[0]
    other = {"r": "l", "l": "r"}[me]
    for k, v, sgn in zip(keys, arm_start_q, ARM_MIRROR_SIGN):
        out[k] = float(v)
        out[other + k[len(me):]] = float(sgn) * float(v)
    for k, v in profile.init_joint_pos.items():
        if k.startswith(f"{me}_hj_"):
            out[other + k[len(me):]] = float(v)
    return out


def build_robot_cfg(profile: _rp.RobotProfile, arm_start_q: tuple, self_collisions: bool,
                    gravity: bool) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSETS_DIR, profile.usd_relpath), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=not gravity, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos=start_joint_pos(profile, arm_start_q), joint_vel={".*": 0.0}),
        actuators={name: ImplicitActuatorCfg(**spec) for name, spec in profile.actuator_specs.items()},
        soft_joint_pos_limit_factor=1.0,
    )


# =============================================================================
# 물리 DR — 초기 범위는 항등, ADR 이 종점까지 선형 확장. 재질은 term 생성 때 버킷을 한 번 뽑아 런타임 확장이
# 무증상 no-op 이라(grasp_s2r) 처음부터 고정 범위다.
# =============================================================================
FRICTION_RANGE = (1.5, 2.5)


def _material_term(asset: str, mu: tuple) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_material, mode="reset",
        params={"asset_cfg": SceneEntityCfg(asset, body_names=".*"),
                "static_friction_range": tuple(mu), "dynamic_friction_range": tuple(mu),
                # ★반발 0 — 1.0(pour cfg 에서 옮겨 온 값)이면 서 있는 컵이 테이블에서 튄다: 손이 닿지 않는데 각속도 p50 0.10 rad/s·
                #   선속도 2.7 cm/s(09.29 probe_rh_aglt_cup_rest, 256 env). 0.0 에서 0.017 rad/s·0.06 cm/s.
                "restitution_range": (0.0, 0.0), "num_buckets": 250})


@configclass
class RhAgltEventCfg:
    robot_material = _material_term("robot", FRICTION_RANGE)
    cup_material = _material_term("cup", FRICTION_RANGE)
    robot_joint_stiffness_and_damping = EventTermCfg(
        func=_mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (1.0, 1.0), "damping_distribution_params": (1.0, 1.0),
                "operation": "scale", "distribution": "uniform"})
    cup_scale_mass = EventTermCfg(
        func=_mdp.randomize_rigid_body_mass, mode="reset",
        # 처음부터 ×0.8–1.25 로 흔들고 ADR 이 ×0.5–2.5 까지 넓힌다(사용자 09.29 "컵 무게 다양화").
        params={"asset_cfg": SceneEntityCfg("cup"), "mass_distribution_params": (0.8, 1.25),
                "operation": "scale", "distribution": "uniform"})


PHYSICS_ADR_TERMINAL = {
    "cup_scale_mass": {"mass_distribution_params": (0.5, 2.5)},
    "robot_joint_stiffness_and_damping": {"stiffness_distribution_params": (0.5, 2.0),
                                          "damping_distribution_params": (0.5, 2.0)},
}


# =============================================================================
@configclass
class RhAgltEnvCfg(DirectRLEnvCfg):
    profile_name: str = DEFAULT_PROFILE
    #: t2r 생성 보상 `compute_reward(ctx)` 절대경로. 비면 영 보상.
    reward_code_path: str = ""

    # ---- 시뮬레이션 ----
    episode_length_s: float = 15.0            # 900 스텝 @60 Hz — cup_grasp 먼 출발과 같다
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0, render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=2 ** 25, gpu_total_aggregate_pairs_capacity=2 ** 23,
            gpu_max_rigid_patch_count=2 ** 22, gpu_max_rigid_contact_count=2 ** 23,
            # 09.29 aglt_l_i03: 기본 2**26(64 MB)에서 e800 부터 collisionStackSize 넘침 17,439회(요구 최대 95 MB,
            # 'Contacts have been dropped') — 형제 RH56F1 트랙(pour_fabric_mimic)과 같은 2**30.
            gpu_collision_stack_size=2 ** 30,
            friction_correlation_distance=0.00625))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    hold_steps: int = 10                      # 리셋 직후 컵 정착 — 팔·손 고정, 액션 무시, 보상 0

    # ---- 물리 스위치 ----
    enable_gravity: bool = True
    gravity_compensation: float = 1.0
    enable_self_collisions: bool = True       # pour_fj(09.28) 와 같다 — 09.15 collider 정리 뒤 켰다
    surface_friction: float = 1.0
    ground_plane_z: float = -0.10
    table_surface_z: float = 0.205

    # ---- 컵 ----
    cup_scale: float = 0.65
    object_spawn_pad: float = 0.005
    # 파생(resolve_cfg 가 CUP_UNIT × cup_scale 로 덮는다 — hydra 로 따로 덮지 말 것)
    cup_origin_offset_z: float = 0.0
    cup_mouth_z: float = 0.0
    cup_bottom_z: float = 0.0
    cup_radius: float = 0.0
    cup_half_height: float = 0.0

    # ---- 10.01 붓기 인계용 쥔 상태 수집(사용자 "aglt 할 때 미리 비드를 넣어 둔 상태로 모으기") — 기본 끔 ----
    #: 컵 안 비드 수(0 = 비드 없음, 학습 경로 그대로). pour_fabric_mimic 과 같은 자산·배치(common.bead_assets)·적층.
    bead_count: int = 0
    bead_z_from_bottom: float = 0.0195         # pour_fabric_mimic 과 같은 값(바닥 윗면 13.0 mm + 비드 반지름 6 + 0.5)
    bead_layer_dz: float = 0.014
    #: 비우지 않으면 grasp_capture.GraspCapture 가 쥐고 든 순간을 npz 로 모은다(play 전용).
    grasp_capture_path: str = ""
    grasp_capture_n: int = 2048
    grasp_capture_lift_min: float = 0.05

    # ---- 먼 출발 · 소환 ----
    #: 활성 팔 시작 관절(절대) — 09.29 사용자 "cup_grasp 처럼 테이블 앞 바깥". URDF IK: palm_sensor (0.08, −0.30, 0.45)
    #:   (상판 앞 가장자리 x 0.07 위, 손끝만 상판 위 x ≤ 0.19) · 관절 한계 여유 10.5° · 손 링크 최저 z 0.400(컵 윗면 0.319 위).
    #:   방향(사용자 규칙): 손바닥 +z(palm_sensor 국소 z) 수평 · 두 손바닥이 마주 봄(우 +y) · 엄지 끝이 새끼 끝보다 71 mm 위.
    #:   손가락 방향은 수평에서 −10°(캘리브 홈 −25° 에서 15° 들어 올림 — 인벨롭 실측의 수평 손가락 축에 가깝게).
    arm_start_q: tuple = (-1.2127, 0.2026, 0.6538, 1.7608, 0.3791, 0.5785, 0.6646)
    #: 컵 소환 = 중심 ± 반폭(env-local) = x 0.10–0.40 · y −0.30–−0.10(사용자 09.29 "x 0.10–0.40, y 폭 0.20").
    #:   IK 격자(손 방향 고정 파지 자세): 이 범위 전체 도달. 상판 구멍 (0.135, −0.30)·(0.27, −0.30) 이 걸려 재추첨한다.
    spawn_center: tuple = (0.25, -0.20)
    spawn_half: tuple = (0.15, 0.10)
    spawn_holes: tuple = TABLE_HOLES_RIGHT
    spawn_hole_margin: float = 0.01
    #: 부팅 가드 — 리셋 직후 손바닥↔컵 원점 거리의 env 평균이 이 대역 밖이면 죽인다(먼 출발이 먼 출발인지).
    start_palm_cup_dist_band: tuple = (0.22, 0.46)

    # ---- 팔: 관절 증분(cup_grasp) ----
    k_arm: float = 0.05                       # rad/스텝
    arm_ema: float = 0.1                      # 포화 slew = 0.1·0.05·60 = 0.3 rad/s

    # ---- 손: pour_bi_rh 와 같은 grip 구간 절대 목표 + EMA + 벤더 속도 + 힘 멈춤 ----
    hand_ema: float = 0.1
    hand_full_range_s: float = 1.0            # RH56F1 벤더 사양 전 범위 1 s
    contact_freeze_threshold: float = 1.0     # N — 손가락 닫기 정지(실기 펌웨어 힘 멈춤)
    contact_force_threshold: float = 1.0      # N — 파지 판정(엄지 AND 다른 손가락)
    joint_err_norm: float = 1.2               # 손 목표−실측 관측 정규화 [rad]

    # ---- 실기 반응(09.30 사용자 "실제 제어 로직으로 학습" 1번) — PD 에 들어가는 목표만 바꾼다(real_response.py) ----
    #: 에피소드마다 U{lo..hi} 정책 스텝(60 Hz). ADR 에 걸지 않는다 — i07b·i08 은 ADR 이 한 번도 오르지 않아 실측 지연을 못 볼 수 있다.
    arm_cmd_delay_steps: tuple = (9, 12)      # 150~200 ms — 로봇 자체(배포 pd FULL), memory cg-l-i01-real-vs-sim-handoff-0928
    hand_cmd_delay_steps: tuple = (3, 5)      # 50~83 ms — 실측 명령→첫 움직임 60~80 ms(RH56F1_TUNING_0930 §2)
    #: 편 쪽 하한 [deg], 구동 관절 순서(thumb_1·thumb_2·index·middle·ring·pinky). 실측(§4) 우 T2 1.2·I 3.6·M 3.2·R 4.9·P 5.2°,
    #:   좌 T2 2.6·I 3.9·M 2.4·R 2.7·P 2.3° — 레지스터가 벤더 범위에서 잘려 0 rad 명령에도 이만큼 굽어 있다. 두 손을 다 덮는 범위.
    hand_open_floor_deg_lo: tuple = (0.0, 1.0, 2.0, 2.0, 2.0, 2.0)
    hand_open_floor_deg_hi: tuple = (0.0, 3.0, 5.5, 5.5, 5.5, 5.5)
    #: 리셋 때 엄지 외전 시작값 U(lo, hi) — 전원 뒤 쉬는 자세 우 1.40·좌 1.51 rad, 정책의 편 손 명령 1.57 은 실기도 닿는다(레지스터 853·917).
    thumb1_reset_range: tuple = (1.40, 1.57)

    # ---- 언더액추 백스톱 ----
    mimic_dep_limit_margin_rad: float = 1.5
    mimic_runaway_dep_qd: float = 100.0
    mimic_runaway_qd_steps: int = 3
    mimic_runaway_err_rad: float = 3.0

    # ---- 목표(cup_grasp 구조) ----
    goal_first_xy_range: float = 0.05
    goal_first_z_range: tuple = (0.10, 0.18)  # 정착고 기준
    goal_delta_distance: float = 0.08
    goal_box_xy_margin: float = 0.0           # 목표 박스 = 소환 박스(x 하한 0.10 — 상판 앞 가장자리 0.07·종료 0.05 여유)
    goal_box_z_range: tuple = (0.08, 0.22)    # 정착고 기준
    goal_success_steps: int = 10
    goal_max: int = 5
    tol_start: float = 0.10
    tol_floor: float = 0.02
    tol_factor: float = 0.9
    tol_interval: int = 3000
    tol_success_threshold: float = 2.0
    tol_eval: float = 0.0
    lift_latch_m: float = 0.10

    # ---- 종료 ----
    runaway_joint_vel: float = 20.0
    drop_below_table_m: float = 0.03
    cup_tilt_max_deg: float = 60.0
    cup_out_x: tuple = (0.05, 0.85)
    cup_out_y: tuple = (-0.60, 0.25)
    hand_floor_depth_m: float = 0.005
    phys_cup_speed_max: float = 3.0
    phys_cup_angvel_max: float = 30.0
    phys_contact_max_n: float = 200.0
    phys_contact_steps: int = 2

    # ---- 관측 잡음·지연(s2r) ----
    obs_noise_qpos: float = 0.002
    obs_noise_qvel: float = 0.05
    obs_noise_body: float = 0.005
    tactile_obs_noise_n: float = 0.1
    tactile_obs_clip_n: float = 10.0
    tactile_obs_tanh_n: float = 3.0
    critic_contact_tanh_n: float = 5.0
    perception_delay_max_steps: int = 3       # ADR 종점: 컵 pose 0..3 정책 스텝 지연
    adr_obs_noise_object_xyz_max: float = 0.015
    adr_obs_noise_object_rot_max_deg: float = 0.0   # 컵은 축대칭 — 회전 잡음 끔(사용자 09.29)

    # ---- 물리 DR · ADR · 외란 ----
    enable_events: bool = True
    events: RhAgltEventCfg = RhAgltEventCfg()
    enable_adr: bool = True
    adr_num_increments: int = 30
    adr_increment_interval: int = 3000
    adr_trigger_threshold: float = 0.4        # 지표 = 직전 에피소드 성공수 평균 / goal_max
    wrench_force_scale_max: float = 5.0
    wrench_torque_scale_max: float = 0.5
    wrench_prob_range: tuple = (0.001, 0.1)
    wrench_lift_min_m: float = 0.03

    console_log_interval: int = 600

    # ---- 파생 ----
    robot_cfg: ArticulationCfg = None
    cup_cfg: RigidObjectCfg = None
    table_spawn: sim_utils.UsdFileCfg = TABLE_SPAWN
    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0

    def __post_init__(self) -> None:
        resolve_cfg(self)


def thumb_abduction_index(profile: _rp.RobotProfile) -> int:
    """구동 손 관절 중 엄지 외전(thumb_1) 자리."""
    hits = [i for i, n in enumerate(profile.hand_joint_names) if n.endswith("_thumb_1")]
    if len(hits) != 1:
        raise RuntimeError(f"[{profile.name}] 엄지 외전 관절을 하나로 못 찾았다: {profile.hand_joint_names}")
    return hits[0]


def validate_cfg(cfg: "RhAgltEnvCfg", profile: _rp.RobotProfile) -> None:
    """조용히 무효가 되는 조합을 cfg 단계에서 죽인다."""
    errs = []
    for name, spec in profile.actuator_specs.items():
        if name.endswith("_hand_mimic") and (spec.get("stiffness"), spec.get("damping")) != (0.0, 0.0):
            errs.append(f"{name} 게인이 0/0 이 아니다 — PhysX mimic 제약과 싸운다(09.07 발산)")
    if len(profile.hand_open_pose) != profile.num_hand_joints or len(profile.hand_grip_pose) != profile.num_hand_joints:
        errs.append("손 open/grip 자세 길이 ≠ 구동 관절 수")
    if any(abs(o - g) < 1e-4 for o, g in zip(profile.hand_open_pose, profile.hand_grip_pose)):
        errs.append("open == grip 인 손 관절 — 그 손가락은 조용히 죽는다")
    for name in ("arm_cmd_delay_steps", "hand_cmd_delay_steps"):
        lo, hi = getattr(cfg, name)
        if not 0 <= int(lo) <= int(hi):
            errs.append(f"{name} 0 ≤ lo ≤ hi: {getattr(cfg, name)}")
    fl, fh = cfg.hand_open_floor_deg_lo, cfg.hand_open_floor_deg_hi
    if len(fl) != profile.num_hand_joints or len(fh) != profile.num_hand_joints:
        errs.append("hand_open_floor_deg_lo/hi 길이 ≠ 구동 손 관절 수")
    elif any(not 0.0 <= float(a) <= float(b) for a, b in zip(fl, fh)):
        errs.append(f"편 손 하한 0 ≤ lo ≤ hi: {fl} / {fh}")
    t_lo, t_hi = (float(v) for v in cfg.thumb1_reset_range)
    ti = thumb_abduction_index(profile)
    if not min(profile.hand_grip_pose[ti], profile.hand_open_pose[ti]) <= t_lo <= t_hi <= max(profile.hand_grip_pose[ti], profile.hand_open_pose[ti]):
        errs.append(f"thumb1_reset_range {cfg.thumb1_reset_range} 가 엄지 외전 grip~open 구간 밖")
    if not 0.0 < float(cfg.tol_floor) <= float(cfg.tol_start):
        errs.append(f"0 < tol_floor ≤ tol_start: {cfg.tol_floor}/{cfg.tol_start}")
    if not float(cfg.goal_box_z_range[0]) <= float(cfg.goal_first_z_range[0]) <= float(cfg.goal_first_z_range[1]) <= float(cfg.goal_box_z_range[1]):
        errs.append(f"goal_first_z_range {cfg.goal_first_z_range} 가 goal_box_z_range {cfg.goal_box_z_range} 밖")
    if float(cfg.goal_first_z_range[0]) < float(cfg.lift_latch_m):
        errs.append("첫 목표 높이 하한이 들기 래치보다 낮다 — 들지 않고 목표에 닿는다")
    cx, cy = (float(v) for v in cfg.spawn_center)
    hx, hy = (float(v) for v in cfg.spawn_half)
    if not (cfg.cup_out_x[0] < cx - hx and cx + hx < cfg.cup_out_x[1]
            and cfg.cup_out_y[0] < cy - hy and cy + hy < cfg.cup_out_y[1]):
        errs.append("소환 박스가 종료 상자(cup_out_x/y) 밖으로 나간다")
    for hx0, hy0, hr in cfg.spawn_holes:
        if ((cx - hx0) ** 2 + (cy - hy0) ** 2) ** 0.5 < hr + float(cfg.cup_radius) + float(cfg.spawn_hole_margin):
            errs.append(f"소환 중심이 상판 구멍 ({hx0}, {hy0}) 에 걸린다 — 재추첨 실패 env 가 거기 놓인다")
    if errs:
        raise RuntimeError("[rh_aglt cfg] " + " · ".join(errs))


def resolve_cfg(cfg: "RhAgltEnvCfg") -> None:
    """프로필·스위치 → 자산 cfg·차원. **멱등** — hydra 오버라이드 뒤 env 가 다시 부른다."""
    profile = PROFILES[cfg.profile_name]
    s = float(cfg.cup_scale)
    cfg.cup_origin_offset_z = -CUP_UNIT["bottom_z"] * s
    cfg.cup_mouth_z = CUP_UNIT["rim_z"] * s
    cfg.cup_bottom_z = CUP_UNIT["bottom_z"] * s
    cfg.cup_radius = CUP_UNIT["outer_r"] * s
    cfg.cup_half_height = 0.5 * (CUP_UNIT["rim_z"] - CUP_UNIT["bottom_z"]) * s
    cfg.robot_cfg = build_robot_cfg(profile, tuple(cfg.arm_start_q), bool(cfg.enable_self_collisions),
                                    bool(cfg.enable_gravity))
    cfg.cup_cfg = build_cup_cfg(s)
    if not bool(cfg.enable_events):
        cfg.events = None
    validate_cfg(cfg, profile)
    a, h, f = profile.num_arm_joints, profile.num_hand_joints, len(profile.finger_sensor_bodies)
    cfg.action_space = a + h
    cfg.observation_space = layout_width(actor_layout(a, h, f))
    cfg.state_space = cfg.observation_space + layout_width(critic_extra_layout(a, h, f))


@configclass
class RhAgltEnvCfg_PLAY(RhAgltEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        # 커리큘럼은 체크포인트에 없다 — 평가 tol 은 학습 종점(tol_floor)으로 고정. 다른 값은 hydra `env.tol_eval=`.
        self.tol_eval = self.tol_floor
        super().__post_init__()
