"""pour_fabric_mimic 환경 설정 — 양팔 **잡기→들기→붓기**, 저차원(언더액추·PhysX mimic) 손 전용.

★`pour_fabric/pour_fabric_env_cfg.py` 사본(09.14). 원본 트랙은 손대지 않는다. 다른 점:
  · 쌍 레지스트리 = 이 트랙의 `bimanual`(RH56F1 Fabrics 프로필, 기본 쌍 "rh").
  · 손 액션 폭 = 구동관절 수(per_finger, RH56F1 6) → 액션 24 = (palm 6 + 손 6) × 2.
  · `oppose_grip_delta_rad = 0.0`(tesollo 엄지 대향 전용 노브 — RH56F1 ch1 은 엄지 굴곡이라
    −0.6 을 주면 grip 이 한계 밖으로 잘려 엄지가 안 접힌다, 09.02 실측).
  · 접촉 동결은 finger 스코프만(손가락당 구동관절 1개).
  · 종속관절 한계 부팅 확장 `mimic_dep_limit_margin_rad`(grasp_fj_rh 09.07 실측 근거).
  · `_validate_mimic_fields` 가 조용히 무효가 되는 조합을 cfg 단계에서 죽인다.

(아래는 원본 설명)

★09.13 재작성 이유. 구판은 warm 뱅크(이미 잡은 상태)에서 시작하고 손을 동결한 채 붓기만
  배웠다. 이번 트랙은 text2reward 방식의 **보상 자동생성**이 목적이라 과제 문장("양팔로
  컵을 각각 잡고 비드 있는 컵을 없는 컵으로 옮긴다")을 통째로 정책이 배워야 한다 —
  테이블 위 컵 두 개에서 시작하고 손 20관절도 정책(시너지)이 제어한다.
  보상은 이 파일에 **없다**: `reward_code_path` 의 생성 코드가 `RewardContext` 를 읽어
  계산한다(`modules/t2r`). 비어 있으면 영 보상(부팅/무작위 롤아웃용).

★09.14 sim2real(라운드 3, 사용자 지시): actor 관측에서 손 관절속도 제거(실기 드라이버 velocity
  는 관절속도가 아니다 — 09.07 실측), 컵 pose 지각 지연+코히런트 노이즈, 관절 노이즈,
  물리 DR(컵 질량·관절 게인·컵 마찰) + 들린 컵 외란 — 전부 grasp_s2r/grasp_kp 모듈 재사용,
  ADR(순간 성공률 트리거)로 중립 → 종점 확장. 충돌 신호(컵끼리·손↔타물체)를 ctx 에 노출.

제어 스택 = grasp_s2r 현행(팔 Fabrics ×2 · 손 관절공간 시너지 + 접촉 동결) 을 양팔로.
물리 블록 = 구 pour_fabric(비드 20개×N env 접촉 버퍼) 그대로.
"""

from __future__ import annotations

import os

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
from openarm.common.bead_assets import DEFAULT_BEAD_COUNT, make_beads_cfg

from . import bimanual as _bm

_ASSETS_DIR = _ob.ASSETS_DIR

# =============================================================================
# 컵 자산 — pour 전용 **SDF** 콜라이더(convex hull 은 컵 속을 메워 비드가 안 담긴다).
# 원점 오프셋·내부 기하는 pour_v1 실측(.usd bbox) — bead_flags 판정과 같은 값.
# =============================================================================
# ★09.14 사용자 결정: 컵 대신 **shaker 계열**(`shaker_closed_rl.usd` = 바닥 플러그 + 열린 상단, SDF 콜라이더,
#   object_bank 09.01 pxr 실측: 바닥 −0.0921 · 림 +0.0829 · 내경 r 0.0432 · 외경 r 0.0440, 벽 0.8 mm).
#   RH56F1 은 엄지–4지 간극 최대 83.7 mm·실사용 ~70 mm(object_bank SHAKER_SMALL 주석) → cup_scale 0.65 = 지름 57 mm
#   (Track B shaker_one 과 같은 크기). 인벨롭 파지는 포기, 손끝 파지로 잡아도 된다(사용자 09.14).
# ★09.22 사용자 결정: 바닥 두께 강화 사본 `shaker_closed_thick_rl.usd`(scripts/tools/make_thick_floor_shaker_asset.py).
#   i11 trace 에서 받는 컵 진입 비드의 2~3 % 가 바닥(SDF 1.2 mm + plug 4 mm, scale 1)을 뚫었고 CCD 로는 안 줄었다.
#   plug 위에 convexHull floor_fill 16 mm 를 쌓아 바닥 윗면 −0.0881 → −0.0721(scale 1). 외곽 프로파일 불변.
POUR_CUP_USD = os.path.join(_ASSETS_DIR, "cup", "shaker_closed_thick_rl.usd")
POUR_CUP_ORIGIN_OFFSET_Z = 0.0921     # 바닥 −0.0921 (scale 1)
POUR_CUP_MASS = 0.134                 # = object_bank.BASE_OBJECT_MASS (스케일 무관 실물값)

SOURCE_CUP_PRIM = "/World/envs/env_.*/SourceCup"
RECEIVER_CUP_PRIM = "/World/envs/env_.*/ReceiverCup"
# ★shaker USD 는 강체(RigidBodyAPI)가 루트가 아니라 `baseLink` 하위 prim 이다(cup_big_sdf 는 루트).
#   접촉 센서 prim 과 손가락 센서의 필터는 **강체 prim** 을 가리켜야 한다 — 어긋나면 PhysX 가 조용히 힘 0 을 준다
#   (관측 #0230: 필터 0개 매칭 = 로그 한 줄 + 전부 0). env 가 부팅에서 매칭 수를 검사한다.
POUR_CUP_BODY_NAME = "baseLink"
SOURCE_CUP_BODY = SOURCE_CUP_PRIM + "/" + POUR_CUP_BODY_NAME
RECEIVER_CUP_BODY = RECEIVER_CUP_PRIM + "/" + POUR_CUP_BODY_NAME
TABLE_PRIM = "/World/envs/env_.*/Table"


# 컵 기하(스케일 1.0 기준) — resolve_cfg 가 `cup_scale` 을 곱해 cfg 필드를 채운다(멱등).
CUP_GEOM_UNIT = {"cup_inner_radius": 0.0432, "cup_inside_z_min": -0.0721, "cup_inside_z_max": 0.0829,
                 "cup_mouth_z": 0.0829, "object_origin_offset_z": POUR_CUP_ORIGIN_OFFSET_Z}


def build_cup_cfg(prim_path: str, scale: float = 1.0) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=POUR_CUP_USD,
            scale=(scale, scale, scale),
            activate_contact_sensors=True,
            # ★질량은 스케일과 무관하게 실물 컵 값(0.134 kg)을 유지한다 — s³ 로 줄이면 0.6 배 컵이 29 g 이 되어
            #   손가락이 스치기만 해도 넘어진다(09.14 실측). 크기만 줄이는 것이 사용자 의도다.
            mass_props=sim_utils.MassPropertiesCfg(mass=POUR_CUP_MASS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=1.0,       # 7218N 스파이크 근거(구 pour)
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.20, 0.30)),
    )


def build_robot_cfg(pair: _bm.BimanualPair, self_collisions: bool,
                    gravity: bool) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSETS_DIR, pair.usd_relpath),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=not gravity,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collisions,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=dict(pair.init_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators={
            name: ImplicitActuatorCfg(**spec)
            for name, spec in pair.actuator_specs.items()
        },
        soft_joint_pos_limit_factor=1.0,
    )


# 작업면 — env_v1.usda (top_plate 상면 z=0.205, 09.05 CAD 정정·실기 줄자 일치).
TABLE_SPAWN = sim_utils.UsdFileCfg(
    usd_path=os.path.join(_ASSETS_DIR, "simulation_setting/env_v1/usd/env_v1.usda"),
)


# =============================================================================
# 물리 DR EventTerm — 초기 범위는 **중립**(항등). ADR 이 종점까지 선형 확장한다
# (`modules/physics_dr.py` 규약; 자산 이름이 두 개라 여기서 따로 정의).
# ★재질 term 은 term 생성 시 버킷을 1회 샘플링해 **런타임 확장이 무증상 no-op** 이다
#   (grasp_s2r `_adr_apply_physics` 주석) → 컵 마찰은 cfg 단계에서 고정 범위로 연다.
# =============================================================================
def _material_term(asset: str, lo: float, hi: float) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_material, mode="reset",
        params={"asset_cfg": SceneEntityCfg(asset, body_names=".*"),
                "static_friction_range": (lo, hi), "dynamic_friction_range": (lo, hi),
                "restitution_range": (1.0, 1.0), "num_buckets": 250})


def _mass_term(asset: str) -> EventTermCfg:
    return EventTermCfg(
        func=_mdp.randomize_rigid_body_mass, mode="reset",
        params={"asset_cfg": SceneEntityCfg(asset), "mass_distribution_params": (1.0, 1.0),
                "operation": "scale", "distribution": "uniform"})


@configclass
class PourFabricEventCfg:
    # 09.17 사용자 결정: 손(로봇) 마찰 2.0 — probe_rh_grasp_lift fric 6조건에서 손-컵 쌍 마찰 ~0.97 → 2.0 일 때만
    #   직립+양쪽 파지가 들기로 이어졌다(1/73 → 25/86). 컵도 cup_friction_range 로 2.0 고정(평균 combine → 쌍 2.0).
    robot_material = _material_term("robot", 2.0, 2.0)
    source_cup_material = _material_term("source_cup", 1.0, 1.0)     # resolve_cfg 가 범위 적용
    receiver_cup_material = _material_term("receiver_cup", 1.0, 1.0)
    robot_joint_stiffness_and_damping = EventTermCfg(
        func=_mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (1.0, 1.0),
                "damping_distribution_params": (1.0, 1.0),
                "operation": "scale", "distribution": "uniform"})
    source_cup_scale_mass = _mass_term("source_cup")
    receiver_cup_scale_mass = _mass_term("receiver_cup")


# ADR 종점(grasp_s2r E1 값) — 질량·게인만 ADR 로 확장(재질은 위 이유로 cfg 고정).
PHYSICS_ADR_TERMINAL = {
    "source_cup_scale_mass": {"mass_distribution_params": (0.5, 2.5)},
    "receiver_cup_scale_mass": {"mass_distribution_params": (0.5, 2.5)},
    "robot_joint_stiffness_and_damping": {
        "stiffness_distribution_params": (0.5, 2.0),
        "damping_distribution_params": (0.5, 2.0)},
}


# =============================================================================
@configclass
class PourFabricMimicEnvCfg(DirectRLEnvCfg):
    """차원은 resolve_cfg 가 pair 로 확정한다."""

    pair_name: str = _bm.DEFAULT_PAIR

    # ---- 보상 (text2reward 생성 코드) ----------------------------------------------
    # 빈 문자열 = 영 보상. 학습 런은 반드시 생성·검증을 거친 파일을 가리켜야 한다.
    reward_code_path: str = ""

    # ---- 시뮬레이션 (물리 = 구 pour_fabric, 비드 버퍼 포함) ---------------------------
    episode_length_s: float = 15.0            # 900 스텝 @60Hz
    decimation: int = 2
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            # 비드 20개 × N env 접촉 폭증 — pour_v1 검증값(128 env overflow 크래시 끝에 확정).
            gpu_found_lost_aggregate_pairs_capacity=64 * 1024 * 1024,
            gpu_total_aggregate_pairs_capacity=16 * 1024 * 1024,
            gpu_max_rigid_patch_count=2 ** 24,
            gpu_max_rigid_contact_count=2 ** 24,
            gpu_collision_stack_size=2 ** 30,
            gpu_max_num_partitions=64,
            friction_correlation_distance=0.00625,
        ),
    )
    # ★09.14 4096 env 실측 43 GB·10.7k fps(서버 RTX PRO 6000). 128 은 pour_v1 시절 값.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True)

    # ---- 물리 스위치 --------------------------------------------------------------
    enable_gravity: bool = True
    gravity_compensation: float = 1.0
    # ★RH56F1 은 자기충돌 OFF(형제 grasp_s2r/grasp_fj/grasp_fj_rh 의 D3 기본 09.01). 09.14 부팅 실측:
    #   ON 이면 시작 자세에서 r_hl_index_tip 에 226 N 의 자기접촉 힘이 상시 걸리고(foreign_force 오염),
    #   폐쇄 0.7 부터 index_2·thumb_3/4 가 이웃 손가락에 밀려 되돌아가 mimic 오차 0.9 rad 가 난다.
    enable_self_collisions: bool = False
    surface_friction: float = 1.0             # 테이블 재질(컵-테이블). grasp_s2r 규약으로 바인딩.
    ground_plane_z: float = -0.10

    # ---- 작업면·컵 기하 -------------------------------------------------------------
    # ★물체 = shaker_closed × cup_scale(위 주석). 아래 기하 필드 5개는
    #   resolve_cfg 가 CUP_GEOM_UNIT × cup_scale 로 **덮어쓴다**(hydra 로 개별 기하를 덮지 말 것). 질량은 실물값 유지.
    cup_scale: float = 0.65
    # 비드 소환: xy 는 검증 배치(bead_offsets_in_cup) 그대로(반경 14/18 mm < 내경 28 mm), z 는 바닥 기준으로 다시 쌓는다
    #   (원본 z 는 cup_big 원점 기준이라 shaker 0.65 에서는 림 위로 나간다). 층 간격 14 mm 유지.
    bead_z_from_bottom: float = 0.0195         # 09.22 두꺼운 바닥: 윗면 13.0 mm(scale 0.65) + 비드 반지름 6 mm + 0.5 mm
    bead_layer_dz: float = 0.014
    table_surface_z: float = 0.205
    object_origin_offset_z: float = POUR_CUP_ORIGIN_OFFSET_Z
    object_spawn_pad: float = 0.005           # 스폰 침투 반동 방지
    object_spawn_range: float = 0.02          # 스폰 중심 xy 균등 ± [m]
    # ★09.30 사용자 "A로 진행": 두 컵 스폰 중심을 x 로 옮긴다(음수 = 로봇 쪽). 기본 0(옛 런 재생 불변).
    #   probe_pour_fj_envelope_reach.py: x 0.38 에서 K7 인벨롭은 팔꿈치 j4 여유 1.2°(팔 다 폄), −0.12 에서 최소 여유 8.7°.
    object_spawn_dx: float = 0.0
    cup_inner_radius: float = 0.041
    cup_inside_z_min: float = -0.070          # bottom(-0.077) + bead 반경 여유
    cup_inside_z_max: float = 0.100           # 림
    cup_mouth_z: float = 0.100
    bead_count: int = DEFAULT_BEAD_COUNT
    # 리셋 직후 비드 정착 대기 — 이 동안 팔은 시작 자세에 고정되고 액션은 무시된다.
    hold_steps: int = 30

    # ---- Fabrics (= grasp_s2r 현행) -------------------------------------------------
    fabrics_dt: float = 1.0 / 60.0
    fabric_decimation: int = 2
    fabrics_damping_gain: float = 10.0
    fabrics_max_objects_per_env: int = 8
    fabric_use_cuda_graph: bool = False
    fabric_velocity_ff_scale: float = 1.0
    use_hand_repulsion: bool = False
    use_body_repulsion_pairs: bool = True
    fabric_table_obstacle: bool = True
    fabric_table_margin_xy: float = 0.10
    fabric_table_thickness: float = 0.05
    fabric_fk_pos_tol: float = 0.005          # 부팅 게이트: fabric FK vs USD palm (2스텝 처짐 1~2mm 가 바닥 — 실측 0.5~1.7mm)

    # ---- 팔 액션: palm 6D = 앵커(시작 자세 palm 실측) + 델타 ---------------------------
    # 델타 박스 (x,y,z [m] · ez,ey,ex [deg]) — lo/hi 비대칭. a=0 이 앵커(=컵 옆 시작 자세).
    # 소스: 붓기 tilt 축 구간을 깊게 연다.
    # ★회전 슬롯 순서 = `_palm_pose_6d` [yaw(ez), pitch(ey), roll(ex)] = fabric "euler_zyx" 지령.
    #   ★09.13 프로브 실측(접근→파지→리프트 뒤 슬롯별 −1 지령): 슬롯 3(yaw)에 −150° 는 도달
    #     불가(회전오차 110°, 컵 tilt 24°)이고 **슬롯 5(roll)** 가 오차 1.3° 로 추종해 컵 tilt 가
    #     지령대로 나온다(−45° → 44.4°). 붓기 축은 **roll = 슬롯 5** 다 — 구 pour cfg 주석
    #     ("첫 슬롯이 roll")은 틀렸다. 리시버(좌)는 별도 cfg(대칭 ±30°)라 미러 문제 없음.
    # ★RH56F1 홈(Track B 캘리브)은 palm 이 컵 원점보다 13.7 cm 위·14.6 cm 바깥이라(09.14 부팅 실측
    #   palm−cup = (−0.08, −0.146, +0.137)) z 하한을 −0.12 → −0.18 로 연다. 나머지는 원본 값.
    src_palm_delta_lo: tuple = (-0.15, -0.10, -0.18, -45.0, -45.0, -150.0)
    src_palm_delta_hi: tuple = (0.15, 0.40, 0.25, 45.0, 45.0, 45.0)
    rcv_palm_delta_lo: tuple = (-0.15, -0.40, -0.18, -30.0, -30.0, -30.0)
    rcv_palm_delta_hi: tuple = (0.15, 0.10, 0.25, 30.0, 30.0, 30.0)
    # 위치는 프로필 palm 박스로 추가 clamp(회전은 델타 박스만).
    # ★09.28 palm 지령 방식(사용자 결정: DG-5F pour_fabric 커밋 9d300ef4 의 증분 방식을 이식).
    #   "absolute" = 구 방식(a → 앵커 + 델타 자세). 보관 체크포인트 재생용 기본값.
    #   "increment" = palm 6D 액션을 EMA 로 거른 뒤 **손바닥 목표 속도**(최대 속도 비율)로 읽어 목표를 누적한다.
    #     목표는 델타 박스·palm 박스 안, 그리고 현재 실제 palm 근처(leash)로 묶인다 — ±1 은 박스 끝 자세가 아니라 최대 속도.
    #   근거(RH, 절대 방식): i04 ep850 오른손 palm 명령 x·z 100 %·roll 59 % 가 |a|>0.95, 명령−실제 간극 z 7 cm·roll 0.9 rad;
    #     i06 ep950 왼손 z·roll 98~100 % 포화. 속도·간극 상한은 DG-5F 좋은 런(i18·i24·t2r2_i01) 실측 값을 그대로 쓴다(같은 OpenArm 팔).
    #   ★실기 정책 노드도 같은 EMA·속도 상한·간극 상한·박스로 목표를 누적해야 한다.
    #   "workspace" = ★09.28 DEXTRAH(dextrah_kuka_allegro) 팔 지령 방식. 위치는 **palm 박스 전체**에 선형 사상
    #     0.5(a+1)(hi−lo)+lo(a=0 = 박스 중심, compute_absolute_action 과 같은 식), 회전은 앵커 ± 델타 박스 회전 구간
    #     (DEXTRAH 는 공칭 ± max_pose_angle 대칭이라 a=0 = 공칭; 우리 소스 roll 구간은 비대칭(−150/+45)이라 부호별 사상으로
    #     a=0 = 앵커를 지킨다). EMA·속도 누적·leash 없음 — 목표까지의 속도·가속·저크 제한과 충돌 회피는 fabric 이 맡는다
    #     (DEXTRAH 와 같은 1/60 s × 2 적분, fabric q·qd 를 PD 위치·속도 목표로). 정책은 fabric 상태를 관측으로 받는다
    #     (`obs_fabric_state`, DEXTRAH obs 의 fabric_q·qd·qdd 와 같은 역할).
    palm_action_mode: str = "absolute"
    # ★09.28 DEXTRAH 관측: 팔마다 fabric 팔 슬라이스의 q(7) 를 actor·critic 에 붙인다(잡음 없음 — fabric 은 정책 노드 안의
    #   내부 상태라 실기에서도 정확히 안다). DEXTRAH 는 fabric q·qd·qdd 칸을 두지만 qd·qdd 에 곱하는
    #   observation_annealing 계수가 ADR 범위 (0, 0) 이라 학습 내내 0 이고, 실기 배포(kuka_allegro_stereo_fgp.py)도
    #   `fabric_qd * 0.`·`fabric_qdd * 0.` 을 넣는다 → 실제로 쓰는 정보는 fabric q 뿐이라 q 만 붙인다.
    #   실기 정책 노드는 같은 fabric 을 같은 dt 로 적분해 이 값을 만들어야 한다.
    obs_fabric_state: bool = False
    # ★09.28 DEXTRAH ADR custom: fabric cspace 감쇠 10 → 20, PD 속도 목표 계수(fabric qd FF) 1 → 0 을 ADR 진행도에 따라 옮긴다
    #   (dextrah_kuka_allegro_env_cfg.py adr_custom_cfg_dict "fabric_damping"·"pd_targets"). 진행도 0 에서는 지금 값 그대로다.
    fabric_adr: bool = False
    # ★09.28 fabric 파라미터 프리셋. "track" = openarm_rh56f1_pose_params.yaml 그대로(보관 체크포인트 재생용).
    #   "dextrah" = 그 yaml 에서 DEXTRAH(kuka_allegro_pose_params.yaml)와 다른 값 4개를 DEXTRAH 값으로 되돌린다:
    #   palm_attractor.conical_gain 80→50, body_repulsion.engage_depth 0.30→0.5, constant_accel 2→1, constant_accel_geom 3→5.
    #   네 값은 07-03 rh56f1 fabric 생성 때 tesollo yaml 에서 복사됐고 이 로봇에서 잰 근거가 없다(git 202f3824).
    #   충돌 구·쌍·관절 가속·저크 한계는 로봇 고유라 그대로 둔다. yaml 은 다른 트랙도 쓰므로 인스턴스에서만 바꾼다.
    fabric_param_preset: str = "track"
    # ★09.28 팔 제어 경로(사용자: "DG5F-M-S grasping 세팅처럼 양팔 full-joint, fabric 제어 안 쓰는 걸로").
    #   "fabric" = palm 6D 지령 → fabric → PD(위 palm_action_mode 가 지령 방식을 고른다). 보관 체크포인트 재생용 기본값.
    #   "joint"  = grasp_fj Track B(= cup_grasp, SimToolReal 방식) 그대로: 팔 액션 7 = 관절 증분
    #     q_raw = clamp(q*_{t-1} + k_arm·a, 관절 한계) → q*_t = α·q_raw + (1−α)·q*_{t-1} → **위치 목표만**(속도 목표 0).
    #     fabric 은 적분하지 않는다(부팅 FK 게이트용으로만 만든다). 팔 액션 폭 6→7 이라 액션 24→26.
    #     정책은 q*_{t-1}(팔 7)을 관측으로 받는다(cup_grasp `_cmd_state` 와 같은 역할).
    #   k_arm·arm_ema 는 cup_grasp 가 상속한 grasp_fj_t2r 값(k 0.05 rad/스텝·α 0.1 → 포화 slew 0.3 rad/s).
    arm_control: str = "fabric"
    k_arm: float = 0.05
    arm_ema: float = 0.1
    # ★09.29 joint 팔 액션의 뜻(사용자: "1,2번이 같이 적용된 게 좋다" — bounds_loss 0.005 + 절대 목표).
    #   "increment" = 위 증분 법칙(f00~f03·cup_grasp·sim2real joint_decoder). 보관 체크포인트 재생용 기본값.
    #   "absolute"  = 시작 자세 기준 절대 목표: a ≥ 0 → home + a·(hi − home), a < 0 → home + a·(home − lo)
    #     (a = 0 이 시작 자세, ±1 이 관절 한계). q*_t = q*_{t−1} + clamp(α·(q_raw − q*_{t−1}), ±arm_abs_vmax·dt).
    #     f03 ep5400 에서 증분 팔이 |mu| 1.4~2.4 로 한계에 96 % 갇혔다 — 절대 목표는 mu 가 줄면 목표가 곧바로 안쪽으로 온다.
    #     속도 상한 0.3 rad/s = 증분 모드의 포화 slew(α·k·60)와 같다. 실기 디코더는 아직 증분이다(sim2real 은 사용자 영역).
    arm_joint_mode: str = "increment"
    arm_abs_vmax: float = 0.3
    # ★09.28 손 제어 경로(사용자 D1 "맞음": cup_grasp 처럼 관절별 절대 목표 + EMA, 결합·게이트·동결 없음).
    #   "synergy" = 구 방식(폐쇄도 누적·4지 결합·close_gate·접촉 동결). 보관 체크포인트 재생용 기본값.
    #   "direct"  = grasp_fj `_hand_targets`(hand_direct) 와 같은 꼴: 구동 관절 j 마다
    #     raw = lo + ½(a+1)(hi−lo)(관절 한계 전 범위) → q*_t = α·raw + (1−α)·q*_{t-1} → 스텝당 변화 ≤ 전 범위/(hand_full_range_s·60)
    #     → clamp(lo, hi). a 는 관절 목표 그 자체다(명령과 결과 사이에 게이트·동결이 없다).
    #   hand_ema 0.1 = cup_grasp(SimToolReal handMovingAverage). 속도 상한 = RH56F1 벤더 전 범위 1 s(이 트랙 CLAUDE.md 09.15) —
    #   실기 구동기가 그보다 빨리 못 움직이므로 sim 도 같은 상한을 둔다.
    hand_control: str = "synergy"
    hand_ema: float = 0.1
    hand_full_range_s: float = 1.0
    # ★09.30 실기 제어 응답(사용자 "실제 제어 로직으로 학습될 수 있게도"): sim2real docs/RH56F1_TUNING_0930.md (SIME2REAL, arm4090 양손 실측)
    #   · HANDOFF_cg_l_i01(09.28, 같은 OpenArm 팔). 모두 기본 끔(옛 런 재생 불변). 지령 법칙(절대 목표·EMA·속도 상한)은 배포와 이미 같다.
    #   hand_cmd_delay_steps  손 명령 → 움직임 순수 지연(실측 60 ms, 드라이버 20 ms 읽기 포함) → 60 Hz 에서 4 스텝.
    #   arm_cmd_delay_steps   팔 명령 지연 [lo, hi] 스텝 균등(에피소드마다) — 로봇 자체 150~200 ms → (9, 12).
    #   hand_vel_cap_rad_s    손 목표 속도 상한을 관절 공통 rad/s 로(배포 pd max_vel_track 2.1; 실측 약 2.4 rad/s). 0 = 관절 전 범위/hand_full_range_s.
    #   hand_thumb_flex_vel_cap_rad_s  thumb_2(엄지 굽힘)만 따로(실측 약 32 °/s = 0.56 rad/s). 0 = 위 값.
    #   hand_finger_open_floor_rad     네 손가락 목표 하한 — 명령 레지스터 1740 까지만 펴진다(관절 2.2~5.2°, 중앙 약 0.065 rad).
    hand_cmd_delay_steps: int = 0
    arm_cmd_delay_steps: tuple = (0, 0)
    hand_vel_cap_rad_s: float = 0.0
    hand_thumb_flex_vel_cap_rad_s: float = 0.0
    hand_finger_open_floor_rad: float = 0.0
    # ★09.28 사용자 "1번, 2.a": direct 손의 범위와 힘 멈춤. 기본값은 첫 full-joint 런(f00~f01) 재생용.
    #   hand_direct_range "limits" = 관절 한계 전 범위, "grip" = 편 자세 ~ 쥔 자세(손가락끼리 짓누름 방지)
    #   hand_direct_contact_freeze = 컵 접촉력 > contact_freeze_threshold(1.0 N) 인 손가락은 더 닫지 않는다(펴기는 늘 허용)
    hand_direct_range: str = "limits"
    hand_direct_contact_freeze: bool = False
    adr_fabric_damping_range: tuple = (10.0, 20.0)
    adr_fabric_velocity_ff_range: tuple = (1.0, 0.0)
    palm_action_ema_alpha: float = 0.25       # 증분 방식에서만: y ← α·a + (1−α)·y (60 Hz 에서 시정수 약 58 ms)
    src_palm_vel_max: tuple = (0.6, 180.0)    # 증분 방식 최대 속도 (m/s, deg/s)
    rcv_palm_vel_max: tuple = (0.4, 120.0)
    src_palm_leash: tuple = (0.06, 30.0)      # 증분 방식 목표−실제 간극 상한 (m, deg)
    rcv_palm_leash: tuple = (0.05, 15.0)

    # ---- 손: 관절공간 시너지 (= grasp_s2r 현행 coupled3) --------------------------------
    synergy_close_speed: float = 0.005        # 폐쇄도 변화율 상한 / 정책 스텝
    # ★09.28 손 지령 방식(사용자: "손가락 제어가 액션을 못 따라가면 액션을 바꾸거나 액추에이터를 맞춰야 한다").
    #   "absolute" = 구 방식(a → 닫힘 목표 (a+1)/2, 스텝당 synergy_close_speed 로 따라감). 보관 체크포인트 재생용 기본값.
    #   "increment" = a 를 **닫힘 속도**(synergy_close_speed 의 비율)로 읽는다: 폐쇄도 += a·speed. ±1 = 최대 속도로 닫기/펴기,
    #     0 = 그대로 유지. 액션이 곧 실제 변화량이라 "명령했는데 안 따라오는" 구간이 없다(게이트·접촉 동결 제외).
    #   근거(i06 ep950 왼손): 엄지 명령 +1 을 계속 냈는데 폐쇄도 t=60 0.15 · t=120 0.45 · t=200 0.85 — 관절은 시너지 목표를
    #     오차 0 으로 따라갔고(자유 손 |목표−실측| 중앙 0.000), 지연은 전부 이 속도 상한에서 났다.
    #   ★실기 노드도 같은 누적·상한을 써야 한다.
    #   최대 속도: 벤더 사양은 전 범위 1 s(이 트랙 CLAUDE.md 09.15 "4지 >10 N·엄지 >15 N·전 범위 1 s") → 60 Hz 에서 1/60 = 0.0167/스텝.
    #   기본값 0.005(3.3 s)는 grasp_s2r(DG-5F) 상속값이라 보관 체크포인트 재생용으로 두고, pour_bi_rh iter_09 부터 launch 가
    #   env.synergy_close_speed=0.0167 을 준다.
    hand_action_mode: str = "absolute"
    synergy_contact_freeze: bool = True       # 닿은 손가락의 구동관절 정지 → 감쌈은 하드웨어 결합이 만든다
    synergy_freeze_scope: str = "finger"      # 이 트랙은 finger 만(손가락당 구동관절 1개)
    couple_four_fingers: bool = True          # 엄지 독립, 나머지 4지 채널별 평균
    finger_residual_scale: float = 0.0
    # 대향 관절(엄지 ch1) grip = open + delta — grasp_s2r D3 기본. **소스 팔 부호 기준**,
    # 리시버(좌)는 미러 부호(thumb_2 축 Z → −1)를 env 가 적용한다.
    # ★RH56F1: 0.0 고정(엄지 자세는 프로필 open/grip 표 1.57→1.20 · 0.0→0.24 가 담는다).
    oppose_grip_delta_rad: float = 0.0
    hand_velocity_ff_scale: float = 1.0
    joint_pos_err_max: float = 1.2            # obs 정규화 [rad]
    # ★09.13 부팅 실측: 시작 자세에서 palm↔컵 중심이 약 0.16 m(손끝이 3 cm 앞) — 반경은
    #   그보다 커야 시작 자세에서 닫을 수 있다. 0.22/램프 0.3 → 0.154 m 안쪽은 게이트 1.0.
    # ★RH56F1 시작 자세: palm(0.31,−0.30,0.42) ↔ 컵 원점(0.38,−0.16,0.28) ≈ 0.21 m(FK 계산) —
    #   원본 0.22/램프 0.3 이면 시작에서 게이트 0.18 이라 닫기가 거의 막힌다. 0.30/0.3 → 0.21 m 에서 1.0.
    #   ⚠probe 로 실측 후 조정(`probe_pour_fabric_mimic_boot.py` 부팅 로그의 palm↔컵 거리).
    close_gate_enabled: bool = True
    close_gate_radius: float = 0.30
    close_gate_ramp: float = 0.3

    # ---- 언더액추 백스톱 (grasp_fj_rh 09.07 실측 근거 그대로) ------------------------------
    # 접촉이 구동관절을 하한 밑으로 역구동하면 mimic 요구치가 종속 한계 밖이 되어 두 제약이 동시에
    # 만족 불가 → 솔버가 에너지를 주입(rh_b1: 222 epoch 중 118 에서 mimic 오차 400~916 rad).
    # 종속관절 한계를 부팅에서 이만큼 더 넓힌다(위치 권한은 mimic 제약, 한계는 백스톱). 0 = 자산 그대로.
    mimic_dep_limit_margin_rad: float = 1.5
    # ★09.14 실측(t2r_rh_i00/_m3): 손끝으로 컵을 누르며 드는 순간 mimic 결합이 깨져 종속관절이 154~483 rad 튀고 속도 500~5e4 rad/s
    #   가 난다. 한계 여유 3.0 rad 로도 재현 → 여유는 처방이 아니다. 깨진 env 는 에피소드 끝까지 무의미한 물리로 남으므로
    #   **종속관절 속도가 이 값을 넘으면 runaway 로 종료**한다(사용자 결정 "폭주 오면 리셋"). 건전 기준선 09.07: 자유 폐쇄 ≤7.2,
    #   접촉 킥 수십~수백 rad/s → 100 은 킥 상단·폭주(500+) 사이.
    mimic_runaway_dep_qd: float = 100.0
    # ★09.14 라운드 1(t2r_rh_i00_r1): 오차가 속도 100 rad/s 아래로 천천히 벌어진 폭주 2회(epoch 321-323 39→236 rad,
    #   348-351 7→18 rad)가 위 기준을 빠져나갔다. 정상 epoch 의 전 env 최대 오차는 ≤1.7 rad → 결합 오차 자체도 종료 조건
    #   (사용자 결정 09.14 "2 추가").
    mimic_runaway_err_rad: float = 3.0
    # ★09.17 사용자 결정: 속도 기준은 **연속 이 스텝 수** 지속일 때만 종료. 컵 충돌 순간 튐 한 번에 끊으면 학습 초반
    #   에피소드 50~90 % 가 끊겨 "컵에 닿으면 죽는다"를 가르친다(정황). 결합 오차·비유한 상태는 즉시 종료 유지.
    mimic_runaway_qd_steps: int = 3

    # ---- 접촉 --------------------------------------------------------------------
    contact_force_threshold: float = 1.0      # N — 파지(대향) 판정(env grasp 플래그)
    # 09.17 리셋 진단: 동결이 파지 판정과 같은 1 N 에 걸려 손가락이 1 N 에서 닫기를 멈췄다 → 힘이 1~2 N 에서
    #   더 안 오르고 플래그가 깜빡이며 들기 중 컵을 놓쳤다(probe_rh_reset_diag). 동결은 따로 4 N(사용자 결정).
    #   실기 RH56F1 펌웨어 forceSet 을 같은 값으로 맞춘다.
    # 09.17 사용자 결정(fric 프로브 뒤): 마찰 2.0 과 함께 1.0 N 으로 되돌린다 — 마찰 2.0 에서 제대로 든 시행이
    #   동결 1.0 N 36건 vs 4.0 N 5건(fric_up 753 clean 중). 위 4 N 근거는 마찰 ~0.97 조건에서만 잰 것이다.
    contact_freeze_threshold: float = 1.0     # N — 손가락 닫기 동결 판정
    contact_obs_clip: float = 20.0
    # ---- 손끝 촉각 actor obs (사용자 결정 09.14 "3 추가") ------------------------------------
    # 실기 출처: RH56F1 TouchData1.finger_forces[5] (정전용량 손끝 법선력, 0.01 N 단위, 1024 = 10.24 N).
    # 실기 센서는 무엇에 닿든 재므로 sim 도 컵 필터가 아닌 손끝 링크 전체 접촉력(net). 배포 시 손가락 순서 재배열 필요.
    tactile_obs_clip_n: float = 10.0
    tactile_obs_noise_n: float = 0.1
    # ★09.28 촉각 관측 압축(사용자 "정규화 진행"): > 0 이면 tanh(F / 이 값). 원시 N 은 초반 대부분 0 이라 rl_games 정규화 통계가
    #   평균 0.23·표준편차 0.83(09.28 pour_fj 실측)에 맞춰지고 ±5 clip 에서 약 4.4 N 위가 한 값으로 뭉개졌다.
    #   3 N: 파지 판정 1 N → 0.32, 1.5 N → 0.46, 5 N → 0.93, clip 10 N → 0.997. 0 = 원시 N(구 id 재생용 기본값).
    tactile_obs_tanh_n: float = 0.0
    # ★09.28 actor 관측 정리(사용자: "권하는 정리로, 남기는 것은 손끝−컵"). 두 칸은 다른 칸에서 정확히 계산된다:
    #   손끝−손바닥 = (손끝−컵) + (컵−손바닥), 주둥이 차 = (리시버 컵−소스 컵) + mouth_z·(두 컵 위쪽 방향 차).
    #   True = 구 배치(구 id 재생용 기본값). full-joint 가족은 둘 다 끈다 → 팔마다 −15, 공통 −3.
    obs_tips_rel_palm: bool = True
    obs_mouth_diff: bool = True
    # ---- 보상 입력(RewardContext) 손바닥 축 — 계약 "palm_axes 앞 3칸 = 손바닥 법선" (09.15) --------------
    # RH56F1 palm_sensor 링크: 열 0 = 손가락이 늘어선 가로(기저 ±y), 열 1 = 손가락 길이(기저 +z), 열 2 = 손바닥 법선(기저 +x,
    # 엄지 기저가 이쪽). 열 0·1 을 그대로 넣으면 iter_03 orient 가 손 옆날을 컵으로 돌리게 보상했다. 정책 obs 는 무관.
    ctx_palm_normal_col: int = 2
    ctx_palm_second_col: int = 1
    collision_force_threshold: float = 1.0    # N — 컵끼리·손↔타물체 충돌 지표 임계
    # ---- 엄지 입구 걸림 접근 계측(로그 전용, 09.15 사용자 "지표로깅으로 확인 가능하게") --------------
    # task/{src,rcv}_near_rate · _thumb_over_rim_near · _thumb_above_rim_mm_near. iter_03 rim_hook 과 같은 기하.
    thumb_rim_near_m: float = 0.10            # palm↔컵 원점 거리가 이 안이면 "접근한 env"
    thumb_rim_band_m: float = 0.02            # 엄지 끝 축방향 높이 ≥ 입구 − 이 값이면 입구 높이
    thumb_rim_radial_margin_m: float = 0.025  # 엄지 끝 반경 < 벽(내경+4 mm) + 이 값이면 입구 위
    # ---- 파지 포켓 계측(로그 전용, 09.16 사용자 "대향 여부를 영상 없이 지표로") -----------
    # task/{src,rcv}_thumb_oppose_near · _tip_gap_mm_near · _cup_in_pocket_near.
    # 라운드 7 영상: 배치가 검지-엄지-컵 이라 손가락을 굽혀도 잡을 수 없는데 보상의 pinch_geo 는
    # 그런 자세에도 0.19~0.38 을 지불했다(합성 검증 09.16). 그래서 대향을 따로 잰다.
    pocket_near_m: float = 0.12               # palm↔컵 원점 거리가 이 안이면 "접근한 env"
    pocket_axis_margin_m: float = 0.005       # 컵 축↔(엄지,4지) 선분 거리 < 내경 + 이 값이면 컵이 사이

    # ---- 성공 판정 (pour_v1 계승) — 보상과 분리된 **기준 지표** ----------------------
    # ★09.23 사용자 결정(목표 = 전량 이송·흘림 없음): 0.50/0.40/10 → 0.80/0.25/20.
    #   근거 i13: 절반만 넘기면 성공이 켜져 in_target 0.39~0.46, in_source 0.48 에서 멈췄다(비드 절반이 소스에 남는다).
    #   성공 이후 더 부을 수입(transferred 4.0 × 비율)이 흘림 위험(-12/알)·성공 상실(-20/스텝)보다 작았다.
    #   ★이 변경으로 success/episode_success 는 이전 런과 직접 비교 불가 — 비교는 in_target·흘림/유출로 한다.
    success_fill_ratio: float = 0.80
    success_spill_max: float = 0.25
    success_xy_thresh: float = 0.20           # 두 컵 중심 xy 거리
    success_hold_steps: int = 20              # 10 → 20 스텝(1/3 s) 유지 — 붓고 나서 안정화까지 본다
    # ★09.13 hacking 차단: 소스 컵을 리시버 입구에 끼워 넣으면 소스 안 비드가 리시버 원통 안에 들어와
    #   in_target 로 세어졌다(ep 600 영상: 붓기 없이 성공 0.73). 원점 거리가 이보다 짧으면 성공 무효.
    #   붓는 자세(소스 입구가 리시버 림 위)에서는 원점 거리가 ≥ 12~15 cm 다.
    cups_nested_dist: float = 0.09

    # ---- 종료 --------------------------------------------------------------------
    runaway_joint_vel: float = 20.0
    drop_below_table_m: float = 0.03          # 컵 원점이 (테이블 상면 + 원점오프셋 − 이 값) 아래면 낙하
    # ★09.29 물리 이상 종료(사용자: "물리스텝이 이상한 것들은 환경을 리셋시키는 게 필요"). 0 = 끔(옛 런 재생 불변).
    #   f03 트레이스 3점(ep1750·2100·5400): 받는 컵 속도 > 3 m/s 또는 손 접촉 > 200 N 이 난 env 가 3~13/64 개였고,
    #   그 뒤 종료 없이 중앙 194~419 스텝(최대 788) 이어졌다. 컵은 40 cm 밖(최대 95 m)으로 날아갔고 손 접촉은 최대 9만 N.
    #   정상 범위(같은 트레이스, 리셋 제외): 컵 속도 p99 ≤ 0.3 m/s, 손가락·손바닥 접촉 p99 ≤ 48 N.
    #   오른팔이 로봇 몸에 부딪힌 foreign 접촉(ep2100 p99 101 N, 최대 3,133 N)도 같은 기준으로 끊는다(실기 안전).
    shape_log_every: int = 4                  # ★09.29 파지 모양 로그 계산 주기(스텝) — 놓침 비율은 매 스텝
    phys_cup_speed_max: float = 0.0          # 컵 선속도 [m/s] 초과 즉시 종료
    phys_cup_angvel_max: float = 0.0         # 컵 각속도 [rad/s] 초과 즉시 종료
    phys_contact_max_n: float = 0.0          # 손(손가락·손바닥·외부 접촉) 힘 [N] 이 phys_contact_steps 스텝 연속 초과면 종료
    phys_contact_steps: int = 2
    cup_leash_m: float = 0.0                 # 컵 원점이 스폰에서 수평으로 이보다 멀면 종료
    # ★09.30 쥔 상태 시작 뱅크(사용자: "잡기 이후부터 스타트", "쥔 상태 80 % + 처음 20 %"). 기본 끔(옛 런 재생 불변).
    #   수집: start_bank_capture_path 를 주고 정책을 재생하면 두 손 쥠·두 컵 start_bank_lift_min 이상 들림·소스 세움·비드 소스 안
    #   순간을 모아 npz 로 쓴다. 복원: start_bank_frac 비율의 리셋을 뱅크 상태에서 시작(hold 건너뜀). start_bank.py.
    start_bank_path: str = ""
    start_bank_frac: float = 0.0
    start_bank_capture_path: str = ""
    start_bank_capture_n: int = 512
    start_bank_lift_min: float = 0.05

    console_log_interval: int = 600

    # ---- s2r 관측 (09.14) -----------------------------------------------------------
    # 상시(ADR 무관) 관절·FK 노이즈. 실측 근거는 grasp_s2r cfg `obs_noise_*` 주석:
    #   qpos 실측 ~0.001 rad(엔코더 LSB), qvel 운동 구간 0.05, body(FK) 0.005 m.
    obs_noise_qpos: float = 0.002
    obs_noise_qvel: float = 0.05
    obs_noise_body: float = 0.005
    # 컵 pose 지각(FP++) — 지연 큐 + 코히런트 노이즈. ADR 축(초기 → 종점):
    perception_delay_max_steps: int = 3       # 종점: 0~3 정책스텝(0~50 ms) 균등
    perception_delay_base_steps: int = 0
    obs_noise_object_xyz: float = 0.0         # 초기 [m]
    adr_obs_noise_object_xyz_max: float = 0.015
    obs_noise_object_rot_deg: float = 0.0
    adr_obs_noise_object_rot_max_deg: float = 3.0

    # ---- 물리 DR / ADR (09.14) --------------------------------------------------------
    enable_events: bool = True
    events: PourFabricEventCfg = PourFabricEventCfg()
    # 재질은 처음부터 고정 범위(런타임 확장 불가). 09.17 사용자 결정: (0.7, 1.2) → 2.0 고정(fric_up 조건과 동일).
    cup_friction_range: tuple = (2.0, 2.0)
    enable_adr: bool = True
    adr_num_increments: int = 30
    adr_increment_interval: int = 3000        # 정책 스텝
    adr_trigger_threshold: float = 0.30       # 순간 성공률(success_now 평균)
    # 들린 컵 외란(질량정규화, grasp_s2r W1 값). ADR 로 0 → 종점 스케일.
    wrench_force_scale_max: float = 5.0       # N/kg
    wrench_torque_scale_max: float = 0.5      # N·m/kg
    wrench_prob_range: tuple = (0.001, 0.1)
    wrench_lift_min_m: float = 0.03           # 이만큼 들렸을 때만 외란

    # ---- 파생 자산 cfg -------------------------------------------------------------
    robot_cfg: ArticulationCfg = None
    source_cup_cfg: RigidObjectCfg = None
    receiver_cup_cfg: RigidObjectCfg = None
    beads_cfg = None
    table_spawn: sim_utils.UsdFileCfg = TABLE_SPAWN
    source_contact_filter: tuple = ()
    receiver_contact_filter: tuple = ()

    observation_space: int = 0
    action_space: int = 0
    state_space: int = 0
    # 파생 폭(로깅·검증용) — resolve_cfg 가 채운다
    num_actions_per_side: int = 0

    def __post_init__(self) -> None:
        resolve_cfg(self)


def _hand_action_width(profile) -> int:
    """per_finger: 구동관절 수 = 액션 슬롯 수 (`hand_finger_channels` 가 1:1 을 준다)."""
    slots = sorted(int(s) for m in profile.hand_finger_channels.values() for s in m.values())
    if slots != list(range(profile.num_hand_joints)):
        raise ValueError(f"[{profile.name}] hand_finger_channels 슬롯 {slots} 이 구동 {profile.num_hand_joints} 과 1:1 이 아니다")
    return profile.num_hand_joints


def _validate_mimic_fields(cfg: "PourFabricMimicEnvCfg", pair) -> None:
    """언더액추 손에서 **조용히 무효가 되는 조합**을 cfg 단계에서 죽인다(grasp_fj_rh 규약)."""
    errs = []
    if float(cfg.oppose_grip_delta_rad) != 0.0:
        errs.append("oppose_grip_delta_rad 는 tesollo 엄지 대향(_2) 전용 — RH56F1 은 0.0")
    if str(cfg.synergy_freeze_scope) != "finger":
        errs.append("synergy_freeze_scope 는 finger 만(손가락당 구동관절 1개)")
    if float(cfg.mimic_dep_limit_margin_rad) < 0.0:
        errs.append("mimic_dep_limit_margin_rad 는 ≥ 0")
    if float(cfg.mimic_runaway_err_rad) <= 0.0:
        errs.append("mimic_runaway_err_rad 는 > 0 (0 이면 모든 env 가 매 스텝 종료)")
    if int(cfg.mimic_runaway_qd_steps) < 1:
        errs.append("mimic_runaway_qd_steps 는 ≥ 1 (1 = 튐 한 번에 종료)")
    cols = (int(cfg.ctx_palm_normal_col), int(cfg.ctx_palm_second_col))
    if not all(0 <= c <= 2 for c in cols) or cols[0] == cols[1]:
        errs.append(f"ctx_palm_normal_col/ctx_palm_second_col 는 0~2 의 서로 다른 열: {cols}")
    for p in (pair.source, pair.receiver):
        for name, spec in p.actuator_specs.items():
            if name.endswith("_hand_mimic") and (spec.get("stiffness"), spec.get("damping")) != (0.0, 0.0):
                errs.append(f"{p.name}:{name} 게인이 0/0 이 아니다 — PhysX mimic 제약과 싸운다(09.07 발산)")
    if errs:
        raise RuntimeError("[pour_fabric_mimic cfg] " + " · ".join(errs))


def resolve_cfg(cfg: "PourFabricMimicEnvCfg") -> None:
    """스위치 → 자산 cfg · 차원 파생. **멱등** — hydra 오버라이드 후 env 가 재호출한다."""
    pair = _bm.get_pair(cfg.pair_name)

    cfg.robot_cfg = build_robot_cfg(pair, self_collisions=cfg.enable_self_collisions,
                                    gravity=cfg.enable_gravity)
    s = float(cfg.cup_scale)
    for k, v in CUP_GEOM_UNIT.items():
        setattr(cfg, k, v * s)
    cfg.source_cup_cfg = build_cup_cfg(SOURCE_CUP_PRIM, s)
    cfg.receiver_cup_cfg = build_cup_cfg(RECEIVER_CUP_PRIM, s)
    cfg.beads_cfg = make_beads_cfg(_ASSETS_DIR, n=int(cfg.bead_count))
    cfg.source_contact_filter = (SOURCE_CUP_BODY,)
    cfg.receiver_contact_filter = (RECEIVER_CUP_BODY,)

    # 물리 DR: 컵 마찰은 고정 범위(런타임 확장 불가) — term 생성 전인 cfg 단계에서 적용.
    if not bool(cfg.enable_events):
        cfg.events = None
    elif cfg.events is not None:
        lo, hi = float(cfg.cup_friction_range[0]), float(cfg.cup_friction_range[1])
        for term in (cfg.events.source_cup_material, cfg.events.receiver_cup_material):
            term.params["static_friction_range"] = (lo, hi)
            term.params["dynamic_friction_range"] = (lo, hi)

    _validate_mimic_fields(cfg, pair)
    for p in (pair.source, pair.receiver):
        if _hand_action_width(p) != _hand_action_width(pair.source):
            raise ValueError("양팔 손 액션 폭이 다르다 — 같은 손 자산이어야 한다")
    hand_w = _hand_action_width(pair.source)
    arm_w = pair.source.num_arm_joints if str(cfg.arm_control) == "joint" else 6
    cfg.num_actions_per_side = arm_w + hand_w
    cfg.action_space = 2 * cfg.num_actions_per_side

    # policy obs (팔마다): arm q/qd(2·A) + hand q(H) + palm_pos 3 + palm_axes 6
    #   + tips_rel_palm 3F + palm_to_cup 3 + cup_to_tips 3F + joint_err H + cup_up 3
    #   ★09.14 hand_qd(H) 는 actor 에서 뺐다(실기 드라이버 velocity ≠ 관절속도) — critic 에만.
    # + 공통: src_cup→rcv_cup 3 + 주둥이→개구 3 + prev_action Dact
    per = 0
    hand_qd_total = 0
    for p in (pair.source, pair.receiver):
        a, h, f = p.num_arm_joints, p.num_hand_joints, len(p.finger_sensor_bodies)
        per += 2 * a + h + 3 + 6 + 3 * f + 3 + 3 * f + h + 3 + f  # tactile
        if not bool(cfg.obs_tips_rel_palm):
            per -= 3 * f                                            # 손끝−손바닥 제거
        if bool(cfg.obs_fabric_state) or str(cfg.arm_control) == "joint":
            per += a                                                # fabric 팔 q (DEXTRAH) | 팔 지령 q*_{t-1} (joint)
        hand_qd_total += h
    cfg.observation_space = per + 3 + (3 if bool(cfg.obs_mouth_diff) else 0) + cfg.action_space
    # critic = policy(clean) + hand_qd(2H) + 비드 분율 4 + 비드 무게중심(rcv 프레임) 3
    #        + 두 컵 lin/ang vel 12 + 진행도 1 + 손가락 접촉력 2F
    f_src = len(pair.source.finger_sensor_bodies)
    f_rcv = len(pair.receiver.finger_sensor_bodies)
    cfg.state_space = cfg.observation_space + hand_qd_total + 4 + 3 + 12 + 1 + f_src + f_rcv


@configclass
class PourFabricMimicEnvCfg_PLAY(PourFabricMimicEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 50
        super().__post_init__()


@configclass
class PourFJEnvCfg(PourFabricMimicEnvCfg):
    """★09.28 pour_bi_rh 4번째 사이클(사용자: "이제부터 제어 방식은 full-joint 으로만"): 양팔 full-joint 13×2 = 26.

    팔 = 관절 증분 + EMA(`arm_control=joint`), 손 = 구동 관절별 절대 목표 + EMA + 벤더 속도 상한(`hand_control=direct`).
    fabric 적분·fabric 관측·fabric ADR·palm 지령 없음. 새 gym id(`open-<pair>_b_pour_fj`)로만 쓴다 — 구 id·체크포인트 재생 불변.
    """

    def __post_init__(self) -> None:
        self.arm_control = "joint"
        self.hand_control = "direct"
        self.obs_fabric_state = False
        self.fabric_adr = False
        self.enable_self_collisions = True
        self.hand_velocity_ff_scale = 0.0     # grasp_fj hand_direct 규약: 손 속도 목표 0(위치 목표만)
        self.tactile_obs_tanh_n = 3.0         # 촉각 관측 tanh(F/3 N)
        self.hand_direct_range = "grip"       # 09.28 사용자 2.a
        self.hand_direct_contact_freeze = True   # 09.28 사용자 1번 (임계 contact_freeze_threshold 1.0 N)
        self.obs_tips_rel_palm = False        # 관측 정리: 손끝−컵 을 남긴다
        self.obs_mouth_diff = False
        # ★사용자(09.28): 잡음·지연 모델은 과제 성공 전까지 모두 끈다 — 관절·FK·촉각 잡음, 컵 지각 지연·잡음(ADR 종점 포함).
        #   물리 DR(마찰·질량 이벤트)과 들린 컵 외란(wrench)은 관측 잡음·지연이 아니라 그대로 둔다.
        self.obs_noise_qpos = 0.0
        self.obs_noise_qvel = 0.0
        self.obs_noise_body = 0.0
        self.tactile_obs_noise_n = 0.0
        self.obs_noise_object_xyz = 0.0
        self.adr_obs_noise_object_xyz_max = 0.0
        self.obs_noise_object_rot_deg = 0.0
        self.adr_obs_noise_object_rot_max_deg = 0.0
        self.perception_delay_base_steps = 0
        self.perception_delay_max_steps = 0
        super().__post_init__()
