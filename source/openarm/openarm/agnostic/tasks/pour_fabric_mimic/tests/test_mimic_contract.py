"""pour_fabric_mimic 계약 — 저차원(언더액추) 손 전용 트랙이 원본 pour_fabric 과 갈라져야 하는 곳을 잠근다.

Isaac 불요(소스·프로필 데이터). 차원 대조는 Isaac 있을 때만.
"""
import io
import re
import tokenize
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openarm.agnostic.modules import robot_profiles as _rp
from openarm.agnostic.tasks.pour_fabric_mimic import bimanual as _bm
from openarm.agnostic.tasks.pour_fabric_mimic import robot_profiles as _local

_PKG = Path(__file__).resolve().parents[1]


def _code_only(src: str) -> str:
    """주석·문자열(독스트링) 제거 — 리터럴 검사는 **코드**에만 건다(설명문은 로봇 이름을 써야 한다)."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


_ENV = (_PKG / "pour_fabric_env.py").read_text()
_RIG = (_PKG / "side_rig.py").read_text()
_CFG = (_PKG / "pour_fabric_env_cfg.py").read_text()
_ORIG = _PKG.parent / "pour_fabric"
_HDGP = Path(__file__).resolve().parents[7]
_ASSET = _HDGP / "assets" / "robot" / "openarm_rh56f1_bi_rl"
assert (_ASSET / "openarm_rh56f1_bi_rl.urdf").exists(), _ASSET


# =============================================================================
# 원본 불변 · 이 트랙은 별도 폴더
# =============================================================================
def test_original_pour_fabric_untouched_by_this_track():
    """원본에 mimic 문자열이 새로 들어가면 사용자 결정("원본 냅두고") 위반."""
    for f in ("side_rig.py", "pour_fabric_env.py", "pour_fabric_env_cfg.py", "bimanual.py"):
        assert "mimic" not in (_ORIG / f).read_text(), f
    assert "openarm_rh56f1_bi_rl" not in (_ORIG / "bimanual.py").read_text()


@pytest.mark.parametrize("name", ["env", "rig", "cfg"])
def test_no_robot_specific_literals(name):
    src = {"env": _ENV, "rig": _RIG, "cfg": _CFG}[name]
    body = _code_only("\n".join(l for l in src.splitlines() if not l.startswith("import fabrics_sim")))
    for lit in ("r_aj_", "l_aj_", "r_hj_", "l_hj_", "r_hl_", "l_hl_", "tesollo", "dg5f", "rh56f1_right", "RH56F1_"):
        assert lit not in body, f"{name}: 로봇 리터럴 '{lit}' — 프로필로 옮길 것"


# =============================================================================
# 프로필 · 쌍
# =============================================================================
def test_pair_rh_is_default_and_only_from_local_profiles():
    assert _bm.DEFAULT_PAIR == "rh" and set(_bm.PAIRS) == {"rh"}
    p = _bm.get_pair("rh")
    assert p.source.name == "rh56f1_right_fab" and p.receiver.name == "rh56f1_left_fab"
    assert p.usd_relpath.endswith("openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.usd")
    # 모듈 레지스트리에는 안 들어간다(grasp config 가 그 dict 로 gym id 를 찍는다)
    assert "rh56f1_right_fab" not in _rp.PROFILES and "rh56f1_left_fab" not in _rp.PROFILES


def test_fabric_enabled_on_both_sides_with_26dof_order():
    p = _bm.get_pair("rh")
    assert p.source.fabric_class == "OpenArmRh56f1PoseFabric"
    assert p.receiver.fabric_class == "OpenArmRh56f1LeftPoseFabric"
    assert len(_local.FABRIC_JOINT_ORDER) == 26
    assert p.source.fabric_joint_order == p.receiver.fabric_joint_order == _local.FABRIC_JOINT_ORDER
    assert _bm.fabric_slots(p.source) == (slice(0, 7), slice(7, 13))
    assert _bm.fabric_slots(p.receiver) == (slice(13, 20), slice(20, 26))
    # 슬라이스 관절 = 프로필 관절(순서까지)
    for prof in (p.source, p.receiver):
        a, h = _bm.fabric_slots(prof)
        order = _local.FABRIC_JOINT_ORDER
        assert all(re.fullmatch(prof.arm_joint_regex, n) for n in order[a])
        assert tuple(order[h]) == prof.hand_joint_names


def test_left_is_mirror_of_right_arm_sign_and_hand_identity():
    """09.14 순수 FK: 팔 (−1,−1,−1,+1,−1,−1,−1) · 손 +1 → 4지 손끝 0.00 mm 일치."""
    p = _bm.get_pair("rh")
    init = p.init_joint_pos
    sign = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
    for i, s in enumerate(sign, start=1):
        assert init[f"l_aj_{i}"] == pytest.approx(s * init[f"r_aj_{i}"])
    for n in ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1"):
        assert init[f"l_hj_{n}"] == pytest.approx(init[f"r_hj_{n}"])
    assert p.receiver.hand_open_pose == p.source.hand_open_pose
    assert p.receiver.hand_grip_pose == p.source.hand_grip_pose
    assert p.receiver.object_spawn_center == (p.source.object_spawn_center[0], -p.source.object_spawn_center[1])
    assert p.receiver.palm_box_min[1] == -p.source.palm_box_max[1]


def test_left_hand_limits_admit_the_mirrored_poses():
    """좌 URDF 한계 안에 open/grip 이 들어가는지(부호 +1 이 성립하는 전제)."""
    urdf = next(_ASSET.glob("*.urdf"))
    lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
           for j in ET.parse(urdf).getroot().iter("joint") if j.find("limit") is not None and j.get("type") == "revolute"}
    p = _bm.get_pair("rh").receiver
    for nm, o, g in zip(p.hand_joint_names, p.hand_open_pose, p.hand_grip_pose):
        lo, hi = lim[nm]
        assert lo - 1e-6 <= o <= hi + 1e-6 and lo - 1e-6 <= g <= hi + 1e-6, (nm, o, g, lo, hi)


def test_dependent_actuators_zero_gain_both_sides():
    p = _bm.get_pair("rh")
    mimic = {k: v for k, v in p.actuator_specs.items() if k.endswith("_hand_mimic")}
    assert set(mimic) == {"src_right_hand_mimic", "rcv_left_hand_mimic"}
    for v in mimic.values():
        assert (v["stiffness"], v["damping"]) == (0.0, 0.0)


def test_per_finger_slots_bijective_with_driven_joints():
    p = _bm.get_pair("rh")
    for prof in (p.source, p.receiver):
        slots = sorted(int(s) for m in prof.hand_finger_channels.values() for s in m.values())
        assert slots == list(range(prof.num_hand_joints))
        assert prof.hand_freeze_suffixes and set(prof.hand_freeze_suffixes) >= {"1", "2"}


# =============================================================================
# 소스 계약 — 원본과 갈라진 곳
# =============================================================================
def test_fabric_ctor_uses_bimanual_class_signature():
    assert 'use_hand_fabric=False, hand_mode="direct")' in _RIG
    code = _code_only(_RIG)
    assert "robot_dir_name" not in code and "tip_per_finger" not in code
    assert "def sync_other" in _RIG and "self.src.sync_other(self.rcv)" in _ENV and "self.rcv.sync_other(self.src)" in _ENV
    assert "self.fabric_q[:, self.arm_sl]" in _RIG and "[:, :n_arm]" not in _RIG


def test_hand_layout_is_per_finger_with_coupling_matrix():
    assert "self.syn_slot" in _RIG and "self.syn_couple" in _RIG
    assert "a_hand.view(N, nf, self.syn_nch)" not in _RIG
    assert "hold = (h_mid | h_dist) & self.syn_flex" in _RIG
    assert 'self.syn_flex = torch.tensor([s in p.hand_freeze_suffixes for s in sfx]' in _RIG
    assert 's == "3"' not in _RIG


def test_mimic_gates_present_and_read_from_asset():
    for fn in ("_load_mimic_pairs", "_assert_mimic_constraints_present", "_assert_hand_pose_usable", "_widen_dependent_joint_limits"):
        assert f"def {fn}" in _ENV
    assert "physxMimicJoint:rotZ:gearing" in _ENV and 'j.find("mimic")' in _ENV
    assert "ctrl/mimic_err_max" in _ENV
    assert "self._widen_dependent_joint_limits()" in _ENV


def test_cfg_kills_tesollo_only_knobs():
    assert re.search(r"oppose_grip_delta_rad:\s*float\s*=\s*0\.0", _CFG)
    assert re.search(r'synergy_freeze_scope:\s*str\s*=\s*"finger"', _CFG)
    m = float(re.search(r"mimic_dep_limit_margin_rad:\s*float\s*=\s*([0-9.]+)", _CFG).group(1))
    assert m >= 1.0
    assert "_validate_mimic_fields(cfg, pair)" in _CFG


def test_asset_actually_carries_mimic_multipliers_both_hands():
    urdf = next(_ASSET.glob("*.urdf"))
    pairs = {j.get("name"): float(j.find("mimic").get("multiplier"))
             for j in ET.parse(urdf).getroot().iter("joint") if j.find("mimic") is not None}
    for s in ("r", "l"):
        assert pairs[f"{s}_hj_thumb_3"] == pytest.approx(1.1425)
        assert pairs[f"{s}_hj_thumb_4"] == pytest.approx(0.7508)
        for f in ("index", "middle", "ring", "pinky"):
            assert pairs[f"{s}_hj_{f}_2"] == pytest.approx(1.1169)


def test_contact_filters_point_at_rigid_body_prim():
    """shaker 강체는 baseLink 하위 — 필터가 루트를 가리키면 힘이 조용히 0(관측 #0230)."""
    assert "cfg.source_contact_filter = (SOURCE_CUP_BODY,)" in _CFG
    assert "prim_path=_cfg.SOURCE_CUP_BODY, filter_prim_paths_expr=[_cfg.RECEIVER_CUP_BODY]" in _ENV
    assert "find_matching_prim_paths(expr)" in _ENV
    from pxr import Usd
    stage = Usd.Stage.Open(str(_HDGP / "assets" / "cup" / "shaker_closed_thick_rl.usd"))
    rb = [p.GetName() for p in stage.Traverse() if "PhysicsRigidBodyAPI" in p.GetAppliedSchemas()]
    assert rb == ["baseLink"], rb


def test_mimic_blowup_terminates_episode():
    """09.14 사용자 결정: mimic 결합이 깨진 env(종속관절 속도 폭주)는 리셋한다 — 한계 여유는 처방이 아니었다."""
    assert re.search(r"mimic_runaway_dep_qd:\s*float\s*=\s*([0-9.]+)", _CFG)
    assert "terminated = runaway | mimic_runaway | self._dropped" in _ENV
    assert 'self.extras["done/mimic_runaway"]' in _ENV


def test_transient_dep_qd_spike_does_not_terminate():
    """09.17 사용자 결정: 컵 충돌 순간의 종속관절 속도 튐 한 번으로는 에피소드를 끝내지 않는다(학습 초반 에피소드
    50~90 % 가 이 종료로 끊겨 "컵에 닿으면 죽는다"를 가르쳤을 정황). 속도 기준은 연속 K 스텝 지속일 때만,
    결합 오차·비유한 상태는 즉시 종료."""
    m = re.search(r"mimic_runaway_qd_steps:\s*int\s*=\s*([0-9]+)", _CFG)
    assert m and int(m.group(1)) >= 2
    code = _code_only(_ENV)
    assert "self . _mim_qd_streak >= int ( cfg . mimic_runaway_qd_steps )" in code
    assert "torch . isfinite" in code.split("def _get_dones")[1].split("def _reset_idx")[0]
    assert "self . _mim_qd_streak [ env_ids ] = 0" in code
    assert 'self.extras["done/mimic_qd_spike"]' in _ENV
    assert "mimic_runaway_qd_steps" in _CFG.split("def _validate_mimic_fields")[1]


def test_slow_mimic_drift_also_terminates():
    """09.14 라운드 1: 오차가 epoch 321-323(39→236 rad)·348-351(7→18 rad) 동안 속도 100 rad/s 아래로 천천히 벌어져
    속도 기준을 빠져나갔다 → 사용자 결정 "2 추가": 결합 오차 자체도 종료 조건."""
    m = re.search(r"mimic_runaway_err_rad:\s*float\s*=\s*([0-9.]+)", _CFG)
    assert m and 1.7 < float(m.group(1)) < 7.0, "정상 epoch 최대 1.7 rad · 느린 폭주 시작 7 rad 사이"
    code = _code_only(_ENV)
    assert "cfg . mimic_runaway_err_rad" in code
    assert "| ( mim_err > float ( cfg . mimic_runaway_err_rad ) ) |" in code
    assert 'self.extras["done/mimic_err_runaway"]' in _ENV
    assert "mimic_runaway_err_rad" in _CFG.split("def _validate_mimic_fields")[1]


def test_fingertip_tactile_obs_is_sim2real_shaped():
    """09.14 사용자 결정 "3 추가": RH56F1 실기 손끝 촉각(TouchData1.finger_forces[5], 0.01 N)에 대응하는 5칸/손.
    실기 센서는 무엇에 닿든 재므로 sim 도 컵 필터(force_matrix_w)가 아니라 손끝 링크 **전체** 접촉력(net_forces_w)."""
    rig = _code_only(_RIG)
    assert "def tip_tactile ( self )" in rig
    body = _RIG.split("def tip_tactile(self)")[1].split("\n    def ")[0]
    assert "net_forces_w" in body and "force_matrix_w" not in body
    assert "self.sensors[f][-1]" in body, "프로필 규약 (중간, 원위, 팁) — 마지막이 팁"
    env = _code_only(_ENV)
    assert env.count("self . _tactile ( self . src , noisy = True )") == 1
    assert env.count("self . _tactile ( self . rcv , noisy = True )") == 1
    assert env.count("self . _tactile ( self . src , noisy = False )") == 1
    assert env.count("self . _tactile ( self . rcv , noisy = False )") == 1
    assert re.search(r"tactile_obs_clip_n:\s*float\s*=\s*10\.0", _CFG), "실기 1024 = 10.24 N 포화"
    assert re.search(r"tactile_obs_noise_n:\s*float\s*=", _CFG)
    assert "+ f  # tactile" in _CFG, "관측 차원식에 손끝 촉각 f 칸"


def test_ctx_palm_axes_put_palm_normal_first():
    """09.15: RewardContext 계약은 "palm_axes 앞 3칸 = 손바닥 법선". RH56F1 palm_sensor 는 열 2 가 법선(URDF, 아래 테스트)이라
    열 0·1 을 그대로 넣으면 생성 보상(iter_03 orient)이 손 옆날을 컵으로 돌리는 방향을 보상한다 → env 가 [법선, 손가락 방향] 순서로 넣는다."""
    m = re.search(r"ctx_palm_normal_col:\s*int\s*=\s*(\d)", _CFG)
    m2 = re.search(r"ctx_palm_second_col:\s*int\s*=\s*(\d)", _CFG)
    assert m and int(m.group(1)) == 2
    assert m2 and int(m2.group(1)) == 1
    code = _code_only(_ENV)
    assert code.count("src_palm_axes = self . _ctx_palm_axes ( self . src )") == 1
    assert code.count("rcv_palm_axes = self . _ctx_palm_axes ( self . rcv )") == 1
    assert "palm_R ( ) [ : , : , 0 ] , self . src . palm_R ( ) [ : , : , 1 ]" not in code
    assert "palm_R ( ) [ : , : , 0 ] , self . rcv . palm_R ( ) [ : , : , 1 ]" not in code


def _rpy_matrix(rpy):
    import math
    r, p_, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p_), math.sin(p_), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


@pytest.mark.parametrize("s", ["r", "l"])
def test_rh56f1_palm_sensor_column2_is_palmar(s):
    """URDF 실측: palm_sensor 열 2 = 손 기저 +x, 엄지 기저가 네 손가락보다 +x 쪽(손바닥 쪽). 열 0 = 손가락 늘어선 가로(±y)."""
    root = ET.parse(_ASSET / "openarm_rh56f1_bi_rl.urdf").getroot()
    J = {j.get("name"): j for j in root.findall("joint")}
    def origin(name):
        o = J[name].find("origin")
        return [float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()]
    for n in (f"{s}_hj_palm_1", f"{s}_hj_palm_2"):
        assert all(abs(v) < 1e-6 for v in origin(n)[1]), f"{n} 회전 없음 가정(palm_2 프레임 = 기저 프레임)"
    assert J[f"{s}_hj_palm_sensor"].find("parent").get("link") == f"{s}_hl_palm_2"
    R = _rpy_matrix(origin(f"{s}_hj_palm_sensor")[1])
    col = lambda k: [R[i][k] for i in range(3)]
    assert all(abs(a - b) < 1e-3 for a, b in zip(col(2), (1.0, 0.0, 0.0))), col(2)
    assert abs(abs(col(0)[1]) - 1.0) < 1e-3, col(0)
    thumb_x = origin(f"{s}_hj_thumb_1")[0][0]
    fingers_x = [origin(f"{s}_hj_{f}_1")[0][0] for f in ("index", "pinky")]
    assert thumb_x > max(fingers_x) + 0.01, "엄지 기저가 +x(손바닥) 쪽"


@pytest.mark.parametrize("s", ["r", "l"])
def test_rh56f1_contact_bodies_carry_the_distal_collider(s):
    """09.15: 손끝 마디 `_2`·`_sensor`·`_tip`(엄지 thumb_4·sensor·tip)은 벤더 STL 이 같은 입체라 collider 는 `_sensor` 에만 둔다.
    env 가 읽는 접촉 링크는 전부 collider 가 있어야 하고(없으면 접촉이 조용히 0), 사본 링크엔 없어야 한다(겹친 강체에 접촉이 나뉜다)."""
    P = pytest.importorskip("openarm.agnostic.tasks.pour_fabric_mimic.robot_profiles")
    prof = P.RH56F1_RIGHT_FAB if s == "r" else P.RH56F1_LEFT_FAB
    root = ET.parse(_ASSET / "openarm_rh56f1_bi_rl.urdf").getroot()
    has_col = {link.get("name"): bool(link.findall("collision")) for link in root.findall("link")}
    bodies = [b for bs in prof.finger_sensor_bodies.values() for b in bs] + [prof.palm_body]
    assert all(has_col[b] for b in bodies), [b for b in bodies if not has_col[b]]
    copies = [f"{s}_hl_thumb_4", f"{s}_hl_thumb_tip",
              *(f"{s}_hl_{f}_{k}" for f in ("index", "middle", "ring", "pinky") for k in ("2", "tip"))]
    assert not any(has_col[c] for c in copies), [c for c in copies if has_col[c]]


# 원본 계약 중 그대로 유지돼야 하는 것(보상 없음 · 성공은 env · a=0 = 앵커)
def test_inherited_contracts_hold():
    assert "load_reward_fn" in _ENV and "RewardContext(" in _ENV
    assert not re.search(r"_weight\b", _ENV)
    assert "success=self._success_now" in _ENV and "& (~self._cups_nested)" in _ENV
    assert "torch.where(a6 >= 0.0, a6 * self.delta_hi, -a6 * self.delta_lo)" in _RIG
    assert "torch.where(delta > 0.0, delta * g, delta)" in _RIG and "hold & (delta > 0.0)" in _RIG


# =============================================================================
# 차원 (Isaac 필요)
# =============================================================================
def test_dims_from_resolve_cfg():
    pytest.importorskip("pxr"); pytest.importorskip("isaaclab")
    from openarm.agnostic.tasks.pour_fabric_mimic import pour_fabric_env_cfg as C
    cfg = C.PourFabricMimicEnvCfg()
    assert cfg.action_space == 2 * (6 + 6) == 24
    per = 0
    for p in (cfg_pair := _bm.get_pair(cfg.pair_name)).source, cfg_pair.receiver:
        a, h, f = p.num_arm_joints, p.num_hand_joints, len(p.finger_sensor_bodies)
        per += 2 * a + h + 3 + 6 + 3 * f + 3 + 3 * f + h + 3 + f  # 09.14 손끝 촉각 f 칸
    assert cfg.observation_space == per + 6 + 24 == 182
    assert cfg.state_space == 182 + 12 + 4 + 3 + 12 + 1 + 10 == 224
    from openarm.agnostic.tasks.pour_fabric_mimic import config as reg
    assert reg.REGISTERED == {"rh": "open-rh_b_pour_fab_mimic"}


def test_thumb_rim_approach_is_logged_not_rewarded():
    """09.15 사용자 "다음부터 지표로깅으로 확인 가능하게": 엄지 입구 걸림 접근을 영상 없이 본다.
    양손 near 비율·near 중 엄지 입구 위 비율·엄지-입구 높이(mm) 를 로그로만 낸다 — obs·RewardContext 에는 넣지 않는다."""
    for side in ("src", "rcv"):
        for k in ("near_rate", "thumb_over_rim_near", "thumb_above_rim_mm_near"):
            assert f'f"task/{{side}}_{k}"' in _ENV or f"task/{side}_{k}" in _ENV
    assert _ENV.count("self._log_thumb_rim(") == 2
    for f in ("thumb_rim_near_m", "thumb_rim_band_m", "thumb_rim_radial_margin_m"):
        assert re.search(rf"{f}:\s*float\s*=", _CFG), f
    ctx_block = _ENV.split("return RewardContext(")[1].split("\n        )")[0]
    assert "thumb_rim" not in ctx_block and "thumb_over_rim" not in ctx_block


def test_t2r_contract_is_track_local_not_shared():
    """09.17: 공용 `modules/t2r` 에 다른 트랙이 필수 필드를 추가해 이 트랙 env 가 기동 즉시 죽었다
    (RewardContext.__init__() missing 'bead_fill_level', 'premature_tilt').
    사용자 결정 "현재 세션을 공용 모듈과 분리" — 계약·로더·검증기·프롬프트는 트랙 안에 둔다."""
    local = _PKG / "t2r"
    for f in ("context.py", "loader.py", "validator.py", "prompts.py", "__init__.py"):
        assert (local / f).exists(), f
    assert "from .t2r.context import RewardContext" in _ENV
    assert "from .t2r.loader import" in _ENV
    assert "modules.t2r" not in _ENV, "공용 t2r 를 다시 import 하면 같은 사고가 재발한다"
    ctx_src = (local / "context.py").read_text()
    for gone in ("bead_fill_level", "premature_tilt"):
        assert gone not in ctx_src, f"{gone} 은 이 트랙 env 가 만들지 않는다"


def test_grasp_pocket_is_logged_not_rewarded():
    """09.16 사용자 "대향 여부를 영상 없이 지표로": 라운드 7 배치는 검지-엄지-컵 이라 잡을 수 없는데
    보상 pinch_geo 는 0.19~0.38 을 지불했다. 대향·엄지↔4지 간극·컵이 사이인지를 로그로만 낸다."""
    for side in ("src", "rcv"):
        for k in ("thumb_oppose_near", "tip_gap_mm_near", "cup_in_pocket_near"):
            assert f'f"task/{{side}}_{k}"' in _ENV or f"task/{side}_{k}" in _ENV
    assert _ENV.count("self._log_grasp_pocket(") == 2
    for f in ("pocket_near_m", "pocket_axis_margin_m"):
        assert re.search(rf"{f}:\s*float\s*=", _CFG), f
    ctx_block = _ENV.split("return RewardContext(")[1].split("\n        )")[0]
    assert "oppose" not in ctx_block and "pocket" not in ctx_block


# =============================================================================
# ★09.28 증분 palm 지령 (DG-5F pour_fabric 9d300ef4 이식)
# =============================================================================
def _rig_helpers():
    """side_rig.py 의 모듈 수준 증분 도우미만 떼어 실행한다(isaaclab 불요)."""
    import ast
    import math
    import torch
    tree = ast.parse(_RIG)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("palm_vmax_vec", "step_palm_increment", "leash_palm_target")]
    ns = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "side_rig_helpers", "exec"), ns)
    return ns, torch


def test_palm_action_mode_default_absolute_and_branch_present():
    assert re.search(r'palm_action_mode:\s*str\s*=\s*"absolute"', _CFG)
    code = _code_only(_ENV)
    assert "rig . step_palm_target (" in code and "rig . compose_palm_target (" in code
    assert "self . _palm_cmd [ env_ids ] = 0.0" in code     # 리셋 때 EMA 상태를 비운다


def test_ctx_carries_palm_linear_velocity():
    code = _code_only(_ENV)
    assert "src_palm_lin_vel = self . src . palm_lin_vel ( )" in code
    assert "rcv_palm_lin_vel = self . rcv . palm_lin_vel ( )" in code


def test_increment_step_accumulates_and_clips_to_delta_box():
    ns, torch = _rig_helpers()
    vmax = ns["palm_vmax_vec"](0.6, 180.0)
    anchor = torch.zeros(1, 6)
    lo = torch.tensor([-0.1] * 3 + [-0.5] * 3); hi = torch.tensor([0.1] * 3 + [0.5] * 3)
    t = anchor.clone()
    for _ in range(6):                                          # +x 최대 속도 6 스텝(60 Hz)
        t = ns["step_palm_increment"](t, torch.tensor([[1.0, 0, 0, 0, 0, 0]]), vmax, 1 / 60, anchor, lo, hi)
    assert abs(float(t[0, 0]) - 0.06) < 1e-6                   # 0.6 m/s x 0.1 s
    for _ in range(60):
        t = ns["step_palm_increment"](t, torch.tensor([[1.0, 0, 0, 0, 0, 0]]), vmax, 1 / 60, anchor, lo, hi)
    assert abs(float(t[0, 0]) - 0.1) < 1e-6                    # 델타 박스 끝에서 멈춘다


def test_leash_ties_target_to_measured_palm():
    ns, torch = _rig_helpers()
    tgt = torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0, 1.0]]); pose = torch.zeros(1, 6)
    out = ns["leash_palm_target"](tgt, pose, 0.06, 30.0)
    assert abs(float(out[0, :3].norm()) - 0.06) < 1e-6
    assert abs(float(out[0, 5]) - __import__("math").radians(30.0)) < 1e-6


def test_hand_action_mode_default_absolute_and_increment_branch():
    assert re.search(r'hand_action_mode:\s*str\s*=\s*"absolute"', _CFG)
    code = _code_only(_RIG)
    assert "delta = a_j * rate" in code                       # 증분: 액션 = 닫힘 속도
    assert "delta = ( cmd_j - self . syn_close ) . clamp ( - rate , rate )" in code   # 절대 방식 유지


# =============================================================================
# ★09.28 DEXTRAH 작업공간 절대 palm 지령 + fabric q 관측 + fabric ADR
# =============================================================================
def _rig_method(name: str):
    """SideRig 의 메서드 하나를 떼어 가짜 self 로 부를 수 있게 한다(isaaclab 불요)."""
    import ast
    import math
    import torch
    cls = next(n for n in ast.parse(_RIG).body if isinstance(n, ast.ClassDef) and n.name == "SideRig")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "side_rig_method", "exec"), ns)
    return ns[name], torch


def _fake_rig(torch):
    from types import SimpleNamespace
    return SimpleNamespace(
        box_lo=torch.tensor([0.10, -0.55, 0.26]), box_hi=torch.tensor([0.55, 0.22, 0.70]),
        fab_to_env=torch.tensor([0.0, 0.0, 0.1]),
        anchor=torch.tensor([[0.3, -0.3, 0.3, 0.1, -0.2, 0.3]]),
        delta_lo=torch.tensor([-0.15, -0.10, -0.18, -0.8, -0.8, -2.6]),
        delta_hi=torch.tensor([0.15, 0.40, 0.25, 0.8, 0.8, 0.8]),
        palm_targets=None)


def test_workspace_target_zero_action_is_box_center_and_anchor_rotation():
    fn, torch = _rig_method("workspace_palm_target")
    r = _fake_rig(torch)
    fn(r, torch.zeros(1, 6), torch.tensor([True]))
    center_fab = 0.5 * (r.box_lo + r.box_hi) - r.fab_to_env
    assert torch.allclose(r.palm_targets[0, :3], center_fab, atol=1e-6)
    assert torch.allclose(r.palm_targets[0, 3:], r.anchor[0, 3:], atol=1e-6)


def test_workspace_target_extremes_hit_box_and_signed_rotation_limits():
    fn, torch = _rig_method("workspace_palm_target")
    r = _fake_rig(torch)
    fn(r, torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0, -1.0]]), torch.tensor([True]))
    pos_env = r.palm_targets[0, :3] + r.fab_to_env
    assert torch.allclose(pos_env, torch.tensor([0.55, -0.55, 0.70]), atol=1e-6)
    rot = r.palm_targets[0, 3:] - r.anchor[0, 3:]
    assert torch.allclose(rot, torch.tensor([0.8, -0.8, -2.6]), atol=1e-6)


def test_workspace_target_hold_env_keeps_anchor():
    fn, torch = _rig_method("workspace_palm_target")
    r = _fake_rig(torch)
    fn(r, torch.ones(1, 6), torch.tensor([False]))
    assert torch.allclose(r.palm_targets, r.anchor)


def test_workspace_mode_branch_obs_and_fabric_adr_wired():
    assert re.search(r"obs_fabric_state:\s*bool\s*=\s*False", _CFG)
    assert re.search(r"fabric_adr:\s*bool\s*=\s*False", _CFG)
    assert "per += a " in _code_only(_CFG) or "per += a\n" in _CFG
    code = _code_only(_ENV)
    assert "rig . workspace_palm_target (" in code
    assert "rig . fabric_arm_q ( )" in code
    assert 'get_param("fabric", "damping")' in _ENV and 'get_param("fabric", "velocity_ff")' in _ENV
    assert "self . _apply_fabric_adr ( )" in code
    rig = _code_only(_RIG)
    assert "self . velocity_ff_scale * self . fabric_qd [ : , self . arm_sl ]" in rig


def test_fabric_preset_dextrah_restores_four_values():
    from types import SimpleNamespace
    fn, torch = _rig_method("_apply_fabric_preset")
    fp = {"palm_attractor": {"conical_gain": 80.0},
          "body_repulsion": {"engage_depth": 0.30, "constant_accel": 2.0, "constant_accel_geom": 3.0}}
    fab = SimpleNamespace(fabric_params=fp, base_fabric_repulsion=SimpleNamespace(allocated_data={"engage_depth": 0.30}))
    r = SimpleNamespace(fabric=fab, role="src")
    fn(r, "track")
    assert fp["palm_attractor"]["conical_gain"] == 80.0
    fn(r, "dextrah")
    assert fp["palm_attractor"]["conical_gain"] == 50.0
    assert (fp["body_repulsion"]["engage_depth"], fp["body_repulsion"]["constant_accel"],
            fp["body_repulsion"]["constant_accel_geom"]) == (0.5, 1.0, 5.0)
    assert fab.base_fabric_repulsion.allocated_data["engage_depth"] == 0.5
    assert re.search(r'fabric_param_preset:\s*str\s*=\s*"track"', _CFG)


# =============================================================================
# ★09.28 joint 팔 제어(grasp_fj Track B = cup_grasp 방식, fabric 없음)
# =============================================================================
def test_joint_arm_step_is_increment_then_ema_and_clamped():
    from types import SimpleNamespace
    fn, torch = _rig_method("step_joint_arm")
    r = SimpleNamespace(env=SimpleNamespace(cfg=SimpleNamespace(k_arm=0.05, arm_ema=0.1, arm_joint_mode="increment"),
                                            step_dt=1.0 / 60.0),
                        arm_q_target=torch.zeros(1, 7), arm_lo=torch.full((7,), -0.02), arm_hi=torch.full((7,), 1.0),
                        reset_q=torch.zeros(7), arm_t=torch.arange(7))
    fn(r, torch.ones(1, 7), torch.tensor([True]))
    assert torch.allclose(r.arm_q_target, torch.full((1, 7), 0.005))      # α·k·a
    fn(r, -torch.ones(1, 7) * 20, torch.tensor([True]))                    # 한계 밖 요구 → 하한으로 묶인 raw 의 EMA
    assert float(r.arm_q_target.min()) >= -0.02 - 1e-9
    fn(r, torch.ones(1, 7), torch.tensor([False]))                         # hold = 시작 자세
    assert torch.allclose(r.arm_q_target, torch.zeros(1, 7))


def test_joint_arm_absolute_mode_home_limits_and_speed_cap():
    """09.29: arm_joint_mode=absolute — a=0 은 시작 자세로, ±1 은 관절 한계로 수렴, 스텝 변화는 vmax·dt 이하."""
    from types import SimpleNamespace
    fn, torch = _rig_method("step_joint_arm")
    cfg = SimpleNamespace(k_arm=0.05, arm_ema=0.1, arm_joint_mode="absolute", arm_abs_vmax=0.3)
    home = torch.tensor([0.5, 0.0, -0.2, 1.0, 0.0, 0.3, 0.0])
    r = SimpleNamespace(env=SimpleNamespace(cfg=cfg, step_dt=1.0 / 60.0),
                        arm_q_target=home.clone().unsqueeze(0), arm_lo=torch.full((7,), -1.5), arm_hi=torch.full((7,), 2.0),
                        reset_q=home.clone(), arm_t=torch.arange(7))
    cap = 0.3 / 60.0
    fn(r, torch.ones(1, 7), torch.tensor([True]))
    assert float((r.arm_q_target - home).abs().max()) <= cap + 1e-9          # 속도 상한
    for _ in range(3000):
        fn(r, torch.ones(1, 7) * 5.0, torch.tensor([True]))                  # |a|>1 은 ±1 로 자른다 → 상한
    assert torch.allclose(r.arm_q_target, torch.full((1, 7), 2.0), atol=1e-4)
    for _ in range(3000):
        fn(r, torch.zeros(1, 7), torch.tensor([True]))                       # a=0 → 시작 자세로 돌아온다(갇히지 않음)
    assert torch.allclose(r.arm_q_target, home.unsqueeze(0), atol=1e-4)
    for _ in range(3000):
        fn(r, -torch.ones(1, 7), torch.tensor([True]))
    assert torch.allclose(r.arm_q_target, torch.full((1, 7), -1.5), atol=1e-4)
    fn(r, torch.ones(1, 7), torch.tensor([False]))                           # hold = 시작 자세
    assert torch.allclose(r.arm_q_target, home.unsqueeze(0))


def test_joint_arm_mode_default_is_increment():
    assert re.search(r'arm_joint_mode:\s*str\s*=\s*"increment"', _CFG)
    assert 'if str(cfg.arm_joint_mode) not in ("increment", "absolute"):' in _ENV


def test_joint_arm_mode_wired_no_fabric_step_position_only():
    assert re.search(r'arm_control:\s*str\s*=\s*"fabric"', _CFG)
    assert "arm_w = pair.source.num_arm_joints if str(cfg.arm_control) == \"joint\" else 6" in _CFG
    code = _code_only(_ENV)
    assert "rig . step_joint_arm ( a [ : , : w ] , active )" in code
    assert "rig . synergy_targets ( a [ : , w : ] , gate )" in code
    assert "if not self . src . joint_arm :" in code
    assert "[ rig . arm_q_target ] if rig . joint_arm" in code
    rig = _code_only(_RIG)
    assert "self . robot . set_joint_velocity_target ( torch . zeros_like ( self . arm_q_target )" in rig
    assert "self . arm_q_target [ env_ids ] = q0 [ : , self . arm_t ]" in rig


# =============================================================================
# ★09.28 full-joint 가족(pour_fj): 손 direct + 팔 joint, 새 gym id
# =============================================================================
def _direct_rig(torch, n=6):
    from types import SimpleNamespace
    return SimpleNamespace(
        env=SimpleNamespace(cfg=SimpleNamespace(hand_ema=0.1, hand_full_range_s=1.0, hand_direct_range="limits",
                                                hand_direct_contact_freeze=False, contact_freeze_threshold=1.0)),
        syn_lo=torch.zeros(n), syn_hi=torch.ones(n) * 1.2, syn_slot=torch.arange(n),
        syn_target=torch.zeros(1, n), syn_open=torch.zeros(n), syn_grip=torch.ones(n), syn_close=None)


def test_direct_hand_is_absolute_target_with_ema_and_vendor_speed_cap():
    fn, torch = _rig_method("direct_hand_targets")
    r = _direct_rig(torch)
    t1 = fn(r, torch.ones(1, 6))                      # raw = hi = 1.2 → EMA 0.12 → 속도 상한 1.2/60 = 0.02
    assert torch.allclose(t1, torch.full((1, 6), 0.02), atol=1e-6)
    r.syn_target = torch.full((1, 6), 0.6)
    t2 = fn(r, torch.zeros(1, 6))                     # raw = 중앙 0.6 → 그대로
    assert torch.allclose(t2, torch.full((1, 6), 0.6), atol=1e-6)
    r.syn_target = torch.full((1, 6), 1.19)
    t3 = fn(r, torch.ones(1, 6))
    assert float(t3.max()) <= 1.2 + 1e-6              # 한계로 자른다


def test_direct_hand_bypasses_gate_freeze_coupling():
    code = _code_only(_RIG)
    body = code.split("def synergy_targets")[1]
    head = body.split("a_j = a_hand")[0]
    assert "return self . direct_hand_targets ( a_hand )" in head   # 결합·게이트·동결보다 먼저 돌아간다


def test_fj_cfg_family_defaults_and_registration():
    assert "class PourFJEnvCfg(PourFabricMimicEnvCfg)" in _CFG
    blk = _CFG.split("class PourFJEnvCfg(PourFabricMimicEnvCfg)")[1]
    for line in ('self.arm_control = "joint"', 'self.hand_control = "direct"', "self.obs_fabric_state = False",
                 "self.fabric_adr = False", "self.enable_self_collisions = True", "self.hand_velocity_ff_scale = 0.0"):
        assert line in blk, line
    assert re.search(r'hand_control:\s*str\s*=\s*"synergy"', _CFG)     # 구 id 기본값 불변
    reg = (_PKG / "config" / "__init__.py").read_text()
    assert '("pour_fj", PourFJEnvCfg, "PourFJ", REGISTERED_FJ)' in reg
    assert '("pour_fab_mimic", PourFabricMimicEnvCfg, "PourFabricMimic", REGISTERED)' in reg


def test_tactile_obs_tanh_only_in_fj_family():
    assert re.search(r"tactile_obs_tanh_n:\s*float\s*=\s*0\.0", _CFG)                 # 구 id 원시 N
    blk = _CFG.split("class PourFJEnvCfg(PourFabricMimicEnvCfg)")[1]
    assert "self.tactile_obs_tanh_n = 3.0" in blk
    code = _code_only(_ENV)
    assert "return torch . tanh ( f / k ) if k > 0.0 else f" in code


def test_fj_obs_cleanup_and_no_noise_delay():
    blk = _CFG.split("class PourFJEnvCfg(PourFabricMimicEnvCfg)")[1]
    for line in ("self.obs_tips_rel_palm = False", "self.obs_mouth_diff = False", "self.obs_noise_qpos = 0.0",
                 "self.obs_noise_qvel = 0.0", "self.obs_noise_body = 0.0", "self.tactile_obs_noise_n = 0.0",
                 "self.adr_obs_noise_object_xyz_max = 0.0", "self.adr_obs_noise_object_rot_max_deg = 0.0",
                 "self.perception_delay_max_steps = 0"):
        assert line in blk, line
    assert re.search(r"obs_tips_rel_palm:\s*bool\s*=\s*True", _CFG) and re.search(r"obs_mouth_diff:\s*bool\s*=\s*True", _CFG)
    code = _code_only(_ENV)
    assert "if bool ( cfg . obs_tips_rel_palm ) else [ ]" in code
    assert code.count("if bool ( self . cfg . obs_mouth_diff ) else [ ]") == 2          # actor·critic 둘 다


def _direct_rig_v2(torch, n=6, rng="grip", freeze=False, touch=None):
    from types import SimpleNamespace
    r = SimpleNamespace(
        env=SimpleNamespace(cfg=SimpleNamespace(hand_ema=1.0, hand_full_range_s=1.0, hand_direct_range=rng,
                                                hand_direct_contact_freeze=freeze, contact_freeze_threshold=1.0)),
        syn_lo=torch.zeros(n), syn_hi=torch.ones(n) * 1.5, syn_slot=torch.arange(n), syn_fi=torch.arange(n),
        syn_flex=torch.ones(n, dtype=torch.bool),
        syn_target=torch.zeros(1, n), syn_open=torch.zeros(n), syn_grip=torch.full((n,), 0.9), syn_close=None)
    t = torch.zeros(1, n) if touch is None else touch
    r.finger_link_forces = lambda: (t, t, t)
    return r


def test_direct_hand_grip_range_caps_at_grip_pose_and_keeps_joint_speed():
    fn, torch = _rig_method("direct_hand_targets")
    r = _direct_rig_v2(torch, rng="grip")
    for _ in range(200):
        r.syn_target = fn(r, torch.ones(1, 6))
    assert torch.allclose(r.syn_target, torch.full((1, 6), 0.9), atol=1e-6)     # 쥔 자세에서 멈춘다(관절 한계 1.5 까지 안 감)
    r2 = _direct_rig_v2(torch, rng="grip")
    t1 = fn(r2, torch.ones(1, 6))
    assert torch.allclose(t1, torch.full((1, 6), 1.5 / 60.0), atol=1e-6)        # 속도 상한은 관절 전 범위/1 s 그대로


def test_direct_hand_contact_freeze_blocks_closing_only():
    fn, torch = _rig_method("direct_hand_targets")
    touch = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    r = _direct_rig_v2(torch, freeze=True, touch=touch)
    r.syn_target = torch.full((1, 6), 0.3)
    t = fn(r, torch.ones(1, 6))                                                  # 닫기 요구
    assert abs(float(t[0, 0]) - 0.3) < 1e-6 and float(t[0, 1]) > 0.3                        # 닿은 손가락만 멈춤
    t2 = fn(r, -torch.ones(1, 6))                                                # 펴기 요구
    assert float(t2[0, 0]) < 0.3                                                 # 펴기는 허용


def test_fj_cfg_grip_range_and_freeze():
    blk = _CFG.split("class PourFJEnvCfg(PourFabricMimicEnvCfg)")[1]
    assert 'self.hand_direct_range = "grip"' in blk and "self.hand_direct_contact_freeze = True" in blk
    assert re.search(r'hand_direct_range:\s*str\s*=\s*"limits"', _CFG) and re.search(r"hand_direct_contact_freeze:\s*bool\s*=\s*False", _CFG)


def test_phys_anomaly_termination_off_by_default_and_wired():
    """09.29: 물리 이상 종료(컵 튕김·손 접촉 폭발·컵 이탈) — 기본 0(끔), 켜면 terminated 에 들어간다."""
    for k in ("phys_cup_speed_max", "phys_cup_angvel_max", "phys_contact_max_n", "cup_leash_m"):
        assert re.search(rf"{k}:\s*float\s*=\s*0\.0", _CFG), k
    code = _code_only(_ENV)
    assert "terminated = runaway | mimic_runaway | self . _dropped | phys [ ] " in code
    assert "self . _phys_f_streak [ env_ids ] = 0" in code


def test_grasp_shape_metrics_wired_per_side():
    """09.29: rh_aglt 파지 모양 지표 사본 — 쥔 스텝만 평균, 두 손 shape/{src,rcv}_* 로 기록, 로그 전용."""
    import importlib.util
    import pathlib
    import torch
    p = pathlib.Path(__file__).resolve().parents[1] / "shape_metrics.py"
    spec = importlib.util.spec_from_file_location("_shape", p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    n = 4
    cup = torch.zeros(n, 3); up = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
    tips = torch.zeros(n, 5, 3); tips[:, 0] = torch.tensor([0.0, 0.03, 0.01]); tips[:, 1:] = torch.tensor([0.0, -0.03, 0.01])
    g = torch.tensor([True, True, False, False])
    out = mod.grasp_shape(grasped=g, palm_pos=torch.tensor([-0.042, 0.0, 0.008]).repeat(n, 1),
                          palm_normal=torch.tensor([1.0, 0.0, 0.0]).repeat(n, 1), finger_dir=torch.tensor([0.0, 1.0, 0.0]).repeat(n, 1),
                          tips_pos=tips, finger_force=torch.full((n, 5), 2.0), prox_force=torch.zeros(n, 5), cup_pos=cup, cup_up=up,
                          cup_radius=0.028, cup_mouth_z=0.054, closure=torch.full((n,), 0.5), thr=1.0, mid_sign=1.0)
    assert abs(float(out["shape/palm_axis_r"]) - 1.5) < 1e-3 and float(out["shape/fingers_touching"]) == 5.0
    assert abs(float(out["shape/thumb_oppose"]) - 1.0) < 1e-6 and "shape/approach_palm_z_abs" in out
    code = _code_only(_ENV)
    assert "self . _log_grasp_shape ( ctx )" in code
    assert 'self.extras["shape/" + side + "_" + k.split("/", 1)[1]] = v' in _ENV


def test_grasp_shape_release_and_cup_frame_logs():
    """09.29 사용자 '영상 없이 로그로(병목 없이)': 놓침 비율은 매 스텝, 모양·컵 좌표 자세·기운 동안 자세는 shape_log_every 주기."""
    assert re.search(r"shape_log_every:\s*int\s*=\s*4", _CFG)
    for key in ('_release_rate"', '_{tag}n_dot_cupaxis"', '_{tag}f_dot_cupaxis"', '_{tag}fingers5"', '_tilted_grasp_frac"'):
        assert key in _ENV, key
    assert "self._shape_tick % max(int(self.cfg.shape_log_every), 1) != 0" in _ENV


def test_start_bank_off_by_default_and_wired():
    """09.30 사용자 '잡기 이후부터 스타트': 쥔 상태 시작 뱅크 — 기본 끔, 켜면 리셋 끝에서 덮고 hold 를 건너뛴다."""
    assert re.search(r'start_bank_path:\s*str\s*=\s*""', _CFG)
    assert re.search(r"start_bank_frac:\s*float\s*=\s*0\.0", _CFG)
    assert re.search(r'start_bank_capture_path:\s*str\s*=\s*""', _CFG)
    code = _code_only(_ENV)
    assert "self . _bank = StartBank ( self )" in code
    assert "self . _bank . restore ( env_ids )" in code
    assert "self . _bank . maybe_capture ( flags . in_source_frac )" in code
    bank = (_PKG / "start_bank.py").read_text()
    assert "env.episode_length_buf[ids] = int(env.cfg.hold_steps)" in bank
    for k in ("joint_pos", "joint_vel", "src_root", "rcv_root", "beads", "src_arm_q_target", "rcv_syn_target", "src_spawn"):
        assert f'"{k}"' in bank, k


def test_start_bank_restore_picks_fraction_and_skips_hold():
    """뱅크 복원 로직(물리 없이): frac 1 이면 전부, 0.0 이면 아무 env 도 덮지 않는다. 필드 누락은 로드에서 멈춘다."""
    import importlib.util
    import types
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("_sb", _PKG / "start_bank.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    n, J, K, A, H = 6, 9, 3, 7, 6

    class _Obj:
        def __init__(self, shape):
            self.data = types.SimpleNamespace(root_state_w=torch.zeros(n, 13), object_state_w=torch.zeros(n, K, 13))
            self.written = None
        def write_root_state_to_sim(self, s, env_ids): self.written = (s.clone(), env_ids.clone())
        def write_object_state_to_sim(self, s, env_ids): self.written = (s.clone(), env_ids.clone())

    class _Robot:
        def __init__(self):
            self.data = types.SimpleNamespace(joint_pos=torch.zeros(n, J), joint_vel=torch.zeros(n, J))
            self.js = None
        def write_joint_state_to_sim(self, p, v, env_ids): self.js = (p.clone(), env_ids.clone())
        def set_joint_position_target(self, p, env_ids): pass

    def _rig(role):
        return types.SimpleNamespace(role=role, arm_q_target=torch.zeros(n, A), syn_target=torch.zeros(n, H),
                                     syn_vel=torch.ones(n, H), syn_close=torch.zeros(n, H), syn_open=torch.zeros(H),
                                     syn_grip=torch.ones(H), arm_t=torch.arange(A))

    def _env(frac, path):
        cfg = types.SimpleNamespace(start_bank_frac=frac, start_bank_path=path, start_bank_capture_path="",
                                    start_bank_capture_n=8, start_bank_lift_min=0.05, bead_count=K, hold_steps=30)
        e = types.SimpleNamespace(cfg=cfg, num_envs=n, device="cpu", robot=_Robot(), source_cup=_Obj(1), receiver_cup=_Obj(1),
                                  beads=_Obj(1), src=_rig("src"), rcv=_rig("rcv"), _src_spawn=torch.zeros(n, 3),
                                  _rcv_spawn=torch.zeros(n, 3), _prev_arm_qd={"src": torch.zeros(n, A), "rcv": torch.zeros(n, A)},
                                  scene=types.SimpleNamespace(env_origins=torch.arange(n).float().unsqueeze(1).repeat(1, 3)),
                                  episode_length_buf=torch.zeros(n, dtype=torch.long))
        return e

    import tempfile, os
    B = 4
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "bank.npz")
        np.savez(p, joint_pos=np.full((B, J), 0.3), joint_vel=np.zeros((B, J)), src_root=np.ones((B, 13)), rcv_root=np.ones((B, 13)),
                 beads=np.ones((B, K, 13)), src_arm_q_target=np.full((B, A), 0.2), rcv_arm_q_target=np.full((B, A), 0.2),
                 src_syn_target=np.full((B, H), 0.5), rcv_syn_target=np.full((B, H), 0.5), src_spawn=np.full((B, 3), 0.1),
                 rcv_spawn=np.full((B, 3), 0.1))
        e = _env(1.0, p); sb = mod.StartBank(e)
        ids = sb.restore(torch.arange(n))
        assert len(ids) == n and bool((e.episode_length_buf == 30).all())
        assert torch.allclose(e.src.arm_q_target, torch.full((n, A), 0.2)) and bool((e.src.syn_vel == 0).all())
        assert torch.allclose(e.src.syn_close, torch.full((n, H), 0.5))
        root, _ = e.source_cup.written
        assert torch.allclose(root[:, 0], 1.0 + e.scene.env_origins[:, 0])          # env-local → world
        e0 = _env(0.0, ""); sb0 = mod.StartBank(e0)
        assert len(sb0.restore(torch.arange(n))) == 0 and bool((e0.episode_length_buf == 0).all())
        bad = os.path.join(td, "bad.npz"); np.savez(bad, joint_pos=np.zeros((B, J)))
        with pytest.raises(ValueError):
            mod.StartBank(_env(0.5, bad))


def test_object_spawn_dx_default_zero_and_applied_to_both_cups():
    """09.30 사용자 'A로 진행': 컵 스폰 x 이동 — 기본 0(옛 런 불변), 두 컵 모두 같은 식."""
    assert re.search(r"object_spawn_dx:\s*float\s*=\s*0\.0", _CFG)
    code = _code_only(_ENV)
    assert "pos [ : , 0 ] = rig . profile . object_spawn_center [ 0 ] + float ( cfg . object_spawn_dx ) + offs [ : , 0 ]" in code


def test_real_control_defaults_off_and_wired():
    """09.30 사용자 '실제 제어 로직으로 학습될 수 있게도': 명령 지연·손 속도 상한·네 손가락 편 한계 — 기본 끔, 물리에는 지연된 목표."""
    assert re.search(r"hand_cmd_delay_steps:\s*int\s*=\s*0", _CFG)
    assert re.search(r"arm_cmd_delay_steps:\s*tuple\s*=\s*\(0,\s*0\)", _CFG)
    for k in ("hand_vel_cap_rad_s", "hand_thumb_flex_vel_cap_rad_s", "hand_finger_open_floor_rad"):
        assert re.search(rf"{k}:\s*float\s*=\s*0\.0", _CFG), k
    code = _code_only(_RIG)
    assert "self . robot . set_joint_position_target ( self . arm_applied , joint_ids = self . arm_ids )" in code
    assert "self . robot . set_joint_position_target ( self . syn_applied , joint_ids = self . syn_ids )" in code
    env = _code_only(_ENV)
    assert "rig . push_command ( )" in env and "rig . init_real_control ( )" in env and "rig . reset_real_control ( env_ids )" in env


def test_real_control_delay_logic():
    """지연 버퍼: 손은 고정 d 스텝 뒤, 팔은 env 마다 [lo, hi] 스텝 뒤의 목표가 물리로 간다. 리셋은 기록을 목표로 채운다."""
    import types
    init, torch = _rig_method("init_real_control")
    push, _ = _rig_method("push_command")
    reset, _ = _rig_method("reset_real_control")
    N = 3
    rig = types.SimpleNamespace()
    rig.N, rig.device = N, "cpu"
    rig.env = types.SimpleNamespace(cfg=types.SimpleNamespace(hand_vel_cap_rad_s=2.1, hand_thumb_flex_vel_cap_rad_s=0.56,
                                                             hand_cmd_delay_steps=4, arm_cmd_delay_steps=(2, 2)))
    rig.robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_names=["r_hj_thumb_1", "r_hj_thumb_2", "r_hj_index_1"]))
    rig.syn_ids = [0, 1, 2]
    rig.arm_q_target = torch.zeros(N, 7)
    rig.syn_target = torch.zeros(N, 3)
    init(rig)
    assert torch.allclose(rig._real_hand_cap, torch.tensor([2.1, 0.56, 2.1]) / 60.0)
    assert rig._finger_mask.tolist() == [False, False, True]
    for t in range(1, 7):
        rig.arm_q_target = torch.full((N, 7), float(t)); rig.syn_target = torch.full((N, 3), float(t))
        push(rig)
        assert float(rig.arm_applied[0, 0]) == max(t - 2, 0) and float(rig.syn_applied[0, 0]) == max(t - 4, 0)
    rig.arm_q_target = torch.full((N, 7), 9.0); rig.syn_target = torch.full((N, 3), 9.0)
    reset(rig, torch.tensor([1]))
    assert float(rig.arm_applied[1, 0]) == 9.0 and float(rig.syn_applied[1, 0]) == 9.0 and float(rig.arm_applied[0, 0]) == 4.0
