"""rh_aglt_l 계약(좌팔 거울상) — Isaac 없이 돈다(순수 torch 모듈 · 자산 URDF · cfg 소스 파싱).

실행: PYTHONPATH=source/openarm /usr/bin/python3 -m pytest source/openarm/openarm/agnostic/tasks/rh_aglt_l/tests -q
"""

from __future__ import annotations

import ast
import io
import math
import re
import sys
import tokenize
from pathlib import Path

import pytest
import torch
import yaml

from openarm.agnostic.modules import keypoint_goal as kg
from openarm.agnostic.modules import robot_profiles as _rp
from openarm.agnostic.tasks.rh_aglt_l import mimic as _mimic
from openarm.agnostic.tasks.rh_aglt_l.profile import ARM_SIGN_L, RH56F1_LEFT
from openarm.agnostic.tasks.rh_aglt_l.arm_action import step_arm_increment
from openarm.agnostic.tasks.rh_aglt_l.goals import GoalParams, GoalState
from openarm.agnostic.tasks.rh_aglt_l.hand_action import closure, grip_range, linear_grip_raw, step_hand_target
from openarm.agnostic.tasks.rh_aglt_l.layout import actor_layout, critic_extra_layout, layout_width
from openarm.agnostic.tasks.rh_aglt_l.real_response import DelayLine, firmware_stop, open_floor, sample_open_floor
from openarm.agnostic.tasks.rh_aglt_l.spawn import sample_cup_xy

_PKG = Path(__file__).resolve().parents[1]
_HDGP = Path(__file__).resolve().parents[7]
_URDF = _HDGP / "assets" / "robot" / "openarm_rh56f1_bi_rl" / "openarm_rh56f1_bi_rl.urdf"
PROFILE = RH56F1_LEFT
SIDE = "l"
_RIGHT_PKG = _PKG.parent / "rh_aglt_r"


# =============================================================================
# cfg 소스에서 기본값을 읽는다(cfg 모듈은 isaaclab 을 import 해서 시스템 python 으로 못 부른다)
# =============================================================================
def _cfg_defaults(pkg: Path = _PKG) -> dict:
    tree = ast.parse((pkg / "rh_aglt_env_cfg.py").read_text())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    out = {}
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RhAgltEnvCfg")
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            if isinstance(node.value, ast.Name) and node.value.id in consts:
                out[node.target.id] = consts[node.value.id]
                continue
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    out["_CUP_UNIT"] = consts["CUP_UNIT"]
    if "arm_start_q" not in out:                      # 좌: 우 자세 × 부호(생성식)
        out["arm_start_q"] = tuple(s * v for s, v in zip(ARM_SIGN_L, consts["_ARM_START_Q_RIGHT"]))
    return out


CFG = _cfg_defaults()
CUP_R = CFG["_CUP_UNIT"]["outer_r"] * CFG["cup_scale"]
CUP_Z0 = CFG["table_surface_z"] - CFG["_CUP_UNIT"]["bottom_z"] * CFG["cup_scale"]
CUP_TOP = CUP_Z0 + CFG["_CUP_UNIT"]["rim_z"] * CFG["cup_scale"]


def _urdf_tools():
    sys.path.insert(0, str(_HDGP.parent / "urdf" / "tools"))
    from solve_arm_reset_pose import Urdf  # noqa: E402
    return Urdf(_URDF)


# =============================================================================
# 팔 · 손 지령
# =============================================================================
def test_arm_increment_is_cup_grasp_law():
    """q_raw = clamp(q + k·a) → q* = α·q_raw + (1−α)·q — 포화 slew α·k·60 = 0.3 rad/s(cup_grasp 와 같다)."""
    lo, hi = -torch.ones(2), torch.ones(2)
    q = torch.zeros(1, 2)
    new = step_arm_increment(q, torch.tensor([[1.0, -1.0]]), k=CFG["k_arm"], alpha=CFG["arm_ema"], q_lo=lo, q_hi=hi)
    assert torch.allclose(new, torch.tensor([[1.0, -1.0]]) * CFG["k_arm"] * CFG["arm_ema"])
    assert abs(CFG["k_arm"] * CFG["arm_ema"] * 60.0 - 0.3) < 1e-9
    held = step_arm_increment(q, torch.zeros(1, 2), k=CFG["k_arm"], alpha=CFG["arm_ema"], q_lo=lo, q_hi=hi)
    assert torch.allclose(held, q), "a = 0 은 제자리"
    edge = step_arm_increment(hi.unsqueeze(0), torch.ones(1, 2), k=0.05, alpha=0.1, q_lo=lo, q_hi=hi)
    assert torch.allclose(edge, hi.unsqueeze(0)), "한계 밖으로 나가면 안 된다"


def test_hand_mapping_is_pour_bi_rh_grip_range():
    """pour_fabric_mimic direct(range=grip): −1 → min(open,grip), +1 → max(open,grip), 0 → 가운데."""
    o, g = torch.tensor(PROFILE.hand_open_pose), torch.tensor(PROFILE.hand_grip_pose)
    lo, hi = grip_range(o, g)
    assert torch.allclose(linear_grip_raw(-torch.ones(1, 6), o, g), lo.unsqueeze(0))
    assert torch.allclose(linear_grip_raw(torch.ones(1, 6), o, g), hi.unsqueeze(0))
    assert torch.allclose(linear_grip_raw(torch.zeros(1, 6), o, g), (0.5 * (lo + hi)).unsqueeze(0))
    # thumb_1(외전)만 open > grip 이라 +1 이 편 쪽이다 — pour_bi_rh 와 같다
    flip = [i for i in range(6) if PROFILE.hand_open_pose[i] > PROFILE.hand_grip_pose[i]]
    assert flip == [0]


def test_hand_freeze_blocks_closing_only():
    o, g = torch.tensor(PROFILE.hand_open_pose), torch.tensor(PROFILE.hand_grip_pose)
    cap = torch.full((6,), 10.0)
    half = (0.5 * (o + g)).unsqueeze(0)
    frz = torch.ones(1, 6, dtype=torch.bool)
    closed = step_hand_target(half, g.unsqueeze(0), alpha=1.0, cap=cap, freeze=frz, q_open=o, q_grip=g)
    assert torch.allclose(closed, half), "닿은 손가락이 더 닫혔다"
    opened = step_hand_target(half, o.unsqueeze(0), alpha=1.0, cap=cap, freeze=frz, q_open=o, q_grip=g)
    assert torch.allclose(opened, o.unsqueeze(0)), "닿은 손가락이 펴지지 않는다"


def test_hand_speed_cap_is_vendor_full_range_one_second():
    o, g = torch.tensor(PROFILE.hand_open_pose), torch.tensor(PROFILE.hand_grip_pose)
    cap = torch.full((6,), 0.01)
    new = step_hand_target(o.unsqueeze(0), g.unsqueeze(0), alpha=1.0, cap=cap,
                           freeze=torch.zeros(1, 6, dtype=torch.bool), q_open=o, q_grip=g)
    assert float((new - o).abs().max()) <= 0.01 + 1e-7


# =============================================================================
# 소환 · 목표
# =============================================================================
def test_spawn_inside_box_and_off_table_holes():
    torch.manual_seed(0)
    xy, _, failed = sample_cup_xy(20000, CFG["spawn_center"], CFG["spawn_half"], CFG["spawn_holes"],
                                  CUP_R, CFG["spawn_hole_margin"], "cpu")
    assert failed == 0
    c, h = torch.tensor(CFG["spawn_center"]), torch.tensor(CFG["spawn_half"])
    assert bool(((xy - c).abs() <= h + 1e-6).all())
    for cx, cy, r in CFG["spawn_holes"]:
        d = (xy - torch.tensor((cx, cy))).norm(dim=-1)
        assert float(d.min()) >= r + CUP_R + CFG["spawn_hole_margin"] - 1e-6


def test_spawn_box_inside_termination_box():
    (cx, cy), (hx, hy) = CFG["spawn_center"], CFG["spawn_half"]
    assert CFG["cup_out_x"][0] < cx - hx and cx + hx < CFG["cup_out_x"][1]
    assert CFG["cup_out_y"][0] < cy - hy and cy + hy < CFG["cup_out_y"][1]


def _goal_state(n: int) -> GoalState:
    (cx, cy), (hx, hy), m = CFG["spawn_center"], CFG["spawn_half"], CFG["goal_box_xy_margin"]
    seq = kg.GoalSeqCfg(first_xy_range=CFG["goal_first_xy_range"], first_z_range=CFG["goal_first_z_range"],
                        delta_distance=CFG["goal_delta_distance"],
                        box_min=(cx - hx - m, cy - hy - m, CUP_Z0 + CFG["goal_box_z_range"][0]),
                        box_max=(cx + hx + m, cy + hy + m, CUP_Z0 + CFG["goal_box_z_range"][1]),
                        success_steps=CFG["goal_success_steps"], max_goals=CFG["goal_max"])
    half = 0.5 * (CFG["_CUP_UNIT"]["rim_z"] - CFG["_CUP_UNIT"]["bottom_z"]) * CFG["cup_scale"]
    return GoalState(n, 5, GoalParams(half_height=half, seq=seq, tol_start=CFG["tol_start"],
                                      tol_floor=CFG["tol_floor"], tol_factor=CFG["tol_factor"],
                                      tol_interval=CFG["tol_interval"],
                                      tol_success_threshold=CFG["tol_success_threshold"],
                                      lift_latch_m=CFG["lift_latch_m"]), "cpu")


def test_first_goal_requires_lifting_above_latch():
    gs = _goal_state(512)
    pos = torch.tensor([[*CFG["spawn_center"], CUP_Z0]]).repeat(512, 1)
    quat = torch.tensor([[1.0, 0, 0, 0]]).repeat(512, 1)
    gs.reset(torch.arange(512), pos, quat)
    dz = gs.goal_pos[:, 2] - CUP_Z0
    assert float(dz.min()) >= CFG["lift_latch_m"] - 1e-6


def test_success_needs_grasp_and_advances_goal():
    gs = _goal_state(2)
    pos = torch.tensor([[*CFG["spawn_center"], CUP_Z0]]).repeat(2, 1)
    quat = torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1)
    gs.reset(torch.arange(2), pos, quat)
    g0 = gs.goal_pos.clone()
    active = torch.ones(2, dtype=torch.bool)
    grasped = torch.tensor([True, False])
    for _ in range(CFG["goal_success_steps"]):
        out = gs.step(g0.clone(), quat, grasped, active)
    assert bool(out["success"][0]) and not bool(out["success"][1])
    assert int(gs.trk.successes[0]) == 1 and int(gs.trk.successes[1]) == 0
    assert not torch.allclose(gs.goal_pos[0], g0[0]) and torch.allclose(gs.goal_pos[1], g0[1])
    far = g0.clone()
    far[:, 2] += 0.20
    out = gs.step(far, quat, torch.tensor([True, True]), active)
    assert not bool(out["success"][1]), "목표를 스친 뒤 멀리서 쥔 순간은 성공이 아니다(리뷰 H1)"
    out = gs.step(g0.clone(), quat, torch.tensor([True, True]), active)
    assert bool(out["success"][1]), "근처 누적이 찬 뒤 목표 근처에서 쥐면 성공이다"


def test_hold_steps_never_succeed_or_latch():
    gs = _goal_state(1)
    pos = torch.tensor([[*CFG["spawn_center"], CUP_Z0]])
    quat = torch.tensor([[1.0, 0, 0, 0]])
    gs.reset(torch.arange(1), pos, quat)
    idle = torch.zeros(1, dtype=torch.bool)
    for _ in range(3 * CFG["goal_success_steps"]):
        out = gs.step(gs.goal_pos.clone(), quat, torch.ones(1, dtype=torch.bool), idle)
    assert not bool(out["success"][0]) and not bool(gs.lifted[0])


# =============================================================================
# 관측 배치 · PPO
# =============================================================================
def test_layout_widths():
    a, h, f = PROFILE.num_arm_joints, PROFILE.num_hand_joints, len(PROFILE.finger_sensor_bodies)
    assert (a, h, f) == (7, 6, 5)
    assert layout_width(actor_layout(a, h, f)) == 96
    assert layout_width(actor_layout(a, h, f)) + layout_width(critic_extra_layout(a, h, f)) == 128
    names = [n for n, _ in actor_layout(a, h, f)]
    assert "hand_qd" not in names, "손 관절속도는 actor 에 넣지 않는다(실기 드라이버 velocity ≠ 관절속도)"


def test_agent_yaml_matches_sibling_tracks():
    y = yaml.safe_load((_PKG / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text())["params"]
    net, c = y["network"], y["config"]
    assert net["rnn"]["units"] == 1024 and net["rnn"]["before_mlp"] is True
    assert net["mlp"]["units"] == [1024, 1024, 512, 512]
    assert (c["horizon_length"], c["seq_length"], c["minibatch_size"], c["mini_epochs"]) == (16, 16, 16384, 2)
    assert (c["gamma"], c["e_clip"], c["entropy_coef"], c["bounds_loss_coef"]) == (0.99, 0.1, 0.0, 0.005)
    assert c["bound_loss_type"] == "bound"
    assert c["save_frequency"] == 200, "사용자 09.29: pth 저장 간격 200 epoch"


# =============================================================================
# 자산 · 프로필 · 먼 출발
# =============================================================================
def test_mimic_pairs_from_urdf():
    pairs = _mimic.parse_mimic_pairs(str(_URDF), SIDE)
    assert len(pairs) == 6
    # 엄지는 연쇄(thumb_4 ← thumb_3 ← thumb_2)다 — 모든 사슬이 구동 관절에서 끝나야 한다.
    for dep in pairs:
        cur, hops = dep, 0
        while cur in pairs:
            cur, hops = pairs[cur][0], hops + 1
            assert hops <= len(pairs), f"{dep}: mimic 순환"
        assert cur in PROFILE.hand_joint_names, f"{dep}: 사슬 끝 {cur} 가 구동 관절이 아니다"


def test_profile_mimic_gains_zero():
    specs = {k: v for k, v in PROFILE.actuator_specs.items() if k.endswith("_hand_mimic")}
    assert specs and all((v["stiffness"], v["damping"]) == (0.0, 0.0) for v in specs.values())


def test_far_start_pose_fk():
    u = _urdf_tools()
    arm = [f"{SIDE}_aj_{i}" for i in range(1, 8)]
    home = {k: PROFILE.init_joint_pos[k] for k in arm}
    start = dict(zip(arm, CFG["arm_start_q"]))
    palm = f"{SIDE}_hl_palm_sensor"
    T0, T1 = u.pose(palm, home), u.pose(palm, start)
    # 손 방향은 test_start_pose_orientation_user_rule 이 사용자 규칙(손바닥 +z 수평·마주 봄·엄지 위)으로 잠근다.
    #   캘리브 홈과의 차이는 손가락을 15° 들어 올린 것뿐이다(09.29 테이블 앞 출발).
    R_err = T0[:3, :3].T @ T1[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (R_err.trace() - 1) / 2))))
    assert ang < 16.0, f"먼 출발 손 방향이 캘리브 홈과 {ang:.2f}° 다르다(설계 15°)"
    for k, v in start.items():
        lo, hi = u.joints[k]["lo"], u.joints[k]["hi"]
        assert min(v - lo, hi - v) > math.radians(10.0), f"{k} 한계 여유 < 10°"
    q = dict(start, **{f"{SIDE}_hj_thumb_1": PROFILE.hand_open_pose[0]})
    zmin = min(u.pose(b, q)[2, 3] for b in (palm, *PROFILE.fingertip_bodies))
    assert zmin > CUP_TOP + 0.05, f"시작 손 최저 z {zmin:.3f} 이 컵 윗면 {CUP_TOP:.3f} 에 가깝다"
    # 부팅 가드 대역 안(소환 표본 평균)
    torch.manual_seed(0)
    xy, _, _ = sample_cup_xy(4096, CFG["spawn_center"], CFG["spawn_half"], CFG["spawn_holes"],
                             CUP_R, CFG["spawn_hole_margin"], "cpu")
    p = torch.tensor(T1[:3, 3], dtype=torch.float32)
    d = (torch.cat([xy, torch.full((4096, 1), CUP_Z0)], dim=1) - p).norm(dim=-1)
    lo, hi = CFG["start_palm_cup_dist_band"]
    assert lo <= float(d.mean()) <= hi


# =============================================================================
# 소스 규칙
# =============================================================================
def _code_only(src: str) -> str:
    return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(src).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))


@pytest.mark.parametrize("fname", ["rh_aglt_env.py", "arm_action.py", "hand_action.py", "goals.py", "spawn.py", "shape_metrics.py",
                                   "layout.py", "reward_context.py", "mimic.py", "real_response.py"])
def test_no_robot_literals_in_code(fname):
    body = _code_only((_PKG / fname).read_text())
    for lit in ("r_aj_", "l_aj_", "r_hj_", "l_hj_", "r_hl_", "l_hl_", "rh56f1_right", "RH56F1_"):
        assert lit not in body, f"{fname}: 로봇 리터럴 '{lit}' — 프로필로"


def test_no_imports_from_other_task_packages():
    """사용자 결정(09.29): env 부터 새로 — 다른 과제 패키지를 import 하지 않는다(중립 modules 만)."""
    for f in _PKG.rglob("*.py"):
        if "tests" in f.parts:
            continue
        src = f.read_text()
        for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", src, re.M):
            mod = m.group(1)
            assert ".tasks." not in mod or mod.startswith("openarm.agnostic.tasks.rh_aglt_l"), f"{f.name}: {mod}"
            assert not mod.startswith("..") or mod.startswith("..rh_aglt_env_cfg"), f"{f.name}: {mod}"


# =============================================================================
# 거울 계약 — 좌우가 다른 곳은 profile·cfg 의 y 값뿐이다
# =============================================================================
_SHARED = ("arm_action.py", "hand_action.py", "goals.py", "spawn.py", "layout.py", "reward_context.py", "mimic.py", "shape_metrics.py",
           "rh_aglt_env.py", "config/agents/rl_games_ppo_lstm_cfg.yaml")


@pytest.mark.parametrize("fname", _SHARED)
def test_shared_files_identical_to_right(fname):
    assert (_PKG / fname).read_bytes() == (_RIGHT_PKG / fname).read_bytes(), f"{fname} 이 우 패키지와 갈렸다"


def test_cfg_is_y_mirror_of_right():
    r = _cfg_defaults(_RIGHT_PKG)
    assert CFG["arm_start_q"] == tuple(s * v for s, v in zip(ARM_SIGN_L, r["arm_start_q"]))
    assert CFG["spawn_center"] == (r["spawn_center"][0], -r["spawn_center"][1])
    assert CFG["spawn_half"] == r["spawn_half"]
    assert sorted(CFG["spawn_holes"]) == sorted((x, -y, rr) for x, y, rr in r["spawn_holes"])
    assert CFG["cup_out_x"] == r["cup_out_x"]
    assert CFG["cup_out_y"] == (-r["cup_out_y"][1], -r["cup_out_y"][0])
    same = [k for k in r if k not in ("arm_start_q", "spawn_center", "spawn_holes", "cup_out_y", "_CUP_UNIT")]
    diff = [k for k in same if CFG.get(k) != r[k]]
    assert not diff, f"y 와 무관한 값이 좌우로 갈렸다: {diff}"


def test_left_profile_mirrors_right_names_and_idle_arm():
    r = _rp.RH56F1_RIGHT
    assert PROFILE.hand_open_pose == r.hand_open_pose and PROFILE.hand_grip_pose == r.hand_grip_pose
    assert all(n.startswith("l_") for n in PROFILE.hand_joint_names + PROFILE.fingertip_bodies)
    assert PROFILE.actuator_specs == r.actuator_specs
    for i, s in enumerate(ARM_SIGN_L, start=1):
        assert PROFILE.init_joint_pos[f"r_aj_{i}"] == s * r.init_joint_pos[f"l_aj_{i}"]


def test_left_far_start_mirrors_right_fk():
    u = _urdf_tools()
    rq = _cfg_defaults(_RIGHT_PKG)["arm_start_q"]
    R = {f"r_aj_{i + 1}": v for i, v in enumerate(rq)}
    L = {f"l_aj_{i + 1}": v for i, v in enumerate(CFG["arm_start_q"])}
    for b in ("index_tip", "middle_tip", "ring_tip", "pinky_tip"):
        pr, pl = u.pose(f"r_hl_{b}", R)[:3, 3], u.pose(f"l_hl_{b}", L)[:3, 3]
        assert abs(pl[0] - pr[0]) < 1e-3 and abs(pl[1] + pr[1]) < 1e-3 and abs(pl[2] - pr[2]) < 1e-3, b


# =============================================================================
# 파지 모양 로그 지표(로그 전용)
# =============================================================================
def test_grasp_shape_only_counts_grasped_steps():
    from openarm.agnostic.tasks.rh_aglt_l.shape_metrics import grasp_shape
    cup = torch.tensor([[0.3, 0.0, 0.26], [0.3, 0.0, 0.26]])
    up = torch.tensor([[0.0, 0.0, 1.0]] * 2)
    # env0: 옆에서 수평 파지(법선 +y, 손가락 +x), 엄지 반대편 · env1: 위에서 내려 집기지만 쥔 판정 없음
    palm = torch.tensor([[0.3, -0.043, 0.265], [0.3, 0.0, 0.40]])
    n = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    f = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    tips = torch.zeros(2, 5, 3)
    tips[0, 0] = torch.tensor([0.3, 0.03, 0.26])
    tips[0, 1:] = torch.tensor([0.3, -0.03, 0.26])
    tips[1] = torch.tensor([0.3, 0.0, 0.33])
    ff = torch.tensor([[2.0, 2.0, 2.0, 0.0, 0.0], [0.0] * 5])
    out = grasp_shape(grasped=torch.tensor([True, False]), palm_pos=palm, palm_normal=n, finger_dir=f, tips_pos=tips,
                      finger_force=ff, prox_force=ff * 0.5, cup_pos=cup, cup_up=up, cup_radius=0.0286,
                      cup_mouth_z=0.0539, closure=torch.tensor([0.6, 0.1]), thr=1.0)
    assert abs(float(out["shape/palm_normal_z"])) < 1e-6 and abs(float(out["shape/finger_dir_z"])) < 1e-6
    assert float(out["shape/fingers_touching"]) == 3.0 and float(out["shape/prox_touching"]) == 0.0
    assert abs(float(out["shape/palm_axis_r"]) - 1.5) < 0.02
    assert float(out["shape/thumb_oppose"]) > 0.99
    assert float(out["shape/thumb"]) == 1.0 and float(out["shape/ring"]) == 0.0
    assert "shape/closure_near_open" not in out          # env1 은 손바닥이 컵 원점에서 14 cm
    none = grasp_shape(grasped=torch.tensor([False, False]), palm_pos=palm, palm_normal=n, finger_dir=f, tips_pos=tips,
                       finger_force=ff, prox_force=ff, cup_pos=cup, cup_up=up, cup_radius=0.0286, cup_mouth_z=0.0539,
                       closure=torch.tensor([0.6, 0.1]), thr=1.0)
    assert "shape/palm_normal_z" not in none, "쥔 env 가 없으면 쥔 스텝 지표를 쓰지 않는다"


def test_start_pose_orientation_user_rule():
    """사용자 09.29: palm_sensor 국소 +z(손바닥 중심 방향)는 중력에 수직, 두 손바닥은 마주 보고(우 +y · 좌 −y),
    엄지 쪽이 새끼 쪽보다 확실히 위(뒤집힘 방지). 상판 앞 가장자리(x 0.07) 근처에서 출발한다."""
    u = _urdf_tools()
    q = {f"{SIDE}_aj_{i + 1}": v for i, v in enumerate(CFG["arm_start_q"])}
    q[f"{SIDE}_hj_thumb_1"] = PROFILE.hand_open_pose[0]
    T = u.pose(PROFILE.palm_body, q)
    z = T[:3, 2]
    assert abs(z[2]) < math.sin(math.radians(2.0)), f"손바닥 +z 가 수평이 아니다: {z}"
    toward_mid = 1.0 if SIDE == "r" else -1.0
    assert z[1] * toward_mid > 0.99, f"손바닥이 반대 손 쪽(중심선)을 보지 않는다: {z}"
    thumb = u.pose(PROFILE.fingertip_bodies[0], q)[2, 3]
    pinky = u.pose(PROFILE.fingertip_bodies[-1], q)[2, 3]
    assert thumb - pinky > 0.05, f"엄지 끝이 새끼 끝보다 5 cm 이상 위가 아니다: {thumb - pinky:.3f}"
    assert abs(T[0, 3] - 0.08) < 0.01, f"손바닥이 상판 앞 가장자리 근처가 아니다: x {T[0, 3]:.3f}"


def test_idle_arm_is_mirror_of_active_start():
    """사용자 09.29 "왼팔 오른팔 모두 대칭적으로" — 쉬는 팔은 활성 팔 먼 출발 자세의 y 거울, 손은 같은 편 손."""
    u = _urdf_tools()
    other = {"r": "l", "l": "r"}[SIDE]
    sign = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
    q_me = {f"{SIDE}_aj_{i + 1}": v for i, v in enumerate(CFG["arm_start_q"])}
    q_ot = {f"{other}_aj_{i + 1}": s * v for i, (s, v) in enumerate(zip(sign, CFG["arm_start_q"]))}
    q_me[f"{SIDE}_hj_thumb_1"] = q_ot[f"{other}_hj_thumb_1"] = PROFILE.hand_open_pose[0]
    for b in ("palm_sensor", "index_tip", "pinky_tip"):
        pm, po = u.pose(f"{SIDE}_hl_{b}", q_me)[:3, 3], u.pose(f"{other}_hl_{b}", q_ot)[:3, 3]
        assert abs(pm[0] - po[0]) < 5e-3 and abs(pm[1] + po[1]) < 5e-3 and abs(pm[2] - po[2]) < 5e-3, b
    src = (_PKG / "rh_aglt_env_cfg.py").read_text()
    assert "out[other + k[len(me):]] = float(sgn) * float(v)" in src


def test_approach_orientation_metrics():
    from openarm.agnostic.tasks.rh_aglt_l.shape_metrics import grasp_shape
    n = 3
    cup = torch.tensor([[0.3, 0.0, 0.26]]).repeat(n, 1)
    up = torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1)
    palm = torch.tensor([[0.1, -0.3, 0.45]]).repeat(n, 1)
    normal = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    tips = torch.zeros(n, 5, 3)
    tips[:, 0, 2], tips[:, 4, 2] = 0.48, 0.41
    tips[1, 0, 2], tips[1, 4, 2] = 0.40, 0.47          # 뒤집힌 손
    out = grasp_shape(grasped=torch.zeros(n, dtype=torch.bool), palm_pos=palm, palm_normal=normal,
                      finger_dir=torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1), tips_pos=tips,
                      finger_force=torch.zeros(n, 5), prox_force=torch.zeros(n, 5), cup_pos=cup, cup_up=up,
                      cup_radius=0.0286, cup_mouth_z=0.0539, closure=torch.zeros(n), thr=1.0, mid_sign=1.0)
    assert abs(float(out["shape/approach_flipped"]) - 1 / 3) < 1e-6
    assert abs(float(out["shape/approach_palm_z_abs"]) - 1 / 3) < 1e-6
    assert abs(float(out["shape/approach_palm_to_mid"]) - 2 / 3) < 1e-6


def test_physx_collision_stack_large_enough():
    """09.29 aglt_l_i03: 기본 64 MB 에서 접촉이 버려졌다(요구 최대 95 MB) — 2**28 이상이어야 한다."""
    src = (_PKG / "rh_aglt_env_cfg.py").read_text()
    m = re.search(r"gpu_collision_stack_size\s*=\s*2\s*\*\*\s*(\d+)", src)
    assert m and int(m.group(1)) >= 28, "gpu_collision_stack_size 가 없거나 2**28 미만"


# =============================================================================
# 실기 반응(09.30 사용자 "실제 제어 로직으로 학습") — real_response.py
# =============================================================================
def test_delay_line_reads_d_steps_back_and_fill_clears():
    line = DelayLine(2, 1, 3, "cpu")
    line.fill(torch.arange(2), torch.tensor([-1.0]))
    for t in range(5):
        line.push(torch.full((2, 1), float(t)))
    got = line.read(torch.tensor([0, 3]))
    assert got[:, 0].tolist() == [4.0, 1.0]
    line.fill(torch.tensor([1]), torch.tensor([9.0]))
    assert line.read(torch.tensor([0, 3]))[:, 0].tolist() == [4.0, 9.0], "리셋 뒤 이전 에피소드 명령이 남았다"


def test_firmware_stop_blocks_closing_only():
    o, g = torch.tensor(PROFILE.hand_open_pose), torch.tensor(PROFILE.hand_grip_pose)
    half = (0.5 * (o + g)).unsqueeze(0)
    touch = torch.ones(1, 6, dtype=torch.bool)
    assert torch.allclose(firmware_stop(half, g.unsqueeze(0), touch, o, g), half), "닿은 손가락이 더 닫혔다"
    assert torch.allclose(firmware_stop(half, o.unsqueeze(0), touch, o, g), o.unsqueeze(0)), "닿은 손가락이 펴지지 않는다"
    free = torch.zeros(1, 6, dtype=torch.bool)
    assert torch.allclose(firmware_stop(half, g.unsqueeze(0), free, o, g), g.unsqueeze(0))


def test_open_floor_keeps_fingers_bent_like_real_hand():
    """실기: 0 rad 명령에도 네 손가락 2.2~5.2°·엄지 굽힘 1.2~2.6° 굽어 있다. 엄지 외전(편 쪽 = 큰 값)은 하한 없음."""
    o, g = torch.tensor(PROFILE.hand_open_pose), torch.tensor(PROFILE.hand_grip_pose)
    floor = sample_open_floor(256, CFG["hand_open_floor_deg_lo"], CFG["hand_open_floor_deg_hi"], "cpu")
    out = open_floor(o.unsqueeze(0).expand(256, -1), o, g, floor)
    depth = torch.rad2deg((out - o) * torch.sign(g - o))
    ti = next(i for i, n in enumerate(PROFILE.hand_joint_names) if n.endswith("_thumb_1"))
    assert float(depth[:, ti].abs().max()) < 1e-5
    four = [i for i, n in enumerate(PROFILE.hand_joint_names) if n.endswith("_1") and "thumb" not in n]
    assert float(depth[:, four].min()) >= 2.0 - 1e-4 and float(depth[:, four].max()) <= 5.5 + 1e-4
    grip = open_floor(g.unsqueeze(0).expand(256, -1), o, g, floor)
    assert torch.allclose(grip, g.unsqueeze(0).expand(256, -1)), "하한이 쥔 쪽 목표를 바꿨다"


def test_real_response_cfg_matches_measurements():
    dt_ms = 1000.0 / 60.0
    a_lo, a_hi = CFG["arm_cmd_delay_steps"]
    h_lo, h_hi = CFG["hand_cmd_delay_steps"]
    assert 145.0 <= a_lo * dt_ms and a_hi * dt_ms <= 205.0, "팔 150~200 ms(memory cg-l-i01)"
    assert h_lo * dt_ms <= 60.0 and 80.0 <= h_hi * dt_ms <= 100.0, "손 명령→첫 움직임 60~80 ms 를 덮어야 한다"
    lo, hi = CFG["hand_open_floor_deg_lo"], CFG["hand_open_floor_deg_hi"]
    measured = {"thumb_2": (1.2, 2.6), "index_1": (3.6, 3.9), "middle_1": (3.2, 2.4), "ring_1": (4.9, 2.7), "pinky_1": (5.2, 2.3)}
    for i, n in enumerate(PROFILE.hand_joint_names):
        for key, vals in measured.items():
            if n.endswith(key):
                assert all(lo[i] <= v <= hi[i] for v in vals), f"{n}: 실측 {vals} 가 [{lo[i]}, {hi[i]}] 밖"
    t_lo, t_hi = CFG["thumb1_reset_range"]
    assert t_lo <= 1.40 and 1.51 <= t_hi <= max(PROFILE.hand_open_pose)


def test_pd_gets_delayed_targets_obs_keeps_commanded():
    """배포처럼 관측은 명령한 목표, PD 는 지연·펌웨어를 거친 목표."""
    body = _code_only((_PKG / "rh_aglt_env.py").read_text())
    assert "set_joint_position_target ( self . arm_q_applied" in body
    assert "set_joint_position_target ( self . hand_applied" in body
    assert "set_joint_position_target ( self . arm_q_target" not in body
    assert "set_joint_position_target ( self . hand_target" not in body
    assert "arm_q , arm_qd , self . arm_q_target ," in body, "관측의 팔 목표가 명령 값이 아니다"
