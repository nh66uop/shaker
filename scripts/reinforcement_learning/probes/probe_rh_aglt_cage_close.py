"""rh_aglt '손바닥 깊이 + 형상으로 걸림' 파지 탐색 프로브 (09.30 사용자: "마디가 다 닫힐 필요는 없고, 손바닥 쪽으로
최대한 들어오게 잡게 하려는 것. 원통이라 팁으로 누르는 것보다 기하적으로 컵이 걸리게").

probe_rh_aglt_envelope_close.py(손 방향 1가지) 확장: 손 방향(손가락 축 기울기 θ, 손가락 축 둘레 굴림 ρ)과 손바닥–컵 축
거리 d 를 훑어, 손을 닫은 뒤
  - 손바닥–컵 축 거리(r 배수, 작을수록 컵이 손바닥 깊이) · 손바닥 패드 접촉
  - 감김: 네 손가락 끝이 컵 축 너머(손바닥 반대편)에 있는 비율 — 원통이 손 밖으로 빠질 수 없는 형상 걸림
  - 엄지 대향, 손가락별 첫마디·손끝 접촉, 쥔 판정
  - 천천히 5 cm 들어 2 s 유지한 뒤 들린 높이·기울기
를 조건별로 적는다. 학습과 같은 지령 경로(팔 증분·EMA, 손 선형 사상·1 N 멈춤).
"""
import argparse
import json
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--per", type=int, default=4)
ap.add_argument("--d_list", default="1.15,1.3,1.5")
ap.add_argument("--pitch_list", default="0,-20", help="손가락 축 아래로 기울기(도, 음수 = 손끝 아래)")
ap.add_argument("--roll_list", default="-20,0,20", help="손가락 축 둘레 굴림(도, + = 새끼 쪽이 컵에 가까워지게)")
ap.add_argument("--h", type=float, default=0.15)
ap.add_argument("--reward", default="reward_gen/rh_aglt_r/iter_07/compute_reward.py")
ap.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_matrix, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.agnostic.tasks.rh_aglt_r.config  # noqa: E402,F401

D = [float(v) for v in args.d_list.split(",")]
PI = [float(v) for v in args.pitch_list.split(",")]
RO = [float(v) for v in args.roll_list.split(",")]
GROUPS = [(d, p, r) for d in D for p in PI for r in RO]
task = "open-rh_r_aglt-lstm"
n_env = len(GROUPS) * args.per
cfg = parse_env_cfg(task, device=args.device, num_envs=n_env)
cfg.reward_code_path = args.reward
cfg.enable_adr = False
cfg.episode_length_s = 20.0                      # 900 스텝 프로브가 시간 초과 리셋에 걸리지 않게
env = gym.make(task, cfg=cfg).unwrapped
env.reset()
dev, N = env.device, env.num_envs
R, MZ = float(env.cfg.cup_radius), float(env.cfg.cup_mouth_z)
gd = torch.tensor([GROUPS[i // args.per][0] for i in range(N)], device=dev)
gp = torch.deg2rad(torch.tensor([GROUPS[i // args.per][1] for i in range(N)], device=dev))
gr = torch.deg2rad(torch.tensor([GROUPS[i // args.per][2] for i in range(N)], device=dev))


def rot_axis(axis: torch.Tensor, ang: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """로드리게스 회전(배치)."""
    k = axis / axis.norm(dim=-1, keepdim=True)
    c, s = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)
    return v * c + torch.cross(k, v, dim=-1) * s + k * (k * v).sum(-1, keepdim=True) * (1 - c)


n0 = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3).clone()       # 법선: 중심선 쪽(+y)
f0 = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3).clone()       # 손가락: +x
# 손가락 축 아래로 기울기: 법선 둘레 회전(손가락 끝이 아래로 → 음의 각)
f1 = rot_axis(n0, -gp, f0)
n1 = n0
# 손가락 축 둘레 굴림: 법선이 아래/위로 — + 면 법선이 아래를 조금 봄(새끼 쪽이 컵에 가까워짐)
n2 = rot_axis(f1, gr, n1)
f2 = f1
x2 = torch.cross(f2, n2, dim=-1)
rot_t = torch.stack([x2, f2, n2], -1)
q_t = quat_from_matrix(rot_t)
e = 0.67 * n2 + 0.73 * f2
e[:, 2] = 0.0
e = e / e.norm(dim=-1, keepdim=True)                                      # 손바닥 → 컵 축(수평)
fixed = env.robot.is_fixed_base
a_close = torch.where(env.q_grip >= env.q_open, 1.0, -1.0)
n_arm = len(env.arm_ids)


def ik_action(pos_t: torch.Tensor) -> torch.Tensor:
    J = env.robot.root_physx_view.get_jacobians()
    bi = env.palm_idx - 1 if fixed else env.palm_idx
    Jb = J[:, bi, 0:6, :][:, :, env.arm_t + (0 if fixed else 6)]
    err_p = pos_t - env._palm_pos()
    err_r = axis_angle_from_quat(quat_mul(q_t, quat_conjugate(env.robot.data.body_quat_w[:, env.palm_idx])))
    err = torch.cat([err_p, err_r], -1)
    JJt = Jb @ Jb.transpose(1, 2) + (0.05 ** 2) * torch.eye(6, device=dev)
    dq = (Jb.transpose(1, 2) @ torch.linalg.solve(JJt, err.unsqueeze(-1))).squeeze(-1)
    q = env.robot.data.joint_pos[:, env.arm_t]
    q_des = (q + 0.5 * dq).clamp(env.arm_lo, env.arm_hi)
    return ((q_des - env.arm_q_target) / float(env.cfg.k_arm)).clamp(-1.0, 1.0)


def snapshot(tag: str, cup0: torch.Tensor) -> dict:
    prox, tip = env._link_forces()
    cp = env._cup_pos()
    qw = env.cup.data.root_quat_w
    up = torch.stack([2 * (qw[:, 1] * qw[:, 3] + qw[:, 0] * qw[:, 2]), 2 * (qw[:, 2] * qw[:, 3] - qw[:, 0] * qw[:, 1]),
                      1 - 2 * (qw[:, 1] ** 2 + qw[:, 2] ** 2)], -1)
    tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1, 1)))
    v = env._palm_pos() - cp
    h = (v * up).sum(-1, keepdim=True)
    palm_r = (v - h * up).norm(dim=-1) / R
    tips = env._tips_pos() - cp.unsqueeze(1)
    tr = tips - (tips * up.unsqueeze(1)).sum(-1, keepdim=True) * up.unsqueeze(1)
    ph = -(v - h * up)
    ph = ph / ph.norm(dim=-1, keepdim=True).clamp(min=1e-6)                 # 손바닥 → 축(컵 기준)
    beyond = ((tr[:, 1:] * ph.unsqueeze(1)).sum(-1) > 0).float()            # 네 손가락 끝이 축 너머
    ut = tr / tr.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    oppose = (-(ut[:, 0] * ut[:, 1:].mean(1)).sum(-1)).clamp(0, 1)
    palm_f = env._mag(env._palm_sensor)
    lift = cp[:, 2] - cup0[:, 2]
    out = {}
    for k, (d, p, r) in enumerate(GROUPS):
        sl = slice(k * args.per, (k + 1) * args.per)
        out[f"d{d:.2f}_p{p:+.0f}_r{r:+.0f}"] = {
            "palm_axis_r": round(float(palm_r[sl].mean()), 2), "palm_touch": round(float((palm_f[sl] > 0.5).float().mean()), 2),
            "wrap_beyond(I,M,R,P)": [round(float(beyond[sl, i].mean()), 2) for i in range(4)],
            "oppose": round(float(oppose[sl].mean()), 2),
            "prox(T,I,M,R,P)": [round(float((prox[sl, i] > 0.5).float().mean()), 2) for i in range(5)],
            "tip(T,I,M,R,P)": [round(float((tip[sl, i] > 0.5).float().mean()), 2) for i in range(5)],
            "grasped": round(float(env._grasp_flag(prox + tip)[sl].float().mean()), 2),
            "lift_cm": round(100 * float(lift[sl].mean()), 1), "tilt_deg": round(float(tilt[sl].mean()), 1)}
    print(json.dumps({"phase": tag, **out}, ensure_ascii=False), flush=True)
    return out


cup0 = env._cup_pos().clone()
pos_goal = cup0 - gd.unsqueeze(1) * R * e
pos_goal[:, 2] = cup0[:, 2] + args.h * MZ
pos_pre = pos_goal - 0.05 * e
result = {}
T_PRE, T_IN, T_CLOSE, T_LIFT = 300, 480, 660, 840
for t in range(T_LIFT + 60):
    a = torch.zeros(N, env.cfg.action_space, device=dev)
    if t < T_PRE:
        tgt, hand = pos_pre, -a_close
    elif t < T_IN:
        tgt, hand = pos_goal, -a_close
    elif t < T_CLOSE:
        tgt, hand = pos_goal, a_close
    else:
        frac = min(1.0, (t - T_CLOSE) / 120.0)
        tgt = pos_goal.clone()
        tgt[:, 2] += 0.05 * frac
        hand = a_close
    a[:, :n_arm] = ik_action(tgt)
    a[:, n_arm:] = hand
    env.step(a)
    if t == T_CLOSE - 1:
        result["closed"] = snapshot("closed", cup0)
    if t == T_LIFT + 59:
        result["lifted_held"] = snapshot("lifted_held", cup0)
if args.out:
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
env.close()
app.close()
