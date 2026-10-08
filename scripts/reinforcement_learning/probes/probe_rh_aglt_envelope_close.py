"""rh_aglt 인벨롭 파지 가능성 프로브 (09.30 사용자 "제일 좋은 파지는 손마디를 쓰는 인벨롭").

지금까지 어떤 정책도 네 손가락 첫마디로 컵을 감싸지 않았다(좌 i05 거울 궤적: 첫마디 검지 0.35·중지~새끼 0).
RH56F1(짧은 손가락)·이 컵(반지름 ~2.9 cm)으로 인벨롭이 물리적으로 되는지, 되면 손바닥을 어디에 두어야 하는지 본다.

방법(오른팔 env, 학습과 같은 지령 경로 — 팔 증분 액션·EMA, 손 선형 사상·1 N 멈춤):
  1) 감쇠 최소제곱 IK 로 손바닥을 컵 옆 사전 자리(목표에서 바깥으로 4 cm)에 편 손으로 둔다.
  2) 목표 자리로 들어간다: 컵 원점 높이 + h·입구, 컵 축에서 d·r, 법선 +y(중심선 쪽)·손가락 +x 수평,
     손바닥→축 방향 = 0.67 법선 + 0.73 손가락(K7, pour_bi_rh i24 실측).
  3) 손을 grip 쪽으로 닫는다(env 의 1 N 멈춤 그대로).
  4) 쥔 채 손바닥을 5 cm 올린다.
env 묶음마다 (d, h) 를 바꿔 손가락별 첫마디(>0.5 N)·손끝 접촉, 쥔 판정, 들린 높이, 컵 기울기를 적는다.
"""
import argparse
import json
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--per", type=int, default=8)
ap.add_argument("--d_list", default="1.3,1.5,1.7,2.0")
ap.add_argument("--h_list", default="0.15,0.45")
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
HH = [float(v) for v in args.h_list.split(",")]
GROUPS = [(d, h) for d in D for h in HH]
task = "open-rh_r_aglt-lstm"
n_env = len(GROUPS) * args.per
cfg = parse_env_cfg(task, device=args.device, num_envs=n_env)
cfg.reward_code_path = args.reward
cfg.enable_adr = False
env = gym.make(task, cfg=cfg).unwrapped
env.reset()
dev, N = env.device, env.num_envs
R, MZ = float(env.cfg.cup_radius), float(env.cfg.cup_mouth_z)
gd = torch.tensor([GROUPS[i // args.per][0] for i in range(N)], device=dev)
gh = torch.tensor([GROUPS[i // args.per][1] for i in range(N)], device=dev)
n_ax = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3)
f_ax = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3)
e = 0.67 * n_ax + 0.73 * f_ax
e = e / e.norm(dim=-1, keepdim=True)
rot_t = torch.stack([torch.cross(f_ax, n_ax, dim=-1), f_ax, n_ax], -1)
q_t = quat_from_matrix(rot_t)
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
    up = env.cup.data.root_quat_w
    zax = torch.stack([2 * (up[:, 1] * up[:, 3] + up[:, 0] * up[:, 2]), 2 * (up[:, 2] * up[:, 3] - up[:, 0] * up[:, 1]),
                       1 - 2 * (up[:, 1] ** 2 + up[:, 2] ** 2)], -1)
    tilt = torch.rad2deg(torch.acos(zax[:, 2].clamp(-1, 1)))
    lift = cp[:, 2] - cup0[:, 2]
    pe = (pos_goal - env._palm_pos()).norm(dim=-1)
    out = {}
    for k, (d, h) in enumerate(GROUPS):
        sl = slice(k * args.per, (k + 1) * args.per)
        out[f"d{d:.1f}r_h{h:.2f}"] = {
            "palm_err_cm": round(100 * float(pe[sl].mean()), 1),
            "prox_touch(T,I,M,R,P)": [round(float((prox[sl, i] > 0.5).float().mean()), 2) for i in range(5)],
            "tip_touch": [round(float((tip[sl, i] > 0.5).float().mean()), 2) for i in range(5)],
            "grasped": round(float(env._grasp_flag(prox + tip)[sl].float().mean()), 2),
            "lift_cm": round(100 * float(lift[sl].mean()), 1), "tilt_deg": round(float(tilt[sl].mean()), 1),
            "cup_shift_cm": round(100 * float((cp[sl, :2] - cup0[sl, :2]).norm(dim=-1).mean()), 1)}
    print(json.dumps({"phase": tag, **out}, ensure_ascii=False), flush=True)
    return out


cup0 = env._cup_pos().clone()
pos_goal = cup0 - gd.unsqueeze(1) * R * e
pos_goal[:, 2] = cup0[:, 2] + gh * MZ
pos_pre = pos_goal - 0.04 * e
pos_pre[:, 2] = pos_goal[:, 2]
result = {}
for t in range(840):
    a = torch.zeros(N, env.cfg.action_space, device=dev)
    if t < 360:
        tgt, hand = pos_pre, -a_close
    elif t < 540:
        tgt, hand = pos_goal, -a_close
    elif t < 720:
        tgt, hand = pos_goal, a_close
    else:
        tgt = pos_goal.clone()
        tgt[:, 2] += 0.05
        hand = a_close
    a[:, :n_arm] = ik_action(tgt)
    a[:, n_arm:] = hand
    env.step(a)
    if t == 539:
        result["at_pose_open"] = snapshot("at_pose_open", cup0)
    if t == 719:
        result["closed"] = snapshot("closed", cup0)
    if t == 839:
        result["lifted"] = snapshot("lifted", cup0)
if args.out:
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
env.close()
app.close()
