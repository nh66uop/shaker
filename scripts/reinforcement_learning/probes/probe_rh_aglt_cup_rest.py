"""rh_aglt 서 있는 컵의 각속도·선속도 — 손이 닿지 않는 상태에서 컵이 흔들리는가(09.29 aglt_r_i00 tip_penalty −0.115/스텝).

팔은 제자리(a=0), 손 액션 −1(손가락 편 쪽). 손은 시작 자세에서 컵보다 9 cm 이상 위라 닿지 않는다.
반발계수(재질 이벤트 restitution_range)를 바꿔 가며 잰다.

    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_aglt_cup_rest.py \
        --task open-rh_r_aglt-lstm --num_envs 256 --steps 300 --restitution 1.0
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_r_aglt-lstm")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--restitution", type=float, default=None, help="재질 이벤트 반발계수(로봇·컵 둘 다). 없으면 cfg 값")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
import openarm.agnostic.tasks.rh_aglt_r.config  # noqa: E402,F401  — 등록 ImportError 를 드러낸다
import openarm.agnostic.tasks.rh_aglt_l.config  # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
if args.restitution is not None:
    for term in (env_cfg.events.robot_material, env_cfg.events.cup_material):
        term.params["restitution_range"] = (args.restitution, args.restitution)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()
N, A = env.num_envs, env.cfg.action_space
act = torch.zeros(N, A, device=env.device)
act[:, 7:] = -1.0
w_hist, v_hist, tilt_hist, foreign = [], [], [], []
for t in range(args.steps):
    env.step(act)
    if t < int(env.cfg.hold_steps) + 5:
        continue
    w_hist.append(env.cup.data.root_ang_vel_w.norm(dim=-1))
    v_hist.append(env.cup.data.root_lin_vel_w.norm(dim=-1))
    up = env._cup_up()
    tilt_hist.append(torch.rad2deg(torch.acos(up[:, 2].clamp(-1, 1))))
    foreign.append(env._foreign_force())
w, v, tilt, fo = (torch.stack(x) for x in (w_hist, v_hist, tilt_hist, foreign))


def q(x: torch.Tensor) -> dict:
    x = x.flatten().float()
    return {k: round(float(torch.quantile(x, p)), 5) for k, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))} | \
        {"max": round(float(x.max()), 5), "mean": round(float(x.mean()), 5)}


res = {"restitution": args.restitution if args.restitution is not None else "cfg",
       "cup_ang_vel_rad_s": q(w), "cup_lin_vel_m_s": q(v), "cup_tilt_deg": q(tilt), "hand_foreign_N": q(fo),
       "tip_penalty_like": round(float((0.6 * torch.tanh(w / 0.3) * (0.5 + 0.5 * torch.tanh(torch.deg2rad(tilt) / 0.2))).mean()), 5)}
print("[RESULT] " + json.dumps(res, ensure_ascii=False), flush=True)
env.close()
app.close()
