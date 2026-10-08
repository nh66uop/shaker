"""붓기 인계 뱅크 검사 — pour_fj env 를 뱅크 100 % 로 리셋하고 '지금 목표 그대로 유지' 액션으로 굴려 파지가 남는지 잰다.

10.01 사용자: "둘로 나누고 100 % 뱅크로 · aglt 할 때 비드를 넣어 둔 상태로 모으기".
뱅크 = rh_aglt i10 우(비드)·좌 쥔 상태 짝(scripts/tools/build_pour_handoff_bank.py). 같은 자산·같은 손 법칙이지만
env 가 다르다(게인·자기충돌·마찰 DR·두 팔 동시) — 옮긴 상태가 pour env 에서 그대로 서 있는지가 이 검사다.
유지 액션: 팔은 `arm_joint_mode`(absolute: 목표를 역사상, increment: 0), 손은 direct grip 사상의 역(목표 그대로).

출력: 항목별 결과 npz(--out_keep 이면 통과 항목만 추린 뱅크) + 요약 json.
    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_handoff_bank.py --headless \
        --bank bank.npz --num_envs 512 --steps 120 --out_keep bank_kept.npz
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_b_pour_fj")
parser.add_argument("--bank", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=120, help="뱅크 리셋 뒤 유지할 스텝(60 Hz)")
parser.add_argument("--reward_code_path", default="")
parser.add_argument("--arm_joint_mode", default="absolute", choices=("absolute", "increment"))
parser.add_argument("--offset", type=int, default=-1, help="0 이상이면 뱅크를 그 위치부터 차례로(전수 검사)")
parser.add_argument("--out_keep", default="", help="통과 항목만 추린 뱅크 경로")
parser.add_argument("--summary", default="")
parser.add_argument("--ok_idx", default="", help="통과한 뱅크 인덱스를 npy 로(여러 번 나눠 돈 결과 합치기용)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import numpy as np           # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
import openarm.agnostic.tasks.pour_fabric_mimic.config  # noqa: E402,F401

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
# f05(t2r_rh5_f05) 기동 인자와 같은 env 값 — 학습 때와 같은 물리·행동 법칙에서 잰다
for k, v in {"object_spawn_dx": -0.12, "hand_cmd_delay_steps": 0, "arm_cmd_delay_steps": (0, 0),
             "hand_vel_cap_rad_s": 2.1, "hand_thumb_flex_vel_cap_rad_s": 0.56, "hand_finger_open_floor_rad": 0.065,
             "arm_joint_mode": args.arm_joint_mode, "phys_cup_speed_max": 3.0, "phys_cup_angvel_max": 30.0,
             "phys_contact_max_n": 200.0, "phys_contact_steps": 2, "cup_leash_m": 0.5,
             "start_bank_path": args.bank, "start_bank_frac": 1.0}.items():
    if not hasattr(cfg, k):
        raise SystemExit(f"cfg 에 {k} 가 없다")
    setattr(cfg, k, v)
if args.reward_code_path:
    cfg.reward_code_path = args.reward_code_path
env = gym.make(args.task, cfg=cfg).unwrapped
N, dev = env.num_envs, env.device
A, h = env.cfg.action_space, env._half
w = env._arm_w


def hold_action() -> torch.Tensor:
    """지금 팔·손 목표를 그대로 유지하는 액션(사상의 역)."""
    a = torch.zeros(N, A, device=dev)
    floor = float(getattr(env.cfg, "hand_finger_open_floor_rad", 0.0))
    for i, rig in enumerate((env.src, env.rcv)):
        if str(env.cfg.arm_joint_mode) == "absolute":
            home = rig.reset_q[rig.arm_t].unsqueeze(0)
            q = rig.arm_q_target
            up = (q - home) / (rig.arm_hi - home).clamp(min=1e-6)
            dn = (q - home) / (home - rig.arm_lo).clamp(min=1e-6)
            a[:, i * h:i * h + w] = torch.where(q >= home, up, dn).clamp(-1.0, 1.0)
        lo = torch.minimum(rig.syn_open, rig.syn_grip).clamp(rig.syn_lo, rig.syn_hi)
        hi = torch.maximum(rig.syn_open, rig.syn_grip).clamp(rig.syn_lo, rig.syn_hi)
        if floor > 0.0:
            lo = torch.where(rig._finger_mask, lo.clamp(min=floor), lo)
            hi = torch.maximum(hi, lo)
        a_j = (2.0 * (rig.syn_target - lo) / (hi - lo).clamp(min=1e-6) - 1.0).clamp(-1.0, 1.0)
        a_hand = torch.zeros_like(a_j)
        a_hand[:, rig.syn_slot] = a_j
        a[:, i * h + w:(i + 1) * h] = a_hand
    return a


def outside_grip(rig) -> torch.Tensor:
    lo = torch.minimum(rig.syn_open, rig.syn_grip).clamp(rig.syn_lo, rig.syn_hi)
    hi = torch.maximum(rig.syn_open, rig.syn_grip).clamp(rig.syn_lo, rig.syn_hi)
    t = rig.syn_target
    return ((t < lo - 1e-4) | (t > hi + 1e-4)).any(dim=1)


if args.offset >= 0:
    env._bank.seq_next = args.offset
env.reset()
idx = env._bank.last_idx.clone()
if (idx < 0).any():
    raise SystemExit("리셋 뒤 뱅크에서 시작하지 않은 env 가 있다 — start_bank_frac 1.0 확인")
out_src, out_rcv = outside_grip(env.src).float().mean().item(), outside_grip(env.rcv).float().mean().item()
alive = torch.ones(N, dtype=torch.bool, device=dev)
for t in range(args.steps):
    _, _, term, trunc, _ = env.step(hold_action())
    alive &= ~(term | trunc)
o = env.scene.env_origins
lift_s = (env.source_cup.data.root_pos_w - o)[:, 2] - env._src_spawn[:, 2]
lift_r = (env.receiver_cup.data.root_pos_w - o)[:, 2] - env._rcv_spawn[:, 2]
in_src = env._prev_in_src
ok = alive & env._src_grasped & env._rcv_grasped & (lift_s > 0.03) & (lift_r > 0.03) & (in_src >= 0.999)
res = {"n_env": N, "steps": args.steps, "arm_joint_mode": args.arm_joint_mode,
       "keep": float(ok.float().mean()), "alive": float(alive.float().mean()),
       "src_grasped": float(env._src_grasped.float().mean()), "rcv_grasped": float(env._rcv_grasped.float().mean()),
       "src_lift_cm_med": float(lift_s.median() * 100), "rcv_lift_cm_med": float(lift_r.median() * 100),
       "in_source_mean": float(in_src.mean()),
       "hand_target_outside_grip_src": out_src, "hand_target_outside_grip_rcv": out_rcv}
print("[handoff] " + json.dumps(res, ensure_ascii=False), flush=True)
if args.summary:
    with open(args.summary, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
if args.ok_idx:
    np.save(args.ok_idx, np.array(sorted(set(idx[ok].tolist())), dtype=np.int64))
    np.save(args.ok_idx.replace(".npy", "_tested.npy"), np.array(sorted(set(idx.tolist())), dtype=np.int64))
if args.out_keep:
    good = sorted(set(idx[ok].tolist()))
    z = np.load(args.bank)
    keep = {k: (z[k][good] if z[k].ndim and len(z[k]) == len(z["joint_pos"]) and not k.endswith("_names") else z[k])
            for k in z.files}
    np.savez_compressed(args.out_keep, **keep)
    print(f"[handoff] 통과 항목 {len(good)} 개(뱅크 {len(z['joint_pos'])} 중) → {args.out_keep}", flush=True)
env.close()
app.close()
