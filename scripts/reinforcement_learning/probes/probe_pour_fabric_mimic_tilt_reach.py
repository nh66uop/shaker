"""붓기 자세 도달 프로브 — "어디에서 컵을 얼마나 기울일 수 있나"를 학습 없이 잰다.

09.25 분석 근거: i17g trace 에서 소스 palm 회전 명령이 박스 한계(−150°)에 붙어 있는데도
컵 기울기는 중앙 24°였고, 컵이 중앙을 넘어간 y +0.06~+0.20 구간에서만 75° 초과가 26 % 나왔다.
관절 한계는 여유(최소 0.095)가 있었으므로 Fabrics 제어기가 자세를 못 만드는 것인지,
그 위치에서 팔이 원래 못 가는 것인지 가른다.

대본: 접근 → 손 닫기 → 들기 → **env 마다 다른 가로(y) 목표로 이동** → 기울기 명령을 0 → −1 로 램프.
env 별로 y 명령을 -1..+1 로 나눠 한 번에 쓸어본다. 각 env 의 달성 기울기·회전 오차·관절 여유를 출력한다.

실행(서버):
  isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_fabric_mimic_tilt_reach.py \
      --num_envs 16 --steps 900 --headless
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--approach_m", type=float, default=0.07)
parser.add_argument("--lift_m", type=float, default=0.12)
parser.add_argument("--tilt_slot", type=int, default=5)
parser.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym                      # noqa: E402
import torch                                 # noqa: E402
import openarm.agnostic.tasks                # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.reward_code_path = ""
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
N, A = env.num_envs, env.cfg.action_space
half = A // 2
hold = int(env.cfg.hold_steps)
dev = env.device

# ★09.25 3차: 가로(y) × 높이(z) 2차원 스윕. env 를 (ny × nz) 격자로 나눠 한 번에 지도를 만든다.
#   측정: 같은 회전 명령에서 달성 roll·회전 오차·관절 여유(특히 r_aj_2)·어느 관절이 병목인지.
#   근거(09.25): j2 는 리셋 자세 여유 0.341 인데 운영 중 하한(−0.17)까지 내려가 박히고,
#   그만큼 손목(j6 93.6 %·j7 98.8 %)이 범위를 다 쓴다. 작업 위치를 바꾸면 이 배분이 달라지는지 본다.
ny, nz = 4, 4
N_need = ny * nz
if N != N_need:
    print(f"[warn] num_envs {N} != {N_need} — 격자를 env 수에 맞춰 자른다", flush=True)
y_grid = torch.linspace(-1.0, 1.0, ny, device=dev)
z_grid = torch.linspace(-1.0, 1.0, nz, device=dev)
y_cmd = y_grid.repeat_interleave(nz)[:N]
z_cmd = z_grid.repeat(ny)[:N]
MOVE_AT, TILT_AT = 60, 260
rows = []
snap = None

for t in range(args.steps):
    a = torch.zeros(N, A, device=dev)
    u = t - hold
    if u >= 0:
        a[:, 2] = args.lift_m / float(env.src.delta_hi[2])
        if u >= MOVE_AT:
            a[:, 1] = y_cmd
            a[:, 2] = torch.where(z_cmd >= 0, z_cmd, z_cmd)   # z 도 env 별로(부호별 델타 매핑은 env 가 처리)
        if u >= TILT_AT:
            a[:, args.tilt_slot] = -min((u - TILT_AT) / 200.0, 1.0)
    env.step(a)
    ex = env.extras
    if t % 150 == 0:
        print(f"[step {t:4d}] rotErr={float(ex['fabric/src_rot_err_deg']):6.1f}deg "
              f"palmErr={float(ex['fabric/src_palm_err'])*1000:6.1f}mm", flush=True)
    if t == min(args.steps, int(env.max_episode_length)) - 30:
        pose = env.src.palm_pose_6d()
        cmd = env.src.palm_targets.clone()
        cmd = torch.cat([cmd[:, :3] + env.src.fab_to_env, cmd[:, 3:]], dim=1)
        q = env.robot.data.joint_pos[:, env.src.arm_t]
        lim = env.robot.data.soft_joint_pos_limits[:, env.src.arm_t]
        mid = 0.5 * (lim[..., 0] + lim[..., 1])
        half_r = (0.5 * (lim[..., 1] - lim[..., 0])).clamp(min=1e-6)
        marg = (1.0 - (q - mid).abs() / half_r).clamp(0.0, 1.0)
        snap = dict(pose=pose.detach().cpu(), cmd=cmd.detach().cpu(), marg=marg.detach().cpu())

# env 별 결과 (에피소드 종료 30 스텝 전 스냅샷)
import math as _math
names = [env.robot.joint_names[i] for i in env.src.arm_ids]
print("\n가로(y)×높이(z) 격자별 palm 자세 도달성")
print(f"{'palm_y':>8s} {'palm_z':>7s} {'실제roll':>9s} {'오차':>7s} {'j2여유':>7s} {'최소여유':>8s} {'병목관절':>10s}")
if snap is not None:
    for i in range(N):
        py, pz = float(snap["pose"][i, 1]), float(snap["pose"][i, 2])
        cr, pr = float(snap["cmd"][i, 5]), float(snap["pose"][i, 5])
        err = _math.degrees(abs((cr - pr + _math.pi) % (2 * _math.pi) - _math.pi))
        mg = snap["marg"][i]
        jmin = int(mg.argmin())
        rows.append(dict(palm_y=py, palm_z=pz, act_roll_deg=_math.degrees(pr), roll_err_deg=err,
                         j2_margin=float(mg[1]), margin_min=float(mg.min()), bottleneck=names[jmin]))
        print(f"{py:+8.3f} {pz:7.3f} {_math.degrees(pr):+9.1f} {err:7.1f} {float(mg[1]):7.3f} {float(mg.min()):8.3f} {names[jmin]:>10s}")
if args.out and rows:
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"[out] {args.out}")
env.close()
app.close()
