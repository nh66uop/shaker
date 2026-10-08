"""좌/우 손을 **같은 조건으로 닫아** 손가락별 참여를 가른다 — 미러 이식 Step 2·3 검증.

★09.20 왜 만들었나: 좌팔 첫 런(`fj_randL_i00b`)에서 컵에 닿는 손가락이 엄지 0.67 · 중지 0.70 · 새끼 0.59 인데
  **검지 0.0004 · 약지 0.0003** 이었다(우판 같은 시점은 다섯 손가락 0.5~0.7). 정책 습관인지, 미러 부호·한계·자산 문제인지
  정책 없이 가른다: 컵을 손바닥 앞 파지 자리에 순간이동시키고 모든 손가락에 **닫기 지령 a=+1**(액션 한계 기준)을 준 뒤
  손가락별 실측 관절각·컵 접촉력을 찍는다. 두 task 를 같은 옵션으로 돌려 비교한다.

    isaaclab.sh -p .../probe_grasp_fj_t2r_close_hand.py --task open-short_l_grasp_fj_t2r_rand-lstm --num_envs 8
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_l_grasp_fj_t2r_rand-lstm")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--settle_steps", type=int, default=5, help="컵을 옮기기 전 대기 스텝")
parser.add_argument("--reward_code_path", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
if args.reward_code_path:
    cfg.reward_code_path = args.reward_code_path
env = gym.make(args.task, cfg=cfg).unwrapped
dev = env.device
env.reset()

fingers = list(env._finger_names)
names = [env.robot.data.joint_names[i] for i in env._syn_ids]
lo, hi = env._act_lo, env._act_hi
# 닫기 지령: 액션 한계에서 "더 굽힌 쪽" = 기본 자세에서 먼 쪽. 기본 자세(리셋 각)와의 거리로 고른다.
reset_q = env._hand_reset_q
close_q = torch.where((hi - reset_q).abs() >= (lo - reset_q).abs(), hi, lo)
a_close = torch.where(close_q >= hi - 1e-9, torch.ones_like(close_q), -torch.ones_like(close_q))

force_max = torch.zeros(args.num_envs, len(fingers), 3, device=dev)
palm_max = torch.zeros(args.num_envs, device=dev)
act = torch.zeros(args.num_envs, int(env.cfg.action_space), device=dev)
act[:, 7:] = a_close.unsqueeze(0)

for step in range(args.steps):
    if step == args.settle_steps:      # 컵을 손바닥 앞 파지 자리로 옮긴다(팔은 그대로)
        R = env._palm_ee_R()
        palm = (env._env_local(env.robot.data.body_pos_w[:, env.palm_idx])
                + torch.einsum("nij,j->ni", R, env._t2r_palm_center_off))
        target = palm + R[:, :, 0] * (env._obj_grasp_r.unsqueeze(1) + 0.005) + R[:, :, 2] * env._obj_grasp_r.unsqueeze(1)
        root = env.object.data.root_state_w.clone()
        root[:, 0:3] = target + env.scene.env_origins
        root[:, 7:13] = 0.0
        env.object.write_root_state_to_sim(root)
    env.step(act)
    links, palm_f = env._link_cup_forces()
    force_max = torch.maximum(force_max, links)
    palm_max = torch.maximum(palm_max, palm_f)

q = env.robot.data.joint_pos[:, env._syn_ids]
prog = ((q - lo.unsqueeze(0)) / (hi - lo).clamp(min=1e-6).unsqueeze(0))
out = {
    "task": args.task,
    "profile": env.cfg.profile_name,
    "hand_side": str(getattr(env.cfg, "hand_side", "r")),
    "finger_force_max_N": {f: round(float(force_max[:, i].max()), 3) for i, f in enumerate(fingers)},
    "finger_touch_frac": {f: round(float((force_max[:, i].amax(-1) > 0.1).float().mean()), 3)
                          for i, f in enumerate(fingers)},
    # 닫기 지령 대비 실측 — 1.0 = 지령한 끝까지 갔다(접촉에 막히면 그 앞에서 멈춘다)
    "joint_closed_frac": {nm: round(float(prog[:, j].mean() if close_q[j] >= hi[j] - 1e-9
                                          else 1.0 - prog[:, j].mean()), 3) for j, nm in enumerate(names)},
    "palm_force_max_N": round(float(palm_max.max()), 3),
}
print("CLOSE_HAND " + json.dumps(out, ensure_ascii=False, indent=1), flush=True)
env.close()
app.close()
