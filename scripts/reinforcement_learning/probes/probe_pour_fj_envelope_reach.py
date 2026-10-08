"""pour_fj 인벨롭 파지 자세 도달성 프로브 (09.30 사용자 "지금도 자세가 이상하고 컵도 제대로 파지를 못하고 있음").

f08·f09 두 런 모두 입구를 비스듬히 위에서 잡았고(손바닥 높이 입구의 +0.6~+0.9, 축 거리 2.1~2.5 r, 손가락·컵 축 −0.6)
어깨 j2 가 한계에 붙었다. K7 인벨롭 자세가 이 로봇·컵 위치에서 관절 한계 안으로 닿는지 먼저 본다.

방법: 컵은 멀리 치우고(접촉 없이), 각 env 에 접근 방위 φ(손바닥 → 컵 축 방향)를 하나씩 준다. 목표 손바닥 자세 =
컵 스폰 원점 기준 높이 +0.15 × 입구, 컵 축에서 1.5 r, 법선·손가락 축 수평, 손바닥→축 방향 = 0.67 법선 + 0.73 손가락 축(K7).
6D 감쇠 최소제곱 IK 로 관절 목표를 구해 팔 절대 목표 액션으로 넣는다(학습과 같은 지령 경로, EMA·속도 상한 포함).
끝에서 위치·방향 오차, 관절별 한계 여유(도), 손 외부 접촉을 방위별로 적는다.
"""
import argparse
import json
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--per_az", type=int, default=6)
ap.add_argument("--steps", type=int, default=720)
ap.add_argument("--out", default="")
ap.add_argument("--spawn_dx", type=float, default=0.0, help="env cfg object_spawn_dx (컵 스폰 자체를 옮김)")
ap.add_argument("--az_list", default="-90,-60,-30,0,30,60,90,120")
ap.add_argument("--dx_list", default="0", help="목표를 컵 스폰에서 x 로 옮김(음수 = 로봇 쪽) — 컵을 가까이 두면 여유가 생기는지")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_conjugate, quat_from_matrix, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.agnostic.tasks.pour_fabric_mimic.config  # noqa: E402,F401
import openarm.tasks  # noqa: E402,F401

AZ_DEG = [int(v) for v in args.az_list.split(",")]   # 소스 기준 방위(0 = 손바닥이 로봇 쪽 −x 에서 +x 로 컵을 봄). 리시버는 y 반전
DX = [float(v) for v in args.dx_list.split(",")]
GROUPS = [(a_, d_) for d_ in DX for a_ in AZ_DEG]
task = "open-rh_b_pour_fj"
n_env = len(GROUPS) * args.per_az
cfg = parse_env_cfg(task, device=args.device, num_envs=n_env)
cfg.arm_joint_mode = "absolute"
cfg.object_spawn_dx = float(args.spawn_dx)
env = gym.make(task, cfg=cfg).unwrapped
env.reset()
dev, N, h = env.device, env.num_envs, env._half
R, MZ = 0.02808, float(env.cfg.cup_mouth_z)
ENV_N, ENV_F = 0.67, 0.73
az = torch.tensor([GROUPS[i // args.per_az][0] for i in range(N)], dtype=torch.float32, device=dev)
dxv = torch.tensor([GROUPS[i // args.per_az][1] for i in range(N)], dtype=torch.float32, device=dev)
jac_view = env.robot.root_physx_view
fixed = env.robot.is_fixed_base
spawn = {"src": env._src_spawn.clone(), "rcv": env._rcv_spawn.clone()}


def target_pose(rig):
    """(pos (N,3) env-local, rot (N,3,3) 열 = 로컬 x·y(손가락)·z(법선))."""
    sgn = 1.0 if rig.role == "src" else -1.0
    phi = torch.deg2rad(az) * sgn
    e = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], -1)      # 손바닥 → 컵 축
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(e)
    ep = torch.cross(ez, e, dim=-1)
    a_n, a_f = math.acos(ENV_N), math.acos(ENV_F)
    best = None
    for s in (1.0, -1.0):
        n = math.cos(a_n) * e + s * math.sin(a_n) * ep
        f = math.cos(a_f) * e - s * math.sin(a_f) * ep
        x = torch.cross(f, n, dim=-1)
        # 실측(f08 재생 접근): 소스 로컬 x 의 z 성분 > 0, 리시버 < 0
        ok = (x[:, 2] * sgn) > 0
        rot = torch.stack([x, f, n], -1)
        best = rot if best is None else torch.where(ok[:, None, None], rot, best)
    cp = spawn[rig.role].clone()
    cp[:, 0] += dxv
    pos = cp - 1.5 * R * e
    pos[:, 2] = cp[:, 2] + 0.15 * MZ
    return pos, best


tgt = {rig.role: target_pose(rig) for rig in env.rigs}
rows = []
for t in range(args.steps):
    # 컵은 멀리(접촉 없이) — 매 스텝 되돌린다
    for cup, dx in ((env.source_cup, 3.0), (env.receiver_cup, 3.5)):
        st = cup.data.default_root_state.clone()
        st[:, :3] = env.scene.env_origins + torch.tensor([dx, 0.0, 0.5], device=dev)
        st[:, 7:] = 0.0
        cup.write_root_state_to_sim(st)
    a = torch.zeros(N, env.cfg.action_space, device=dev)
    J = jac_view.get_jacobians()
    for i, rig in enumerate(env.rigs):
        pos_t, rot_t = tgt[rig.role]
        pos_c = rig.palm_pos()
        q_c = env.robot.data.body_quat_w[:, rig.palm_idx]
        err_p = pos_t - pos_c
        q_t = quat_from_matrix(rot_t)
        err_r = axis_angle_from_quat(quat_mul(q_t, quat_conjugate(q_c)))
        err = torch.cat([err_p, err_r], -1)
        bi = rig.palm_idx - 1 if fixed else rig.palm_idx
        Jb = J[:, bi, 0:6, :][:, :, rig.arm_t + (0 if fixed else 6)]
        lam = 0.05
        JJt = Jb @ Jb.transpose(1, 2) + (lam ** 2) * torch.eye(6, device=dev)
        dq = (Jb.transpose(1, 2) @ torch.linalg.solve(JJt, err.unsqueeze(-1))).squeeze(-1)
        q = env.robot.data.joint_pos[:, rig.arm_t]
        q_des = (q + 0.5 * dq).clamp(rig.arm_lo, rig.arm_hi)
        home = rig.reset_q[rig.arm_t].unsqueeze(0)
        act = torch.where(q_des >= home, (q_des - home) / (rig.arm_hi - home).clamp(min=1e-6),
                          (q_des - home) / (home - rig.arm_lo).clamp(min=1e-6))
        a[:, i * h:i * h + 7] = act.clamp(-1.0, 1.0)
        a[:, i * h + 7:(i + 1) * h] = -1.0                       # 손 편 채
    env.step(a)
    if (t + 1) % 120 == 0 or t == args.steps - 1:
        row = {"t": t + 1}
        for rig in env.rigs:
            pos_t, rot_t = tgt[rig.role]
            pe = (pos_t - rig.palm_pos()).norm(dim=-1)
            rc = matrix_from_quat(env.robot.data.body_quat_w[:, rig.palm_idx])
            cos_a = (((rc.transpose(1, 2) @ rot_t).diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            ang = torch.rad2deg(torch.acos(cos_a))
            q = env.robot.data.joint_pos[:, rig.arm_t]
            mar = torch.rad2deg(torch.minimum(q - rig.arm_lo, rig.arm_hi - q))
            ff = rig.foreign_force()
            per = {}
            for k, (azd, dxd) in enumerate(GROUPS):
                sl = slice(k * args.per_az, (k + 1) * args.per_az)
                per[f"{azd}@{dxd:+.2f}"] = {"pos_cm": round(100 * float(pe[sl].mean()), 1), "ang_deg": round(float(ang[sl].mean()), 1),
                                 "margin_deg": [round(float(v), 1) for v in mar[sl].mean(0)],
                                 "min_margin_deg": round(float(mar[sl].min(1).values.mean()), 1),
                                 "foreign": round(float((ff[sl] > 1.0).float().mean()), 2)}
            row[rig.role] = per
        rows.append(row)
        print(json.dumps({"t": row["t"], "src": {k: (v["pos_cm"], v["ang_deg"], v["min_margin_deg"]) for k, v in row["src"].items()},
                          "rcv": {k: (v["pos_cm"], v["ang_deg"], v["min_margin_deg"]) for k, v in row["rcv"].items()}}), flush=True)
if args.out:
    json.dump(rows, open(args.out, "w"), indent=1)
env.close()
app.close()
