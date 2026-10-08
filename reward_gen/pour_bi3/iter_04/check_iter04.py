"""iter_04 grip_slip 검사: (1) 강체 파지(컵이 손과 같이 병진·회전)면 0, (2) 상대속도에 단조·포화,
(3) 결정론 trace 에서 i03f(왼손 풀림) 대 i24 지급액 — 컵 좌표계 손바닥 위치의 유한차분으로 근사."""
import numpy as np
import torch

S = "/tmp/claude-1000/-home-user-rl-ws/97645183-1013-4081-a16e-1eee7f0beebb/scratchpad/"
W, FREE, SCALE = 1.0, 0.02, 0.05
LIFT_W, LIFTOFF_H, LIFT_T = 3.0, 0.03, 0.10


def slip(palm_v, cup_v, cup_w, palm_p, cup_p):
    return torch.norm(palm_v - cup_v - torch.cross(cup_w, palm_p - cup_p, dim=-1), dim=-1)


def term(v):
    return -W * np.tanh(np.clip(v - FREE, 0, None) / SCALE)


# (1) 강체: 컵이 v, w 로 움직이고 손바닥이 같은 강체 위 점이면 손바닥 속도 = v + w x r
g = torch.Generator().manual_seed(0)
cup_p, r = torch.randn(1000, 3, generator=g), 0.1 * torch.randn(1000, 3, generator=g)
v, w = torch.randn(1000, 3, generator=g), 3 * torch.randn(1000, 3, generator=g)
print("강체 파지 최대 slip 속도:", float(slip(v + torch.cross(w, r, dim=-1), v, w, cup_p + r, cup_p).max()))
vs = np.linspace(0, 0.3, 301)
hs = np.linspace(0, 0.15, 151)
sh = 0.5 * (1 - np.exp(-hs / 0.03)) + 0.5 * np.clip(hs / 0.10, 0, 1)
net = 3.0 * 0.5 * sh - W * sh                      # 들기 최저 계수(src_up=0, 감쌈 0) - 최대 벌점
print("최악(최대 떨림·src_up=0)에서 들기 순수익 높이에 대해 증가:", bool(np.all(np.diff(net) > 0)), f"10 cm {net[100]:+.3f}")
print("단조 비증가(벌점)·상한 −W:", bool(np.all(np.diff(term(vs)) <= 1e-12)), round(float(term(np.array([10.0]))[0]), 3))


def qrot_inv(q, x):
    w_, a, b, c = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([np.stack([1-2*(b*b+c*c), 2*(a*b-c*w_), 2*(a*c+b*w_)], -1),
                  np.stack([2*(a*b+c*w_), 1-2*(a*a+c*c), 2*(b*c-a*w_)], -1),
                  np.stack([2*(a*c-b*w_), 2*(b*c+a*w_), 1-2*(a*a+b*b)], -1)], -2)
    return np.einsum("...ji,...j->...i", R, x)


for name, f in (("i03f ep2600", "t2r3_i03f_ep2600_trace64_adr0_ctl.npz"), ("i24 best", "i24best_preset_trace64_adr0.npz")):
    z = np.load(S + f)
    for side in ("src", "rcv"):
        p = qrot_inv(z[f"{side}_cup_quat"], z[f"{side}_palm_pose"][..., :3] - z[f"{side}_cup_pos"])
        sp = np.linalg.norm(np.diff(p, axis=0), axis=-1) * 60
        h = z[f"{side}_cup_pos"][1:, :, 2] - z[f"{side}_cup_pos"][0, :, 2]
        hp = np.clip(h, 0, None)
        shape = 0.5 * (1 - np.exp(-hp / LIFTOFF_H)) + 0.5 * np.clip(h / LIFT_T, 0, 1)
        t = term(sp) * (z[f"{side}_grasped"][1:] > 0.5) * shape
        print(f"{name:12s} {side}: grip_slip 평균 {t[300:].mean():+.3f}/스텝 (t300~) · 비드 유입 t150~300 {t[150:300].mean():+.3f}")
