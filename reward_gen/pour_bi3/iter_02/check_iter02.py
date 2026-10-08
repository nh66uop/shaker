"""iter_02 palm_lift 성질 검사: (1) 결정론 trace 에서 i01b(손끝) 대 i24(손바닥) 청구 크기,
(2) 손바닥 0 N 일 때도 들기 순수익(lift_src 최저 계수 - palm_lift)이 높이에 대해 단조 증가하는지."""
import sys
import numpy as np

S = "/tmp/claude-1000/-home-user-rl-ws/97645183-1013-4081-a16e-1eee7f0beebb/scratchpad/"
LIFTOFF_H, LIFT_T, W, PMIN, LIFT_W = 0.03, 0.10, 1.5, 3.0, 3.0


def shape(h):
    hp = np.clip(h, 0, None)
    return 0.5 * (1 - np.exp(-hp / LIFTOFF_H)) + 0.5 * np.clip(h / LIFT_T, 0, 1)


for name, f in (("t2r3_i01b ep3500", "t2r3_i01b_ep3500_trace64_adr0.npz"), ("i24 best", "i24best_preset_trace64_adr0.npz")):
    z = np.load(S + f)
    for side in ("src", "rcv"):
        h = z[f"{side}_cup_pos"][..., 2] - z[f"{side}_cup_pos"][0, :, 2]
        pen = -W * z[f"{side}_grasped"] * shape(h) * np.clip((PMIN - z[f"{side}_f_palm"]) / PMIN, 0, 1)
        print(f"{name:18s} {side}: palm_lift 평균 {pen[300:].mean():+.3f} (t300~, 스텝당)")

hs = np.linspace(0, 0.15, 151)
net = LIFT_W * 0.5 * shape(hs) - W * shape(hs)          # 최악: src_up=0, 감쌈 보너스 0, 손바닥 0 N
print("손바닥 0 N·최악 계수 순수익 단조 비감소:", bool(np.all(np.diff(net) >= -1e-12)), f"(10 cm {net[100]:+.3f})")
net_typ = LIFT_W * 1.0 * shape(hs) - W * shape(hs)
print("손바닥 0 N·src_up=1 순수익 10 cm:", f"{net_typ[100]:+.3f}", "| 손바닥 3 N 이상 10 cm:", f"{LIFT_W * shape(hs)[100]:+.3f}")
