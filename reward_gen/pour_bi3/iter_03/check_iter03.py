"""iter_03 palm_lift(상점형) 성질 검사: (1) 손바닥 0 N 이면 항이 0 이라 들기 순수익이 iter_01 과 같다,
(2) 손바닥 힘에 대해 단조 증가·3 N 에서 상한, (3) 높이에 대해 단조 비감소, (4) 결정론 trace 에서 i01b 대 i24 지급액."""
import numpy as np

S = "/tmp/claude-1000/-home-user-rl-ws/97645183-1013-4081-a16e-1eee7f0beebb/scratchpad/"
LIFTOFF_H, LIFT_T, W, PMIN = 0.03, 0.10, 1.5, 3.0


def shape(h):
    hp = np.clip(h, 0, None)
    return 0.5 * (1 - np.exp(-hp / LIFTOFF_H)) + 0.5 * np.clip(h / LIFT_T, 0, 1)


def term(h, f, held=1.0):
    return W * held * shape(h) * np.clip(f / PMIN, 0, 1)


hs = np.linspace(0, 0.15, 151)
fs = np.linspace(0, 6, 61)
print("손바닥 0 N 에서 항 = 0 (iter_01 과 같음):", bool(np.all(term(hs, 0.0) == 0)))
print("높이에 대해 단조 비감소(3 N):", bool(np.all(np.diff(term(hs, 3.0)) >= -1e-12)))
tf = term(0.10, fs)
print("손바닥 힘에 대해 단조 증가(0~3 N)·3 N 이상 상한:", bool(np.all(np.diff(tf[fs <= 3]) > 0)), bool(np.allclose(tf[fs >= 3], tf[fs >= 3][0])))
print("테이블 위(h=0) 지급:", float(term(0.0, 3.0)))
for name, f in (("t2r3_i01b ep3500", "t2r3_i01b_ep3500_trace64_adr0.npz"), ("i24 best", "i24best_preset_trace64_adr0.npz")):
    z = np.load(S + f)
    for side in ("src", "rcv"):
        h = z[f"{side}_cup_pos"][..., 2] - z[f"{side}_cup_pos"][0, :, 2]
        v = term(h, z[f"{side}_f_palm"], z[f"{side}_grasped"])
        print(f"{name:18s} {side}: palm_lift 평균 {v[300:].mean():+.3f} (t300~, 스텝당)")
