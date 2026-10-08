"""rh_aglt 좌↔우 rl_games 체크포인트 거울 변환(09.30).

정책 π_src 를 반대 팔 env 에서 쓰려면 a_dst = S_a · π_src(D_o · o_dst). D_o·S_a 는 ±1 대각이라
정규화 평균(μ' = D μ, 분산 그대로)·LSTM 입력 가중치 열(W_ih · D)·mu 행(S_a)·critic 첫 층 열에 접어 넣으면 정확하다.
Adam 1차 모멘트도 가중치와 같은 부호로 바꾼다(2차는 그대로).

거울 = 세계 XZ 평면 반사 M = diag(1,−1,1). 관측 칸(layout.py actor_layout/critic_extra_layout)별 부호:
  팔 관절(각·속도·지령·직전 액션)  ARM_MIRROR_SIGN (−1,−1,−1,1,−1,−1,−1)   ← rh_aglt_env_cfg 와 FK 로 검증
  손 관절·촉각·접촉·스칼라         +1
  위치·선속도·방향 벡터(월드)        M = (1,−1,1)
  손바닥 회전 x 열                  −M = (−1,1,−1)  (고유 회전이라 반사 후 x 축이 뒤집힌다, FK 확인)
  손바닥 회전 y 열                  M
  컵 각속도(유사 벡터)               −M = (−1,1,−1)

사용: python3 scripts/tools/mirror_rh_aglt_ckpt.py SRC.pth DST.pth [--selftest]
"""

from __future__ import annotations

import argparse
import sys

import torch

ARM_MIRROR_SIGN = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
M = (1.0, -1.0, 1.0)
M_NEG = (-1.0, 1.0, -1.0)
A, H, F = 7, 6, 5


def actor_sign() -> list[float]:
    s: list[float] = []
    s += list(ARM_MIRROR_SIGN) * 3                  # arm_q, arm_qd, arm_q_target
    s += [1.0] * H * 2                              # hand_q, hand_err
    s += list(M)                                    # palm_pos
    s += list(M_NEG) + list(M)                      # palm_rot = [R[:,:,0], R[:,:,1]]
    s += list(M)                                    # cup_rel_palm
    s += list(M)                                    # cup_up
    s += list(M) * F                                # tips_rel_cup
    s += list(M) * 4                                # kp_rel_goal (4 keypoints)
    s += list(M)                                    # goal_rel_palm
    s += [1.0] * F                                  # tactile
    s += list(ARM_MIRROR_SIGN) + [1.0] * H          # prev_action
    assert len(s) == 96, len(s)
    return s


def critic_sign() -> list[float]:
    s = actor_sign()
    s += [1.0] * H                                  # hand_qd
    s += list(M) + list(M_NEG) + list(M)            # cup_lin_vel, cup_ang_vel(유사 벡터), palm_lin_vel
    s += [1.0] * (F + F + 1)                        # contact_prox, contact_tip, contact_palm
    s += [1.0] * 6                                  # lifted, grasped, progress, goal_successes, kp_dist, near_goal_steps
    assert len(s) == 128, len(s)
    return s


def action_sign() -> list[float]:
    return list(ARM_MIRROR_SIGN) + [1.0] * H


def _sign_rules(D96: torch.Tensor, D128: torch.Tensor, Sa: torch.Tensor) -> dict:
    """파라미터/버퍼 이름 → 적용 함수. 없는 이름은 그대로."""
    return {
        "running_mean_std.running_mean": lambda t: t * D96,
        "a2c_network.rnn.rnn.weight_ih_l0": lambda t: t * D96.unsqueeze(0),
        "a2c_network.mu.weight": lambda t: t * Sa.unsqueeze(1),
        "a2c_network.mu.bias": lambda t: t * Sa,
    }


def _cv_rules(D128: torch.Tensor) -> dict:
    return {
        "model.running_mean_std.running_mean": lambda t: t * D128,
        "model.a2c_network.actor_mlp.0.weight": lambda t: t * D128.unsqueeze(0),
    }


def mirror(ck: dict) -> dict:
    D96 = torch.tensor(actor_sign())
    D128 = torch.tensor(critic_sign())
    Sa = torch.tensor(action_sign())
    out = dict(ck)
    model = dict(ck["model"])
    rules = _sign_rules(D96, D128, Sa)
    for k, fn in rules.items():
        if k not in model:
            raise KeyError(f"체크포인트에 {k} 가 없다 — 신경망 구조가 예상과 다르다")
        model[k] = fn(model[k].to(torch.float32)).to(model[k].dtype)
    out["model"] = model
    cv = dict(ck["assymetric_vf_nets"])
    for k, fn in _cv_rules(D128).items():
        if k not in cv:
            raise KeyError(f"central value 에 {k} 가 없다")
        cv[k] = fn(cv[k].to(torch.float32)).to(cv[k].dtype)
    out["assymetric_vf_nets"] = cv
    # 옵티마이저: 파라미터 순서 = model 의 a2c_network 파라미터(버퍼 제외) 순서. 1차 모멘트만 부호 변환.
    opt = ck.get("optimizer")
    if opt is not None:
        names = [k for k in ck["model"] if k.startswith("a2c_network.")]
        st = opt["state"]
        if len(st) and len(names) == len(opt["param_groups"][0]["params"]):
            new_state = {}
            for i, name in enumerate(names):
                s = dict(st.get(i, {}))
                if name in rules and "exp_avg" in s:
                    s["exp_avg"] = rules[name](s["exp_avg"].to(torch.float32)).to(s["exp_avg"].dtype)
                new_state[i] = s
            out["optimizer"] = {**opt, "state": new_state}
        else:
            out.pop("optimizer")                    # 대응을 확신할 수 없으면 버린다(새 옵티마이저)
            print("[mirror] optimizer 상태 대응 불확실 → 제거", file=sys.stderr)
    return out


def _actor_forward(m: dict, obs: torch.Tensor) -> torch.Tensor:
    """rl_games before_mlp LSTM actor 의 mu (한 스텝, 은닉 0) — 자가검사 전용 재현."""
    mean, var = m["running_mean_std.running_mean"], m["running_mean_std.running_var"]
    x = ((obs - mean) / torch.sqrt(var + 1e-5)).clamp(-5.0, 5.0)
    lstm = torch.nn.LSTM(96, 1024)
    lstm.load_state_dict({k.split("rnn.rnn.")[1]: v.float() for k, v in m.items() if "rnn.rnn." in k})
    y, _ = lstm(x.unsqueeze(0))
    y = torch.nn.functional.layer_norm(y[0], (1024,), m["a2c_network.layer_norm.weight"], m["a2c_network.layer_norm.bias"])
    for i in (0, 2, 4, 6):
        y = torch.nn.functional.elu(y @ m[f"a2c_network.actor_mlp.{i}.weight"].T + m[f"a2c_network.actor_mlp.{i}.bias"])
    return y @ m["a2c_network.mu.weight"].T + m["a2c_network.mu.bias"]


def selftest(src: dict, dst: dict) -> float:
    torch.manual_seed(0)
    D = torch.tensor(actor_sign())
    Sa = torch.tensor(action_sign())
    ms = {k: v.float() for k, v in src["model"].items() if torch.is_tensor(v)}
    md = {k: v.float() for k, v in dst["model"].items() if torch.is_tensor(v)}
    std = torch.sqrt(ms["running_mean_std.running_var"])
    o = ms["running_mean_std.running_mean"] * D + torch.randn(64, 96) * std   # 대상 쪽 관측(분포 안)
    with torch.no_grad():
        a_dst = _actor_forward(md, o)
        a_ref = _actor_forward(ms, o * D) * Sa
    return float((a_dst - a_ref).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    out = mirror(ck)
    if a.selftest:
        err = selftest(ck, out)
        print(f"[mirror] selftest max|π_dst(o) − S·π_src(D·o)| = {err:.3e}")
        if err > 1e-4:
            print("[mirror] ✗ 자가검사 실패", file=sys.stderr)
            return 1
    torch.save(out, a.dst)
    print(f"[mirror] {a.src} → {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
