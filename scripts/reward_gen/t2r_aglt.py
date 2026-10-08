#!/usr/bin/env python3
"""rh_aglt 트랙 t2r 도구 — ingest(응답 → compute_reward.py + 검증) · ledger(결정론 단계 원장).

기존 t2r_fj/t2r_ms/t2r_rh 는 각자 과제의 RewardContext 를 전제로 해서 이 트랙(tasks/rh_aglt_{r,l}/reward_context.py)을
검증하지 못한다. 이 스크립트는 Isaac 없이 시스템 python 으로 돈다(보상은 torch·math 만 쓴다).

    python3 scripts/reward_gen/t2r_aglt.py ingest --iter reward_gen/rh_aglt_r/iter_00 --response response_c1.md
    python3 scripts/reward_gen/t2r_aglt.py ledger --iter reward_gen/rh_aglt_r/iter_00 [--reward compute_reward_c1.py]

ledger 행(오른팔 장면, 컵 (0.35, −0.16), 좌우 대칭이라 우만 잰다): 한 단계 앞 행이 늘 스텝당 더 받아야 하고(단조),
주차 행(쥔 주먹으로 컵 옆·컵 쓰러뜨리기·테이블 위에서 쥔 채 머물기)은 다음 단계보다 싸야 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

_HDGP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP / "source" / "openarm"))
from openarm.agnostic.tasks.rh_aglt_r.reward_context import (  # noqa: E402
    RewardContext, call_reward_fn, load_reward_fn)

# ---- 장면 상수(이 트랙 cfg 기본값과 같다 — tests 가 cfg 를 잰다) ----
TABLE_Z = 0.205
CUP_R = 0.0440 * 0.65
CUP_MOUTH = 0.0829 * 0.65
CUP_BOTTOM = -0.0921 * 0.65
CUP_C = torch.tensor([0.35, -0.16, TABLE_Z - CUP_BOTTOM], dtype=torch.float64)
START_PALM = torch.tensor([0.08, -0.30, 0.45], dtype=torch.float64)   # 09.29 시작 자세(테이블 앞, cfg arm_start_q FK)
GOAL = CUP_C + torch.tensor([0.03, 0.02, 0.14], dtype=torch.float64)
LIFT_LATCH = 0.10
GOAL_TOL = 0.10
SLOPPY_RATIO = 2.0     # 서툰 첫 파지 ≥ 2 × 컵 옆 머묾 — 실측: 1.38(pour R5)·1.49(iter_02) 머묾, 1.86(pour)·2.3(iter_01)·3.3(iter_00) 쥠
TIP_AZ_DEG = (58.2, -27.7, -32.5, -45.4, -73.1)   # 엄지·검지·중지·약지·새끼 끝 방위(손바닥→축 기준, 09.30 good_mirror 실측)
TIP_R = (1.37, 1.38, 1.31, 1.08, 0.83)            # 끝 반경 / r (새끼는 컵 아래 좁은 곳)
#: 10.01 사용자 "붓기 이후 작업을 생각하면 입구보다 더 낮게 잡아야" — 손바닥 0.15 × 입구(붓기 트랙 h_grip 과 같음).
#:   끝 높이는 우 i10 ep4600 결정론 실측(손바닥 0.41 · 엄지 0.64 · 검지 0.55 · 중지 0.14 · 약지 −0.30 · 새끼 −0.59)을
#:   손바닥 0.15 로 0.26 내린 값. 이전 값(0.74, 0.53, 0.11, −0.26, −0.58)은 손바닥 0.15 와 손 모양이 안 맞았다.
TIP_H = (0.38, 0.29, -0.12, -0.56, -0.85)         # 끝 높이 / 입구(낮은 파지)
HIGH_DH = 0.26                                    # 우 i10 실제 파지 = 낮은 파지 + 0.26 × 입구(손 전체)
HIGH_MARGIN = 0.85                                # 높은 파지(i10 실측)는 같은 상태 낮은 파지의 85 % 미만
THUMB_STEER_H = 0.40                              # 쥐기 전 엄지 유도 목표 높이 / 입구(이전 0.62)
#: 10.01 다섯 손가락(사용자 "iter11 로 5지 모두 참여") — 우 i10 결정론: 약지 접촉 0.00, 약지 명령 mu −1.37(완전 펴짐, 클립 밖)
RING_OFF_MARGIN = 0.80                            # 약지만 빠진 파지는 다섯 손가락의 80 % 미만(iter_10 보상 0.745 — 이미 통과)
#: ★iter_10 보상은 실제 i10 궤적(든 스텝)에서 약지 1.2 N 을 더하면 +29 % 인데도 정책이 약지를 완전히 편다 — 접촉 전에는
#:   약지를 벽 쪽으로 닫아도 지배 수입(목표 유지)이 안 오른다(finger_wall 0.10 뿐). 쥔 채 약지 끝을 1.8 → 1.25 r 로 옮기면
#:   RING_NEAR_GAIN 배 이상 올라야 한다(접촉 전 기울기를 지배 수입에).
RING_NEAR_GAIN = 1.03
FAR_PALM_R, FAR_PALM_H = 2.08, 0.59                # 실측 손끝 파지의 손바닥 자리(좌 i05) — 컵이 손바닥에서 멀다
CAGE_MARGIN = 0.75                                 # 손바닥 먼 손끝 파지는 손바닥 가까운 감싸기의 75 % 미만
THUMB_OUT_R = 1.18        # 바깥 벽 엄지 끝 반경 / r (09.30 probe, 좌 i05)
THUMB_IN_R, THUMB_IN_H = 0.95, 0.89   # 입구 안 엄지 끝 반경 / r, 높이 / 입구 (09.30 probe, 우 i05 ep3800, in_cup 0.97)
THUMB_IN_MARGIN = 0.60    # 입구 안 엄지 행은 인벨롭의 60 % 미만(09.30 우 i05: 72 % 에서 그 파지를 택함)
ENVELOPE_MARGIN = 0.90  # 인벨롭 판별 행은 기준 행의 90 % 미만
#: 10.01 관절 한계(우 i08c 붕괴: 이송 습관이 시작으로 번져 손목 j6·어깨 j2 한계에 흡수). margin 1 = 범위 중앙, 0 = 한계.
#:   실측 먼 출발 여유(URDF, 한계 반 범위 대비) — j1 0.075 가 가장 작다. 좋은 파지(good_mirror) 이송 중앙값 — 손목 j6 0.00.
START_MARGIN = [0.075, 0.216, 0.584, 0.559, 0.759, 0.263, 0.577]
GOOD_CARRY_MARGIN = [0.30, 0.58, 0.74, 0.36, 0.56, 0.00, 0.74]
LIMIT_APPROACH_MARGIN = 0.75   # 파지 전 관절 하나가 한계면 같은 자리 여유 있는 상태의 75 % 미만
LIMIT_HELD_BAND = (0.70, 0.97)
#: 10.01 네 손가락(사용자 "약지·새끼 팁이라도 닿았으면") — 우 i09 결정론: 약지 0.02·새끼 0.00 접촉, 새끼 명령 닫힘 0.00
FINGER3_MARGIN = 0.70           # 약지·새끼가 빠진 세 손가락 파지는 네 손가락의 70 % 미만(iter_09 0.79× 에서 정책이 세 손가락을 택함)
FINGER_TIP_GAIN = 1.10          # 약지·새끼 끝만 가볍게(0.5 N) 닿아도 빠진 것보다 10 % 이상
JITTER_TOL = 0.01      # 떨림 행이 출발 행과 다를 수 있는 폭(스텝당) — 접근 수입의 1/3 보다 작게
# 컵·테이블 치수만 본다 — 소환 좌표(0.35 등)는 흔한 가중치 값과 겹쳐 거짓 양성이 났다(iter_00 c1 의 접근 가중치 0.35).
FORBIDDEN_LITERALS = ("0.0286", "0.028", "0.057", "0.0539", "0.0599", "0.134", "0.265", "0.205")


# =============================================================================
# ingest
# =============================================================================
def extract_python(md: str) -> str:
    blocks = re.findall(r"```python\n(.*?)```", md, re.S)
    if not blocks:
        raise SystemExit("응답에 ```python 블록이 없다")
    return blocks[-1]


def _row_ctx(n: int = 64, seed: int = 0) -> RewardContext:
    """무작위지만 물리적으로 말이 되는 배치(형상·유한성 검증용)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    cup = CUP_C.float().repeat(n, 1) + (r(n, 3) - 0.5) * torch.tensor([0.16, 0.14, 0.0])
    cup[:, 2] += r(n) * 0.2
    palm = cup + (r(n, 3) - 0.5) * 0.6
    up = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    f = r(n, 5) * 3.0
    return RewardContext(
        table_z=TABLE_Z, cup_radius=CUP_R, cup_mouth_z=CUP_MOUTH, cup_bottom_z=CUP_BOTTOM, lift_latch_m=LIFT_LATCH,
        goal_tol=GOAL_TOL, goal_max=5,
        palm_pos=palm, palm_lin_vel=(r(n, 3) - 0.5), palm_axes=torch.cat([_unit(r(n, 3) - 0.5), _unit(r(n, 3) - 0.5)], 1),
        tips_pos=palm.unsqueeze(1) + (r(n, 5, 3) - 0.5) * 0.2, hand_closure=r(n), hand_q_target_closure=r(n, 6),
        finger_force=f, finger_prox_force=f * 0.4, finger_tip_force=f * 0.6, palm_force=r(n), hand_foreign_force=r(n),
        grasped=r(n) > 0.5, hand_min_z=palm[:, 2] - 0.07, arm_q=r(n, 7), arm_qd=r(n, 7) - 0.5, arm_limit_margin=r(n, 7),
        cup_pos=cup, cup_quat=q, cup_up=up, cup_tilt=r(n) * 0.5, cup_lin_vel=r(n, 3) - 0.5, cup_ang_vel=r(n, 3) - 0.5,
        cup_spawn_pos=CUP_C.float().repeat(n, 1), lifted=r(n) > 0.5, goal_pos=GOAL.float().repeat(n, 1),
        goal_kp_dist=r(n) * 0.3, near_goal=r(n) > 0.8, goal_success=r(n) > 0.95, goal_successes=(r(n) * 5).floor(),
        episode_progress=r(n), actions=r(n, 13) * 2 - 1, prev_actions=r(n, 13) * 2 - 1)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def validate(path: Path) -> dict:
    src = path.read_text()
    errs, warns = [], []
    for m in re.finditer(r"^\s*(?:import|from)\s+(\w+)", src, re.M):
        if m.group(1) not in ("torch", "math", "__future__", "typing"):
            errs.append(f"허용되지 않은 import: {m.group(1)}")
    if re.search(r"\bglobal\b", src):
        errs.append("global 사용(보상은 순수 함수여야 한다)")
    body = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    lits = [lit for lit in FORBIDDEN_LITERALS if re.search(rf"(?<![\d.]){re.escape(lit)}(?!\d)", body)]
    if lits:
        errs.append(f"장면·컵 치수 리터럴 하드코딩 의심: {lits} — ctx 값을 쓸 것")
    fn, _ = load_reward_fn(str(path))
    out = {}
    for seed in (0, 1, 2):
        ctx = _row_ctx(seed=seed)
        try:
            total, terms = call_reward_fn(fn, ctx)
        except Exception as e:  # noqa: BLE001 — 생성 코드 오류를 그대로 보고
            errs.append(f"호출 실패(seed {seed}): {type(e).__name__}: {e}")
            break
        if not torch.isfinite(total).all():
            errs.append(f"비유한 보상(seed {seed})")
        for k, v in terms.items():
            if not torch.isfinite(v).all():
                errs.append(f"비유한 항 {k}(seed {seed})")
        out = {"n_terms": len(terms), "terms": sorted(terms), "total_mean": float(total.mean())}
    if out and out["n_terms"] > 20:
        warns.append(f"항이 {out['n_terms']}개 — 15개 내외 원칙")
    return {"status": "PASS" if not errs else "FAIL", "errors": errs, "warnings": warns, **out}


# =============================================================================
# ledger
# =============================================================================
def _ctx_rows(rows: list[dict]) -> RewardContext:
    """행 dict 목록 → 배치 ctx(float32). 행에 없는 값은 기본(정지·편 손·접촉 없음)."""
    n = len(rows)

    def col(key, default):
        return torch.stack([torch.as_tensor(r.get(key, default), dtype=torch.float32) for r in rows])

    up = col("cup_up", [0.0, 0.0, 1.0])
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return RewardContext(
        table_z=TABLE_Z, cup_radius=CUP_R, cup_mouth_z=CUP_MOUTH, cup_bottom_z=CUP_BOTTOM, lift_latch_m=LIFT_LATCH,
        goal_tol=GOAL_TOL, goal_max=5,
        palm_pos=col("palm", START_PALM.tolist()), palm_lin_vel=col("palm_v", [0.0] * 3),
        palm_axes=col("axes", [0.0, 1.0, 0.0, 0.985, 0.0, -0.174]), tips_pos=col("tips", [[0.0] * 3] * 5),
        hand_closure=col("closure", 0.0), hand_q_target_closure=col("closure6", [0.0] * 6),
        finger_force=col("ff", [0.0] * 5), finger_prox_force=col("fp", [0.0] * 5), finger_tip_force=col("ft", [0.0] * 5),
        palm_force=col("palm_f", 0.0), hand_foreign_force=col("foreign", 0.0), grasped=col("grasped", 0.0) > 0.5,
        hand_min_z=col("hand_min_z", 0.40), arm_q=col("arm_q", [0.0] * 7), arm_qd=col("arm_qd", [0.0] * 7),
        arm_limit_margin=col("margin", [0.6] * 7), cup_pos=col("cup", CUP_C.tolist()), cup_quat=q, cup_up=up,
        cup_tilt=torch.acos(up[:, 2].clamp(-1, 1)), cup_lin_vel=col("cup_v", [0.0] * 3), cup_ang_vel=col("cup_w", [0.0] * 3),
        cup_spawn_pos=CUP_C.float().repeat(n, 1), lifted=col("lifted", 0.0) > 0.5, goal_pos=GOAL.float().repeat(n, 1),
        goal_kp_dist=col("kp", float((CUP_C - GOAL).norm())), near_goal=col("near", 0.0) > 0.5,
        goal_success=col("succ", 0.0) > 0.5, goal_successes=col("n_succ", 0.0),
        episode_progress=col("prog", 0.3), actions=torch.zeros(n, 13), prev_actions=torch.zeros(n, 13))


def _hand(palm: torch.Tensor, closed: float, cup: torch.Tensor | None = None) -> dict:
    """손끝 5개: 편 손은 손바닥 앞으로 뻗고, 쥔 손은 컵 둘레(반지름 CUP_R)에 둔다(엄지는 반대편)."""
    fdir = torch.tensor([0.985, 0.0, -0.174], dtype=torch.float64)   # 시작 자세: 손가락 10° 아래
    if cup is None or closed < 0.5:
        tips = [palm + fdir * 0.10 + torch.tensor([0, 0.03 if i == 0 else 0.0, -0.012 * i], dtype=torch.float64)
                for i in range(5)]
        if closed >= 0.5:                                   # 주먹: 손끝이 손바닥 쪽으로 말린다
            tips = [palm + fdir * 0.04 + torch.tensor([0, 0.02, -0.01 * i], dtype=torch.float64) for i in range(5)]
    else:
        # 09.30 실측 배치(궤적 원장 good_mirror = 좌 i05 거울, 들고 있는 스텝 중앙값): 손바닥→축 방향 u 기준 방위각·반경·높이.
        #   네 손가락 끝이 축 너머(방위 −28~−73°)로 감긴다. 이전 배치(엄지 +y, 4지 −40~−100° 세계각)는 4지가 손바닥 쪽
        #   반쪽에 있어 실제 파지와 달랐다.
        base = math.atan2(0.67, 0.73)                       # u = 0.67 n + 0.73 f, n=(0,1,0), f=(1,0,0)
        tips = [cup + torch.tensor([rr * CUP_R * math.cos(base + math.radians(az)), rr * CUP_R * math.sin(base + math.radians(az)),
                                    hh * CUP_MOUTH], dtype=torch.float64)
                for az, rr, hh in zip(TIP_AZ_DEG, TIP_R, TIP_H)]
    return {"tips": torch.stack(tips).tolist(), "closure": closed, "closure6": [closed] * 6}


def ledger_rows() -> tuple[list, list, list, list, list]:
    n = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)              # 손바닥 법선(컵 쪽)
    f = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)              # 수평 손가락 축(R2 인벨롭)
    u = (0.67 * n + 0.73 * f) / (0.67 * n + 0.73 * f).norm()
    grasp_palm = CUP_C + torch.tensor([0, 0, 0.15 * CUP_MOUTH], dtype=torch.float64) - 1.5 * CUP_R * u
    pre_palm = grasp_palm - 0.04 * u
    mid_palm = 0.5 * (START_PALM + pre_palm)
    ax_h = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    held = {"ff": [2.0, 2.0, 2.0, 1.2, 1.2], "fp": [0.8] * 5, "ft": [1.2, 1.2, 1.2, 0.4, 0.4], "grasped": 1.0,
            "palm_f": 0.0, "axes": ax_h}

    def up(dz: float, lifted: bool, dxy: tuple = (0.0, 0.0), **kw) -> dict:
        off = torch.tensor([dxy[0], dxy[1], dz], dtype=torch.float64)
        cup = CUP_C + off
        d = {**held, **_hand(grasp_palm + off, 0.7, cup), "palm": (grasp_palm + off).tolist(), "cup": cup.tolist(),
             "lifted": float(lifted), "kp": float((cup - GOAL).norm()), "hand_min_z": 0.215 + dz}
        d.update(kw)
        return d

    rows = [
        ("0_start", {"palm": START_PALM.tolist(), **_hand(START_PALM, 0.0)}),
        ("1_halfway_open", {"palm": mid_palm.tolist(), **_hand(mid_palm, 0.0), "axes": ax_h}),
        ("2_pregrasp_open", {"palm": pre_palm.tolist(), **_hand(pre_palm, 0.0), "axes": ax_h, "hand_min_z": 0.22}),
        ("3_grasped_on_table", {**up(0.0, False)}),
        ("4_lift_5cm", {**up(0.05, False)}),
        ("5_lift_goal_height", {**up(0.14, True)}),
        ("6_at_goal_held", {**up(0.14, True, (0.03, 0.02), near=1.0)}),
    ]
    park = [
        ("2p_fist_at_pregrasp", {"palm": pre_palm.tolist(), **_hand(pre_palm, 1.0), "axes": ax_h, "hand_min_z": 0.22}, "2_pregrasp_open"),
        ("2t_open_tipping_cup", {"palm": grasp_palm.tolist(), **_hand(grasp_palm, 0.0), "axes": ax_h, "ft": [0, 0.4, 0, 0, 0],
                                 "ff": [0, 0.4, 0, 0, 0], "cup_up": [0.0, math.sin(0.35), math.cos(0.35)],
                                 "cup_w": [1.5, 0, 0]}, "2_pregrasp_open"),
        ("3v_vertical_pinch_on_table", {**up(0.0, False), "axes": [0.0, 0.0, -1.0, 0.0, -1.0, 0.0],
                                        "ff": [2.0, 2.0, 0, 0, 0], "fp": [0] * 5, "ft": [2.0, 2.0, 0, 0, 0]}, "3_grasped_on_table"),
        ("5d_dropped_after_lift", {"palm": (grasp_palm + torch.tensor([0, 0, 0.14], dtype=torch.float64)).tolist(),
                                   **_hand(grasp_palm, 0.7), "lifted": 1.0, "cup_v": [0, 0, -0.8]}, "3_grasped_on_table"),
    ]
    # ★인벨롭 판별 행(09.29 사용자 영상 판정 "들기는 했지만 인벨롭이 아님") — 테이블 위 수평 인벨롭 파지(3)보다 낮아야 한다
    g3 = dict(rows[3][1])
    rim_tips = [list(t) for t in g3["tips"]]
    rim_tips[0] = (CUP_C + torch.tensor([0.0, -0.3 * CUP_R, CUP_MOUTH + 0.005], dtype=torch.float64)).tolist()  # 엄지가 입구 위
    park += [
        ("3r_thumb_on_rim", {**g3, "tips": rim_tips}, "3_grasped_on_table"),
        ("3f_fingertips_only", {**g3, "fp": [0.0] * 5, "ft": [2.0, 2.0, 2.0, 1.2, 1.2]}, "3_grasped_on_table"),
        ("3x_held_cup_tilted20", {**g3, "cup_up": [0.0, math.sin(0.35), math.cos(0.35)]}, "3_grasped_on_table"),
    ]
    # ★접근 방향 행(09.29 사용자 "접근 중 방향 유지 필요", 방향 규칙: 손바닥 +z 수평·컵(중심선) 쪽, 엄지가 새끼보다 위)
    h1, h2 = dict(rows[1][1]), dict(rows[2][1])
    flip_tips = [list(t) for t in h1["tips"]]
    flip_tips[0][2], flip_tips[4][2] = flip_tips[4][2] - 0.02, flip_tips[0][2]          # 엄지 끝이 새끼 끝 아래
    g3f = [list(t) for t in g3["tips"]]
    g3f[0][2], g3f[4][2] = g3f[4][2] - 0.02, g3f[0][2]
    park += [
        ("1u_halfway_palm_up", {**h1, "axes": [0.0, 0.6, 0.8, 1.0, 0.0, 0.0]}, "1_halfway_open"),
        ("1f_halfway_flipped", {**h1, "tips": flip_tips}, "1_halfway_open"),
        ("2a_pregrasp_palm_away", {**h2, "axes": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0]}, "2_pregrasp_open"),
        ("3u_grasp_thumb_below", {**g3, "tips": g3f}, "3_grasped_on_table"),
    ]
    # ★사용자 영상 판정(09.29 i02): 오른손 — 손바닥은 컵 가까이인데 엄지가 컵을 지나쳐 인벨롭이 안 됨,
    #   왼손 — 엄지·검지 사이에 컵이 있으나 손바닥 면이 기울어 있음. 둘 다 인벨롭(3)의 90 % 미만이어야 한다.
    ov_tips = [list(t) for t in g3["tips"]]
    ov_tips[0] = (CUP_C + torch.tensor([2.2 * CUP_R * math.cos(1.2), 2.2 * CUP_R * math.sin(1.2), 0.01],
                                        dtype=torch.float64)).tolist()                     # 엄지 끝이 벽 너머로 지나침
    tl = math.radians(25.0)                                                                   # 손바닥 면 25° 기울어짐
    park += [
        ("3o_thumb_overshoots_wall", {**g3, "tips": ov_tips, "fp": [1.2, 0.8, 0.8, 0.8, 0.8], "ft": [0.0, 1.2, 1.2, 0.4, 0.4]},
         "3_grasped_on_table"),
        ("3p_grasp_palm_tilted25", {**g3, "axes": [0.0, math.cos(tl), math.sin(tl), math.cos(tl), 0.0, -math.sin(tl)]},
         "3_grasped_on_table"),
    ]
    # ★서툰 첫 파지(09.29 iter_02: 컵 옆 83 % 도달·파지 0 이 750 epoch — 첫 파지가 컵 옆에 머무는 것보다 싸다 [가설])
    #   엄지와 검지 끝만 1.2 N(첫마디 0, 나머지 0), 엄지가 손가락과 같은 쪽(대향 0), 손 반쯤 닫힘. 파지 판정은 선다.
    sl_tips = [list(t) for t in g3["tips"]]
    # 09.30: 서툰 첫 파지의 엄지도 실측 바깥 벽 위치(1.2 r, 입구 0.6)에 둔다 — 대향만 없는 상태(손가락과 같은 쪽 −0.6 rad)
    sl_tips[0] = (CUP_C + torch.tensor([1.2 * CUP_R * math.cos(-0.6), 1.2 * CUP_R * math.sin(-0.6), 0.6 * CUP_MOUTH],
                                       dtype=torch.float64)).tolist()
    sloppy = {**g3, "tips": sl_tips, "ff": [1.2, 1.2, 0, 0, 0], "fp": [0.0] * 5, "ft": [1.2, 1.2, 0, 0, 0],
              "closure": 0.5, "closure6": [0.5] * 6}
    hover = {**h2, "closure": 0.6, "closure6": [0.6] * 6}
    sloppy_rows = [("3s_sloppy_first_grasp", sloppy, "2h_hover_half_closed"), ("2h_hover_half_closed", hover, None)]
    # ★09.30 우 i07/i07b: 실제 첫 파지는 엄지 끝이 입구 근처(0.9)에 있다 — iter_07 에서 머묾의 1.26× 뿐이라 1200 epoch 머묾.
    #   입구 근처 엄지 첫 파지도 머묾의 2× 이상. 또 엄지를 목표 상자에 둔 채 떠 있는 머묾(엄지 유도 수입 최대)보다도 2× 이상.
    nr_tips = [list(t) for t in sl_tips]
    nr_tips[0] = (CUP_C + torch.tensor([1.25 * CUP_R * math.cos(-0.6), 1.25 * CUP_R * math.sin(-0.6), 0.9 * CUP_MOUTH],
                                       dtype=torch.float64)).tolist()
    hb_tips = [list(t) for t in hover["tips"]]
    hb_tips[0] = (CUP_C + torch.tensor([1.3 * CUP_R * math.cos(math.pi * 0.5), 1.3 * CUP_R * math.sin(math.pi * 0.5),
                                        0.62 * CUP_MOUTH], dtype=torch.float64)).tolist()
    sloppy_rows += [("3n_first_grasp_thumb_near_rim", {**sloppy, "tips": nr_tips}, "2h_hover_half_closed"),
                    ("3s_sloppy_vs_box_hover", sloppy, "2k_hover_thumb_in_box"),
                    ("2k_hover_thumb_in_box", {**hover, "tips": hb_tips}, None)]
    # ★기울어진 채 컵 옆에 떠 있기(09.29 aglt_l_i03 ep1700 영상·로그: 접촉 0.005 손가락, 손바닥 법선 |z| 0.40,
    #   엄지가 입구 위를 가로지름, 파지 0 이 800 epoch) — 같은 자리·같은 닫힘의 수평 머묾(2h)보다 10 % 이상 낮아야 한다.
    tl2 = math.asin(0.40)
    om_tips = [list(t) for t in hover["tips"]]
    om_tips[0] = (CUP_C + torch.tensor([0.0, 0.0, CUP_MOUTH + 0.01], dtype=torch.float64)).tolist()   # 엄지 끝이 입구 위
    tilted_hover = {**hover, "tips": om_tips, "axes": [0.0, math.cos(tl2), -math.sin(tl2), 1.0, 0.0, 0.0]}
    park.append(("2l_hover_tilted_thumb_over_mouth", tilted_hover, "2h_hover_half_closed"))
    # ★09.30 사용자 영상 판정 — 우 i03 "입구 안으로 엄지를 넣고 있음", 좌 i04 "손이 약간 기울어 인벨롭, 소지가 닿지 않음"
    #   (로그: 새끼 접촉 0.01, 손가락 방향 z −0.39). 테이블 위(3) 와 목표에서 든 상태(6) 둘 다에서 인벨롭의 90 % 미만이어야 한다
    #   — 들기·이송·목표 유지 수입이 결함과 무관하면 정책은 결함 파지로 옮긴다.
    def _defects(base: dict, cup: torch.Tensor, tag: str, ref: str) -> list:
        ins = [list(t) for t in base["tips"]]
        ins[0] = (cup + torch.tensor([0.5 * CUP_R, 0.0, CUP_MOUTH - 0.01], dtype=torch.float64)).tolist()  # 엄지 끝이 입구 안
        a23 = math.radians(23.0)
        inj = [list(t) for t in base["tips"]]
        inj[0] = (cup + torch.tensor([THUMB_IN_R * CUP_R, 0.0, THUMB_IN_H * CUP_MOUTH], dtype=torch.float64)).tolist()
        return [
            (f"{tag}i_thumb_inside_mouth", {**base, "tips": ins}, ref),
            (f"{tag}j_thumb_just_inside_wall", {**base, "tips": inj}, ref),
            (f"{tag}k_pinky_off", {**base, "ff": base["ff"][:4] + [0.0], "fp": base["fp"][:4] + [0.0],
                                   "ft": base["ft"][:4] + [0.0]}, ref),
            (f"{tag}d_fingers_down23", {**base, "axes": [0.0, 1.0, 0.0, math.cos(a23), 0.0, -math.sin(a23)]}, ref),
        ]
    park += _defects(g3, CUP_C, "3", "3_grasped_on_table")
    # ★09.30 사용자 "손바닥 쪽으로 최대한 들어오게, 원통이라 기하적으로 걸리게": 실측 손끝 파지(손바닥 2.08 r·입구 0.59,
    #   네 손가락 첫마디 0 N)는 같은 손끝 배치로 손바닥이 컵에 붙은 파지보다 확실히 낮아야 한다(CAGE_MARGIN).
    def _far(base: dict, cup: torch.Tensor, tag: str, ref: str) -> tuple:
        uh = torch.tensor([math.cos(math.atan2(0.67, 0.73)), math.sin(math.atan2(0.67, 0.73)), 0.0], dtype=torch.float64)
        palm = cup - FAR_PALM_R * CUP_R * uh
        palm[2] = cup[2] + FAR_PALM_H * CUP_MOUTH
        return (f"{tag}m_far_palm_tip_grasp", {**base, "palm": palm.tolist(), "fp": [5.0, 0.0, 0.0, 0.0, 0.0],
                                                "ft": [0.4, 1.5, 1.4, 1.3, 1.2], "ff": [5.4, 1.5, 1.4, 1.3, 1.2]}, ref)
    park.append(_far(g3, CUP_C, "3", "3_grasped_on_table"))
    # ★09.30 궤적 실측(우 i06 ep2600 '입구 위 덮기'): 쥔 판정 47 %, 들기 0, 엄지 끝 1.48 r · 입구 1.05, 검지·새끼만 접촉.
    #   이 상태가 컵 옆 머묾보다 벌면 주차장이 된다 — 머묾(2h)보다 낮아야 한다(판정은 ledger() 의 hat 규칙).
    hat_tips = [list(t) for t in g3["tips"]]
    hat_tips[0] = (CUP_C + torch.tensor([1.48 * CUP_R, 0.0, 1.05 * CUP_MOUTH], dtype=torch.float64)).tolist()
    hat = {**g3, "tips": hat_tips, "ff": [1.5, 1.5, 0.0, 0.0, 1.5], "fp": [0.0, 0.8, 0.0, 0.0, 0.8],
           "ft": [1.5, 0.7, 0.0, 0.0, 0.7]}
    park.append(("3h_hat_thumb_over_rim", hat, "2h_hover_half_closed"))
    g6 = dict(rows[6][1])
    park += _defects(g6, torch.tensor(g6["cup"], dtype=torch.float64), "6", "6_at_goal_held")
    park.append(_far(g6, torch.tensor(g6["cup"], dtype=torch.float64), "6", "6_at_goal_held"))
    # 서 있는 컵의 솔버 떨림(손 접촉 없음, 반발 0): 각속도 p99 0.094 rad/s(09.29 probe_rh_aglt_cup_rest) — 출발 행과 거의 같아야 한다
    jitter = [("0j_start_cup_jitter", {**dict(rows[0][1]), "cup_w": [0.0, 0.0, 0.094], "cup_v": [0.003, 0.0, 0.0]}, "0_start"),
              ("0k_start_cup_jitter_x", {**dict(rows[0][1]), "cup_w": [0.094, 0.0, 0.0]}, "0_start")]
    # ★10.01 관절 한계 행(우 i08c 붕괴) — (이름, 행, 기준, 판정): lt = 기준 × LIMIT_APPROACH_MARGIN 미만,
    #   band = 기준 × LIMIT_HELD_BAND 안, ge = 기준 × 0.98 이상, gt = 기준보다 큼(한계에서 빠져나오는 기울기)
    def _lim(base: dict, margins: list) -> dict:
        return {**base, "margin": margins}
    one0 = [0.6] * 7
    one0[5] = 0.0
    two0 = [0.6] * 7
    two0[1], two0[5] = 0.0, 0.0
    m002, m005 = list(one0), list(one0)
    m002[5], m005[5] = 0.02, 0.05
    r0, r1, r6 = dict(rows[0][1]), dict(rows[1][1]), dict(rows[6][1])
    limit_rows = [
        ("0L_start_real_margins", _lim(r0, START_MARGIN), "0_start", "ge"),
        ("0M_start_two_joints_at_limit", _lim(r0, two0), "0_start", "lt"),
        ("1L_halfway_joint_at_limit", _lim(r1, one0), "1_halfway_open", "lt"),
        ("1L2_halfway_margin_002", _lim(r1, m002), "1L_halfway_joint_at_limit", "gt"),
        ("1L5_halfway_margin_005", _lim(r1, m005), "1L2_halfway_margin_002", "gt"),
        ("6L_goal_joint_at_limit", _lim(r6, one0), "6_at_goal_held", "band"),
        ("6G_goal_good_carry_margins", _lim(r6, GOOD_CARRY_MARGIN), "6_at_goal_held", "band"),
    ]
    # ★10.01 손가락 행 — (이름, 행, 기준, 판정): lt3 = 기준 × FINGER3_MARGIN 미만, tip = 기준 × FINGER_TIP_GAIN 이상, gt = 기준보다 큼
    def _f(base: dict, ff: list, fp: list, ft: list) -> dict:
        return {**base, "ff": ff, "fp": fp, "ft": ft}
    f3 = _f(r6, [2.0, 2.0, 2.0, 0.0, 0.0], [0.8, 0.8, 0.8, 0.0, 0.0], [1.2, 1.2, 1.2, 0.0, 0.0])
    ft = _f(r6, [2.0, 2.0, 2.0, 0.5, 0.5], [0.8, 0.8, 0.8, 0.0, 0.0], [1.2, 1.2, 1.2, 0.5, 0.5])
    # 쥐기 전(판정 없음·접촉 없음) 파지 자리 손끝 배치에서 새끼 끝만 벽 바깥 1.8 r → 1.45 r(1 cm) — 높이 −0.3 × 입구(창 안)
    pre = dict(rows[3][1])
    pre.update({"grasped": 0.0, "ff": [0.0] * 5, "fp": [0.0] * 5, "ft": [0.0] * 5, "closure": 0.5, "closure6": [0.5] * 6})
    base_az = math.atan2(0.67, 0.73) + math.radians(TIP_AZ_DEG[4])
    def _pinky(rr: float) -> dict:
        tips = [list(t) for t in pre["tips"]]
        tips[4] = (CUP_C + torch.tensor([rr * CUP_R * math.cos(base_az), rr * CUP_R * math.sin(base_az), -0.3 * CUP_MOUTH],
                                        dtype=torch.float64)).tolist()
        return {**pre, "tips": tips}
    # ★10.01 약지(다섯 손가락)·낮은 파지 행
    rq = _f(r6, [2.0, 2.0, 2.0, 0.0, 1.2], [0.8, 0.8, 0.8, 0.0, 0.8], [1.2, 1.2, 1.2, 0.0, 0.4])
    rqt = _f(r6, [2.0, 2.0, 2.0, 0.5, 1.2], [0.8, 0.8, 0.8, 0.0, 0.8], [1.2, 1.2, 1.2, 0.5, 0.4])
    ring_az = math.atan2(0.67, 0.73) + math.radians(TIP_AZ_DEG[3])
    def _tip(i: int, az: float, rr: float, hh: float) -> dict:
        tips = [list(t) for t in pre["tips"]]
        tips[i] = (CUP_C + torch.tensor([rr * CUP_R * math.cos(az), rr * CUP_R * math.sin(az), hh * CUP_MOUTH],
                                        dtype=torch.float64)).tolist()
        return {**pre, "tips": tips}
    def _high(base: dict) -> dict:
        dz = torch.tensor([0.0, 0.0, HIGH_DH * CUP_MOUTH], dtype=torch.float64)
        return {**base, "palm": (torch.tensor(base["palm"], dtype=torch.float64) + dz).tolist(),
                "tips": [(torch.tensor(t, dtype=torch.float64) + dz).tolist() for t in base["tips"]]}
    r3 = dict(rows[3][1])
    def _held_ring(rr: float) -> dict:
        # 쥔 채 목표에서 약지 접촉 0, 약지 끝만 축에서 rr × r — 접촉 전에도 벽 쪽으로 닫는 기울기가 지배 수입에 실려야 한다
        cup6 = torch.tensor(r6["cup"], dtype=torch.float64)
        tips = [list(t) for t in rq["tips"]]
        tips[3] = (cup6 + torch.tensor([rr * CUP_R * math.cos(ring_az), rr * CUP_R * math.sin(ring_az), TIP_H[3] * CUP_MOUTH],
                                       dtype=torch.float64)).tolist()
        return {**rq, "tips": tips}
    finger_rows = [
        ("6Q_goal_ring_off", rq, "6_at_goal_held", "ring"),
        # 10.01 사용자 "약지만 따로 더하면 약지만 닿고 다른 손가락이 빠질 수 있다" — 어느 한 손가락이 빠져도 같은 값을 치른다
        #   (좌 i10m: iter_10 의 평균 기반 인자에서 약지 0→0.52 가 되며 검지 0.99→0.39·새끼 0.94→0.38 로 맞바뀜)
        ("6Ki_goal_index_off", _f(r6, [2.0, 0.0, 2.0, 1.2, 1.2], [0.8, 0.0, 0.8, 0.8, 0.8], [1.2, 0.0, 1.2, 0.4, 0.4]), "6_at_goal_held", "ring"),
        ("6Km_goal_middle_off", _f(r6, [2.0, 2.0, 0.0, 1.2, 1.2], [0.8, 0.8, 0.0, 0.8, 0.8], [1.2, 1.2, 0.0, 0.4, 0.4]), "6_at_goal_held", "ring"),
        ("6Kl_goal_little_off", _f(r6, [2.0, 2.0, 2.0, 1.2, 0.0], [0.8, 0.8, 0.8, 0.8, 0.0], [1.2, 1.2, 1.2, 0.4, 0.0]), "6_at_goal_held", "ring"),
        ("6Qt_goal_ring_tip_light", rqt, "6Q_goal_ring_off", "tip"),
        ("3Q_table_ring_off", _f(r3, rq["ff"], rq["fp"], rq["ft"]), "3_grasped_on_table", "ring"),
        ("2Rf_pregrasp_ring_far", _tip(3, ring_az, 1.8, TIP_H[3]), None, None),
        ("2Rn_pregrasp_ring_near", _tip(3, ring_az, 1.45, TIP_H[3]), "2Rf_pregrasp_ring_far", "gt"),
        ("2Lf_pregrasp_little_low_far", _tip(4, base_az, 1.8, TIP_H[4]), None, None),
        ("2Ln_pregrasp_little_low_near", _tip(4, base_az, 1.45, TIP_H[4]), "2Lf_pregrasp_little_low_far", "gt"),
        ("6Rf_goal_ring_off_tip_far", _held_ring(1.8), None, None),
        ("6Rn_goal_ring_off_tip_near", _held_ring(1.25), "6Rf_goal_ring_off_tip_far", "gain"),
        ("6H_goal_held_high", _high(r6), "6_at_goal_held", "high"),
        ("3H_table_grasp_high", _high(r3), "3_grasped_on_table", "high"),
        ("6F3_goal_ring_little_off", f3, "6_at_goal_held", "lt3"),
        ("6F4t_goal_ring_little_tips_light", ft, "6F3_goal_ring_little_off", "tip"),
        ("2Pf_pregrasp_little_far", _pinky(1.8), None, None),
        ("2Pn_pregrasp_little_near", _pinky(1.45), "2Pf_pregrasp_little_far", "gt"),
    ]
    return rows, park, jitter, sloppy_rows, limit_rows + finger_rows


def ledger(reward_path: Path) -> dict:
    fn, _ = load_reward_fn(str(reward_path))
    rows, park, jitter, sloppy, limit = ledger_rows()
    names = [r[0] for r in rows] + [p[0] for p in park] + [j[0] for j in jitter] + [x[0] for x in sloppy] + [x[0] for x in limit]
    ctx = _ctx_rows([r[1] for r in rows] + [p[1] for p in park] + [j[1] for j in jitter] + [x[1] for x in sloppy]
                    + [x[1] for x in limit])
    total, terms = call_reward_fn(fn, ctx)
    val = {k: float(v) for k, v in zip(names, total)}
    fails = []
    for (a, _), (b, _) in zip(rows, rows[1:]):
        if not val[b] > val[a]:
            fails.append(f"단조 위반: {b} {val[b]:.3f} ≤ {a} {val[a]:.3f}")
    for name, _, must_be_below in park:
        # 인벨롭 판별 행(3r·3f·3x)은 10 % 이상 낮아야 한다 — 같거나 몇 % 차이면 정책이 가리지 않는다(iter_00 c1r: 0 %·12 %·4 %)
        lim = CAGE_MARGIN * val[must_be_below] if name[1:2] == "m" and name[0] in "36" else \
            THUMB_IN_MARGIN * val[must_be_below] if name[1:2] in ("i", "j") and name[0] in "36" else \
            ENVELOPE_MARGIN * val[must_be_below] if name[:2] in ("3r", "3f", "3x", "1u", "1f", "2a", "3u", "3o", "3p", "2l", "3i", "3k", "3d", "6i", "6k", "6d") else val[must_be_below]
        if not val[name] < lim:
            fails.append(f"주차: {name} {val[name]:.3f} ≥ 기준 {lim:.3f} ({must_be_below} {val[must_be_below]:.3f})")
    for name, _, ref in jitter:
        if abs(val[name] - val[ref]) > JITTER_TOL:
            fails.append(f"떨림: {name} {val[name]:.3f} vs {ref} {val[ref]:.3f} (허용 ±{JITTER_TOL})")
    # 서툰 첫 파지는 컵 옆에 반쯤 닫고 머무는 자리보다 SLOPPY_RATIO 배 이상 벌어야 쥐는 쪽으로 간다(R5: 1.38 배에서 머묾)
    for name, _, ref in sloppy:
        if ref and not val[name] >= SLOPPY_RATIO * val[ref]:
            fails.append(f"첫 파지: {name} {val[name]:.3f} < {SLOPPY_RATIO}× {ref} {val[ref]:.3f}")
    for name, _, ref, rule in limit:
        if ref is None:
            continue
        a, b = val[name], val[ref]
        ok = {"lt": a < LIMIT_APPROACH_MARGIN * b, "ge": a >= 0.98 * b, "gt": a > b + 1e-6,
              "band": LIMIT_HELD_BAND[0] * b <= a < LIMIT_HELD_BAND[1] * b,
              "lt3": a < FINGER3_MARGIN * b, "tip": a >= FINGER_TIP_GAIN * b,
              "ring": a < RING_OFF_MARGIN * b, "high": a < HIGH_MARGIN * b, "gain": a >= RING_NEAR_GAIN * b}[rule]
        if not ok:
            fails.append(f"한계·손가락({rule}): {name} {a:.3f} vs {ref} {b:.3f} ({a / max(b, 1e-9):.2f}×)")
    by_term = {k: {n: round(float(v[i]), 4) for i, n in enumerate(names)} for k, v in terms.items()}
    return {"status": "PASS" if not fails else "FAIL", "fails": fails, "rows": {k: round(v, 4) for k, v in val.items()},
            "terms": by_term}


# =============================================================================
# 궤적 원장(09.30 사용자 "왼손처럼 결과가 나오도록, 우연히 찾게 하지 말고") — 실제 정책 궤적 위에서 후보 보상을 다시 계산한다.
#   좋은 궤적(좌 i05 거울, 우 env)의 스텝 평균 보상이 나쁜 궤적(우 i03 입구 걸치기·i05 입구 안 엄지·i06 입구 덮기)보다
#   TRACE_MARGIN 배 이상 커야 한다 — 스텝 평균 ≈ 할인 전 에피소드 수익이라 PPO 가 실제로 비교하는 값이다.
TRACE_MARGIN = 0.60
TRACE_DIR = _HDGP.parent / "our_source" / "rh_aglt_traces"


def _trace_ctx(npz_path: Path) -> tuple[RewardContext, dict]:
    import dataclasses
    import numpy as np
    z = np.load(npz_path)
    kw, n = {}, None
    for fld in dataclasses.fields(RewardContext):
        a = z[f"ctx.{fld.name}"]
        if a.ndim == 1:                                    # (T,) 스칼라 필드
            v = a[0].item()
            kw[fld.name] = int(v) if fld.name == "goal_max" else float(v)
            continue
        t = torch.from_numpy(a.reshape(-1, *a.shape[2:]))
        kw[fld.name] = t if t.dtype == torch.bool else t.float()
        n = t.shape[0]
    ctx = RewardContext(**kw)
    info = {"steps": int(z[f"ctx.palm_pos"].shape[0]), "envs": int(z[f"ctx.palm_pos"].shape[1]), "samples": n}
    return ctx, info


STEER_UP_MIN = 0.80


def _thumb_toward_box_gain(fn, tp: Path) -> float:
    """쥐기 전 컵 근처 스텝에서 엄지 끝을 목표 점(같은 방위, 1.3 r · 입구 THUMB_STEER_H)으로 5 mm 옮겼을 때 보상이 오르는 비율."""
    import dataclasses
    ctx, _ = _trace_ctx(tp)
    m = ((ctx.palm_pos - ctx.cup_pos).norm(dim=-1) < 0.10) & ~ctx.grasped
    if not bool(m.any()):
        return float("nan")
    idx = m.nonzero().squeeze(1)
    sub = {}
    for fld in dataclasses.fields(ctx):
        v = getattr(ctx, fld.name)
        sub[fld.name] = v[idx] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == m.shape[0] else v
    c0 = RewardContext(**sub)
    up = c0.cup_up / c0.cup_up.norm(dim=-1, keepdim=True)
    tv = c0.tips_pos[:, 0] - c0.cup_pos
    h = (tv * up).sum(-1, keepdim=True)
    rv = tv - h * up
    rdir = rv / rv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    tgt = c0.cup_pos + rdir * (1.3 * c0.cup_radius) + up * (THUMB_STEER_H * c0.cup_mouth_z)
    step = tgt - c0.tips_pos[:, 0]
    step = step / step.norm(dim=-1, keepdim=True).clamp(min=1e-9) * 0.005
    tips1 = c0.tips_pos.clone()
    tips1[:, 0] = tips1[:, 0] + step
    c1 = dataclasses.replace(c0, tips_pos=tips1)
    r0, _ = call_reward_fn(fn, c0)
    r1, _ = call_reward_fn(fn, c1)
    return float((r1 > r0 + 1e-7).float().mean())


def trace_eval(reward_path: Path, traces: list[Path]) -> dict:
    fn, _ = load_reward_fn(str(reward_path))
    res = {}
    for tp in traces:
        ctx, info = _trace_ctx(tp)
        total, terms = call_reward_fn(fn, ctx)
        g, l = ctx.grasped, ctx.lifted & ctx.grasped
        m = lambda x, k: float(x[k].mean()) if bool(k.any()) else float("nan")
        res[tp.stem] = {**info, "mean_all": float(total.mean()), "mean_held": m(total, g), "mean_lifted": m(total, l),
                        "frac_held": float(g.float().mean()), "frac_lifted": float(l.float().mean()),
                        "terms_all": {k: round(float(v.mean()), 4) for k, v in terms.items()}}
    good = [k for k in res if k.startswith("good")]
    fails = []
    # 09.30 파지 전 유도의 기울기(iter_07 c1: 입구 위·벽 안쪽에서 유도 항이 정확히 0 → aglt_r_i07 유도 0.000):
    #   컵 근처(손바닥 10 cm)·쥐기 전 스텝에서 엄지 끝을 좋은 자리(1.3 r, 입구 0.62)로 5 mm 옮기면 보상이 올라야 한다.
    for tp in traces:
        up_frac = _thumb_toward_box_gain(fn, tp)
        res[tp.stem]["steer_up_frac"] = up_frac
        if tp.stem.startswith("bad") and not up_frac >= STEER_UP_MIN:
            fails.append(f"유도 기울기: {tp.stem} 쥐기 전 컵 근처에서 엄지를 좋은 자리로 5 mm 옮겨 보상이 오르는 비율 {up_frac:.2f} < {STEER_UP_MIN}")
    hover = ledger(reward_path)["rows"]["2h_hover_half_closed"]
    for k in [k for k in res if "hat" in k]:
        if not res[k]["mean_held"] < hover:
            fails.append(f"궤적: {k} 쥔 스텝 평균 {res[k]['mean_held']:.3f} ≥ 컵 옆 머묾(2h) {hover:.3f} — 입구 위 덮기가 주차장")
    for b in [k for k in res if k.startswith("bad")]:
        for gk in good:
            if not res[b]["mean_all"] < TRACE_MARGIN * res[gk]["mean_all"]:
                fails.append(f"궤적: {b} 스텝 평균 {res[b]['mean_all']:.3f} ≥ {TRACE_MARGIN} × {gk} {res[gk]['mean_all']:.3f}")
    return {"status": "PASS" if not fails else "FAIL", "fails": fails, "traces": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("--iter", required=True)
    a.add_argument("--response", required=True)
    b = sub.add_parser("ledger")
    b.add_argument("--iter", required=True)
    b.add_argument("--reward", default="compute_reward.py")
    c = sub.add_parser("trace")
    c.add_argument("--iter", required=True)
    c.add_argument("--reward", default="compute_reward.py")
    c.add_argument("--traces", nargs="*", default=None, help="기본: our_source/rh_aglt_traces/*.npz")
    args = ap.parse_args()
    it = (_HDGP / args.iter) if not Path(args.iter).is_absolute() else Path(args.iter)
    if args.cmd == "ingest":
        code = extract_python((it / args.response).read_text())
        out = it / "compute_reward.py"
        out.write_text(code)
        res = validate(out)
        (it / "validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["status"] == "PASS" else 1
    if args.cmd == "trace":
        tr = [Path(x) for x in args.traces] if args.traces else sorted(TRACE_DIR.glob("*.npz"))
        res = trace_eval(it / args.reward, tr)
        print(json.dumps({"status": res["status"], "fails": res["fails"],
                          "traces": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "terms_all"}
                                     for k, v in res["traces"].items()}}, ensure_ascii=False, indent=1))
        (it / f"trace_{Path(args.reward).stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if res["status"] == "PASS" else 1
    res = ledger(it / args.reward)
    print(json.dumps({k: res[k] for k in ("status", "fails", "rows")}, ensure_ascii=False, indent=1))
    (it / f"ledger_{Path(args.reward).stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
