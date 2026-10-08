# iter_03 (4번째 사이클) 운영 노트 — 2026-09-29

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다(K7 인벨롭 기하, f03 행).

## 사용자 결정 (09-29)
- 보상 B: 2번째 사이클 iter_04(런 t2r_rh2_i04, 이 트랙에서 양손 파지가 처음 선 보상)의 손끝 기준 파지 구조로 돌아가고,
  K7 인벨롭 방향과 속도형 밀림 벌점을 더한다.
- 팔 액션은 절대 목표(a=0 시작 자세·±1 관절 한계, 0.3 rad/s) — 로봇 설명 절. PPO bounds_loss 0.005. 물리 이상은 env 가 종료한다.

## 측정 사실
- t2r_rh2_i04(iter_04, 손바닥 지령·시너지 손, i03 ep2000 이어받음): 왼손 파지 0 → 0.54, 양손 0.52/0.54(e1550). 결정론 4/4 env 양손 파지 유지,
  엄지 3~5 N · 나머지 합 2.5~7 N · 손바닥 0.
- 같은 원장(손끝도 손바닥과 같이 이동)에서 iter_04: 파지(양쪽 1.5 N, 판정 켜짐) 7.82 / 벽에 닿고 닫힘 0.7 4.20 = 1.86 배.
  iter_02: 9.96 / 7.23 = 1.38 배. iter_04 는 손 방향을 보지 않는다(2f·2v = 2, 2.15).
- iter_04 의 깔때기는 파지점보다 4.5 cm 위에 먼저 도착하게 한다(ARRIVE_RISE): 원장 1 절반 접근 1.10 → 1b 옆 12 cm·잡는 높이 0.86 (골).
  3번째 사이클 i04 에서도 같은 형태의 깔때기가 '위에서 끼우기'를 가르쳤다(브리프 4절).
- iter_04 의 slide 는 스폰에서 밀린 거리 상태형이라 원장 3p 2.62 < 3 2.90 (골). i09 파지 붕괴와 같은 형태(브리프 4절).
- f03(iter_02): 왼손 세로 집기(손가락 축 z −0.95, 엄지·검지만), 들 때 10 곳 중 9 곳 미끄러짐. 오른손은 옆 접촉 뒤 물러남.
- iter_04 원문(파지 전·들기 부분, 붓기 부분 제외):

```python
def _close_in(d):
    """Tip-to-wall shaping: long scale for the approach plus a short scale so the last 10 mm still pay
    (10 mm -> 0.47, 6.5 mm -> 0.59, 2 mm -> 0.83, contact -> 1)."""
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the cup frame (axis = cup_up, so it stays valid while the cup is tilted)."""
    ARRIVE_RISE = 0.045    # arrive 45 mm above grasp height (0/336 knock-overs at +45 mm) ...
    FUNNEL_R0 = 0.01       # ... and be at grasp height once the cup is centred within 1 cm
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z        # pinch height above cup origin (~13 mm, inside the 0-25 mm lift group)
    thumb_hi = 0.5 * cup_mouth_z        # thumb band 0..27 mm above origin (no lift with thumb >= 50 mm)

    up = cup_up.unsqueeze(1)                                   # (N,1,3)
    rel = tips_pos - cup_pos.unsqueeze(1)                      # (N,F,3)
    h_tip = (rel * up).sum(dim=-1)                             # (N,F) height along the cup axis
    rad_vec = rel - h_tip.unsqueeze(-1) * up                   # (N,F,3)
    rad = rad_vec.norm(dim=-1)                                 # (N,F)
    # gap is zero only at the measured contact radius (tip frame 26-28 mm, cup outer 28.5 mm);
    # the old +10 mm pad made 33-37 mm count as "on the wall" and killed the gradient
    gap = torch.clamp(rad - tip_r, min=0.0)                    # (N,F)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)        # (N,)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values   # nearest non-thumb tip, (N,)

    # opposition: 1 when the thumb and the other fingers sit on opposite sides of the cup axis
    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))                # (N,)

    # pinch centre and the descending funnel target
    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))  # (N,3)
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup_terms(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
                    cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok, lift_target):
    """Unweighted per-arm terms, each in [0, 1]. pose_ok (N,) = cup attitude is acceptable for this stage."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    # A. reach: long-range funnel term + per-side fingertip-on-wall term.
    #    Each side is scored on its own (sum form) so the thumb keeps its full gradient while the finger
    #    side is already done, and measured force completes a side: reach saturates only with contact
    #    force on BOTH sides (the parked pose of i01 had thumb 0.00 N and reach 0.97).
    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach_tips = 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])
    reach = 0.4 * reach_far + 0.6 * reach_tips

    # B. grasp: balanced two-sided pinch. Before the flag, finger-first touch earns 0.15 and any thumb force
    #    added to it can only raise the value (0.5 N -> +0.19); thumb-only still earns nothing.
    #    Flag boundary (1 N / 1 N): pre_grasp 0.355 -> flagged 0.75, a step up, never a valley.
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)          # lifts had thumb ~2.3 N vs fingers ~2.7 N
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    contact_q = torch.where(g, 0.4 + 0.6 * pinch, pre_grasp)
    grasp = contact_q * (0.1 + 0.9 * geo["thumb_band"]) * pose_ok

    # closure shaping: measured grasps close to 0.68-0.98, the parked hand sat at 0.19 with the closure
    # channels driven open; pays only once the pinch centre is at the grasp pose (contact freeze makes
    # closing there harmless), saturates at 0.7 so over-squeezing is not rewarded
    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = pose_ok * near_grip * torch.clamp(hand_closure / 0.7, max=1.0)

    # C. lift: measured cup elevation only, clamped, and only while really held (explosion-proof)
    h_lift = cup_pos[:, 2] - cup_spawn_pos[:, 2]
    h_pos = torch.clamp(h_lift - 0.005, min=0.0)                # 5 mm dead band: tipping on the rim is not lifting
    v_lin = cup_lin_vel.norm(dim=-1)
    calm = torch.exp(-(v_lin / 0.6) ** 2)
    held = g_f * torch.exp(-(geo["d_grip"] / 0.08) ** 2) * calm
    lift_prog = 0.5 * torch.tanh(h_pos / 0.02) + 0.5 * torch.clamp(h_pos / lift_target, max=1.0)
    lift = held * pose_ok * lift_prog
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)               # 0 on the table, 1 from 2.5 cm up
    up_w = held * torch.clamp(h_pos / 0.05, max=1.0)            # "held and at least 5 cm up"

    # penalties (unweighted, bounded)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)       # 1 N dead band: a slightly stronger thumb is free
    finger_excess = torch.clamp(f_o - f_t, min=0.0)             # finger-only contact: adding thumb force lowers this
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)   # knock-overs came at ~0.09 m/s hand speed
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"reach": reach, "grasp": grasp, "close": close, "lift": lift, "up_w": up_w, "slide": slide,
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling,
            "held": held, "far": torch.tanh(4.0 * geo["d_funnel"])}
```

## iter_03 에서 할 것 (이전 코드 iter_02 위의 변경)
1. 팔마다 들기 전 구조를 iter_04 형태로 바꾼다: 손끝 기준 접근(집는 중심 → 파지점 + 엄지·나머지 손끝-벽 거리, 한쪽씩 합, 힘이 닿으면 완성, 대향 가중),
   파지는 **env 파지 판정(엄지 > 1 N AND 다른 손가락 > 1 N) 뒤에** 크게 주고 판정 전은 작은 사전 항(판정 경계는 오르는 계단, 골 없음),
   닫기는 파지점 근처 × 폐쇄(작은 무게), 들기는 held = 판정 × 파지점 근처 × 컵 속도 차분.
2. 깔때기는 **옆·잡는 높이로 도착**한다(파지점 위 도착 없음). 접근 거리는 컵 원점 기준·높이 차 가중(브리프 5-1 원칙).
3. K7 인벨롭 방향 점수(iter_02 의 env_score)를 **접근·닫기·파지 모두에** 바닥 있는 인자로 건다(들기에는 걸지 않는다 — 들기 경계 연속).
4. 밀림 벌점은 속도형만(iter_02 형태). iter_04 의 밀린 거리 상태형은 쓰지 않는다.
5. 두 팔의 들기 전 구조·무게는 같게(역할별 떠남 벌점 없음). 붓기 이후(배치·조준·회전·배달·흘림·충돌)는 iter_02 그대로.
6. 팔 액션 크기 비용은 두지 않는다(로봇 설명 절). 변화율·팔 속도 비용은 유지.
7. 검사 통과 조건: 원장 단조(0 < 1 < 1b < 2 < 3 < 4 < 5 < 6), 파지 5 / 벽에 닿음 4 ≥ 1.6, 5v(같은 파지 힘·세로 손) < 5 − 1.0,
   2f·2v < 2, 3p ≥ 3, 0p ≥ 0, 1h < 1b, [35] 인벨롭 > 손끝 겨눔·수직 손, [37] 팔 액션 크기 비용 0, [15] 원거리 접근 기울기 > 0, [33] 제어 비용.
