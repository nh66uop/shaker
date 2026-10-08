# iter_02 운영 노트 — 2026-09-27 (세 번째 사이클 2 라운드)

확인된 사실만 적는다. 추정은 [가설] 로 표시한다. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다.

## 사용자 지시 (09-27)
- "접근·파지·리프트가 **fresh 에서도 무조건** 되게 해야 한다."
- 이번 사이클 결정 유지: LSTM(horizon 16), self-collision ON, 붓기 구조는 DG-5F pour_bi2 iter_02 이식(iter_01 c5).

## iter_01 (t2r_rh4_i01, fresh LSTM, c5 보상) 결과 — e1436 에서 사용자 지시로 종료
| 지표 (20 epoch 평균) | e0 | e1100 | e1400 |
|---|---|---|---|
| 소스 / 리시버 파지 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| approach_src (최대 0.4) | 0.016 | 0.040 | 0.037 |
| 소스 손 컵 근처 비율 | 0.008 | 0.000 | 0.000 |
| 리시버 손 컵 근처 비율 | 0.009 | 0.085 | 0.032 |
| 손 닫힘 src / rcv | 0.08 / 0.10 | 0.03 / 0.03 | 0.03 / 0.03 |
| 흘림 / 컵 충돌 / 손 접촉 벌점 | −0.131 / −0.102 / −0.075 | −0.006 / −0.013 / −0.004 | −0.002 / −0.006 / −0.002 |

- 9,400 만 프레임 동안 한 번도 컵을 잡지 못했다. 손을 편 채 컵에서 멀어졌고, 보상 상승은 전부 벌점 회피였다.
- 원인(코드 대조): c5 의 파지 쪽은 `approach`(손바닥·엄지와 컵 중심 거리, 0.4) + `grasp_hold`(성립해야만 0.5) 뿐이다.
  **손끝을 컵 벽에 대는 중간 수입, 손을 닫는 수입, 성립 전 접촉에 대한 단계적 수입이 없다.**
  반면 흘림은 파지 전부터 **비드당 −40** 이다. 컵은 한쪽 접촉 0.3~0.4 N 에 넘어진다(09.17 프로브).
  "근처에 가서 건드리면 큰 손해, 닫아도 이득 없음" 이라 가지 않는 쪽이 최적이었다.
- DG-5F iter_02 는 파지하는 체크포인트에서 이어받는 전제였다(t2r2_i02 = i01 ep250 웜). fresh 에는 파지 항이 따로 필요하다.

## 이 트랙에서 파지를 실제로 배운 보상 (측정)
| 런 | 시작 | 소스 파지 첫 0.05 / 0.3 | 리시버 파지 | 끝 src / rcv |
|---|---|---|---|---|
| t2r_rh_i00_r1 (1 차 사이클) | fresh | e248 / e474 | 없음 | 0.37 / 0.00 |
| t2r_rh_i03 (1 차 사이클) | fresh | e556 / e831 | 없음 | 0.27 / 0.00 |
| t2r_rh2_i03 | rh2_i01 ep1250(fresh, 접근 0.93·파지 0) 이어받음 | +e1753 / +e2154 | 없음 | 0.84 / 0.00 |
| t2r_rh2_i04 | rh2_i03 ep2000 | 곧바로 | **e363** | 0.52 / 0.55 |

- 위 런은 모두 MLP · horizon 64 · self-collision OFF 였다. 이번은 LSTM · horizon 16 · self-collision ON 이다.
- rh2_i01(fresh)은 접근 0.93 에서 **엄지 힘 0 N 으로 주차**했다. rh2_i03 은 이것을 "한쪽 면(엄지 / 나머지)마다 따로 점수를 매기고,
  **접촉력이 생겨야 그 면이 완성**" 되게 바꿔 소스 파지 0.84 를 만들었다.
- 리시버는 fresh 에서 한 번도 파지하지 못했다. rh2_i03 에서는 **소스 들기 수입이 켜지자(e2000→2500) 리시버가 컵을 버렸다**
  (근처 비율 0.87 → 0.05). rh2_i04 는 **소스의 큰 수입(들기 이후)을 리시버 진행도로 게이트**하고, 리시버 근처 벌점 합을
  파지 자세 수입보다 확실히 작게 줄여 리시버 파지 0.55 를 만들었다.
- 컵 사실: 바닥 지름 37.8 mm, 0.134 kg, **한쪽 접촉 0.3~0.4 N 에 넘어진다. 양쪽 동시 접촉은 넘어지지 않는다.**

## iter_02 에서 할 것 (접근·파지·들기를 fresh 에서 확실히)
1. **파지 블록은 rh2_i03 의 형태를 그대로 쓴다**(아래 원문): `_grip_geometry`(손끝 컵 축 기준 높이·반경·맞은편 배치·깔때기),
   `_hand_cup_terms` 의 reach(먼 깔때기 + 면별 손끝-벽, 힘으로 완성) · close(파지 자세에서 닫힘) · grasp(성립 전 접촉도 단계적) ·
   lift / held / up_w · slide · one_sided · palm_push · rush · fling. 가중 W_REACH 1 · W_CLOSE 0.3 · W_GRASP 2 · W_LIFT 3 의
   **자릿수와 사다리 순서(접근 < 파지 < 들기)를 유지**한다. TIP_R 같은 손 치수 상수는 rh2_i03 주석의 측정값을 그대로 쓰고 출처를 적는다.
2. **리시버를 버리지 않게 한다**(rh2_i04 교훈): 소스의 들기 이후 수입(들기·수렴·조준·회전)은 **리시버 진행도**(리시버 reach → 파지 → 들기)로
   게이트한다(곱이 아니라 바닥이 있는 계수로 — 곱 사다리 금지). 리시버의 컵 근처 벌점 합은 리시버 파지 자세 수입보다 확실히 작게 둔다.
3. **파지 전 벌점은 rh2_i03 수준으로 낮춘다.** 흘림은 파지 성립 전 비드당 약 −3(rh2_i03 W_SPILL 60), 두 컵을 쥔 뒤에는 c5 의 −40.
   컵 충돌·손 접촉도 파지 전에는 rh2_i03 의 크기(cup_collision 1.0 · hand_foreign 0.5)를 넘지 않게.
4. **붓기 구조는 iter_01 c5(아래 원문)를 그대로** 두 컵을 쥐고 든 뒤에 적용한다: converge_cups · aim_lip · pour_rot(0° 부터 선형,
   조준 바닥 0.3) · pour_delta 30 · spill −40 · success 20 · finish_upright · premature · pour_dir · rcv_keep. c5 의
   approach_src/rcv · grasp_hold · lift_src/lift_rcv 는 1 번의 rh2_i03 블록으로 **대체**한다(중복 금지).
5. **주차 금지**(memory): 상태 비례 수입은 컵에 손을 댈 때까지 커지지 않게 한다 — rh2_i01 이 reach 0.93 에서 주차했다.
   alive 를 쓰면 rh2_i03 처럼 1.0 이하로 두고, alive + reach 만으로 버는 수입이 파지 자세 수입보다 작아야 한다.
6. 항 수 30 개 안팎까지 허용한다(파지 블록이 팔별로 늘어나므로). 새 ctx 필드를 가정하지 않는다.

## rh2_i04 가 받은 운영자 요구 (원문, 참고)
## What the operator wants from iter_04
1. The receiver must keep approaching and must grasp. Make receiver progress a precondition for the big source pay: e.g. cap or gate lift_src (and later align/tilt) by receiver stage (reach_rcv, then rcv grasp), so that lifting the source while the receiver is away is worth little.
2. Near-cup penalties for the receiver (rush, slide, tilt, palm push, one-sided) must in total stay clearly below what the receiver earns by being in the grasp pose, and leaving the cup must cost clearly more than staying. Do not remove knock-over protection; rescale it.
3. Keep the source-hand reach/grasp/lift formulation that worked.
4. After both cups are held, there must be a gradient that brings the source mouth over the receiver (cups_center_dist stayed 0.34 m with align 0).



## 참고 코드 1 — rh2_i03 보상 원문 (파지 블록을 여기서 가져온다)
```text
import torch
import math


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


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
            "one_sided": one_sided, "palm_push": palm_push, "rush": rush, "fling": fling}


def _pour_geometry(ctx, outer_r):
    """Distance of the source pouring lip (lowest rim point) from a point above the receiver rim centre."""
    POUR_CLEARANCE = 0.05   # lip 5 cm above the receiver rim: no cup-cup contact, origins stay > 9 cm apart
    ez = torch.tensor([0.0, 0.0, 1.0], device=ctx.src_cup_up.device, dtype=ctx.src_cup_up.dtype)
    # (up * up_z - ez) has norm sin(tilt): zero when upright, straight down by one radius when horizontal
    lip = ctx.src_cup_mouth_pos + outer_r * (ctx.src_cup_up * ctx.src_cup_up[:, 2:3] - ez)
    delta = lip - (ctx.rcv_cup_mouth_pos + POUR_CLEARANCE * ez)
    d_xy = delta[:, :2].norm(dim=-1)
    d_align = torch.sqrt(d_xy ** 2 + (0.5 * delta[:, 2]) ** 2)          # height error counts half
    align_gate = torch.exp(-(d_align / 0.06) ** 2)                       # wide: tilting is allowed here
    pour_gate = torch.exp(-(d_align / max(ctx.cup_radius, 0.015)) ** 2)  # narrow: beads would land inside
    return d_align, align_gate, pour_gate


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Stage ladder (per step, cumulative): reach 1 < grasp 2 < lift 3 per arm, then align 4 < tilt 5.
    # A later stage keeps every earlier term, so progress never lowers the reward.
    W_REACH, W_GRASP, W_LIFT = 1.0, 2.0, 3.0
    W_CLOSE = 0.3         # closure shaping at the grasp pose; small next to grasp 2.0, only a gradient helper
    W_ALIGN, W_TILT = 4.0, 5.0
    W_POUR = 150.0        # x d_in_target: 7.5 per bead with 20 beads, one-time, dominates the dense terms at that step
    W_SPILL = 60.0        # x d_spill: 3 per bead, below the pour pay so an imperfect pour is still worth trying
    W_SUCCESS = 20.0      # per step while the env reports success
    W_ALIVE = 1.0         # with reach it outweighs all persistent penalties, so ending the episode never pays
    W_SLIDE, W_TILT_PEN = 0.5, 0.3
    W_ONE_SIDED, W_PALM, W_RUSH, W_FLING = 0.3, 0.2, 0.2, 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    W_ACT_RATE = 0.05     # x mean squared action change (<= 4), regularisation only
    CUP_OUTER_R = 0.0285  # cup wall, used for the pouring lip
    TIP_R = 0.027         # tip-frame radius at measured two-sided contact (thumb 27.9 mm, nearest finger 26.2 mm)
    UPRIGHT_TOL = 0.35    # rad, = the 20 deg success limit on the receiver cup
    TILT_FULL = math.radians(100.0)   # half the beads are out by 86 deg, reward saturates a little past that
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.15, 0.08   # source ends above the receiver

    dtype = ctx.src_cup_tilt.dtype
    d_align, align_gate, pour_gate = _pour_geometry(ctx, CUP_OUTER_R)
    upright_src = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = align_gate + (1.0 - align_gate) * upright_src   # source may tilt only over the receiver
    pose_ok_rcv = upright_rcv

    src = _hand_cup_terms(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                          ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up,
                          ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_src, LIFT_TARGET_SRC)
    rcv = _hand_cup_terms(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                          ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up,
                          ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z,
                          TIP_R, pose_ok_rcv, LIFT_TARGET_RCV)

    nested = ctx.cups_nested.to(dtype)
    both_up = src["up_w"] * rcv["up_w"] * (1.0 - nested)
    align = both_up * (0.5 * (1.0 - torch.tanh(4.0 * d_align)) + 0.5 * torch.exp(-25.0 * d_align))
    pour_tilt = both_up * pour_gate * torch.clamp(ctx.src_cup_tilt / TILT_FULL, max=1.0)

    tilt_pen_src = (1.0 - align_gate) * torch.tanh(ctx.src_cup_tilt / UPRIGHT_TOL)
    tilt_pen_rcv = torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL)
    act_rate = ((ctx.actions - ctx.prev_actions) ** 2).mean(dim=-1)

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "reach_src": W_REACH * src["reach"],
        "reach_rcv": W_REACH * rcv["reach"],
        "close_src": W_CLOSE * src["close"],
        "close_rcv": W_CLOSE * rcv["close"],
        "grasp_src": W_GRASP * src["grasp"],
        "grasp_rcv": W_GRASP * rcv["grasp"],
        "lift_src": W_LIFT * src["lift"],
        "lift_rcv": W_LIFT * rcv["lift"],
        "align_mouths": W_ALIGN * align,
        "pour_tilt": W_TILT * pour_tilt,
        "pour_delta": W_POUR * ctx.d_in_target,
        "spill_delta": -W_SPILL * ctx.d_spill,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype),
        "cup_disturb_src": -(W_SLIDE * src["slide"] + W_TILT_PEN * tilt_pen_src),
        "cup_disturb_rcv": -(W_SLIDE * rcv["slide"] + W_TILT_PEN * tilt_pen_rcv),
        "contact_push": -(W_ONE_SIDED * (src["one_sided"] + rcv["one_sided"])
                          + W_PALM * (src["palm_push"] + rcv["palm_push"])),
        "rush_near_cup": -W_RUSH * (src["rush"] + rcv["rush"]),
        "cup_fling": -W_FLING * (src["fling"] + rcv["fling"]),
        "cup_collision": -W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0),
        "hand_foreign": -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                                      + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)),
        "nested": -W_NESTED * nested,
        "action_rate": -W_ACT_RATE * act_rate,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value
    return reward, comps

```

## 참고 코드 2 — iter_01 c5 원문 (붓기 구조를 여기서 가져온다)
```text
import torch
import math


def _dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.sqrt((d * d).sum(dim=-1) + 1e-12)


def _sat(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.clamp(x, lo, hi)


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x * x).mean(dim=-1) + 1e-12)


def compute_reward(ctx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """RH 양팔 붓기. DG-5F pour_bi2 iter_02 구조를 이식하고 파지 항만 RH(iter_00) 형태를 쓴다.
    모든 길이는 ctx.cup_radius 배수(들기 높이 허용만 절대값 예외), 각도는 LIM 배수."""
    tilt_src = ctx.src_cup_tilt
    zeros = torch.zeros_like(tilt_src)
    dt = tilt_src.dtype

    # ---- 공통 척도 ---------------------------------------------------------------
    r_cup = float(ctx.cup_radius)
    two_r = 2.0 * r_cup                      # lip 수평 허가 반경
    wide = 10.0 * r_cup                      # WIDE 커널 (DG-5F 그대로)
    # premature_tilt_limit 대체 스칼라: RH 실측 방출 101° - 20° (DG-5F 는 유출 93.6° 대 한계 p50 73.6°)
    LIM = math.radians(81.0)
    active = float(max(int(ctx.num_beads), 1))   # bead_active_count 대체 (RH 20 개 고정)

    src_g = ctx.src_grasped.to(dt)
    rcv_g = ctx.rcv_grasped.to(dt)
    both_g = src_g * rcv_g
    ones = torch.ones_like(tilt_src)

    in_frac = _sat(ctx.bead_in_target_frac)
    state_decay = 1.0 - 0.5 * in_frac        # iter_00 파지 항의 감쇠 (담김이 오르면 상태 수입 축소)

    # ---- RH 에 없는 DG-5F 필드를 RH ctx 로 계산 (env pour_rules.py 와 같은 식) -----
    d_xy = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    d_n = torch.sqrt((d_xy * d_xy).sum(dim=-1) + 1e-12)
    d_hat = d_xy / d_n.unsqueeze(-1)                                   # pour_dir_xy
    up = ctx.src_cup_up
    theta = torch.atan2((up[:, :2] * d_hat).sum(dim=-1), up[:, 2])    # src_tilt_toward_rcv (부호 있음)
    lip_xy = ctx.src_cup_mouth_pos[:, :2] + r_cup * torch.cos(theta).unsqueeze(-1) * d_hat
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r_cup * torch.sin(theta)    # src_pour_lip_pos
    e_lip = lip_xy - ctx.rcv_cup_mouth_pos[:, :2]
    lip_d = torch.sqrt((e_lip * e_lip).sum(dim=-1) + 1e-12)
    dz_lip = lip_z - ctx.rcv_cup_mouth_pos[:, 2]

    # ---- A. 접근 (RH iter_00 형태, 팔당 0.4: fresh 학습이라 DG-5F 자릿수) ---------
    d_src = 0.5 * (_dist(ctx.src_palm_pos, ctx.src_cup_pos) + _dist(ctx.src_tips_pos[:, 0, :], ctx.src_cup_pos))
    d_rcv = 0.5 * (_dist(ctx.rcv_palm_pos, ctx.rcv_cup_pos) + _dist(ctx.rcv_tips_pos[:, 0, :], ctx.rcv_cup_pos))
    near_src = 1.0 - torch.tanh(d_src / (4.0 * r_cup))
    near_rcv = 1.0 - torch.tanh(d_rcv / (4.0 * r_cup))
    approach_src = 0.4 * torch.where(ctx.src_grasped, ones, near_src) * state_decay
    approach_rcv = 0.4 * torch.where(ctx.rcv_grasped, ones, near_rcv) * state_decay

    # ---- A. 파지 유지 (팔당 0.5, DG-5F grasp 자릿수) ------------------------------
    grasp_hold = 0.5 * (src_g + rcv_g) * state_decay

    # ---- B. 들기 (RH iter_00 형태; 합 상한 0.5 = DG-5F lift_both 규모) -------------
    h_src = ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = ctx.rcv_cup_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    # 소스는 높게(K3), 과들기 세금은 충분히 높은 곳에서만 (높이 허용은 절대값 예외)
    lift_src = (0.3 * _sat(h_src / 0.28) - 0.15 * _sat((h_src - 0.45) / 0.20)) * src_g * state_decay
    # 리시버는 떼기만, 낮게
    lift_rcv = 0.2 * _sat(h_rcv / two_r) * torch.exp(-torch.clamp(h_rcv - 0.15, min=0.0) / 0.10) \
        * rcv_g * state_decay
    lifted = torch.minimum(_sat(h_src / two_r), _sat(h_rcv / two_r))

    # ---- C. 두 컵 수렴 (1.2, WIDE 커널, DG-5F 그대로) -----------------------------
    converge_cups = 1.2 * both_g * lifted * (
        1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / wide))

    # ---- D. lip 조준 (0.8, DG-5F 그대로) ------------------------------------------
    h_score = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (0.5 * r_cup))
    xy_far = 1.0 - torch.tanh(lip_d / wide)
    xy_fine = 1.0 - torch.tanh(lip_d / two_r)
    xy_score = 0.5 * xy_far + 0.5 * xy_fine
    aim_lip = 0.8 * both_g * lifted * (0.6 * xy_score + 0.4 * h_score)

    # ---- 비드 장부: pour_target_frac = 1.0 (RH 는 전량 이송) -----------------------
    remaining = torch.clamp(1.0 - ctx.bead_in_target_frac, min=0.0)
    need_frac = torch.minimum(remaining, torch.clamp(ctx.bead_in_source_frac, min=0.0))
    want = _sat(need_frac * active)

    # ---- F. 회전 구동 (8.0): 0° 부터 선형, 1.4*LIM(약 113°)에서 포화 ---------------
    lip_gate = 1.0 - torch.tanh(torch.clamp(lip_d - two_r, min=0.0) / two_r)
    rot_prog = _sat(torch.clamp(theta, min=0.0) / (1.4 * LIM))
    aim_xy = 1.0 - torch.tanh(torch.clamp(lip_d - r_cup, min=0.0) / two_r)
    aim_h = 1.0 - torch.tanh(torch.abs(dz_lip - 0.25 * r_cup) / (1.5 * r_cup))
    aim_quality = _sat(0.65 * aim_xy + 0.35 * aim_h)
    AIM_FLOOR = 0.3                          # 바닥 위 가산, 정밀 게이트 아님
    aim_factor = AIM_FLOOR + (1.0 - AIM_FLOOR) * aim_quality
    # grasp_ref_valid 는 RH 에 없어 곱 인자에서 뺀다
    pour_rot = 8.0 * src_g * want * lip_gate * aim_factor * rot_prog

    # ---- F. 실제 도착한 비드 (알당 30) --------------------------------------------
    gained = torch.clamp(ctx.d_in_target, min=0.0) * active
    pour_delta = 30.0 * want * gained

    # ---- 흘림 (알당 40 > 배달 30: 조준 불량 붓기는 순손실) ------------------------
    spill = -40.0 * _sat(torch.clamp(ctx.d_spill, min=0.0) * active, 0.0, 5.0)

    # ---- 마무리: 다 부은 뒤 소스를 세우고 쥐고 있기 (1.5 < 회전 수입 8) ------------
    finish_upright = 1.5 * (1.0 - want) * both_g * (1.0 - torch.tanh(tilt_src / (0.3 * LIM)))

    # ---- 성공 보너스 (env 판정, 재정의 못 함; 기울인 채 버티기 최대 8 보다 크게) ----
    success_bonus = 20.0 * ctx.success.to(dt)

    # ---- 조기 기울임: 벽(0.95~1.0 LIM)에서만, lip 이 2r 밖일 때만 (래치 부분 제외) -
    pre_x = torch.clamp(tilt_src - 0.95 * LIM, min=0.0) / (0.05 * LIM)
    pre_ramp = _sat(torch.tanh(2.0 * pre_x))
    out_gate = _sat((lip_d - two_r) / (0.5 * two_r))
    premature = -12.0 * src_g * pre_ramp * out_gate

    # ---- 실기 안전: 컵끼리 접촉, 포개기 (cup_hit 은 RH 에 없어 뺀다) ----------------
    cup_clash = -(2.0 * torch.tanh(ctx.cup_cup_force / 5.0)
                  + 3.0 * ctx.cups_nested.to(dt))

    # ---- 손은 자기 컵만 (self-collision ON 에서 팔끼리 접촉도 여기로 잡힌다) --------
    hand_foreign = -1.0 * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                           + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    # ---- 붓는 방향이 몸 바깥(+x)을 향하지 않게 ------------------------------------
    dir_ramp = _sat((d_hat[:, 0] - 0.15) / 0.15)
    pour_dir = -1.5 * lifted * dir_ramp

    # ---- 리시버 컵: 직립, 중심선 기준 자기 쪽(+y) ---------------------------------
    side_lo = 0.75 * r_cup
    side_bad = _sat((side_lo - ctx.rcv_cup_pos[:, 1]) / side_lo)   # rcv_side_margin = env-local y
    rcv_keep = 0.5 * rcv_g * (1.0 - torch.tanh(ctx.rcv_cup_tilt / (0.3 * LIM))) - 1.5 * side_bad

    # ---- G. 관절 여유: 붓는 중에만 작은 가산 (최대 0.3, 회전 구동에 곱하지 않음) ----
    m_src = ctx.src_arm_limit_margin
    margin = _sat(0.5 * (m_src.mean(dim=-1) + torch.min(m_src, dim=-1).values))
    posture_margin = 0.3 * src_g * want * lip_gate * rot_prog * margin

    # ---- 제어 정규화 (DG-5F 그대로) ------------------------------------------------
    da = ctx.actions - ctx.prev_actions
    action_rate = -0.05 * (da * da).mean(dim=-1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=-1)
    action_flip = -0.30 * flip
    qd_rcv = _rms(ctx.rcv_arm_qd)
    qd_src = _rms(ctx.src_arm_qd)
    arm_tremor = -(0.25 * torch.tanh(qd_rcv / 1.0) + 0.10 * torch.tanh(qd_src / 1.0))
    # adr_progress 없음: 상수 0.1 포화 벌점 (DG-5F 원형대로 전 차원 평균)
    sat = _sat((ctx.actions.abs() - 0.9) / 0.1).mean(dim=-1)
    action_reg_adr = -0.1 * sat

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_hold": grasp_hold,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "converge_cups": converge_cups,
        "aim_lip": aim_lip,
        "pour_rot": pour_rot,
        "pour_delta": pour_delta,
        "spill": spill,
        "finish_upright": finish_upright,
        "success_bonus": success_bonus,
        "premature": premature,
        "cup_clash": cup_clash,
        "hand_foreign": hand_foreign,
        "pour_dir": pour_dir,
        "rcv_keep": rcv_keep,
        "posture_margin": posture_margin,
        "action_rate": action_rate,
        "action_flip": action_flip,
        "arm_tremor": arm_tremor,
        "action_reg_adr": action_reg_adr,
    }

    reward = zeros.clone()
    for v in components.values():
        reward = reward + v
    return reward, components

```
