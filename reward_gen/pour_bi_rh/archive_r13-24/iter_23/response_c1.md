## 1. 과제의 의미와 단계

오른손(소스)은 비드 20 알이 든 컵을, 왼손(리시버)은 빈 컵을 손끝 집기로 잡아 둘 다 들어 올린다. 소스 컵의 주둥이(기울였을 때 비드가 떠나는 림 점, 코드의 `lip`)를 리시버 컵 입구 위로 가져가 받는 컵 쪽으로 기울이면 비드가 공중을 지나 리시버 컵에 떨어진다. 리시버 컵은 직립(20° 이하)으로 유지하고, 컵을 놓치지 않으며, 흘림을 최소로 한다. 컵을 끼워 넣는 것(`cups_nested`)은 이송이 아니다.

| 단계 | 로봇이 해야 할 일 | 보상에서 맡는 항 |
|---|---|---|
| A 파지 | 양손 손끝 파지, 컵을 밀거나 넘어뜨리지 않기 | grasp_src, grasp_rcv (상태, 작게) |
| B 들기 | 소스는 높게(K3: 높은 손이 회전을 잘 만든다), 리시버는 낮게·직립 | lift_src, lift_rcv |
| C 배치 | 리시버 입구를 절대 높이 목표(테이블 위 컵 림 + 0.15 m)·y ≈ 0·소스 주둥이 아래로 | place_rcv (차분) |
| D 조준 | 소스 주둥이를 입구 위 0.7 r ~ 4 r, 수평 1 r 안으로 | aim_lip (차분) |
| E 방향·기울이기 | 받는 컵 쪽으로 기울이기. 각도 크기는 학습이 찾는다 | tilt_dir (비용), tilt_open (첫 비드 전 작은 차분 신호) |
| F 결과 | 비드 증분 수입, 흘림·조준 밖 방출 벌점, 채움 수준과 전량 이송 | pour_delta, transferred, transfer_tail, spill_delta, unaimed_release, success_bonus |
| 상시 | 안정 속도·부드러운 명령·충돌 없음 | G 제어 항 전부, cup_disturb, safety |

## 2. i22 피드백 분석

| 지표 | 곡선 (처음 → 끝) | 해석 |
|---|---|---|
| task/episode_success, success_bonus | 전 구간 0 | 결과 단계에 도달하지 못했다 |
| bead/in_target | 최대 3.3e-4 (20 알 기준 0.007 알) | 결과 수입 사실상 0 |
| reward/aim_lip | 0.021 → 0.166 (8 배) | 그런데 diag_aim_dist_r 7.32 → 7.32 r, diag_lip_xy 0.195 → 0.191 m: 주둥이는 가까워지지 않았다. 노트 trace: ep700 지불 201 대 위치 차분 92/에피소드 |
| reward/pour_sink | 0.0046 → 0.0705 (15 배) | diag_h_org 는 0.05 ~ 0.10 m 에서 방향 없이 흔들린다. 노트 trace: ep700 +153 대 위치 차분 +18/에피소드 |
| diag_dir_err_deg | 34.3 (3 번째 표본) → 45.8 | 좌우로 비켜 선 주둥이에 aim 이 돈을 준 것과 같은 방향 (노트 원인 1) |
| diag_release_beads vs diag_unaimed_beads | 끝 0.0022 vs 0.0020 | 방출 거의 전부가 조준 창 밖. 노트: 152 알 중 151 알이 회전 급등(|ω| 중앙 65 rad/s) 사고 방출 |
| diag_state_income | 약 3/스텝, 안정 | 예산 규칙(상태 < 결과)은 그대로 성립 |

원인 (노트 확인 사항):

1. aim_lip·place_rcv 의 변화율이 `v_lip = _point_vel(..., lip_pt)` 였다. lip 은 "d_hat 쪽 림 점"이라는 기하 점이라 컵이 자기 축으로 돌면 제자리인데, 강체 점 속도 공식은 ω_축 × (lip − 원점) 을 접근 속도로 센다. 축 회전 부호와 좌우 치우침 부호가 맞는 배치에서 +0.196/스텝이 나오고, ep700 은 그 배치에 91 % 머물렀다.
2. pour_sink 는 높이 변화 없이 벌었다(축 회전 부호와 무관한 초과, [가설] 선속도 수직 성분 불일치).
3. pour_sink 의 켜짐 구역(소스 원점이 입구 위 1.7 r 안)은 70° 부근에서만 조준 창과 겹친다. 더 기울이면 창 바닥 아래로 가거나 구역을 떠나 수입을 잃으므로 70° 에서 멈췄다.

## 3. iter_23 변경 (노트의 3 개만)

1. **aim_lip·place_rcv 변화율 = 주둥이 기하의 정확한 미분.** `_tilt_open_potential` 이 이미 계산하던 `dlip_xy`·`dlip_dz` 를 돌려받아 쓴다. 이 값은 소스 입구(축 위의 점) 속도 + dθ/dt + d(d_hat)/dt 로 만들어지고 du = ω × up 이므로 자기 축 회전 기여는 정확히 0 이다. `v_lip`·`dv` 는 삭제했다. 무게·창·대칭 clamp(±1)·V_PROG 는 그대로다. aim 의 미분은 d|lip_xy|/dt·lip_xy + (above − below)·d(lip_dz)/dt 로 퍼텐셜 `d_aim` 의 정확한 시간 미분이다.
2. **pour_sink 를 합에서 뺀다.** 합산 항 26 → 25. 높이 하한은 조준 창 바닥(aim 의 `below`, 창의 `w_lo`)이 맡는다. 하강 비용만 남기는 안은 쓰지 않았다(비대칭 clamp 는 잡음을 순손실로, ctrl_gate 곱은 컵을 놓으면 0: i20 교훈). 관련 상수(W_POUR_SINK, V_Z_SCALE, SINK_KEEP_R)도 지웠고 `h_org` 는 기존 진단용으로만 남겼다.
3. **진단 3 개(합산 뒤, 보상 불변).** `diag_src_axial_w` = ω·up, `diag_src_n_contact`·`diag_rcv_n_contact` = 힘 > 1.0 N 인 손가락 수.

바꾸지 않은 것: 각도 처방 없음(목표·대역·상한 없음), G 제어·safety 의 값과 게이트 전부, 창의 r 배수(허가 1 ~ 3 r, 높이 바닥 0.7 r·천장 4 r, 배치 창), y ≈ 0 중심선, 예산 규칙, tilt_open·tilt_dir(0.3 / 1.5)·첫 비드 흘림 완화·배치 목표·결과 무게.

## 4. 다음 런에서 볼 것

- aim_lip·place_rcv 에피소드 합이 `diag_aim_dist_r`·`diag_place_dist_r` 스텝 차분 재계산과 일치하는가. 축 회전 초과는 없어져야 한다. 그래도 차이가 남으면 [가설] 선속도 수직 성분(원인 2 의 가설)이 aim 의 높이 부분(`below`/`above` 활성 시 `dlip_dz` 가 `src_cup_lin_vel` z 를 읽는다)과 place 의 `ez`(리시버 v_z)·`dclear` 에 남은 것이다.
- `diag_src_axial_w` 가 창 안에서 0 쪽으로 가는가, 회전 급등(|ω| > 5 rad/s) 스텝 수와 그때의 `diag_src_n_contact`.
- `ep/toward_max` 가 70° 정지를 벗어나는가(>85° 스텝 수), `ep/lipdz_at_max` > 0.7 r, `ep/dir_err_at_max` < 30°, 조용한 붓기 수와 적중률.
- 웜스타트는 노트대로 사용자 결정 사항이다(ep250 은 TaskSpec 예외 확인 필요, 없으면 rh2_i07-r3 ep400).

```python
import torch
import math


# =====================================================================================================
# iter_23 = iter_22 design + ONLY the three round-23 operator changes (09.26):
#   (1) aim_lip / place_rcv rates = the EXACT time derivative of the spout geometry. dlip_xy / dlip_dz come from
#       _tilt_open_potential (velocity of the source MOUTH, an on-axis point, + d theta/dt + d d_hat/dt, with
#       du = w x u), so spin of the cup about its own axis leaves the geometric lip where it is and pays exactly 0.
#       The body-point velocity v_lip = _point_vel(..., lip_pt) is no longer used (i22 ep700: aim paid 201 vs a
#       position difference of 92 per episode, the excess from axial spin with the spout offset sideways).
#       Weights, windows, the symmetric +-1 clamp and V_PROG unchanged.
#   (2) pour_sink removed from the sum (25 summed terms). It paid +153/episode at ep700 against +18 of real height
#       change, and its active zone (source origin < 1.7 r above the receiver rim) overlaps the aimed window only
#       near 70 deg, which parked the pour there. The window floor (aim `below`, window w_lo) keeps the spout high.
#   (3) diagnostics only, after the sum: diag_src_axial_w, diag_src_n_contact, diag_rcv_n_contact.
# Unchanged on purpose: grasp, lift, placement target, tilt_open, tilt_dir (0.3 / 1.5), first-bead spill relief,
# every outcome weight, every speed / control / safety penalty and its gate.
# =====================================================================================================


def _band(x, lo, hi, soft):
    """1 inside [lo, hi], Gaussian fall-off of width `soft` outside."""
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _smoothstep(x):
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_slope(x):
    """d smoothstep / dx (exactly 0 outside [0, 1])."""
    t = torch.clamp(x, 0.0, 1.0)
    return 6.0 * t * (1.0 - t)


def _late_ramp(x, lo, hi):
    """0.5*t + 0.5*t^2: never flat above lo, last part worth 3x the first."""
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _close_in(d):
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _progress(rate_closing, v_scale):
    """Difference-form payment from an exact potential rate, positive = getting closer.
    Symmetric clamp (iter_21): +-1 per step, reached at the stable-speed limit v_scale, so going faster than
    allowed buys nothing and exploration jitter nets zero (i20: an asymmetric clamp made jitter a net loss).
    Used by place_rcv / aim_lip only; tilt_open has its own asymmetric bound (iter_22 repair 1)."""
    return torch.clamp(rate_closing / v_scale, -1.0, 1.0)


def _limit_pen(margin, free):
    d = torch.clamp((free - torch.clamp(margin, 0.0, 1.0)) / free, 0.0, 1.0)
    worst = d.max(dim=-1).values
    return 0.7 * worst * worst + 0.3 * (d * d).mean(dim=-1)


def _shake_pen(qdd, free_mean, scale_mean, free_peak, scale_peak):
    q = qdd.abs()
    mean_part = torch.tanh(torch.clamp(q.mean(dim=-1) - free_mean, min=0.0) / scale_mean)
    peak_part = torch.tanh(torch.clamp(q.max(dim=-1).values - free_peak, min=0.0) / scale_peak)
    return 0.65 * mean_part + 0.35 * peak_part


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
    """Fingertip geometry in the CUP frame. VERBATIM since iter_04 (grasp 0.75/0.82 measured)."""
    ARRIVE_RISE = 0.045
    FUNNEL_R0 = 0.01
    FUNNEL_R1 = 0.04
    EPS = 1e-6
    grasp_h = 0.25 * cup_mouth_z
    thumb_hi = 0.5 * cup_mouth_z

    up = cup_up.unsqueeze(1)
    rel = tips_pos - cup_pos.unsqueeze(1)
    h_tip = (rel * up).sum(dim=-1)
    rad_vec = rel - h_tip.unsqueeze(-1) * up
    rad = rad_vec.norm(dim=-1)
    gap = torch.clamp(rad - tip_r, min=0.0)

    thumb_h = h_tip[:, 0]
    h_err_t = torch.clamp(-thumb_h, min=0.0) + torch.clamp(thumb_h - thumb_hi, min=0.0)
    d_thumb = torch.sqrt(gap[:, 0] ** 2 + h_err_t ** 2)
    h_oth = h_tip[:, 1:]
    h_err_o = torch.clamp(-0.4 * cup_mouth_z - h_oth, min=0.0) + torch.clamp(h_oth - 0.8 * cup_mouth_z, min=0.0)
    d_oth = torch.sqrt(gap[:, 1:] ** 2 + h_err_o ** 2).min(dim=1).values

    e_t = rad_vec[:, 0] / (rad[:, 0:1] + EPS)
    o_vec = rad_vec[:, 1:].mean(dim=1)
    e_o = o_vec / (o_vec.norm(dim=-1, keepdim=True) + EPS)
    opp = 0.5 * (1.0 - (e_t * e_o).sum(dim=-1))

    mid = 0.5 * (tips_pos[:, 0] + tips_pos[:, 1:].mean(dim=1))
    rel_m = mid - cup_pos
    h_m = (rel_m * cup_up).sum(dim=-1)
    rad_m = (rel_m - h_m.unsqueeze(-1) * cup_up).norm(dim=-1)
    rise = ARRIVE_RISE * torch.clamp((rad_m - FUNNEL_R0) / (FUNNEL_R1 - FUNNEL_R0), 0.0, 1.0)
    d_funnel = torch.sqrt(rad_m ** 2 + (h_m - grasp_h - rise) ** 2)
    d_grip = torch.sqrt(rad_m ** 2 + (h_m - grasp_h) ** 2)
    thumb_band = _band(thumb_h, 0.0, thumb_hi, 0.02)
    return {"d_thumb": d_thumb, "d_oth": d_oth, "opp": opp,
            "d_funnel": d_funnel, "d_grip": d_grip, "thumb_band": thumb_band}


def _hand_cup(tips_pos, finger_force, palm_force, hand_closure, grasped, arm_qd, cup_pos, cup_up,
              cup_lin_vel, cup_ang_vel, cup_spawn_pos, cup_mouth_z, tip_r, pose_ok):
    """Per-arm stage-A value in [0, 1] plus the table-disturbance penalties (iter_21 VERBATIM).
    ONE monotone ladder, grasped (>= 0.4) always above not-grasped (<= 0.39)."""
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.bool()
    g_f = g.to(cup_pos.dtype)
    f_t = finger_force[:, 0]
    f_o = finger_force[:, 1:].sum(dim=1)

    reach_far = 1.0 - torch.tanh(4.0 * geo["d_funnel"])
    side_t = torch.maximum(_close_in(geo["d_thumb"]), torch.tanh(f_t / 0.5))
    side_o = torch.maximum(_close_in(geo["d_oth"]), torch.tanh(f_o / 0.5))
    reach = 0.4 * reach_far + 0.6 * 0.5 * (side_t + side_o) * (0.3 + 0.7 * geo["opp"])

    near_grip = torch.exp(-(geo["d_grip"] / 0.03) ** 2)
    close = near_grip * torch.clamp(hand_closure / 0.7, max=1.0)
    pre_grasp = 0.15 * torch.tanh(f_o / 1.0) + 0.25 * torch.tanh(torch.minimum(f_t, f_o) / 0.5)
    pinch = torch.tanh(torch.minimum(f_t, f_o) / 1.5)
    contact_q = (0.4 + 0.6 * pinch) * (0.1 + 0.9 * geo["thumb_band"])
    balance = torch.minimum(f_t, f_o) / (torch.maximum(f_t, f_o) + 1e-6)
    keep = near_grip * (0.3 + 0.7 * balance)
    held_val = 0.4 + 0.3 * contact_q + 0.3 * keep                       # in [0.4, 1.0]
    free_val = 0.25 * reach + 0.10 * close + 0.10 * pre_grasp           # at most 0.39
    value = torch.where(g, held_val, free_val) * pose_ok

    h_pos = torch.clamp(cup_pos[:, 2] - cup_spawn_pos[:, 2] - 0.005, min=0.0)
    lifted_w = torch.clamp(h_pos / 0.02, max=1.0)
    v_lin = cup_lin_vel.norm(dim=-1)
    disp = (cup_pos[:, :2] - cup_spawn_pos[:, :2]).norm(dim=-1)
    v_xy = cup_lin_vel[:, :2].norm(dim=-1)
    slide = (1.0 - lifted_w) * (0.6 * torch.tanh(disp / 0.03) + 0.4 * torch.tanh(v_xy / 0.05))
    thumb_excess = torch.clamp(f_t - f_o - 1.0, min=0.0)
    finger_excess = torch.clamp(f_o - f_t, min=0.0)
    one_sided = (1.0 - lifted_w) * torch.tanh((thumb_excess + 0.5 * finger_excess) / 2.0)
    palm_push = torch.tanh(palm_force / 2.0)
    near = torch.exp(-(geo["d_grip"] / 0.10) ** 2)
    rush = near * (1.0 - g_f) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) \
        + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"value": value, "h": h_pos, "v_lin": v_lin, "slide": slide, "one_sided": one_sided,
            "palm_push": palm_push, "rush": rush, "fling": fling, "keep": keep, "balance": balance}


def _pour_geometry(ctx, r):
    """Directed-spout geometry, iter_08/09 VERBATIM.
      d_hat : horizontal unit vector source cup origin -> receiver mouth
      theta : SIGNED tilt toward the receiver (locates the lip; never a target)
      lip   : rim point the beads leave from; lip_xy / lip_dz relative to the receiver mouth.
    iter_23: lip_pt is still returned but no longer differentiated as a body point (change 1)."""
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    sep = v.norm(dim=-1)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / 0.03, max=1.0)          # azimuth undefined when coaxial (3 cm floor)
    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    cos_t = torch.cos(torch.clamp(theta_eff, 0.0, math.pi))
    lip_drop = r * sin_t + ctx.cup_mouth_z * (1.0 - cos_t)
    lip_xy_vec = ctx.src_cup_mouth_pos[:, :2] \
        + (r * torch.cos(theta)).unsqueeze(-1) * d_hat - ctx.rcv_cup_mouth_pos[:, :2]
    lip_xy = lip_xy_vec.norm(dim=-1)
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r * torch.sin(theta)
    lip_dz = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    lip_pt = torch.cat([ctx.rcv_cup_mouth_pos[:, :2] + lip_xy_vec, lip_z.unsqueeze(-1)], dim=-1)
    return {"d_hat": d_hat, "sep_ok": sep_ok, "theta": theta, "lip_drop": lip_drop,
            "lip_xy_vec": lip_xy_vec, "lip_xy": lip_xy, "lip_dz": lip_dz, "lip_pt": lip_pt}


def _point_vel(lin_vel, ang_vel, origin, point):
    return lin_vel + torch.cross(ang_vel, point - origin, dim=-1)


def _tilt_direction_error(ctx, d_hat):
    """iter_19 VERBATIM: psi = angle between the cup's lean (horizontal part of its up axis) and d_hat.
    dir_err = (1 - cos psi)/2 is 0 when leaning at the receiver at ANY tilt - no magnitude is valued."""
    u_xy = ctx.src_cup_up[:, :2]
    u_h = u_xy.norm(dim=-1)
    cos_psi = torch.clamp((u_xy * d_hat).sum(dim=-1) / torch.clamp(u_h, min=1e-6), -1.0, 1.0)
    return 0.5 * (1.0 - cos_psi), torch.acos(cos_psi)


def _spout_window(lip_xy, lip_dz, dlip_xy, dlip_dz, r_in, r_out, z_lo, z_hi, z_fade):
    """iter_22: spout window in [0, 1] and its exact time derivative (same state function both ways).
    horizontal: 1 inside r_in, smoothstep to 0 at r_out (iter_13 window, unchanged);
    height    : 0 -> 1 linearly over [0, z_lo] (iter_17 floor, unchanged), 1 up to the ceiling z_hi,
                smoothstep to 0 at z_hi + z_fade."""
    t_xy = (lip_xy - r_in) / (r_out - r_in)
    w_xy = 1.0 - _smoothstep(t_xy)
    dw_xy = -_smoothstep_slope(t_xy) * dlip_xy / (r_out - r_in)
    w_lo = torch.clamp(lip_dz / z_lo, 0.0, 1.0)
    dw_lo = ((lip_dz > 0.0) & (lip_dz < z_lo)).to(lip_dz.dtype) * dlip_dz / z_lo
    t_hi = (lip_dz - z_hi) / z_fade
    w_hi = 1.0 - _smoothstep(t_hi)
    dw_hi = -_smoothstep_slope(t_hi) * dlip_dz / z_fade
    w_z = w_lo * w_hi
    dw_z = dw_lo * w_hi + w_lo * dw_hi
    return {"w_xy": w_xy, "w_z": w_z, "win": w_xy * w_z, "dwin": dw_xy * w_z + w_xy * dw_z}


def _tilt_open_potential(ctx, geo, r, v_mouth, du, r_in, r_out, z_lo, z_hi, z_fade):
    """iter_22 repair 2: potential PHI = win * theta+ * max(cos psi, 0) and its EXACT time derivative.
    Every rate is the derivative of the very formula that gives the value: theta and psi from the cup up axis
    u and d_hat (d_hat turns when either cup moves), the lip point of _pour_geometry and the window of
    _spout_window. theta+ alone reaches pi/2 for a cup lying on its side at ANY azimuth within 90 deg of d_hat;
    the cos psi factor keeps only the part of the lean that points at the receiver. No angle target, no band.
    du = d(u)/dt = w x u (assumption (가설): ang_vel in the world frame, as _point_vel already assumes).
    iter_23 change 1: also returns dlip_xy = d|lip_xy|/dt and dlip_dz = d(lip_dz)/dt. They are built from the
    source MOUTH velocity (an on-axis point), dtheta and dd_hat only, so spin about the cup's own axis gives 0."""
    up = ctx.src_cup_up
    d_hat = geo["d_hat"]
    theta = geo["theta"]
    dtype = theta.dtype
    # d_hat = v / |v|, v = receiver mouth - source origin (horizontal)
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    sep = torch.clamp(v.norm(dim=-1), min=1e-4)
    dv = v_mouth[:, :2] - ctx.src_cup_lin_vel[:, :2]
    dd_hat = (dv - d_hat * (d_hat * dv).sum(dim=-1, keepdim=True)) / sep.unsqueeze(-1)
    # theta = atan2(a, b), a = u_xy . d_hat, b = u_z
    a = (up[:, :2] * d_hat).sum(dim=-1)
    da = (du[:, :2] * d_hat).sum(dim=-1) + (up[:, :2] * dd_hat).sum(dim=-1)
    b = up[:, 2]
    dtheta = (b * da - a * du[:, 2]) / torch.clamp(a * a + b * b, min=1e-3)
    # cos psi = a / |u_xy| (same formula as _tilt_direction_error)
    h = torch.clamp(up[:, :2].norm(dim=-1), min=1e-6)
    dh = (up[:, :2] * du[:, :2]).sum(dim=-1) / h
    cos_psi = torch.clamp(a / h, -1.0, 1.0)
    dcos = (da - cos_psi * dh) / h
    # lip: xy = src mouth + r cos(theta) d_hat - rcv mouth, z = src mouth z - r sin(theta) (see _pour_geometry)
    v_m = _point_vel(ctx.src_cup_lin_vel, ctx.src_cup_ang_vel, ctx.src_cup_pos, ctx.src_cup_mouth_pos)
    dlip_vec = v_m[:, :2] - (r * torch.sin(theta) * dtheta).unsqueeze(-1) * d_hat \
        + (r * torch.cos(theta)).unsqueeze(-1) * dd_hat - v_mouth[:, :2]
    dlip_xy = (geo["lip_xy_vec"] * dlip_vec).sum(dim=-1) / torch.clamp(geo["lip_xy"], min=1e-4)
    dlip_dz = v_m[:, 2] - r * torch.cos(theta) * dtheta - v_mouth[:, 2]
    wnd = _spout_window(geo["lip_xy"], geo["lip_dz"], dlip_xy, dlip_dz, r_in, r_out, z_lo, z_hi, z_fade)
    # PHI and dPHI/dt (product rule over the three factors)
    th_p = torch.clamp(theta, min=0.0)
    c_p = torch.clamp(cos_psi, min=0.0)
    dth_p = (theta > 0.0).to(dtype) * dtheta
    dc_p = (cos_psi > 0.0).to(dtype) * dcos
    win, dwin = wnd["win"], wnd["dwin"]
    phi = win * th_p * c_p
    phi_rate = dwin * th_p * c_p + win * dth_p * c_p + win * th_p * dc_p
    return {"phi": phi, "phi_rate": phi_rate, "dtheta": dtheta, "wnd": wnd,
            "dlip_xy": dlip_xy, "dlip_dz": dlip_dz}


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants and their sources
    W_ALIVE = 1.0            # constant survival floor, identical in every state (not A-D income)
    # A grasp (iter_21 verbatim)
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35       # rad, the ENV's own receiver-upright success limit (20 deg)
    # B lift (iter_21 verbatim; brief 5-1 B keeps the raised source allowance)
    W_LIFT_SRC = 1.5
    LIFTOFF_H = 0.03
    LIFT_TARGET_SRC = 0.30
    LIFT_FREE_SRC0 = 0.40
    LIFT_SIGMA_SRC = 0.08
    W_OVER_LIFT = 1.5
    OVER_LIFT_RANGE = 0.20
    W_LIFT_RCV = 0.5
    LIFT_TARGET_RCV = 0.08
    CARRY_H = 0.03
    # C placement (iter_22 verbatim): WORLD-FIXED height for the receiver mouth, horizontal vs the SPOUT
    RCV_POUR_LIFT = 0.15     # m, height allowance (the only absolute constant brief 5-3 permits). Mouth target
                             # = rim of a cup standing on the table + 0.15 m. With the spout window 0.7 r..4 r
                             # above that mouth, the source origin sits at lift ~0.23-0.34 m while pouring
                             # (70-90 deg), inside the lift_src paid range, so placement and lift_src do not fight.
    Z_TOL_R = 1.0            # +-1 r free band around the mouth height target
    PLACE_WIN_R = 1.0        # horizontal: receiver mouth pulled until the spout is within 1 r (= PERM_IN_R)
    BODY_CLEAR_R = 1.0       # until the source BOTTOM is BODY_CLEAR_R*r above the receiver RIM the pull stops at
                             # lip_xy = (PLACE_WIN_R + BODY_CLEAR_R) r = 2 r, so it never drags the receiver into
                             # the source body. The free radius shrinks linearly (slope 1) over that 1 r.
    PLACE_AZ = math.radians(68.0)       # K2 pourable side, kept as a GUARD only
    PLACE_AZ_TOL = math.radians(45.0)   # K1 palm-yaw box +-45 deg: 23..113 deg free, outside costs distance
    CENTER_FREE_R = 2.0      # |y| of the receiver mouth (= pour point) free up to 2 r (y ~= 0 centreline)
    W_PLACE = 1.0
    # D aim
    W_AIM = 1.5
    V_PROG = 0.15            # m/s: progress terms saturate at the stable cup speed limit (SRC_V_FREE)
    LIP_Z_HI_R = 0.7         # spout floor above the rim, 0.7 r ~= 2 cm (iter_17 permission height). With
                             # pour_sink gone (iter_23 change 2) this floor alone keeps the spout above the mouth.
    LIP_Z_MAX_R = 4.0        # spout CEILING 4 r ~= 11 cm above the rim (iter_22)
    LIP_Z_FADE_R = 2.0       # window fades out between 4 r and 6 r
    # E direction (iter_22 verbatim)
    W_TILT_DIR_OUT = 1.5     # outside the window: iter_19/21 strength
    W_TILT_DIR_IN = 0.3      # inside the window: a fifth, the spout is already over the mouth
    # tilt_open (iter_22 verbatim)
    W_TILT_OPEN = 0.1        # per step at most +0.1; whole-episode income bounded by
                             # W_TILT_OPEN * PHI_max / (TILT_RATE_SCALE * dt) ~= one bead (dt = 1/60 s, 가설)
    OPEN_BEADS = 1.0         # active until the first bead has LEFT the source
    OPEN_CHARGE_CAP = 100.0  # charges ungated, capped only at 100x the payment cap (physics glitches)
    # F outcome - unchanged weights (per-bead units with 20 beads)
    W_POUR = 150.0           # +7.5 per bead delivered
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0            # -2 per extra bead dumped in the same step
    W_SPILL = 240.0          # -12 per bead spilled
    SPILL_RELIEF = 0.25      # x0.25 (-3/bead) inside the AIMED window, first bead(s) only, at most one bead
    SETTLE_FRAC = 0.03
    W_UNAIMED = 60.0         # -3 per bead released outside the permission window
    W_TRANSFER = 8.0         # level term: 0.9 in, 0.05 spill -> 5.76/step (> A+B, brief 5-2 (3))
    W_TAIL = 4.0
    TAIL_LO, TAIL_HI = 0.10, 0.90
    EMPTY_LO = 0.20
    W_SUCCESS = 20.0
    SPILL_ZERO = 0.25        # env spill limit
    PERM_IN_R, PERM_OUT_R = 1.0, 3.0      # spout window in cup radii (iter_13, kept)
    LEAD_FULL_R, LEAD_ZERO_R = 1.75, 5.0  # pour-phase window for the stricter spin limit (kept)
    # G control - KEEP LIST values (unchanged, user principle: no speed / control relaxation)
    SRC_V_FREE, SRC_V_SCALE, W_SRC_SPEED = 0.15, 0.15, 0.5
    SRC_W_FREE, SRC_W_SCALE, W_SRC_SPIN = 1.5, 1.5, 0.5
    ROT_POUR_FREE, W_ROT_POUR = 1.0, 1.0
    RCV_W_FREE, RCV_W_SCALE, W_RCV_SPIN = 0.5, 1.0, 0.5
    STILL_FREE, STILL_SCALE, W_STILL_POUR = 0.05, 0.15, 0.5
    W_TILT_PEN_RCV = 0.3
    W_FLIP_PALM, W_FLIP_HAND = 0.3, 0.2
    W_RATE_PALM_SRC, W_RATE_PALM_RCV, W_RATE_HAND = 0.03, 0.05, 0.01
    W_ACT_RATE = 0.05
    BANG_DEAD, W_BANG_SRC, W_BANG_RCV = 0.1, 1.5, 2.0
    SAT_LO, W_PALM_SAT_SRC, W_PALM_SAT_RCV = 0.7, 3.0, 1.5
    MARGIN_FREE, W_ARM_LIMIT = 0.30, 0.4
    QDD_FREE, QDD_SCALE = 2.5, 10.3
    QDD_PEAK_FREE, QDD_PEAK_SCALE = 12.8, 12.8
    W_ARM_SHAKE = 0.4
    TILT_RATE_SCALE = ROT_POUR_FREE   # rad/s: tilt_open payment saturates at the pour-phase spin limit
    # safety / disturbance (unchanged)
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    # diagnostics only (iter_23 change 3)
    CONTACT_F = 1.0          # N, a finger counts as touching its cup above this force (user: log 5-finger grasp)

    dtype = ctx.src_cup_tilt.dtype
    r = ctx.cup_radius
    geo = _pour_geometry(ctx, r)
    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested

    # ------------------------------------------------------------------ A. grasp (iter_21 verbatim)
    upright_rcv = torch.exp(-(ctx.rcv_cup_tilt / UPRIGHT_TOL) ** 2)
    h_src_raw = torch.clamp(ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2] - 0.005, min=0.0)
    carry_src = g_src * torch.clamp(h_src_raw / CARRY_H, max=1.0)
    src_table_upright = torch.exp(-(ctx.src_cup_tilt / UPRIGHT_TOL) ** 2)
    pose_ok_src = carry_src + (1.0 - carry_src) * src_table_upright
    src = _hand_cup(ctx.src_tips_pos, ctx.src_finger_force, ctx.src_palm_force, ctx.src_hand_closure,
                    ctx.src_grasped, ctx.src_arm_qd, ctx.src_cup_pos, ctx.src_cup_up, ctx.src_cup_lin_vel,
                    ctx.src_cup_ang_vel, ctx.src_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, pose_ok_src)
    rcv = _hand_cup(ctx.rcv_tips_pos, ctx.rcv_finger_force, ctx.rcv_palm_force, ctx.rcv_hand_closure,
                    ctx.rcv_grasped, ctx.rcv_arm_qd, ctx.rcv_cup_pos, ctx.rcv_cup_up, ctx.rcv_cup_lin_vel,
                    ctx.rcv_cup_ang_vel, ctx.rcv_cup_spawn_pos, ctx.cup_mouth_z, TIP_R, upright_rcv)
    h_src, h_rcv = src["h"], rcv["h"]
    carry_rcv = g_rcv * torch.clamp(h_rcv / CARRY_H, max=1.0)
    carry_both = carry_src * carry_rcv * not_nested
    up_w_src = g_src * torch.clamp(h_src / 0.05, max=1.0)
    up_w_rcv = g_rcv * torch.clamp(h_rcv / 0.05, max=1.0)
    ctrl_gate = up_w_src * up_w_rcv

    # ------------------------------------------------------------------ B. lift (iter_21 verbatim)
    lift_free_src = LIFT_FREE_SRC0 + geo["lip_drop"]
    liftoff_src = 1.0 - torch.exp(-h_src / LIFTOFF_H)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / LIFT_SIGMA_SRC) ** 2)
    over_lift = torch.clamp((h_src - lift_free_src) / OVER_LIFT_RANGE, 0.0, 1.0)
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_band - W_OVER_LIFT * over_lift
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * (0.5 + 0.5 * upright_rcv)

    # ------------------------------------------------------------------ bead bookkeeping (gates)
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    has_beads = torch.clamp(fill / 0.10, max=1.0)        # >= 2 beads left
    stage_cd = carry_both * has_beads                     # placement / aim live while there is a pour ahead
    n_beads = float(max(int(ctx.num_beads), 1))
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    released = 1.0 - fill                                 # beads no longer in the source (receiver, air, spill)
    first_out = torch.clamp(OPEN_BEADS - n_beads * released, 0.0, 1.0)   # 1 until the first bead LEFT

    # receiver mouth velocity (an on-axis point, so its own axial spin adds nothing) and d(up)/dt of the source
    v_mouth = _point_vel(ctx.rcv_cup_lin_vel, ctx.rcv_cup_ang_vel, ctx.rcv_cup_pos, ctx.rcv_cup_mouth_pos)
    du_src = torch.cross(ctx.src_cup_ang_vel, ctx.src_cup_up, dim=-1)   # d(up)/dt (가설: world-frame ang_vel)

    # ------------------------------------------------------------------ spout window (a GATE, never a state income)
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    z_lo, z_hi = LIP_Z_HI_R * r, LIP_Z_MAX_R * r
    opn = _tilt_open_potential(ctx, geo, r, v_mouth, du_src, r_in, r_out, z_lo, z_hi, LIP_Z_FADE_R * r)
    # iter_23 change 1: the spout rates used by place / aim are the exact derivatives of the lip GEOMETRY
    # (the same ones the window and tilt_open already use). Axial spin of the source cup contributes exactly 0.
    dlip_xy = opn["dlip_xy"]                              # d|lip_xy|/dt, relative to the receiver mouth
    dlip_dz = opn["dlip_dz"]                              # d(lip_dz)/dt
    wnd = opn["wnd"]
    win = wnd["win"]                                      # position window (horizontal x height)
    perm_xy = wnd["w_xy"]
    toward = (geo["theta"] > 0.0).to(dtype)
    perm = win * toward                                   # aimed window: position + leaning toward
    lead_gate = 1.0 - _smoothstep((geo["lip_xy"] - LEAD_FULL_R * r) / ((LEAD_ZERO_R - LEAD_FULL_R) * r))

    # ------------------------------------------------------------------ C. placement (iter_22 target, iter_23 rate)
    # Receiver mouth: world-fixed height (not tied to the source), horizontally until the SPOUT is within
    # PLACE_WIN_R*r, |y| near the centreline, and not outside the pourable side of the source (guard).
    # The horizontal pull reaches PLACE_WIN_R*r only once the source BOTTOM centre is BODY_CLEAR_R*r above the
    # receiver RIM centre. Potential = norm of the excesses; paid as its exact rate -> 0 when parked.
    z_mouth_tgt = ctx.table_z - ctx.cup_bottom_z + ctx.cup_mouth_z + RCV_POUR_LIFT
    dz_m = ctx.rcv_cup_mouth_pos[:, 2] - z_mouth_tgt
    ez = torch.sign(dz_m) * torch.clamp(dz_m.abs() - Z_TOL_R * r, min=0.0)
    z_bot = ctx.src_cup_pos[:, 2] + ctx.cup_bottom_z * ctx.src_cup_up[:, 2]      # source bottom centre
    clear = z_bot - ctx.rcv_cup_mouth_pos[:, 2]                                   # height above the receiver rim
    dclear = ctx.src_cup_lin_vel[:, 2] + ctx.cup_bottom_z * du_src[:, 2] - v_mouth[:, 2]
    ramp = BODY_CLEAR_R * r
    r_free = PLACE_WIN_R * r + ramp - torch.clamp(clear, 0.0, ramp)
    dr_free = -((clear > 0.0) & (clear < ramp)).to(dtype) * dclear
    eh = torch.clamp(geo["lip_xy"] - r_free, min=0.0)
    y_m = ctx.rcv_cup_mouth_pos[:, 1]
    ey = torch.clamp(y_m.abs() - CENTER_FREE_R * r, min=0.0)
    rel = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    drel = v_mouth[:, :2] - ctx.src_cup_lin_vel[:, :2]
    a_lo, a_hi = PLACE_AZ - PLACE_AZ_TOL, PLACE_AZ + PLACE_AZ_TOL
    n_lo_x, n_lo_y = math.sin(a_lo), -math.cos(a_lo)      # outward normal of the 23 deg edge
    n_hi_x, n_hi_y = -math.sin(a_hi), math.cos(a_hi)      # outward normal of the 113 deg edge
    es_lo = torch.clamp(rel[:, 0] * n_lo_x + rel[:, 1] * n_lo_y, min=0.0)
    es_hi = torch.clamp(rel[:, 0] * n_hi_x + rel[:, 1] * n_hi_y, min=0.0)
    des_lo = drel[:, 0] * n_lo_x + drel[:, 1] * n_lo_y
    des_hi = drel[:, 0] * n_hi_x + drel[:, 1] * n_hi_y
    d_place = torch.sqrt(eh ** 2 + ez ** 2 + ey ** 2 + es_lo ** 2 + es_hi ** 2)
    dd_place = (eh * (dlip_xy - dr_free) + ez * v_mouth[:, 2] + ey * torch.sign(y_m) * v_mouth[:, 1]
                + es_lo * des_lo + es_hi * des_hi) / torch.clamp(d_place, min=1e-4)
    place = stage_cd * _progress(-dd_place, V_PROG)

    # ------------------------------------------------------------------ D. aim (3D, difference form)
    # Potential: horizontal spout->mouth distance + how far the spout is BELOW its floor + how far it is ABOVE
    # its ceiling. iter_23 change 1: its rate is the exact derivative of that potential,
    #   d d_aim/dt = (lip_xy * d|lip_xy|/dt + (above - below) * d(lip_dz)/dt) / d_aim,
    # so spinning the cup about its own axis (the lip stays put) pays nothing.
    below = torch.clamp(z_lo - geo["lip_dz"], min=0.0)
    above = torch.clamp(geo["lip_dz"] - z_hi, min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2 + above ** 2)
    dd_aim = (geo["lip_xy"] * dlip_xy + (above - below) * dlip_dz) / torch.clamp(d_aim, min=1e-4)
    aim = stage_cd * _progress(-dd_aim, V_PROG)

    # ------------------------------------------------------------------ tilt_open (iter_22 verbatim)
    # PHI = win * theta+ * max(cos psi, 0); phi_rate is its exact derivative. Per step:
    #   first_out * ( carry_both * sep_ok * clamp(rate/scale, 0, +1) + clamp(rate/scale, -OPEN_CHARGE_CAP, 0) )
    # Payments <= rate and charges = rate, so every closed cycle sums to <= 0 while first_out is constant.
    rate_n = opn["phi_rate"] / TILT_RATE_SCALE
    open_pay = carry_both * geo["sep_ok"] * torch.clamp(rate_n, 0.0, 1.0)
    open_charge = torch.clamp(rate_n, -OPEN_CHARGE_CAP, 0.0)
    tilt_open = first_out * (open_pay + open_charge)

    # ------------------------------------------------------------------ E. direction (cost only, iter_22 verbatim)
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)          # magnitude only as a multiplier, no target
    dir_w = W_TILT_DIR_IN * win + W_TILT_DIR_OUT * (1.0 - win)   # strong only outside the window
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err * dir_w

    # ------------------------------------------------------------------ H. removed (iter_23 change 2)
    # pour_sink is no longer summed: it paid for vertical velocity readings without height change and its zone
    # parked the pour near 70 deg. The spout floor of the aim potential / window keeps the height instead.
    h_org = ctx.src_cup_pos[:, 2] - ctx.rcv_cup_mouth_pos[:, 2]   # kept for the existing diagnostic only

    # ------------------------------------------------------------------ F. outcome (the beads, iter_22 verbatim)
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    # A spilled bead has necessarily LEFT the source; the relief gate uses the count WITHOUT this step's spill and
    # relieves at most one bead's worth: x0.25 (-3 instead of -12, still a loss vs +7.5 delivered).
    spill_now = torch.clamp(ctx.d_spill, min=0.0)
    first_spill = torch.clamp(OPEN_BEADS - n_beads * (released - spill_now), 0.0, 1.0)
    relieved = perm * first_spill * torch.clamp(spill_now, max=OPEN_BEADS / n_beads)
    spill_step = settled * (spill_now - (1.0 - SPILL_RELIEF) * relieved)
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)
    emptied = 1.0 - fill
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)
    fill_ok = torch.clamp(in_t / TAIL_LO, max=1.0)       # emptying onto the table pays nothing
    finish_q = 0.5 * (fill_tail + empty_q * fill_ok)
    release_step = torch.clamp(d_in, min=0.0) + torch.clamp(ctx.d_spill, min=0.0)
    unaimed = settled * release_step * (1.0 - perm)
    out_gate = not_nested * upright_rcv * clean_f

    # ------------------------------------------------------------------ G. control (iter_21 verbatim)
    v_src = src["v_lin"]
    w_src = ctx.src_cup_ang_vel.norm(dim=-1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=-1)
    pour_phase = ctrl_gate * lead_gate * perm_xy
    src_speed = ctrl_gate * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = W_SRC_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE) \
        + W_ROT_POUR * pour_phase * torch.tanh(torch.clamp(w_src - ROT_POUR_FREE, min=0.0) / SRC_W_SCALE)
    rcv_still_over = torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    rcv_steady = W_TILT_PEN_RCV * torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL) \
        + W_RCV_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE) \
        + W_STILL_POUR * ctrl_gate * perm_xy * rcv_still_over

    a_abs = ctx.actions.abs()
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), min=0.0)
    flip_palm_src = flips[:, 0:6].sum(dim=-1)
    flip_palm_rcv = flips[:, 12:18].sum(dim=-1)
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    act_flip = W_FLIP_PALM * (flip_palm_src + flip_palm_rcv) \
        + W_FLIP_HAND * flip_hand
    both_big = (a_abs > BANG_DEAD) & (ctx.prev_actions.abs() > BANG_DEAD)
    bang = (both_big & (ctx.actions * ctx.prev_actions < 0.0)).to(dtype)
    bang_src = bang[:, 0:6].mean(dim=-1)
    bang_rcv = bang[:, 12:18].mean(dim=-1)
    palm_bang = W_BANG_SRC * bang_src + W_BANG_RCV * bang_rcv
    sat_src = torch.mean(_smoothstep((a_abs[:, 0:6] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    sat_rcv = torch.mean(_smoothstep((a_abs[:, 12:18] - SAT_LO) / (1.0 - SAT_LO)), dim=-1)
    palm_sat = W_PALM_SAT_SRC * sat_src + W_PALM_SAT_RCV * sat_rcv
    cmd_rate = W_RATE_PALM_SRC * (da[:, 0:6] ** 2).sum(dim=-1) \
        + W_RATE_PALM_RCV * (da[:, 12:18] ** 2).sum(dim=-1) \
        + W_RATE_HAND * ((da[:, 6:12] ** 2).sum(dim=-1) + (da[:, 18:24] ** 2).sum(dim=-1)) \
        + W_ACT_RATE * (da ** 2).mean(dim=-1)
    arm_posture = W_ARM_LIMIT * (_limit_pen(ctx.src_arm_limit_margin, MARGIN_FREE)
                                 + _limit_pen(ctx.rcv_arm_limit_margin, MARGIN_FREE)) \
        + W_ARM_SHAKE * (_shake_pen(ctx.src_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE)
                         + _shake_pen(ctx.rcv_arm_qdd, QDD_FREE, QDD_SCALE, QDD_PEAK_FREE, QDD_PEAK_SCALE))

    # ------------------------------------------------------------------ safety / disturbance (verbatim)
    cup_disturb = W_SLIDE_SRC * src["slide"] + W_SLIDE_RCV * rcv["slide"] \
        + W_ONE_SIDED_SRC * src["one_sided"] + W_ONE_SIDED_RCV * rcv["one_sided"] \
        + W_PALM_SRC * src["palm_push"] + W_PALM_RCV * rcv["palm_push"] \
        + W_RUSH_SRC * src["rush"] + W_RUSH_RCV * rcv["rush"] \
        + W_FLING * (src["fling"] + rcv["fling"])
    safety = W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0) \
        + W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)) \
        + W_NESTED * nested

    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        # A
        "grasp_src": W_GRASP_SRC * src["value"],
        "grasp_rcv": W_GRASP_RCV * rcv["value"],
        # B
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        # C / D / E / open: difference forms and a cost - none of them pays for standing still or for axial spin
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -tilt_dir,
        "tilt_open": W_TILT_OPEN * tilt_open,
        # F
        "pour_delta": W_POUR * pour_paid - W_FLOW * flow_excess,
        "transferred": W_TRANSFER * in_t * out_gate,
        "transfer_tail": W_TAIL * finish_q * out_gate,
        "spill_delta": -W_SPILL * spill_step,
        "unaimed_release": -W_UNAIMED * unaimed,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype) * clean_f,
        # G
        "src_speed": -W_SRC_SPEED * src_speed,
        "src_spin": -src_spin,
        "rcv_steady": -rcv_steady,
        "act_flip": -act_flip,
        "palm_bang": -palm_bang,
        "palm_sat": -palm_sat,
        "cmd_rate": -cmd_rate,
        "arm_posture": -arm_posture,
        # safety
        "cup_disturb": -cup_disturb,
        "safety": -safety,
    }
    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value

    # diagnostics only (added AFTER the sum)
    comps["diag_place_dist_r"] = d_place / r
    comps["diag_place_az_deg"] = torch.atan2(rel[:, 1], rel[:, 0]) * (180.0 / math.pi)
    comps["diag_side_excess_r"] = (es_lo + es_hi) / r
    comps["diag_rcv_mouth_dz_r"] = dz_m / r
    comps["diag_rcv_mouth_y"] = y_m
    comps["diag_body_clear_r"] = clear / r
    comps["diag_place_free_r"] = r_free / r
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_lip_dz_r"] = geo["lip_dz"] / r
    comps["diag_h_org"] = h_org
    comps["diag_win"] = win
    comps["diag_perm"] = perm
    comps["diag_toward"] = toward
    comps["diag_theta_rate"] = opn["dtheta"]
    comps["diag_open_phi"] = opn["phi"]
    comps["diag_open_rate"] = opn["phi_rate"]
    comps["diag_first_out"] = first_out
    comps["diag_first_spill"] = first_spill
    comps["diag_relieved_beads"] = settled * relieved * n_beads
    comps["diag_dir_err_deg"] = psi * (180.0 / math.pi)
    comps["diag_src_tilt_deg"] = ctx.src_cup_tilt * (180.0 / math.pi)
    comps["diag_lift_budget"] = lift_free_src
    comps["diag_ctrl_gate"] = ctrl_gate
    comps["diag_stage_cd"] = stage_cd
    comps["diag_release_beads"] = release_step * n_beads
    comps["diag_unaimed_beads"] = unaimed * n_beads
    comps["diag_clean"] = clean_f
    comps["diag_finish_q"] = finish_q
    comps["diag_grip_keep_src"] = src["keep"]
    comps["diag_balance_src"] = src["balance"]
    comps["diag_palm_sat_src"] = sat_src
    comps["diag_src_w"] = w_src
    comps["diag_src_v"] = v_src
    comps["diag_src_margin_min"] = torch.clamp(ctx.src_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_state_income"] = comps["grasp_src"] + comps["grasp_rcv"] + comps["lift_src"] + comps["lift_rcv"]
    comps["diag_outcome_income"] = comps["transferred"] + comps["transfer_tail"] + comps["success_bonus"]
    # iter_23 change 3: axial spin reading of the source cup and the number of fingers touching each own cup
    comps["diag_src_axial_w"] = (ctx.src_cup_ang_vel * ctx.src_cup_up).sum(dim=-1)
    comps["diag_src_n_contact"] = (ctx.src_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    comps["diag_rcv_n_contact"] = (ctx.rcv_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    return reward, comps
```
