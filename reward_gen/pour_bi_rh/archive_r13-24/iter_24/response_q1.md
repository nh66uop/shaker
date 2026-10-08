

```python
import torch
import math


# =====================================================================================================
# iter_25 design: Pure-function rewrite addressing iter_23/24 feedback.
# Key changes:
#   (1) Grasp value now rewards finger contact count (n_on > 0) to prevent 3-finger drops during tilt.
#   (2) tilt_open weight raised to 0.5; uses state potential PHI directly (no velocity-derived rates).
#   (3) aim_lip / place_rcv switched from rate-form to bounded state potentials to eliminate noisy
#       velocity/acceleration taxes and false progress signals.
#   (4) Budget rule enforced: state-income (A-D) capped below outcome-income (F).
#   (5) Strictly pure: no globals, no caches, no state between calls.
# =====================================================================================================

def _band(x, lo, hi, soft):
    outside = torch.clamp(lo - x, min=0.0) + torch.clamp(x - hi, min=0.0)
    return torch.exp(-(outside / soft) ** 2)


def _smoothstep(x):
    t = torch.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_slope(x):
    t = torch.clamp(x, 0.0, 1.0)
    return 6.0 * t * (1.0 - t)


def _close_in(d):
    return 0.5 * torch.exp(-30.0 * d) + 0.5 * torch.exp(-160.0 * d)


def _limit_pen(margin, free):
    d = torch.clamp((free - torch.clamp(margin, 0.0, 1.0)) / free, 0.0, 1.0)
    worst = d.max(dim=-1).values
    return 0.7 * worst * worst + 0.3 * (d * d).mean(dim=-1)


def _shake_pen(qdd, free, scale):
    q = qdd.abs()
    mean_part = torch.tanh(torch.clamp(q.mean(dim=-1) - free, min=0.0) / scale)
    peak_part = torch.tanh(torch.clamp(q.max(dim=-1).values - free, min=0.0) / scale)
    return 0.65 * mean_part + 0.35 * peak_part


def _late_ramp(x, lo, hi):
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 * t + 0.5 * t * t


def _point_vel(lin_vel, ang_vel, origin, point):
    return lin_vel + torch.cross(ang_vel, point - origin, dim=-1)


def _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r):
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
    geo = _grip_geometry(tips_pos, cup_pos, cup_up, cup_mouth_z, tip_r)
    g = grasped.to(tips_pos.dtype)
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

    # iter_25 change 1: reward finger contact count to prevent 3-finger drops during tilt
    n_on = (finger_force[:, 1:] > 1.0).sum(dim=-1)  # fingers (excl thumb) touching > 1N
    held_val = 0.4 + 0.2 * contact_q + 0.2 * keep + 0.2 * (n_on / 4.0)  # max 1.0
    free_val = 0.25 * reach + 0.10 * close + 0.10 * pre_grasp  # max ~0.39
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
    rush = near * (1.0 - g) * torch.tanh(arm_qd.norm(dim=-1) / 0.5)
    w_ang = cup_ang_vel.norm(dim=-1)
    fling = 0.5 * torch.tanh(torch.clamp(v_lin - 0.5, min=0.0) / 0.5) + 0.5 * torch.tanh(torch.clamp(w_ang - 4.0, min=0.0) / 4.0)
    return {"value": value, "h": h_pos, "v_lin": v_lin, "slide": slide, "one_sided": one_sided,
            "palm_push": palm_push, "rush": rush, "fling": fling, "keep": keep, "balance": balance}


def _pour_geometry(ctx, r):
    v = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    sep = v.norm(dim=-1)
    d_hat = v / torch.clamp(sep, min=1e-6).unsqueeze(-1)
    sep_ok = torch.clamp(sep / 0.03, max=1.0)
    u_d = (ctx.src_cup_up[:, :2] * d_hat).sum(dim=-1)
    theta = torch.atan2(u_d, ctx.src_cup_up[:, 2])
    theta_eff = torch.minimum(torch.clamp(theta, min=0.0), ctx.src_cup_tilt)
    sin_t = torch.sin(torch.clamp(theta_eff, 0.0, 0.5 * math.pi))
    cos_t = torch.cos(torch.clamp(theta_eff, 0.0, math.pi))
    lip_drop = r * sin_t + ctx.cup_mouth_z * (1.0 - cos_t)
    lip_xy_vec = ctx.src_cup_mouth_pos[:, :2] + (r * torch.cos(theta)).unsqueeze(-1) * d_hat - ctx.rcv_cup_mouth_pos[:, :2]
    lip_xy = lip_xy_vec.norm(dim=-1)
    lip_z = ctx.src_cup_mouth_pos[:, 2] - r * torch.sin(theta)
    lip_dz = lip_z - ctx.rcv_cup_mouth_pos[:, 2]
    return {"d_hat": d_hat, "sep_ok": sep_ok, "theta": theta, "lip_drop": lip_drop,
            "lip_xy_vec": lip_xy_vec, "lip_xy": lip_xy, "lip_dz": lip_dz}


def _tilt_direction_error(ctx, d_hat):
    u_xy = ctx.src_cup_up[:, :2]
    u_h = u_xy.norm(dim=-1)
    cos_psi = torch.clamp((u_xy * d_hat).sum(dim=-1) / torch.clamp(u_h, min=1e-6), -1.0, 1.0)
    return 0.5 * (1.0 - cos_psi), torch.acos(cos_psi)


def _spout_window(lip_xy, lip_dz, r_in, r_out, z_lo, z_hi, z_fade):
    t_xy = (lip_xy - r_in) / (r_out - r_in)
    w_xy = 1.0 - _smoothstep(t_xy)
    w_lo = torch.clamp(lip_dz / z_lo, 0.0, 1.0)
    t_hi = (lip_dz - z_hi) / z_fade
    w_hi = 1.0 - _smoothstep(t_hi)
    w_z = w_lo * w_hi
    win = w_xy * w_z
    return {"w_xy": w_xy, "w_z": w_z, "win": win}


def _tilt_open_potential(ctx, geo, r, r_in, r_out, z_lo, z_hi, z_fade):
    up = ctx.src_cup_up
    d_hat = geo["d_hat"]
    theta = geo["theta"]
    dtype = theta.dtype
    h = torch.clamp(up[:, :2].norm(dim=-1), min=1e-6)
    a = (up[:, :2] * d_hat).sum(dim=-1)
    cos_psi = torch.clamp(a / h, -1.0, 1.0)
    wnd = _spout_window(geo["lip_xy"], geo["lip_dz"], r_in, r_out, z_lo, z_hi, z_fade)
    win = wnd["win"]
    th_p = torch.clamp(theta, min=0.0)
    c_p = torch.clamp(cos_psi, min=0.0)
    phi = win * th_p * c_p
    return {"phi": phi, "wnd": wnd, "dtheta": 0.0}  # rate omitted per iter_25 pure-function constraint


def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ------------------------------------------------------------------ constants
    W_ALIVE = 1.0
    W_GRASP_SRC, W_GRASP_RCV = 1.5, 1.5
    TIP_R = 0.027
    UPRIGHT_TOL = 0.35  # rad
    W_LIFT_SRC, W_LIFT_RCV = 1.5, 0.5
    LIFT_TARGET_SRC, LIFT_TARGET_RCV = 0.30, 0.08
    CARRY_H = 0.03
    W_PLACE, W_AIM = 1.0, 1.5
    W_TILT_DIR_OUT, W_TILT_DIR_IN = 1.5, 0.3
    W_TILT_OPEN = 0.5  # iter_25: increased to give meaningful tilt signal without rate taxes
    OPEN_BEADS = 1.0
    W_POUR = 150.0
    FLOW_CAP_BEADS = 1.0
    W_FLOW = 40.0
    W_SPILL = 240.0
    SPILL_RELIEF = 0.25
    SETTLE_FRAC = 0.03
    W_UNAIMED = 60.0
    W_TRANSFER = 8.0
    W_TAIL = 4.0
    TAIL_LO, TAIL_HI = 0.10, 0.90
    EMPTY_LO = 0.20
    W_SUCCESS = 20.0
    SPILL_ZERO = 0.25
    PERM_IN_R, PERM_OUT_R = 1.0, 3.0
    LIP_Z_HI_R, LIP_Z_MAX_R, LIP_Z_FADE_R = 0.7, 4.0, 2.0
    # Control
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
    W_SLIDE_SRC, W_SLIDE_RCV = 0.5, 0.25
    W_ONE_SIDED_SRC, W_ONE_SIDED_RCV = 0.3, 0.1
    W_PALM_SRC, W_PALM_RCV = 0.2, 0.1
    W_RUSH_SRC, W_RUSH_RCV = 0.2, 0.05
    W_FLING = 0.3
    W_CUP_CUP, W_FOREIGN, W_NESTED = 1.0, 0.5, 2.0
    CONTACT_F = 1.0

    dtype = ctx.src_cup_tilt.dtype
    r = ctx.cup_radius
    geo = _pour_geometry(ctx, r)
    g_src = ctx.src_grasped.to(dtype)
    g_rcv = ctx.rcv_grasped.to(dtype)
    nested = ctx.cups_nested.to(dtype)
    not_nested = 1.0 - nested

    # ------------------------------------------------------------------ A. grasp
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

    # ------------------------------------------------------------------ B. lift
    lift_free_src = 0.40 + geo["lip_drop"]
    liftoff_src = 1.0 - torch.exp(-h_src / 0.03)
    lift_shape_src = 0.5 * liftoff_src + 0.5 * torch.clamp(h_src / LIFT_TARGET_SRC, max=1.0)
    src_band = torch.exp(-(torch.clamp(h_src - lift_free_src, min=0.0) / 0.08) ** 2)
    over_lift = torch.clamp((h_src - lift_free_src) / 0.20, 0.0, 1.0)
    lift_src = W_LIFT_SRC * g_src * lift_shape_src * src_band - 1.5 * over_lift
    lift_shape_rcv = 0.5 * torch.tanh(h_rcv / 0.02) + 0.5 * torch.clamp(h_rcv / LIFT_TARGET_RCV, max=1.0)
    lift_rcv = W_LIFT_RCV * g_rcv * lift_shape_rcv * (0.5 + 0.5 * upright_rcv)

    # ------------------------------------------------------------------ bead gates
    fill = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    has_beads = torch.clamp(fill / 0.10, max=1.0)
    stage_cd = carry_both * has_beads
    n_beads = float(max(int(ctx.num_beads), 1))
    in_t = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    released = 1.0 - fill
    first_out = torch.clamp(OPEN_BEADS - n_beads * released, 0.0, 1.0)

    # ------------------------------------------------------------------ C. placement (state potential, bounded)
    RCV_POUR_LIFT = 0.15
    Z_TOL_R = 1.0
    PLACE_WIN_R = 1.0
    BODY_CLEAR_R = 1.0
    CENTER_FREE_R = 2.0
    PLACE_AZ = math.radians(68.0)
    PLACE_AZ_TOL = math.radians(45.0)
    z_mouth_tgt = ctx.table_z - ctx.cup_bottom_z + ctx.cup_mouth_z + RCV_POUR_LIFT
    dz_m = ctx.rcv_cup_mouth_pos[:, 2] - z_mouth_tgt
    ez = torch.sign(dz_m) * torch.clamp(dz_m.abs() - Z_TOL_R * r, min=0.0)
    z_bot = ctx.src_cup_pos[:, 2] + ctx.cup_bottom_z * ctx.src_cup_up[:, 2]
    clear = z_bot - ctx.rcv_cup_mouth_pos[:, 2]
    ramp = BODY_CLEAR_R * r
    r_free = PLACE_WIN_R * r + ramp - torch.clamp(clear, 0.0, ramp)
    eh = torch.clamp(geo["lip_xy"] - r_free, min=0.0)
    y_m = ctx.rcv_cup_mouth_pos[:, 1]
    ey = torch.clamp(y_m.abs() - CENTER_FREE_R * r, min=0.0)
    rel = ctx.rcv_cup_mouth_pos[:, :2] - ctx.src_cup_pos[:, :2]
    a_lo, a_hi = PLACE_AZ - PLACE_AZ_TOL, PLACE_AZ + PLACE_AZ_TOL
    n_lo_x, n_lo_y = math.sin(a_lo), -math.cos(a_lo)
    n_hi_x, n_hi_y = -math.sin(a_hi), math.cos(a_hi)
    es_lo = torch.clamp(rel[:, 0] * n_lo_x + rel[:, 1] * n_lo_y, min=0.0)
    es_hi = torch.clamp(rel[:, 0] * n_hi_x + rel[:, 1] * n_hi_y, min=0.0)
    d_place = torch.sqrt(eh ** 2 + ez ** 2 + ey ** 2 + es_lo ** 2 + es_hi ** 2)
    place = stage_cd * torch.exp(-2.0 * d_place / r)  # bounded state potential

    # ------------------------------------------------------------------ D. aim (state potential)
    r_in, r_out = PERM_IN_R * r, PERM_OUT_R * r
    z_lo, z_hi = LIP_Z_HI_R * r, LIP_Z_MAX_R * r
    below = torch.clamp(z_lo - geo["lip_dz"], min=0.0)
    above = torch.clamp(geo["lip_dz"] - z_hi, min=0.0)
    d_aim = torch.sqrt(geo["lip_xy"] ** 2 + below ** 2 + above ** 2)
    aim = stage_cd * torch.exp(-3.0 * d_aim / r)  # bounded state potential

    # ------------------------------------------------------------------ E. tilt_open & direction
    opn = _tilt_open_potential(ctx, geo, r, r_in, r_out, z_lo, z_hi, LIP_Z_FADE_R * r)
    win = opn["wnd"]["win"]
    tilt_open_val = carry_both * geo["sep_ok"] * opn["phi"]  # bounded potential, no rate taxes
    dir_err, psi = _tilt_direction_error(ctx, geo["d_hat"])
    tilt_n = ctx.src_cup_tilt / (0.5 * math.pi)
    dir_w = W_TILT_DIR_IN * win + W_TILT_DIR_OUT * (1.0 - win)
    tilt_dir = carry_both * geo["sep_ok"] * tilt_n * dir_err * dir_w

    # ------------------------------------------------------------------ F. outcome
    flow_cap = FLOW_CAP_BEADS / n_beads
    d_in = ctx.d_in_target
    pour_paid = torch.clamp(d_in, -1.0, flow_cap)
    flow_excess = torch.clamp(d_in - flow_cap, min=0.0)
    settled = _smoothstep(ctx.episode_progress / SETTLE_FRAC)
    spill_now = torch.clamp(ctx.d_spill, min=0.0)
    first_spill = torch.clamp(OPEN_BEADS - n_beads * (released - spill_now), 0.0, 1.0)
    relieved = win * first_spill * torch.clamp(spill_now, max=OPEN_BEADS / n_beads)
    spill_step = settled * (spill_now - (1.0 - SPILL_RELIEF) * relieved)
    spill_lvl = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)
    clean_f = torch.clamp(1.0 - spill_lvl / SPILL_ZERO, 0.0, 1.0)
    out_gate = not_nested * upright_rcv * clean_f
    emptied = 1.0 - fill
    fill_tail = _late_ramp(in_t, TAIL_LO, TAIL_HI)
    empty_q = _late_ramp(emptied, EMPTY_LO, 1.0)
    fill_ok = torch.clamp(in_t / TAIL_LO, max=1.0)
    finish_q = 0.5 * (fill_tail + empty_q * fill_ok)
    release_step = torch.clamp(d_in, min=0.0) + torch.clamp(ctx.d_spill, min=0.0)
    unaimed = settled * release_step * (1.0 - win)

    # ------------------------------------------------------------------ G. control
    v_src = src["v_lin"]
    w_src = ctx.src_cup_ang_vel.norm(dim=-1)
    w_rcv = ctx.rcv_cup_ang_vel.norm(dim=-1)
    pour_phase = ctrl_gate * win * torch.clamp(geo["lip_xy"] / r, max=1.0)
    src_speed = ctrl_gate * torch.tanh(torch.clamp(v_src - SRC_V_FREE, min=0.0) / SRC_V_SCALE)
    src_spin = W_SRC_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_src - SRC_W_FREE, min=0.0) / SRC_W_SCALE) \
        + W_ROT_POUR * pour_phase * torch.tanh(torch.clamp(w_src - ROT_POUR_FREE, min=0.0) / SRC_W_SCALE)
    rcv_still_over = torch.tanh(torch.clamp(rcv["v_lin"] - STILL_FREE, min=0.0) / STILL_SCALE)
    rcv_steady = W_TILT_PEN_RCV * torch.tanh(ctx.rcv_cup_tilt / UPRIGHT_TOL) \
        + W_RCV_SPIN * ctrl_gate * torch.tanh(torch.clamp(w_rcv - RCV_W_FREE, min=0.0) / RCV_W_SCALE) \
        + W_STILL_POUR * ctrl_gate * win * rcv_still_over

    a_abs = ctx.actions.abs()
    da = ctx.actions - ctx.prev_actions
    flips = torch.clamp(-(ctx.actions * ctx.prev_actions), min=0.0)
    flip_palm_src = flips[:, 0:6].sum(dim=-1)
    flip_palm_rcv = flips[:, 12:18].sum(dim=-1)
    flip_hand = flips[:, 6:12].sum(dim=-1) + flips[:, 18:24].sum(dim=-1)
    act_flip = W_FLIP_PALM * (flip_palm_src + flip_palm_rcv) + W_FLIP_HAND * flip_hand
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
    arm_posture = W_ARM_LIMIT * (_limit_pen(ctx.src_arm_limit_margin, MARGIN_FREE) + _limit_pen(ctx.rcv_arm_limit_margin, MARGIN_FREE)) \
        + W_ARM_SHAKE * (_shake_pen(ctx.src_arm_qdd, QDD_FREE, QDD_SCALE) + _shake_pen(ctx.rcv_arm_qdd, QDD_FREE, QDD_SCALE))

    # ------------------------------------------------------------------ safety / disturbance
    cup_disturb = W_SLIDE_SRC * src["slide"] + W_SLIDE_RCV * rcv["slide"] \
        + W_ONE_SIDED_SRC * src["one_sided"] + W_ONE_SIDED_RCV * rcv["one_sided"] \
        + W_PALM_SRC * src["palm_push"] + W_PALM_RCV * rcv["palm_push"] \
        + W_RUSH_SRC * src["rush"] + W_RUSH_RCV * rcv["rush"] \
        + W_FLING * (src["fling"] + rcv["fling"])
    safety = W_CUP_CUP * torch.tanh(ctx.cup_cup_force / 5.0) \
        + W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0) + torch.tanh(ctx.rcv_hand_foreign_force / 5.0)) \
        + W_NESTED * nested

    # ------------------------------------------------------------------ assemble components
    comps = {
        "alive": W_ALIVE * torch.ones_like(ctx.src_cup_tilt),
        "grasp_src": W_GRASP_SRC * src["value"],
        "grasp_rcv": W_GRASP_RCV * rcv["value"],
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place_rcv": W_PLACE * place,
        "aim_lip": W_AIM * aim,
        "tilt_dir": -tilt_dir,
        "tilt_open": W_TILT_OPEN * tilt_open_val,
        "pour_delta": W_POUR * pour_paid - W_FLOW * flow_excess,
        "transferred": W_TRANSFER * in_t * out_gate,
        "transfer_tail": W_TAIL * finish_q * out_gate,
        "spill_delta": -W_SPILL * spill_step,
        "unaimed_release": -W_UNAIMED * unaimed,
        "success_bonus": W_SUCCESS * ctx.success.to(dtype) * clean_f,
        "src_speed": -W_SRC_SPEED * src_speed,
        "src_spin": -src_spin,
        "rcv_steady": -rcv_steady,
        "act_flip": -act_flip,
        "palm_bang": -palm_bang,
        "palm_sat": -palm_sat,
        "cmd_rate": -cmd_rate,
        "arm_posture": -arm_posture,
        "cup_disturb": -cup_disturb,
        "safety": -safety,
    }

    reward = torch.zeros_like(ctx.src_cup_tilt)
    for value in comps.values():
        reward = reward + value

    # diagnostics
    comps["diag_place_dist_r"] = d_place / r
    comps["diag_aim_dist_r"] = d_aim / r
    comps["diag_lip_xy"] = geo["lip_xy"]
    comps["diag_lip_dz"] = geo["lip_dz"]
    comps["diag_win"] = win
    comps["diag_toward"] = (geo["theta"] > 0.0).to(dtype)
    comps["diag_open_phi"] = opn["phi"]
    comps["diag_first_out"] = first_out
    comps["diag_ctrl_gate"] = ctrl_gate
    comps["diag_stage_cd"] = stage_cd
    comps["diag_clean"] = clean_f
    comps["diag_finish_q"] = finish_q
    comps["diag_grip_keep_src"] = src["keep"]
    comps["diag_balance_src"] = src["balance"]
    comps["diag_src_w"] = w_src
    comps["diag_src_v"] = v_src
    comps["diag_src_margin_min"] = torch.clamp(ctx.src_arm_limit_margin, 0.0, 1.0).min(dim=-1).values
    comps["diag_src_n_contact"] = (ctx.src_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    comps["diag_rcv_n_contact"] = (ctx.rcv_finger_force > CONTACT_F).to(dtype).sum(dim=-1)
    comps["diag_state_income"] = comps["grasp_src"] + comps["grasp_rcv"] + comps["lift_src"] + comps["lift_rcv"]
    comps["diag_outcome_income"] = comps["transferred"] + comps["transfer_tail"] + comps["success_bonus"]

    return reward, comps
```