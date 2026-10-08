iter_01 후보 c3 — c2 에서 진행도(시간) 감쇠만 뺀 것.

c2 의 `pour_time = 1 - 0.85·clamp((progress - 0.35)/0.35)` 은 "늦게 붓는 회전 수입"을 0.15 배로 깎아
되받기를 막지만, 시간 상수를 처방하는 형태다. i00 의 첫 배달이 t ≈ 375/900(진행도 0.42)에서 일어났으므로
학습 초기에는 이 감쇠가 첫 회전 경사까지 같이 깎을 위험이 있다([가설]).
c3 는 담김+흘림 기반 차단(`pour_need`)과 낮춘 회전 가중(W_POUR 2.0)만으로 같은 게이트를 통과하는지 본다.

```python
import torch
import math


def _norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(N,K) -> (N,) L2 노름 (0 에서 안전)."""
    return torch.sqrt((v * v).sum(dim=1) + eps)


def compute_reward(ctx) -> "tuple[torch.Tensor, dict[str, torch.Tensor]]":
    eps = 1e-8
    r = ctx.cup_radius                      # 조준·배치의 유일한 길이 스케일 (컵 반지름 배수만 쓴다)

    # ---- 가중치 (예산 규칙 5-1) -------------------------------------------------
    # 상태 비례 수입 합 상한 ≈ 1.4/step, 회전·배달 수입은 붓는 동안 4~20/step.
    W_APPROACH = 0.15    # 팔별 접근 (파지 ckpt 가 이미 안다 — 작게)
    W_GRASP = 0.15       # 팔별 파지 유지
    W_LIFT_SRC = 0.20    # 소스는 높게 (K3: 높이가 회전 여유를 만든다)
    W_OVERLIFT = 0.10    # 과들기 세금은 충분히 높은 곳에서만
    W_LIFT_RCV = 0.10    # 리시버는 테이블에서 떼기만, 낮게
    W_PLACE = 0.25       # 배치 유지 (차분형의 빈틈 보완, i23·i24)
    W_AIM_PROG = 1.5     # 조준 차분(진행도), 대칭 clamp ±0.3 m/s → ±0.45
    W_AIM_HOLD = 0.25    # 도달한 조준 자리를 지키는 작은 상태형 수입
    W_DIR = 0.8          # 방향 비용 = 기울기 크기 × 방향 어긋남 (i17 에서 지웠다가 실패)
    W_POUR = 8.0         # ★ 방출 영역 회전 수입 — 나머지 전부의 합보다 크게
    W_POUR_RATE = 0.75    # ★ 받는 컵 쪽으로 도는 속도 (첫 한 알 전의 경사)
    W_MARGIN = 0.5       # 한계 여유를 지키며 기울인 자세를 더 값싸게 (K3·K7, i15 식 직접 벌점 금지)
    W_DELIVER = 300.0    # ★ 실제 이송 증분: 비드 1 알 = 0.05 → 15
    # iter_01 변경 2: 1.0 → 6.0. 담김 0.9 유지 = 5.4/step 로, 회전 수입 최대 8.0 과 같은 자릿수.
    W_IN_LEVEL = 8.0     # ★ 담긴 양 유지 (되받으면 즉시 잃는 수입)
    # iter_01 변경 2: 3.0 → 8.0. 유지(5.4)+성공(8.0)=13.4 > 회전 최대 8.0 → 되받기는 순손실.
    W_SUCCESS = 8.0      # ★ 성공 유지 보너스 (env 판정, 재정의 못 함)
    W_SPILL = 120.0      # 흘림 증분: 1 알 = 6 < 배달 1 알 = 15 (i10~12 과벌점 붕괴 회피)
    W_RCV_UP = 0.8       # 받는 컵 직립 (성공 조건)
    W_NESTED = 1.5       # 컵 포개기는 이송이 아니다
    W_FALL = 1.0         # 컵을 떨어뜨림
    W_CUPCUP = 1.0       # 실기 안전: 컵끼리 접촉
    W_FOREIGN = 0.75     # 실기 안전: 남의 것 건드리기 (팔별)
    W_CUPVEL = 0.25      # 컵 병진 속도 (팔별)
    W_SPIN = 0.25        # 컵 자기축 회전 — 붓기에 무용 (i22)
    W_QD = 0.25          # 팔 관절 속도 (팔별)
    W_QDD = 0.15         # 팔 관절 가속 (팔별)
    W_ACT_RATE = 0.3     # 명령 요동
    W_ACT_SAT = 0.3      # palm 포화
    W_ACT_FLIP = 0.3     # 부호 뒤집기

    T_REL = math.radians(70.0)   # 환경 설명: 가득 찬 컵에서 첫 알이 약 70°
    T_REL_W = math.radians(12.0) # 80° 1/5 · 86° 절반 을 덮는 폭 (상한·벌점 없음)
    F_TARGET = 0.8               # env 성공 기준의 담김 임계 — 회전 수입을 끄는 지점

    # ---- 기본 량 ----------------------------------------------------------------
    tilt = ctx.src_cup_tilt                                  # (N,)
    zeros = torch.zeros_like(tilt)
    ones = torch.ones_like(tilt)

    src_pos = ctx.src_cup_pos
    rcv_pos = ctx.rcv_cup_pos
    lip = ctx.src_cup_mouth_pos                              # 소스 주둥이(림) 중심
    mouth = ctx.rcv_cup_mouth_pos                            # 받는 컵 입구 중심

    src_held = ctx.src_grasped.to(tilt.dtype)
    rcv_held = ctx.rcv_grasped.to(tilt.dtype)
    both_held = src_held * rcv_held
    valid = 1.0 - ctx.cups_nested.to(tilt.dtype)             # 포개짐은 이송 무효
    src_frac = torch.clamp(ctx.bead_in_source_frac, 0.0, 1.0)
    in_frac = torch.clamp(ctx.bead_in_target_frac, 0.0, 1.0)
    spill_frac = torch.clamp(ctx.bead_spill_frac, 0.0, 1.0)

    # ★iter_01 c2: 회전 수입의 스케일에서 src_frac(소스 잔량)을 뺀다.
    #   src_frac 은 되받으면 1.0 으로 복구되므로 왕복이 이득이었다(측정).
    #   대신 "컵에서 나간 비율"(담김 + 흘림; 흘림은 영구적이라 되받아도 안 돌아온다)로 끈다.
    released = torch.clamp(in_frac + spill_frac, 0.0, 1.0)
    pour_need = torch.clamp(1.0 - released / F_TARGET, 0.0, 1.0)   # 담김 0.8 이면 회전 수입 0
    # c3: 진행도(시간) 감쇠를 쓰지 않는다 — 시간 처방 없이 담김·흘림만으로 끈다.
    pour_scale = pour_need

    # 상태 비례 수입은 담김이 오르면 줄어든다 (5-1)
    state_decay = 1.0 - 0.5 * in_frac

    # ---- D 조준: 주둥이가 입구 위(lip_dz>0) · 수평으로 입구 안 (r 배수 창) -------
    e_xy = (mouth - lip)[:, :2]
    lip_xy = torch.sqrt((e_xy * e_xy).sum(dim=1) + eps)      # 주둥이-입구 수평 거리
    lip_dz = lip[:, 2] - mouth[:, 2]                         # 기울면 r·sinθ 만큼 내려감(K6)이 이미 반영됨
    gate_xy = torch.exp(-(lip_xy / (2.0 * r)) ** 2)          # 좁은 방출 허가 창 (i13 골격)
    gate_dz = torch.clamp(lip_dz / r, 0.0, 1.0)              # 입구보다 위일 때만
    aim_gate = gate_xy * gate_dz

    # ---- E 방향: 컵 up 축의 수평 성분이 기우는 방위 -----------------------------
    up_xy = ctx.src_cup_up[:, :2]
    up_xy_n = torch.sqrt((up_xy * up_xy).sum(dim=1) + eps)
    to_rcv_xy = (mouth - src_pos)[:, :2]
    to_rcv_n = torch.sqrt((to_rcv_xy * to_rcv_xy).sum(dim=1) + eps)
    dir_cos = (up_xy * to_rcv_xy).sum(dim=1) / (up_xy_n * to_rcv_n)   # 직립 근처에서는 잡음 → 아래서 tilt 로 가중
    align = torch.clamp(dir_cos, 0.0, 1.0)

    # ---- A 파지 ------------------------------------------------------------------
    d_src = 0.5 * (_norm(ctx.src_palm_pos - src_pos) + _norm(ctx.src_tips_pos[:, 0, :] - src_pos))
    d_rcv = 0.5 * (_norm(ctx.rcv_palm_pos - rcv_pos) + _norm(ctx.rcv_tips_pos[:, 0, :] - rcv_pos))
    near_src = 1.0 - torch.tanh(d_src / (4.0 * r))
    near_rcv = 1.0 - torch.tanh(d_rcv / (4.0 * r))
    approach_src = W_APPROACH * torch.where(ctx.src_grasped, ones, near_src) * state_decay
    approach_rcv = W_APPROACH * torch.where(ctx.rcv_grasped, ones, near_rcv) * state_decay
    grasp_hold = W_GRASP * (src_held + rcv_held) * state_decay

    # ---- B 들기 (높이 허용만 절대 상수 예외) ------------------------------------
    h_src = src_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = rcv_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_src = W_LIFT_SRC * torch.clamp(h_src / 0.28, 0.0, 1.0) \
        - W_OVERLIFT * torch.clamp((h_src - 0.45) / 0.20, 0.0, 1.0)
    lift_src = lift_src * src_held * state_decay
    lift_rcv = W_LIFT_RCV * torch.clamp(h_rcv / (2.0 * r), 0.0, 1.0) \
        * torch.exp(-torch.clamp(h_rcv - 0.15, min=0.0) / 0.10) * rcv_held * state_decay

    # ---- C 배치: 소스 기준 상대 좌표, 컵 반지름 배수 (합 형태 — 곱 사다리 회피) ---
    p_z = torch.clamp((src_pos[:, 2] - mouth[:, 2]) / (2.0 * r), 0.0, 1.0)          # 받는 컵이 더 낮게
    d_cup_xy = torch.sqrt(((src_pos - rcv_pos)[:, :2] ** 2).sum(dim=1) + eps)
    p_xy = torch.exp(-((d_cup_xy - 3.0 * r) / (2.0 * r)) ** 2)                      # 붓기 가능한 간격, 포개짐 아님
    y_mid = 0.5 * (src_pos[:, 1] + rcv_pos[:, 1])
    p_mid = torch.exp(-(y_mid / (4.0 * r)) ** 2)                                    # 양팔 작업은 중심선 근처
    place_hold = W_PLACE * (0.4 * p_z + 0.4 * p_xy + 0.2 * p_mid) * both_held * state_decay

    # 조준 차분(진행도): -lip_xy 의 시간미분 ≈ 상대속도의 접근 성분, clamp 대칭(i20)
    dir_hat = e_xy / lip_xy.unsqueeze(1)
    v_rel_xy = (ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel)[:, :2]
    closing = (v_rel_xy * dir_hat).sum(dim=1)
    aim_progress = W_AIM_PROG * torch.clamp(closing, -0.3, 0.3) * both_held
    aim_hold = W_AIM_HOLD * aim_gate * both_held * state_decay

    # ---- E 방향 비용: 기울기 크기 × 방향 어긋남 ---------------------------------
    tilt_norm = torch.clamp(tilt / (0.5 * math.pi), 0.0, 1.0)
    dir_cost = -W_DIR * tilt_norm * (0.5 * (1.0 - dir_cos)) * src_frac

    # ---- F 회전·배달 (주역) -----------------------------------------------------
    # 방출 영역: 70° 중심 단조 증가, 목표각·대역·상한을 처방하지 않는다.
    rel = 0.5 * (1.0 + torch.tanh((tilt - T_REL) / T_REL_W))
    pour_gate = both_held * valid * aim_gate * align          # 창 안 + 방향 일치일 때만 지불
    # ★iter_01 c3: src_frac → pour_scale. 다 부었으면(담김+흘림 ≥ 0.8) 0.
    pour_open = W_POUR * pour_gate * rel * pour_scale

    # 받는 컵 쪽으로 기울이는 각속도 성분: 축 = ẑ × t̂ (t̂ = 받는 컵 수평 방향).
    # 대칭 clamp 로 되돌림 펌프가 닫힌 경로에서 이득이 안 되게 한다.
    t_hat = to_rcv_xy / to_rcv_n.unsqueeze(1)
    w = ctx.src_cup_ang_vel
    tilt_rate = -w[:, 0] * t_hat[:, 1] + w[:, 1] * t_hat[:, 0]
    pour_rate = W_POUR_RATE * torch.clamp(tilt_rate, -2.0, 2.0) * both_held * valid * aim_gate * pour_scale

    deliver_delta = W_DELIVER * torch.clamp(ctx.d_in_target, -1.0, 1.0) * valid
    in_target_level = W_IN_LEVEL * in_frac * valid * both_held   # ★ 유지 수입 (W 1.0 → 8.0, 두 컵 파지 중에만)
    success_bonus = W_SUCCESS * ctx.success.to(tilt.dtype)    # ★ 성공 유지 (W 3.0 → 8.0)

    # ---- G 자세 여유: 한계에 기대지 않은 기울임을 더 값싸게 (직접 벌점 아님) ------
    m_src = ctx.src_arm_limit_margin
    margin = 0.5 * (m_src.mean(dim=1) + torch.min(m_src, dim=1).values)
    posture_margin = W_MARGIN * rel * aim_gate * both_held * torch.clamp(margin, 0.0, 1.0) * pour_scale

    # ---- H 흘림·안전·제어 --------------------------------------------------------
    spill_delta = -W_SPILL * torch.clamp(ctx.d_spill, 0.0, 1.0)
    rcv_upright = -W_RCV_UP * torch.tanh(ctx.rcv_cup_tilt / math.radians(20.0))
    nested_pen = -W_NESTED * ctx.cups_nested.to(tilt.dtype)
    fell = ((src_pos[:, 2] < ctx.src_cup_spawn_pos[:, 2] - 0.5 * r).to(tilt.dtype)
            + (rcv_pos[:, 2] < ctx.rcv_cup_spawn_pos[:, 2] - 0.5 * r).to(tilt.dtype))
    cup_fall = -W_FALL * fell
    cup_cup = -W_CUPCUP * torch.tanh(ctx.cup_cup_force / 5.0)
    foreign = -W_FOREIGN * (torch.tanh(ctx.src_hand_foreign_force / 5.0)
                            + torch.tanh(ctx.rcv_hand_foreign_force / 5.0))

    spin_src = torch.abs((ctx.src_cup_ang_vel * ctx.src_cup_up).sum(dim=1))   # 자기축 회전은 붓기에 무용
    spin_rcv = torch.abs((ctx.rcv_cup_ang_vel * ctx.rcv_cup_up).sum(dim=1))
    cup_motion = -W_CUPVEL * (torch.tanh(_norm(ctx.src_cup_lin_vel) / 0.8)
                              + torch.tanh(_norm(ctx.rcv_cup_lin_vel) / 0.8)) \
        - W_SPIN * (torch.tanh(spin_src / 4.0) + torch.tanh(spin_rcv / 4.0))

    qd_s = torch.sqrt((ctx.src_arm_qd ** 2).mean(dim=1) + eps)
    qd_r = torch.sqrt((ctx.rcv_arm_qd ** 2).mean(dim=1) + eps)
    qdd_s = torch.sqrt((ctx.src_arm_qdd ** 2).mean(dim=1) + eps)
    qdd_r = torch.sqrt((ctx.rcv_arm_qdd ** 2).mean(dim=1) + eps)
    arm_motion = -W_QD * (torch.tanh(qd_s / 3.0) + torch.tanh(qd_r / 3.0)) \
        - W_QDD * (torch.tanh(qdd_s / 40.0) + torch.tanh(qdd_r / 40.0))

    da = ctx.actions - ctx.prev_actions
    act_rate = -W_ACT_RATE * da.abs().mean(dim=1)
    sat = (torch.clamp(ctx.actions.abs() - 0.85, min=0.0) / 0.15).mean(dim=1)
    flip = torch.clamp(-ctx.actions * ctx.prev_actions, min=0.0).mean(dim=1)
    act_sat = -W_ACT_SAT * sat - W_ACT_FLIP * flip

    components = {
        "approach_src": approach_src,
        "approach_rcv": approach_rcv,
        "grasp_hold": grasp_hold,
        "lift_src": lift_src,
        "lift_rcv": lift_rcv,
        "place_hold": place_hold,
        "aim_progress": aim_progress,
        "aim_hold": aim_hold,
        "dir_cost": dir_cost,
        "pour_open": pour_open,
        "pour_rate": pour_rate,
        "posture_margin": posture_margin,
        "deliver_delta": deliver_delta,
        "in_target_level": in_target_level,
        "success_bonus": success_bonus,
        "spill_delta": spill_delta,
        "rcv_upright": rcv_upright,
        "nested_pen": nested_pen,
        "cup_fall": cup_fall,
        "cup_cup": cup_cup,
        "foreign": foreign,
        "cup_motion": cup_motion,
        "arm_motion": arm_motion,
        "act_rate": act_rate,
        "act_sat": act_sat,
    }

    reward = torch.zeros_like(tilt)
    for v in components.values():
        reward = reward + v
    return reward, components
```
