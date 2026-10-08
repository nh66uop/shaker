Brief 에 가설 id 줄이 없어 prompt 전체를 한 후보로 구현했습니다. 따른 것은 가장 최근 절인 "iter_01 변경 (최종)"(prompt.md:623-644)입니다. 이 절이 앞의 변경 1·2·3(541-561)을 대체합니다.

**구현하며 판단한 것 (충돌과 한계)**
1. **유체 수면 모형은 쓰지 않았다.** 더미 표면이 물처럼 수평이 된다고 보면, 채움 0.30 에서 첫 알이 약 68° 에 나온다는 계산이 된다. 이는 측정 3 과 맞지 않는다. 측정 3 에서는 90~100° 까지 소스 잔량이 1.000 이고, 방출 중앙이 101° 다. 이 모형으로는 검사 [9](85° 가 만점 미만)도 통과하지 못한다.
   - 그래서 더미는 수평이 되지 않는다고 보았다(가설). 기준점은 아래쪽 벽에 닿은 더미 윗모서리다.
   - 결과식은 `(H - h0)·cosθ` 이고, 기하상 90° 에서 0 을 지난다. 경사 척도는 0.5r 이다.
   - 이 식으로 85° 는 약 0.29, 105° 는 약 0.93 이다. 70°·100° 같은 각도 상수는 넣지 않았다.
2. **되받은 상태는 무상태 함수로 막을 수 없다.** 되받은 상태(담김 0, 소스 1.0)는 에피소드 시작 상태와 ctx 상으로 구별되지 않는다. 그래서 되받기는 경제성으로 막는다.
   - 부어 놓고 유지하는 수입(`in_target_level` + `success_bonus`)을 기울임 수입보다 크게 두었다.
   - `d_in_target` 은 단조라 다시 부어도 배달 수입이 다시 붙지 않는다.
3. **리시버 팔 관절 여유도 30% 반영했다.** 변경 3 은 소스 팔 중심으로 적혀 있다. 측정 1 에 리시버 포화(0.48~0.82)도 있어 함께 넣었다.

```python
import torch
import math


def _norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(N,K) -> (N,) L2 노름 (0 에서 안전)."""
    return torch.sqrt((v * v).sum(dim=1) + eps)


def _limit_margin(m: torch.Tensor) -> torch.Tensor:
    """(N,A) 관절 한계 여유 -> (N,) [0,1]. 평균과 최솟값을 반씩 (한 관절이라도 한계에 붙으면 크게 준다)."""
    return torch.clamp(0.5 * (m.mean(dim=1) + torch.min(m, dim=1).values), 0.0, 1.0)


def compute_reward(ctx) -> "tuple[torch.Tensor, dict[str, torch.Tensor]]":
    eps = 1e-8
    r = ctx.cup_radius                      # 조준·배치·방출 경사의 유일한 길이 스케일 (컵 반지름 배수만 쓴다)

    # ---- 가중치 --------------------------------------------------------------------
    # 예산: 붓지 않고 기울인 채 버티기(스텝당 약 3~4) < 부어 놓고 유지(약 6.5, 성공 시 약 11.5) -> 격차 2 배 이상.
    W_APPROACH = 0.15    # 팔별 접근 (파지 ckpt 가 이미 안다 - 작게)
    W_GRASP = 0.15       # 팔별 파지 유지
    W_LIFT_SRC = 0.20    # 소스는 높게 (K3: 높이가 회전 여유를 만든다)
    W_OVERLIFT = 0.10    # 과들기 세금은 충분히 높은 곳에서만
    W_LIFT_RCV = 0.10    # 리시버는 테이블에서 떼기만, 낮게
    W_PLACE = 0.25       # 배치 유지 (차분형의 빈틈 보완, i23·i24)
    W_AIM_PROG = 1.5     # 조준 차분(진행도), 대칭 clamp +-0.3 m/s -> +-0.45
    W_AIM_HOLD = 0.25    # 도달한 조준 자리를 지키는 작은 상태형 수입
    W_DIR = 0.8          # 방향 비용 = 기울기 크기 x 방향 어긋남 (i17 에서 지웠다가 실패)
    W_POUR = 4.0         # ★ 방출 조건 회전 수입 (iter_00 8.0 -> 4.0): 유지 수입(W_IN_LEVEL 6 + W_SUCCESS 5)의 절반 아래
    POUR_BASE = 0.4      # 회전 수입 중 관절 여유와 무관한 몫. 나머지 0.6 은 관절 여유에 비례 (변경 3)
    W_POUR_RATE = 1.0    # 받는 컵 쪽으로 도는 속도 (첫 한 알 전의 경사, 대칭 clamp 라 왕복은 0)
    W_DELIVER = 300.0    # 실제 이송 증분: 비드 1 알 = 0.05 -> 15 (env 단조 증분, 재붓기에는 재지급 없음)
    W_IN_LEVEL = 6.0     # ★ 담긴 양 유지 (iter_00 1.0 -> 6.0): 담김 0.9 유지 5.4 > 최대 회전 수입 4.0
    W_SUCCESS = 5.0      # ★ 성공 유지 보너스 (iter_00 3.0 -> 5.0, env 판정, 재정의 못 함)
    W_SPILL = 120.0      # 흘림 증분: 1 알 = 6 < 배달 1 알 = 15 (i10~12 과벌점 붕괴 회피)
    W_RCV_UP = 0.8       # 받는 컵 직립 (성공 조건 20 도)
    W_NESTED = 1.5       # 컵 포개기는 이송이 아니다
    W_FALL = 1.0         # 컵을 떨어뜨림
    W_CUPCUP = 1.0       # 실기 안전: 컵끼리 접촉
    W_FOREIGN = 0.75     # 실기 안전: 남의 것 건드리기 (팔별, self-collision ON 이면 팔끼리 접촉도 손 foreign 힘으로 잡힐 수 있음)
    W_CUPVEL = 0.25      # 컵 병진 속도 (팔별)
    W_SPIN = 0.25        # 컵 자기축 회전 - 붓기에 무용 (i22)
    W_QD = 0.25          # 팔 관절 속도 (팔별)
    W_QDD = 0.15         # 팔 관절 가속 (팔별)
    W_ACT_RATE = 0.3     # 명령 요동
    W_ACT_SAT = 0.3      # palm 포화
    W_ACT_FLIP = 0.3     # 부호 뒤집기

    # 처음 채움 비율: 이 트랙 측정값 (측정 3: 비드 20 개 더미 29.9 mm / 컵 내부 100.8 mm = 0.30)
    FILL0 = 0.30

    # ---- 기본 량 --------------------------------------------------------------------
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

    # 컵에서 나간 비율(담김 + 흘림, 흘림은 영구). 오를수록 회전 수입이 꺼진다 (변경 2).
    out_frac = torch.clamp(in_frac + spill_frac, 0.0, 1.0)
    remain_gate = 1.0 - out_frac

    # 상태 비례 수입은 담김이 오르면 줄어든다 (5-1)
    state_decay = 1.0 - 0.5 * in_frac

    # ---- D 조준: 주둥이가 입구 위(lip_dz>0) · 수평으로 입구 안 (r 배수 창) -------------
    e_xy = (mouth - lip)[:, :2]
    lip_xy = torch.sqrt((e_xy * e_xy).sum(dim=1) + eps)      # 주둥이-입구 수평 거리
    lip_dz = lip[:, 2] - mouth[:, 2]                         # 기울면 r·sin(theta) 만큼 내려감(K6)이 이미 반영됨
    gate_xy = torch.exp(-(lip_xy / (2.0 * r)) ** 2)          # 좁은 방출 허가 창 (i13 골격)
    gate_dz = torch.clamp(lip_dz / r, 0.0, 1.0)              # 입구보다 위일 때만
    aim_gate = gate_xy * gate_dz

    # ---- E 방향: 컵 up 축의 수평 성분이 기우는 방위 ---------------------------------
    up_xy = ctx.src_cup_up[:, :2]
    up_xy_n = torch.sqrt((up_xy * up_xy).sum(dim=1) + eps)
    to_rcv_xy = (mouth - src_pos)[:, :2]
    to_rcv_n = torch.sqrt((to_rcv_xy * to_rcv_xy).sum(dim=1) + eps)
    dir_cos = (up_xy * to_rcv_xy).sum(dim=1) / (up_xy_n * to_rcv_n)   # 직립 근처에서는 잡음 -> 아래서 tilt 로 가중
    align = torch.clamp(dir_cos, 0.0, 1.0)

    # ---- A 파지 ----------------------------------------------------------------------
    d_src = 0.5 * (_norm(ctx.src_palm_pos - src_pos) + _norm(ctx.src_tips_pos[:, 0, :] - src_pos))
    d_rcv = 0.5 * (_norm(ctx.rcv_palm_pos - rcv_pos) + _norm(ctx.rcv_tips_pos[:, 0, :] - rcv_pos))
    near_src = 1.0 - torch.tanh(d_src / (4.0 * r))
    near_rcv = 1.0 - torch.tanh(d_rcv / (4.0 * r))
    approach_src = W_APPROACH * torch.where(ctx.src_grasped, ones, near_src) * state_decay
    approach_rcv = W_APPROACH * torch.where(ctx.rcv_grasped, ones, near_rcv) * state_decay
    grasp_hold = W_GRASP * (src_held + rcv_held) * state_decay

    # ---- B 들기 (높이 허용만 절대 상수 예외) ----------------------------------------
    h_src = src_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2]
    h_rcv = rcv_pos[:, 2] - ctx.rcv_cup_spawn_pos[:, 2]
    lift_src = W_LIFT_SRC * torch.clamp(h_src / 0.28, 0.0, 1.0) \
        - W_OVERLIFT * torch.clamp((h_src - 0.45) / 0.20, 0.0, 1.0)
    lift_src = lift_src * src_held * state_decay
    lift_rcv = W_LIFT_RCV * torch.clamp(h_rcv / (2.0 * r), 0.0, 1.0) \
        * torch.exp(-torch.clamp(h_rcv - 0.15, min=0.0) / 0.10) * rcv_held * state_decay

    # ---- C 배치: 소스 기준 상대 좌표, 컵 반지름 배수 (합 형태) -----------------------
    p_z = torch.clamp((src_pos[:, 2] - mouth[:, 2]) / (2.0 * r), 0.0, 1.0)          # 받는 컵이 더 낮게
    d_cup_xy = torch.sqrt(((src_pos - rcv_pos)[:, :2] ** 2).sum(dim=1) + eps)
    p_xy = torch.exp(-((d_cup_xy - 3.0 * r) / (2.0 * r)) ** 2)                      # 붓기 가능한 간격, 포개짐 아님
    y_mid = 0.5 * (src_pos[:, 1] + rcv_pos[:, 1])
    p_mid = torch.exp(-(y_mid / (4.0 * r)) ** 2)                                    # 양팔 작업은 중심선 근처
    place_hold = W_PLACE * (0.4 * p_z + 0.4 * p_xy + 0.2 * p_mid) * both_held * state_decay

    # 조준 차분(진행도): -lip_xy 의 시간미분 = 상대속도의 접근 성분, clamp 대칭(i20)
    dir_hat = e_xy / lip_xy.unsqueeze(1)
    v_rel_xy = (ctx.src_cup_lin_vel - ctx.rcv_cup_lin_vel)[:, :2]
    closing = (v_rel_xy * dir_hat).sum(dim=1)
    aim_progress = W_AIM_PROG * torch.clamp(closing, -0.3, 0.3) * both_held
    aim_hold = W_AIM_HOLD * aim_gate * both_held * state_decay

    # ---- E 방향 비용: 기울기 크기 x 방향 어긋남 -------------------------------------
    tilt_norm = torch.clamp(tilt / (0.5 * math.pi), 0.0, 1.0)
    dir_cost = -W_DIR * tilt_norm * (0.5 * (1.0 - dir_cos)) * src_frac

    # ---- F 방출 조건: 림 최저점 vs 소스 안 비드 더미 높이 (변경 1, 각도 상수 없음) -----
    # 컵 내부 깊이(원점->림 - 원점->바닥), 현재 잔량으로 추정한 축 방향 더미 높이.
    depth_in = ctx.cup_mouth_z - ctx.cup_bottom_z
    h_pile = FILL0 * depth_in * src_frac                                     # (N,)
    up_z = ctx.src_cup_up[:, 2]
    sin_t = torch.sin(tilt)
    rim_low_z = lip[:, 2] - r * sin_t                                        # 림 원의 가장 낮은 점 (기운 쪽)
    # 더미가 수평으로 퍼지지 않고 바닥 쪽에 뭉쳐 있다고 본다 (가설: 비드 지름이 컵 반지름 규모라서).
    # 수평 수면 모형은 이 채움에서 약 68 도 방출을 예측하는데, 측정 3 은 90~100 도까지 잔량 1.000 이다.
    # 그래서 기준점은 더미 윗면이 기운 쪽(아래쪽) 벽과 만나는 점이다.
    pile_z = src_pos[:, 2] + (ctx.cup_bottom_z + h_pile) * up_z - r * sin_t
    rim_clear = rim_low_z - pile_z                                           # >0: 림이 더미보다 위 (아직 못 나감)
    # 조건에 가까워질수록, 그리고 넘을수록 단조 증가. 폭은 컵 반지름의 절반.
    rel = 0.5 * (1.0 + torch.tanh(-rim_clear / (0.5 * r)))

    # 회전 수입: 창 안 + 방향 일치 + 양손 파지 + 포개짐 아님 + 아직 컵에서 안 나간 비율.
    # 관절 여유가 클수록 더 받는다 (변경 3): 전체 = W_POUR·gate·rel·remain·(0.4 + 0.6·m).
    # pour_open 은 0.4 몫, posture_margin 은 0.6·m 몫으로 나눠 로깅한다.
    m_src = _limit_margin(ctx.src_arm_limit_margin)
    m_rcv = _limit_margin(ctx.rcv_arm_limit_margin)
    m_pose = 0.7 * m_src + 0.3 * m_rcv          # 붓는 팔 위주, 리시버 포화(측정 1: 0.51~0.82)도 일부 반영
    pour_gate = both_held * valid * aim_gate * align
    pour_core = W_POUR * pour_gate * rel * remain_gate
    pour_open = POUR_BASE * pour_core
    posture_margin = (1.0 - POUR_BASE) * pour_core * m_pose

    # 받는 컵 쪽으로 기울이는 각속도 성분: 축 = z x t (t = 받는 컵 수평 방향).
    # 대칭 clamp 로 기울였다 세우는 왕복이 이득이 안 되게 한다. 나간 비율이 오르면 꺼진다.
    t_hat = to_rcv_xy / to_rcv_n.unsqueeze(1)
    w = ctx.src_cup_ang_vel
    tilt_rate = -w[:, 0] * t_hat[:, 1] + w[:, 1] * t_hat[:, 0]
    pour_rate = W_POUR_RATE * torch.clamp(tilt_rate, -2.0, 2.0) * both_held * valid * aim_gate * remain_gate

    deliver_delta = W_DELIVER * torch.clamp(ctx.d_in_target, -1.0, 1.0) * valid
    in_target_level = W_IN_LEVEL * in_frac * valid
    success_bonus = W_SUCCESS * ctx.success.to(tilt.dtype)

    # ---- H 흘림·안전·제어 ------------------------------------------------------------
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

    reward = zeros
    for v in components.values():
        reward = reward + v
    return reward, components
```
