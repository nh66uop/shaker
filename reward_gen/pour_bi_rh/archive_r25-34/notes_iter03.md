# iter_03 운영 노트 — 2026-09-27 (세 번째 사이클 3 라운드)

확인된 사실만 적는다. 추정은 [가설]. 앞의 DESIGN_BRIEF 가 트랙 전체 교훈이다.

## 사용자 지시 (09-27)
- "접근·파지·리프트가 fresh 에서도 **빨리** 되어야 후속 작업(붓기)을 할 수 있다."
- "왼손이 이상하게 오른손 컵으로 가는 느낌" — 측정으로 확인(아래).
- "다음 실험은 에피소드를 생각해서 엔트로피를 적절히 유지" — 학습 설정 쪽(보상 밖)에서 따로 다룬다.
- 유지: LSTM(→LN→MLP, concat 끔), horizon 16, self-collision ON, 붓기 구조(c5 = DG-5F iter_02 이식).

## iter_02 보상으로 돌린 두 런 (fresh LSTM, 같은 보상)
| | t2r_rh4_i02 (원래 LSTM yaml) | t2r_rh4_i02m (LSTM→LN→MLP, concat 끔, lr 3e-4 linear) |
|---|---|---|
| 종료 epoch | 1739 | 1588 |
| 오른손 컵 근처 비율 최대 | 0.000 | **0.858**(e1011) → 0.015(e1371) |
| 왼손 컵 근처 비율 최대 | 0.68(e1250) → 0.31 | 0.000 |
| 파지 플래그 | 0 | 1.2e-5 |
| entropy (e1011 / e1371) | 27.4 / — | 12.5 / 8.3 |

i02m ep1200 결정론 재생(64 env, 900 스텝):
- 오른손: 2 s 에 손끝이 컵 벽(축에서 2.4~3.4 cm, 컵 반지름 2.8 cm)에 붙어 10 s 유지. **손끝 높이 4~6 cm**(잡는 높이 약 1.3 cm 보다 높다), 손 닫힘 0.45 에서 멈춤, 엄지 5.6 %·중지 2.1 % 접촉. 한 번이라도 파지한 env 3.1 %.
- 왼손: 1 s 에 자기 컵 5.7 cm 까지 왔다가 **자기 컵(y +0.165)을 지나 중심선 쪽 y +0.074** 로 가서 멈춤.
  자기 컵까지 12~13 cm 로 그대로, 소스 컵까지 46 → 25 cm. 손은 닫힌(0.47) 채 공중.
- 두 손 모두 "다가가서 반쯤 닫고 멈춤". 내려가서 쥐고 드는 단계가 없다.

## 같은 저장소 cup_grasp 트랙 — fresh LSTM 으로 접근·파지·들기를 끝냈다 (09-27 조사)
- 오른팔 iter_00: grasp_ep ≥ 0.5 e2392(3.2 h), lift ≥ 0.5 e2789, 최종 grasp 0.90 · lift 0.88. 왼팔 iter_01: grasp 0.5 e1157(1.6 h), lift 0.5 e1421.
- 학습 설정: LSTM 1024 → LN → MLP [1024,1024,512,512], horizon 16, seq 16, minibatch 16384, **mini_epochs 2, lr 1e-4 adaptive(kl 0.016), e_clip 0.1, entropy_coef 0**, critic lr 1e-4, gamma 0.99.
  entropy 36.7 → 파지 직전 18.6(e2250) → 12.3(e5000). 우리 i02m 은 e1011 에 이미 12.5.
- 손은 DG-5F(19 관절 절대 명령)라 손 모양 항(링크별 접촉·curl)은 RH 에 못 옮긴다. **옮길 수 있는 골격**:
  1. **곱 게이트 없는 원거리 접근**: 손바닥-컵 표면 거리 d, `0.35 · fade · (0.6(1−tanh(d/0.20)) + 0.4(1−tanh(d/0.05)))`.
     무작위 초기에서 접근 수입이 충분히 커야 한다(할인 수익: 서 있기 +1.14 대 접근 +8.11).
     접근에 다른 인자를 곱하면(옆벽·편손 바닥 0.15) 손바닥 간극이 0.22 → 0.37 m 로 물러났다.
  2. **편 손은 작은 가산항**(0.04), 손바닥 8 cm 밖에서 전액·2 cm 에서 0·쥐면 0. 편 손 주차 수입 < 한 손가락 접촉 수입.
  3. **연속 held**: `clip((n_touch−2)/2) · (0.3+0.7·palm) · (0.5+0.5·opp)` — 이진 래치가 아니다.
  4. **들기는 품질로 곱하지 않는다**(2.1 · held · upright · (0.5 tanh(h/0.03) + 0.5 h/h_need) + 상승 속도 0.4). 들기가 파지보다 먼저 나왔다.
  5. **테이블 위에서는 들기 이후 수입 0**(off_table), 에피소드 후반 테이블 위 파지 수입 감쇠 · stall 비용 — 테이블 위에서 쥐고 버티기 방지.
  6. **래치 없는 벌점**: drop 은 떨어지는 동안·안 쥔 상태에서만. 높이 래치 비례 벌점은 들기를 지웠다.
  7. 기동 전 **결정론 원장**: L1 서 있기 < 접근, L2 편 손 주차 < 한 손가락, L3 테이블 < 3 cm < 8 cm, L4 누름 < 감기.
  8. (왼팔 iter_03) 들기 전 수입을 held 가 아니라 **컵 높이로 끈다** — held 로 끄면 테이블 위에서 파지를 완성하는 것이 손해가 됐다.
- 주의: 이 컵(pour_mimic)은 한쪽 접촉 0.3~0.4 N 에 넘어진다. cup_grasp 의 shove 면제·손바닥 안착 보너스는 그대로 옮기지 않는다.

## iter_03 에서 할 것
1. **팔별 접근·파지·들기를 cup_grasp 골격으로 다시 쓴다**(팔별 합, 두 팔 사이 곱 사슬 금지).
   - 접근: 손바닥-자기 컵 **표면** 거리(컵 반지름 빼기), 두 척도(20 cm · 5 cm), 곱 인자 없음, 들면 꺼짐.
     **무작위 초기(손바닥 15~16 cm)에서도 경사가 커야 한다** — iter_02 는 20 cm→6 cm 수입 차가 +0.05/스텝이었다.
   - 파지 자세로 내려오기: RH 손끝(`*_tips_pos`)과 컵 축 기준 높이·반경으로 "잡는 높이까지 내려와 양쪽에서 감싸기"를
     단계적으로 준다(iter_02 rh2_i03 블록의 grip_geometry 를 재사용해도 된다). 측정상 오른손은 벽에 붙고도 손끝이 입구 높이에 머물렀다.
   - 닫기·접촉: RH 는 손당 6 채널(0 폄·1 쥠)이고 손가락 모양은 고를 수 없다. `*_hand_closure`, `*_finger_force`(5), `*_palm_force` 만 쓴다.
     성립 전 접촉을 단계적으로, 엄지+다른 손가락 양쪽 접촉과 손바닥 접촉이 오를수록 더(cup_grasp 연속 held 형태).
   - 들기: 품질로 곱하지 않는다. 테이블 위 수입(들기 이후)은 0. 들기 전 수입은 컵 높이로 끈다.
2. **왼손(리시버)이 자기 컵으로 가게 한다.** 측정: 왼손은 자기 컵을 지나 중심선 쪽 9 cm 에 머물렀다.
   리시버 접근을 소스와 같은 크기로 독립 지급하고, 파지 전 리시버 손바닥이 자기 컵보다 중심선 쪽(y 가 컵 y 보다 작은 쪽)으로
   넘어가면 작은 비용을 둔다(손바닥 y 기준, 컵 반지름 배수). 소스의 들기 이후 수입은 리시버 진행도 게이트(바닥 있는)를 유지한다.
3. 파지 전 벌점은 작게: 흘림 약 3/알, 컵 건드림(미끄러짐·기울임)은 넘어뜨림 방지용으로만. 래치 금지.
4. **붓기 구조는 iter_02 의 c5 부분을 그대로**(두 컵을 쥐고 든 뒤: converge·aim·pour_rot·pour_delta 30·spill 40·success 20 ·
   finish·premature·pour_dir·rcv_keep·posture_margin·제어 정규화). 이름을 바꾸지 않는다(로그 비교).
5. 항 수는 30 개 안팎 허용. 새 ctx 필드를 가정하지 않는다. 컵 치수는 ctx 에서 읽는다.

## 참고 1 — cup_grasp 설계 브리프 원문(20 라운드 교훈)
```text
# Verified lessons from 20 earlier rounds of the same task (cup_pick, both arms, 09.20-09.26)

Every line is a measurement from a finished run.

1. From random weights the approach was learned when the long-range reach was shaped on plain palm-to-surface distance (weight 0.35, 0.6 far + 0.4 near, times (0.45 + 0.55 grip), fading once held): palm gap 0.045 / 0.037 m and grasp episodes 0.83 / 0.98 (left / right).
2. Any factor that shrinks the long-range reach income at random initialisation made the policy retreat or park. A side-wall factor on the far reach: palm gap rose 0.22 -> 0.37 m. An open-hand factor with floor 0.15 multiplied onto the reach with weight 0.22: reach income 0.003-0.007 per step against 0.037 for the plain reach in the same epoch window, while the action and arm-speed penalties stayed at -0.023 and -0.013; contacts fell 36 -> 1.5 and the palm gap froze at 0.20-0.23 m.
3. Without any open-hand signal the hand arrived already curled: grasp episodes 0.008, palm contact 0 after 2,075 epochs.
4. An open-hand or palm-proximity income that paid without digit contact produced a flat open hand pressed on the wall: +0.195 per step, hand curl 0.000, no digit contact for 1,400 epochs.
5. While digit contact was zero, any term sitting near its maximum became the resting point (seat + palm terms 1.17 per step against 0.07 for the fingers, contact 0.004).
6. When the object could earn income while still on the table, the policy held it there: 0.54-0.69 per step of goal-shaped income with the object on the table, lift episodes 0.007 after 5,000 epochs.
7. A drop penalty proportional to the episode-long height latch became the largest term (-0.75 per step) and the policy unlearned lifting; charging it only while the object falls removed the collapse.
8. When the downstream gate saturated at four touching digits, a four-digit grasp became optimal: the fifth digit was worth at most 0.14 per step, under 1 % of the downstream income.
9. Multiplying the downstream income by an envelope quality raised the right arm's envelope to 0.86; on the left arm the same reward came with contact force x2.6, arm runaway terminations 5 % -> 42 % and grasp 0.86 -> 0.38.
10. Fingertip pressing, measured by replay on 256 episodes: finger-root joints at 0.00-0.17 of their range, last joints at 0.85-1.00, 66 % of held digit-steps are root-straight / tip-folded, 93-98 % of the fingertip force is normal to the wall. Before lift the last joints sit at 0.11-0.63 and this shape is 4-31 %; after lift it is 47-100 % per digit.
11. A curl measure based on the share of force on the "inner" links paid for pressing, because the second sensed point of each digit is the distal link itself and the fingertip pad is fixed to it: a pad lying on the wall scores full curl; a genuinely rolled finger scored 0.16. The finger's first link (moved by its root joint) has no contact sensor and no ctx point.
12. A link-count wrap depth also scores fingertip pressing as full depth when the distal link and the pad touch together.
13. The spread of each digit's three surface distances (max - min of ctx.link_surface_dist over its three points) separates the shapes in replay: rolled 1.1 cm (thumb 0.6), middle-joint hook 1.9 cm, fingertip press 2.0-2.5 cm.
14. The same reward on the mirrored arm gave different grasps: left full-hand grasp with palm 0.38, right three-digit pinch with palm 0.001.

```

## 참고 2 — cup_grasp_r iter_00 보상 원문(DG-5F 손 — 손 모양 항은 RH 에 없는 필드를 쓴다. 골격만 가져올 것)
```text
import torch
import math


def _soft_contact(force: torch.Tensor, scale: float = 0.15) -> torch.Tensor:
    """0 off the object, ~0.6 at 0.1 N, ->1 above a few tenths of a newton.
    Saturating, so solver force spikes cannot dominate and extra force buys nothing."""
    return torch.tanh(force.clamp(min=0.0) / scale)


def _unit(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe normalisation: a zero vector stays zero (its dot products are then 0)."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _sidewall(normal_z: torch.Tensor) -> torch.Tensor:
    """How HORIZONTAL a surface normal is: 1 on a vertical side wall, 0 on a rim face, a lid or the
    table, ~0.3 on a 45 deg rim edge. Used only from a few centimetres in, always with a floor;
    NEVER on the long-range approach (lesson 2)."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Per-step ceilings:
    #   reach            ~0.60  pre-contact only, x (1 - held)
    #                           = 0.35 PLAIN long-range reach (lesson 1 form, NO open-hand factor)
    #                           + 0.24 palm seating (NO open-hand factor) + 0.14 closing
    #                           + 0.04 ADDITIVE open-hand term (far from the wall only)
    #   keep the grasp   1.20   links-per-digit + count ladder + palm contact + opposition
    #   lift             2.50   x uprightness x real hold (`held`); NOT x env_mult (hypothesis B)
    #   transport        2.50   x off_table x env_mult
    #   hold at rest    10.00   real hold, upright, at the goal, still, x env_mult
    #   first success   60.00   one-off; later successes 2.0, x env_mult
    # =============================================================================================
    W_REACH = 0.35        # long range, PLAIN palm-to-surface distance (lesson 1), fades once held
    W_OPEN = 0.04         # ADDITIVE open-pose income on the way in: below the first digit-contact
    #                       increment (~0.05 at the wall) and far below the 0.195 of lesson 4
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty)
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what a hold needs
    P_TABLE = 0.60        # palm or finger links driven onto the table top
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: small, so lifting and carrying beat standing still
    P_CUP_SPIN = 0.06
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # seating scale
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link
    REACH_FAR = 0.20      # far scale of the plain reach: covers the 0.15-0.49 m start distances
    REACH_NEAR = 0.05     # near scale of the plain reach (last few centimetres)
    OPEN_MEAN_SCALE = 0.12   # mean deviation from the default open pose
    OPEN_MAX_SCALE = 0.30    # and the WORST joint, which is what actually blocks a wrap
    OPEN_NEAR = 0.08      # open-pose term pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # and is zero by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale
    TILT_DEAD = math.radians(5.0)
    TILT_SCALE = math.radians(10.0)
    SQ_TOTAL = 40.0
    SQ_LINK = 15.0
    FALL_DEAD = 0.03
    FALL_SCALE = 0.20
    SET_DEAD = 0.02
    SET_SCALE = 0.05
    SET_LIFTOFF = 0.005
    SET_BAND = 0.04
    GOAL_TOL_MULT = 3.0
    GOAL_TOL_MIN = 0.03
    GOAL_FINE = 0.04
    SETTLE_FLOOR = 0.40
    DWELL_BASE = 0.70
    DWELL_SCALE = 1.5
    ENV_FLOOR = 0.10
    ENV_W_COUNT = 0.30
    ENV_W_DEPTH = 0.35
    ENV_W_PALM = 0.25
    ENV_W_OPP = 0.10
    ENV_DEPTH_LINKS = 2.0
    ENV_NEAR_SHARE = 0.25
    ENV_NEAR_SCALE = 0.01
    CURL_FLOOR = 0.50        # env_q keeps half at zero curl
    FOLLOW_DEAD = 0.010      # m: resolution of the surface distance, NOT a pose target
    FOLLOW_SCALE = 0.006     # m: 1.1 cm -> 0.85, 1.9 cm -> 0.22, 2.0-2.5 cm -> 0.19-0.08
    FLOOR_BAND_TOP = 0.020
    FLOOR_BAND_FULL = 0.010

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0
    digit_joint[1, 4:7] = 1.0
    digit_joint[2, 8:11] = 1.0
    digit_joint[3, 12:15] = 1.0
    digit_joint[4, 17:19] = 1.0
    n_digit_joint = digit_joint.sum(-1)

    # ---- height and vertical velocity ------------------------------------------------------------
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)
    off_table = torch.tanh(height / TABLE_RISE)
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)

    # ---- contact and grasp quality ---------------------------------------------------------------
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3)
    digit_touch = link_c.amax(dim=-1)                              # (N,5)
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)
    finger_any = digit_touch[:, 1:].amax(dim=-1)
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    link_side = _sidewall(ctx.link_surface_normal[..., 2])
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side
    link_cw = link_c * link_side_w
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- open-pose measure (mean AND worst movable joint) ----------------------------------------
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))

    # ---- measured per-digit flexion --------------------------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint
    flex = (closure / 0.5).clamp(max=1.0)

    # ---- palm against the object's real surface --------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    cq = 0.35 + 0.65 * seat

    # open-pose gate: full at 8 cm or more, zero by 2 cm (used ONLY by the additive open term)
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)

    # ---- opposition from the surface normals at the touching links -------------------------------
    normals = ctx.link_surface_normal
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)

    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # ---- envelope quality (multiplier on transport, hold and success only) ------------------------
    digit_gap = (ctx.link_surface_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)
    count_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)
    env_depth = count_depth.mean(dim=-1)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)

    # wall-following curl (lesson 13), a FACTOR of env_q, same formula for every digit
    link_dist = ctx.link_surface_dist
    lift_off = link_dist.amax(dim=-1) - link_dist.amin(dim=-1)
    follow = torch.exp(-(lift_off - FOLLOW_DEAD).clamp(min=0.0) / FOLLOW_SCALE)
    digit_curl = digit_touch * follow
    hand_curl = digit_curl.mean(dim=-1)
    curl_factor = CURL_FLOOR + (1.0 - CURL_FLOOR) * hand_curl
    env_q = env_base * curl_factor
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- fading out once the object is really held
    # =============================================================================================
    fade = loose
    # PLAIN reach of lesson 1: no open-hand factor, no face factor, no side-wall factor (lesson 2)
    reach_far = 1.0 - torch.tanh(palm_d / REACH_FAR)
    reach_near = 1.0 - torch.tanh(palm_d / REACH_NEAR)
    r_reach = W_REACH * fade * (0.6 * reach_far + 0.4 * reach_near) * (0.45 + 0.55 * grip)

    # ADDITIVE open-hand income: only while the palm is far from the surface (full at 8 cm, zero by
    # 2 cm) and zero once held, so it can never become the resting point at the wall (lesson 4)
    r_open = W_OPEN * fade * open_gate * open_pose

    # palm seating paired with finger engagement (no open-hand factor any more)
    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = W_PALM_IN * fade * palm_near * (0.3 + 0.7 * face) * palm_side_w * seat_engage

    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total
    # =============================================================================================
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_oppose = r_oppose * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total. Hypothesis B: gated by `held` and uprightness only, NOT by env_mult, so the
    # first lift is not throttled by an envelope the policy has not formed yet.
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)
    r_lift = W_LIFT * held * upright * lift_shape
    r_rise = W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, x off_table, keeps env_mult
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, x off_table, keeps env_mult
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded, no latch
    # =============================================================================================
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    p_shove = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                          + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    hand_rel = ctx.hand_z_min - ctx.table_z
    floor_t = ((FLOOR_BAND_TOP - hand_rel) / (FLOOR_BAND_TOP - FLOOR_BAND_FULL)).clamp(min=0.0, max=1.0)
    floor_pen = floor_t * floor_t * (3.0 - 2.0 * floor_t)
    p_table = P_TABLE * 0.5 * (palm_pen + floor_pen)
    p_arm_vel = P_ARM_VEL * (0.4 + 0.6 * held) * torch.tanh(ctx.arm_qd.norm(dim=-1) / 1.0)
    p_arm_fast = P_ARM_FAST * torch.tanh((ctx.arm_qd.abs().amax(dim=-1) - 3.0).clamp(min=0.0) / 3.0)
    p_cup_vel = held * (P_CUP_VEL * torch.tanh(cup_speed / 0.5)
                        + P_CUP_SPIN * torch.tanh(spin / 3.0))
    act_rate = (ctx.actions - ctx.prev_actions).pow(2).mean(dim=-1)
    p_action = (P_ACT_RATE * torch.tanh(act_rate / 0.1)
                + P_ARM_ACT * ctx.actions[:, :7].pow(2).mean(dim=-1))

    reward = (
        r_reach + r_open + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_oppose
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero

    components = {
        "reach": r_reach,
        "open_hand": r_open,
        "palm_in": r_palm_in,
        "close": r_close,
        "grasp_count": r_count,
        "grasp_digit": r_digit,
        "grasp_palm": r_palm,
        "grasp_oppose": r_oppose,
        "lift_height": r_lift,
        "lift_rise": r_rise,
        "goal_reach": r_goal,
        "goal_progress": r_progress,
        "goal_hold": r_hold,
        "success_bonus": r_success,
        "pen_drop": -p_drop,
        "pen_setdown": -p_setdown,
        "pen_stall": -p_stall,
        "pen_drift": -p_drift,
        "pen_descend": -p_descend,
        "pen_tilt": -p_tilt,
        "pen_shove": -p_shove,
        "pen_squeeze": -p_squeeze,
        "pen_table": -p_table,
        "pen_arm_vel": -p_arm_vel,
        "pen_arm_fast": -p_arm_fast,
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components

```
