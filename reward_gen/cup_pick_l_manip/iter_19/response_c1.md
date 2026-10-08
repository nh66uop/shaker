# Round 19 응답 (cup_pick_l_manip, 왼팔)

## 1. 과제의 의미와 로봇이 거쳐야 할 단계

과제: 7자유도 왼팔 OpenArm 과 Tesollo DG-5F 손으로, 테이블 위 무작위 위치(x 0.10-0.40 m, y 0.00-0.30 m)에 선
컵(환경마다 반경 44-81 mm, 반높이 42-65 mm)을 집어 들고, 스폰 높이보다 0.21-0.28 m 위(수평 5 cm 이내)의 목표점까지
똑바로 세운 채 옮긴 뒤, 목표 0.15 m 안에서 가해지는 외력과 토크(5 N/kg 에서 시작해 성공할수록 20 N/kg 까지)에
버티며 그 자리에 머문다. 성공은 환경이 판정한다: goal_dist <= success_tol 이 success_hold_steps 동안 이어지고, 그
순간 손가락이 컵 몸통을 감싸고 있어야 한다. 성공마다 다음 목표가 주어지고(최대 5개), tolerance 는 0.1125 m 에서
0.015 m 쪽으로 줄어든다.

로봇이 거쳐야 할 단계:

1. 접근: 손을 기본 자세로 편 채 컵 쪽으로 이동한다. 멀리서 가장 가까운 표면점은 림이므로 측벽 선호는 가까이서만 둔다.
2. 안착: 손바닥을 측벽에 붙이고(0.08 -> 0.02 m 구간이 승부처) 엄지를 손가락 반대편에 둔다.
3. 감싸기: 다섯 digit 이 모두 닿고, 손끝이 아니라 안쪽 두 마디(k = 0, 1)에 힘을 싣는다(operator 요구). 손바닥 접촉과
   엄지·손바닥 대향이 함께 있어야 한다.
4. 들어올리기: 컵 자신의 스폰 높이 대비 상승, 기울지 않게. 테이블 위에 쥔 채 머무는 것은 손해여야 한다.
5. 운반: goal_dist 를 줄이는 방향으로 옮긴다. 목표에서 멀어지는 움직임은 비용이다.
6. 정지와 방어: 목표에서 멈추고, 외란에 밀려도 자리를 지키며 놓치지 않는다.
7. 연쇄: 성공 뒤 다음 목표로 이어간다. 첫 성공만 큰 보너스, 이후 수입은 hold 항에서 나온다.

현재 정책은 1-7 을 모두 수행한다(round 15 부터 lift, carry, hold; round 18 e1250 에서 3-goal 에피소드 0.67, 5-goal
0.56). 남은 품질 과제는 3번의 "안쪽 마디로 감싸기" 하나이고, round 18 이 그 방향으로 움직였다.

## 2. Round 18 결과: 빠지거나 틀린 단계가 있는가

결론: reward 안에서 빠지거나 틀린 단계를 찾지 못했다. round 18 코드를 그대로 유지한다. 아래 코드는 round 18 파일과
주석까지 한 글자도 다르지 않다(좌우 byte-identical 비교와 diff 확인을 쉽게 하려고 주석도 바꾸지 않았다).

### 2.1 round 18 의 두 변경은 의도대로 작동했다

| 항목 | 시작 e1000 | e450 | e800 | e1250 | 비고 |
|---|---|---|---|---|---|
| 고정 tol 성공률 (512 ep) | 0.785 | 0.811 | 0.842 | 0.785 | e800 이 최고, round 17 e800 은 0.703 |
| 3-goal 에피소드 | 0.43 | 0.59 | 0.66 | 0.67 | 에피소드당 성공 2.41 -> 3.36 |
| index 안쪽 마디 힘 비율 (성공 시점) | 0.315 | 0.253 | 0.338 | 0.307 | round 17: 0.010 |
| 다섯 digit 평균 안쪽 비율 | 0.188 | - | - | 0.212 | z +2.6, ring 0.13 -> 0.19 |
| 파지 전 쓰러뜨림 중 손 낮음 | 13/21 (r17) | - | 0/15 | 2/14 | table band (rule h) 효과 |

학습 곡선도 같은 방향이다: 성공 시점 접촉 digit 4.75 -> 4.97, 성공 시점 접촉 link 6.05 -> 6.36(round 15 는 오른팔
5.4, 왼팔 4.8), index 접촉 0.61 -> 0.72(round 15 는 오른팔 0.000, 왼팔 0.10), tolerance 0.100 -> 0.059, tilt
5.3 -> 4.0 deg, total reward 1.43 -> 1.67.

### 2.2 네 가지 관찰과 reward 로 고칠 일인지

| 관찰 | 측정 | 판단 | reward 조치 |
|---|---|---|---|
| 조기 종료 0.248 -> 0.336 | 초과분 전부가 lift 뒤 y 이탈(0.051 -> 0.105 -> 0.160). 다른 원인은 평탄하거나 감소. 운반 leg 당 종료 위험 0.054 -> 0.063 (z 0.7-0.8) | 왼팔이 오른팔 경계(y <= 0.25 m)를 쓰고 goal box 는 0.35 m 까지. 목표를 더 많이 잇는 정책일수록 경계 너머 목표를 더 만난다. reward 는 world axis 나 경계를 이름 붙일 수 없다(mirror-safe 규칙) | 없음. 왼팔 경계를 거울 대칭으로 맞추는 것은 환경 설정이며 operator 판단 |
| 그립 71 -> 65 -> 59 N | ring 이 힘을 손끝(9.2 -> 5.2 N)에서 안쪽으로 옮김(curl factor 가 의도한 변화). 엄지는 squeeze charge 아래서 가벼워진 손가락에 맞춰 덜 누름. drop 0.002, 최악 tilt 20 -> 16 deg, 종료·비종료 에피소드 파지력 동일(60 vs 55 N) | 약한 파지가 아니다. 힘 하한은 절대 힘 목표라 금지 | 없음. curl factor, inner-force trust, squeeze charge 유지 |
| 파지한 컵의 0.6-0.8 m/s 옆 이동 | 경계 쪽과 반대쪽 모두에서 발생, 경계는 한쪽에만 있음 | 목표에서 멀어지는 움직임은 이미 부호 있는 r_progress 와 p_drift 가 과금한다. 속도 charge 를 올리면 모든 운반이 과세되어 정지가 다시 경쟁력을 얻는다(round 10 의 주차 함정) | 없음 |
| e800 -> e1250 성공률 0.842 -> 0.785 (z -2.3) | 같은 구간 y 이탈 0.105 -> 0.160 | reward 신호로 볼 근거가 없다 | 없음. best snapshot 은 e800 유지 |

### 2.3 코드 규칙 점검 (round 18 코드 그대로)

- (a) transport, progress, hold, success 는 모두 off_table 을 곱한다. 컵이 테이블 위에 있으면 0 이다.
- (b) 테이블 위에서 얻을 수 있는 수입은 grasp 1.2 가 상한이고 후반에는 0.54 로 줄며 stall 이 최대 -0.4 를 뺀다. 들어올린
  상태는 지금 grasp(env_mult 약 0.57, held 약 0.9, hold_grip 약 0.86)로도 lift 약 1.1 + transport 약 0.8 + hold
  약 3.4-4.9 + grasp 1.2 이다(추정). lift 가 여전히 유일하고 가장 큰 문이다.
- (c)(d) open-pose 는 접근 구간에만, palm seating 은 engagement 와 짝. 학습 중 reach 0.006, palm_in 0.002 로 파지
  뒤에는 사라진다.
- (e) 지속 latch 에 비례하는 penalty 가 없다. lifted 는 r_rise 수입의 gate 로만 쓰인다.
- (f)(g) env_q 는 digit 을 정렬·평균으로만 다루고 digit 이름이 없다. curl 은 곱셈 factor 라 palm 이나 ladder 로 되살 수
  없다.
- (h) floor band 2 cm -> 1 cm. 측벽을 감싼 손(가장 낮은 link 약 8-9 cm)은 과금되지 않는다.
- penalty 크기(학습 평균): squeeze -0.22 가 가장 크지만 평탄하고, drift -0.085, arm_vel -0.057, cup_vel -0.037,
  tilt -0.036, action -0.023, drop -0.018. 어느 것도 downstream 수입(goal_hold 0.68 + lift 0.67 + goal_reach 0.23)을
  압도하지 않는다.
- 형태: RewardContext 목록 밖 필드 없음, 모든 반환 텐서 (N,), torch 와 math 만 사용, 상태 보관 없음.

### 2.4 다음에 볼 것 (reward 밖)

- operator 영상 판정: index 0.31-0.34, 다섯 digit 평균 0.21 의 안쪽 비율이 요구한 "감싸기" 인지. 판정이 "아직
  손끝" 이면 다음 라운드 후보는 CURL_FULL_SHARE 또는 CURL_FLOOR 조정이지만 이번 라운드에서는 건드리지 않는다.
- 왼팔 y 경계를 고친 뒤 lift 뒤 조기 종료가 시작점 수준(약 0.25)으로 돌아오는지, done/out_xy 가 줄어드는지.
- hand_curl, env_mult, index 안쪽 비율, pen_squeeze 추세.
- 재개 시 success_tol 이 0.1125 로 다시 시작하는 문제는 환경 쪽이며 launch 에서 처리한다.

## 3. 개선 코드 (round 18 과 동일, 변경 없음)

```python
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
    table, ~0.3 on a 45 deg rim edge. Only the z component of a context normal is used, and gravity
    is shared by the left arm and the mirrored right arm, so this stays mirror-safe.

    It is NEVER applied to the long-range approach (round 11: from 0.15-0.49 m away the nearest
    point of an open cup is its rim, the approach paid ~0 and both arms retreated). It is used only
    from a few centimetres in, always with a floor."""
    return (1.0 - normal_z.abs()).clamp(min=0.0, max=1.0)


def compute_reward(ctx: "RewardContext") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = ctx.goal_dist.device
    dtype = ctx.goal_dist.dtype

    # =============================================================================================
    # WEIGHTS. Object-agnostic and arm-agnostic: surface distances/normals, contact forces, object
    # pose/velocity, goal. No approach/envelope latch, no radius/axis/half-height, no world-frame
    # direction constant (only normal z components and heights, identical under the mirror).
    #
    # Per-step ceilings (the budget IS the design, unchanged since round 15):
    #   reach            0.60   pre-contact only, x (1 - held); it vanishes once the object is held
    #                           = 0.22 long range (PLAIN distance) + 0.24 palm seating + 0.14 closing
    #   keep the grasp   1.20   links-per-digit + count ladder (only the TOP rungs need the palm)
    #                           + palm contact (0.30) + opposition
    #   lift             2.50   x uprightness x real hold, rising to the height the goal needs
    #   transport        2.50   1.60 on goal_dist over 0-0.30 m + 0.90 on the signed RATE it falls
    #   hold at rest    10.00   real hold, upright, at the goal, still, growing with successes
    #   first success   60.00   one-off; later successes 2.0, so staying pays, not re-arriving
    #   lift, rise, transport, progress, hold and success are ALL x env_mult in [0.10, 1] (round 16).
    #
    # STRUCTURAL RULES, every one of them a scar from an earlier round:
    #  (a) Everything from "transport" down is x `off_table`, a smooth function of the object's rise
    #      above its OWN spawn height: nothing there pays while the object rests (round 10).
    #  (b) Stage one is ALWAYS profitable: no factor zeroes the long-range approach and no penalty
    #      exceeds the income of the stage the policy is in (round 11).
    #  (c) The hand is paid to stay OPEN only on the way in: the open-pose factor fades with palm
    #      distance (round 12 arrived curled; round 13 parked a flat open hand on the wall).
    #  (d) Palm seating is paired with finger engagement, never paid on proximity alone (round 13).
    #  (e) NO PENALTY IS PROPORTIONAL TO A PERSISTENT LATCH (round 14). Losing the object costs a
    #      bounded amount ONLY WHILE IT IS FALLING, setting it down only WHILE IT COMES DOWN onto the
    #      table; the main cost of a drop is the downstream income that stops. `lifted` survives
    #      only as the switch that stops the rise bonus from being paid twice (income, never negative).
    #  (f) THE DOWNSTREAM INCOME SCALES WITH ONE ENVELOPE QUALITY `env_q` (round 16), built only from
    #      quantities that treat the digits alike, on the same soft contact forces as `held`; no
    #      term names a digit. `held` keeps every GATE (reach fade, drop, stall, drift, descend).
    #  (g) ROUND 18: CURL IS A FACTOR OF env_q, NOT A SUMMAND. Round 17 weighted the depth summand by
    #      each digit's inner-link force share; the policy moved the index grip onto its fingertip
    #      (env_q -0.006) and bought it back with palm seating (+0.019), and even a full curl of every
    #      digit was worth only 8-9 % of the downstream income. Now: per digit, curl = share of its
    #      OWN cup force on its two inner links (k=0, k=1) against its fingertip (k=2), full at one
    #      half, x a smooth trust that its INNER links press; a digit that does not touch gives 0 (no
    #      fallback). hand_curl = mean over the five digits; env_q = env_base x (CURL_FLOOR +
    #      (1 - CURL_FLOOR) x hand_curl), so the ladder, palm and opposition can no longer pay back a
    #      fingertip press. The depth summand is round 16's link count again (the force share enters
    #      once, as this factor).
    #  (h) ROUND 18: THE TABLE CHARGE FIRES WHERE A LINK ACTUALLY TOUCHES THE TABLE. hand_z_min is a
    #      point inside the links, so a link resting on the table reads ~1-2 cm; the round-17 band
    #      (2 mm) never fired (pen_table 0.0000) while the hand came down onto the table. The floor
    #      half now starts 2 cm above the top and is full by 1 cm. Same 0.6 cap, same palm half,
    #      current height only (no latch, bounded).
    #
    # Steady states (estimated from the round-17 replay): grasp parked on the table ~1.2 early, ~0.2
    # late; round-17 grasp (hand_curl ~0.27) at rest on the goal ~0.56 of full downstream income
    # (round 17 paid it ~0.73); every further digit that curls +~0.065 of full; five digits curled
    # with the palm -> ~0.82 of full, i.e. all of what env_base allows; a first-success spike <= 60.
    # =============================================================================================
    W_REACH = 0.22        # long range, PLAIN palm-to-surface distance, x open-pose  } 0.60 total,
    W_PALM_IN = 0.24      # palm seating: steep across 0.08 -> 0.02 m, wall-weighted } x (1 - held)
    W_CLOSE = 0.14        # measured flexion x link nearness to a wall x palm seat   }
    W_COUNT = 0.40        # digit-count ladder, sorted; only the TOP rungs need the palm }
    W_DIGIT = 0.30        # per-digit LINK COUNT, every digit weighted alike            } 1.20:
    W_PALM = 0.30         # palm contact: a pinch must not score like a whole-hand wrap } build and
    W_OPPOSE = 0.20       # thumb/palm against the fingers (surface normals)            } keep a grasp
    W_LIFT = 2.10         # height above the object's own spawn height, capped at goal height }
    W_RISE = 0.40         # upward speed, only until `lifted` latches (income gate, not penalty) } 2.50
    W_GOAL = 1.60         # goal_dist, constant slope over the whole 0-0.30 m        } 2.50 total,
    W_PROGRESS = 0.90     # signed rate at which goal_dist falls                     } x off_table
    W_HOLD = 10.0         # AT REST AT THE GOAL, upright, really held: the dominant income
    W_SUCCESS_FIRST = 60.0  # FIRST success of an episode
    W_SUCCESS_LATER = 2.0   # later successes are cheap: staying must pay through the hold term

    P_DROP = 2.00         # per step, ONLY while the object is falling without a hold (below one
    #                       step of lift income; a whole drop costs a bounded amount once)
    P_SETDOWN = 1.00      # per step, ONLY while the object is coming down onto the table
    P_STALL = 0.40        # time pressure, gated on actually being ON the object
    P_DRIFT = 1.50        # moving AWAY from the goal near it: costs more than the hold it gives up
    P_DESCEND = 0.60      # lowering a held object BELOW goal height < keeping it still < raising it
    P_TILT = 0.25         # small: uprightness is enforced as a MULTIPLIER, not as a penalty
    P_SHOVE = 0.50        # shoving / tipping the object while it is still on the table
    P_SQUEEZE = 0.50      # force beyond what a hold needs (kept as is, by instruction)
    P_TABLE = 0.60        # palm or finger links driven onto the table top (cap unchanged)
    P_ARM_VEL = 0.12      # smoothness; ~0.05 at the commanded ~0.3 rad/s, far below the lift income
    P_ARM_FAST = 0.80     # only above 3 rad/s: runaway states only
    P_CUP_VEL = 0.10      # object speed while held: 0.02 at 0.1 m/s against +0.7 of transport income,
    P_CUP_SPIN = 0.06     #   so lifting and carrying always beat standing still
    P_ACT_RATE = 0.02     # action jitter
    P_ARM_ACT = 0.01      # arm action magnitude only (the fingers must be free to reach +-1)

    PALM_DEAD = 0.02      # palm_surface_dist reads up to ~1 cm high at the surface (512 samples)
    PALM_STEEP = 0.02     # seating scale: 0.05 at 0.08 m, 0.22 at 0.05 m, 0.61 at 0.03 m, 1 at 0.02
    LINK_DEAD = 0.025     # link points sit inside the links: a touching link reads 1-3 cm (1.8 avg)
    CLOSE_IN = 0.03       # "close in" scale: the side-wall preference fades in inside a few cm
    SIDE_FLOOR_PALM = 0.25   # a rim still pays a quarter of a wall, never zero
    SIDE_FLOOR_LINK = 0.35   # ditto per finger link (contact geometry is noisier)
    OPEN_FLOOR = 0.15     # a fully curled hand still earns 15 % of the approach: rule (b)
    OPEN_MEAN_SCALE = 0.12   # mean deviation from the default open pose
    OPEN_MAX_SCALE = 0.30    # and the WORST joint, which is what actually blocks a wrap
    OPEN_NEAR = 0.08      # open-pose pays in full at this palm-surface distance or beyond
    OPEN_CONTACT = 0.02   # and its credit is gone by here, where contact starts
    PALMIN_ENGAGE_FLOOR = 0.35   # seating credit an entirely disengaged hand still keeps
    TABLE_RISE = 0.04     # smooth "off the table" scale: 0 at 0, 0.46 at 2 cm, 0.96 at 8 cm
    TILT_DEAD = math.radians(5.0)    # a few degrees of wobble is not a tilted carry
    TILT_SCALE = math.radians(10.0)  # 15 deg -> x0.37, 30 deg -> x0.08 of lift/transport/hold
    SQ_TOTAL = 40.0       # measured: a grip that holds fine paid -0.45/step at a 15 N threshold
    SQ_LINK = 15.0
    # ---- round 15 constants -------------------------------------------------------------------
    FALL_DEAD = 0.03      # m/s: solver jitter and a slow deliberate lowering are not a fall
    FALL_SCALE = 0.20     # m/s: free fall passes this within ~2 steps, so a real drop is charged fully
    SET_DEAD = 0.02       # m/s: a resting object never triggers the set-down cost
    SET_SCALE = 0.05      # m/s
    SET_LIFTOFF = 0.005   # m: the set-down band starts 5 mm above the spawn height (0 when resting)
    SET_BAND = 0.04       # m: and fades out over the last ~6 cm above the table
    GOAL_TOL_MULT = 3.0   # broad hold region = 3 tolerances: 0.89 at one tolerance, 0.64 at two
    GOAL_TOL_MIN = 0.03   # never narrower than 3 cm, however small the tolerance gets
    GOAL_FINE = 0.04      # fixed fine ramp for the last centimetres
    SETTLE_FLOOR = 0.40   # a push that moves the object costs part of the hold income, not 2/3 of it
    DWELL_BASE = 0.70     # hold income before the first success of the episode
    DWELL_SCALE = 1.5     # -> 0.87 after one success, 0.96 after two
    # ---- round 16 constants: the envelope quality (rule (f)) -----------------------------------
    ENV_FLOOR = 0.10      # env_mult = 0.10 + 0.90 * env_q: never zero, a four-digit hold keeps earning
    ENV_W_COUNT = 0.30    # sorted engagement ladder (the fifth digit is its biggest rung)  } sum 1.0
    ENV_W_DEPTH = 0.35    # links in contact per digit (round-16 link count again)         }
    ENV_W_PALM = 0.25     # palm contact                                                    }
    ENV_W_OPP = 0.10      # thumb/palm opposition, same saturation as `held`                }
    ENV_DEPTH_LINKS = 2.0  # "two or more links each" is full, a fingertip alone half
    ENV_NEAR_SHARE = 0.25  # a digit hovering at the surface is worth at most 1/4 of a touching one
    ENV_NEAR_SCALE = 0.01  # m: pre-contact on-ramp, about the last 1-2 cm before a link touches
    # ---- ROUND 18 constants: curl as a factor (rule (g)) and the table band (rule (h)) ---------
    CURL_FULL_SHARE = 0.50   # inner links carrying half or more of the digit's OWN cup force = curled
    CURL_FORCE_REF = 0.30    # N: trust that the INNER links press, tanh(inner force / 0.3): 0.76 at
    #                          0.3 N. On the inner force, not the digit total, so adding fingertip
    #                          force can never raise a digit's curl (c1r repair of c1)
    CURL_FLOOR = 0.50        # env_q keeps half at zero curl. Sized on the round-17 replay: its grasp
    #                          (hand_curl ~0.27) keeps ~3/4 of what it earns now, all five digits
    #                          curled earn all of env_base, each further curled digit ~+6.5 % of the
    #                          full downstream income (more than the 5-8 % the palm buy-back paid)
    FLOOR_BAND_TOP = 0.020   # m above the table top: the floor charge starts here
    FLOOR_BAND_FULL = 0.010  # m: and is full here (a link resting on the table reads ~1-2 cm)

    zero = torch.zeros_like(ctx.goal_dist)
    lifted = ctx.lifted.to(dtype)
    success = ctx.success.to(dtype)
    n_succ = ctx.num_successes.to(dtype)

    # movable finger joints per digit (ranges <= 0.05 rad are locked; their normalisation is noise)
    # locked = thumb_2(0), index_1(3), middle_1(7), ring_1(11), pinky_1(15), pinky_2(16)
    movable = torch.tensor(
        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        device=device, dtype=dtype,
    )
    n_movable = movable.sum().clamp(min=1.0)
    digit_joint = torch.zeros((5, 19), device=device, dtype=dtype)
    digit_joint[0, 1:3] = 1.0      # thumb_3, thumb_4
    digit_joint[1, 4:7] = 1.0      # index_2, _3, _4
    digit_joint[2, 8:11] = 1.0     # middle_2, _3, _4
    digit_joint[3, 12:15] = 1.0    # ring_2, _3, _4
    digit_joint[4, 17:19] = 1.0    # pinky_3, pinky_4
    n_digit_joint = digit_joint.sum(-1)                                        # (5,)

    # =============================================================================================
    # HEIGHT AND VERTICAL VELOCITY: the spine of this reward.
    # =============================================================================================
    height = (ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.0)      # rise above ITS start
    h_need = (ctx.goal_pos[:, 2] - ctx.cup_spawn_pos[:, 2]).clamp(min=0.05)    # read from the goal
    off_table = torch.tanh(height / TABLE_RISE)     # smooth, exactly 0 on the table
    on_table = 1.0 - off_table
    vz = ctx.cup_lin_vel[:, 2]

    # uprightness: ONE multiplier on every lift / transport / hold income (tilt fell 28 -> 5 deg when
    # it became a multiplier instead of a small additive penalty).
    upright = torch.exp(-(ctx.cup_tilt - TILT_DEAD).clamp(min=0.0) / TILT_SCALE)     # (N,) in (0,1]

    # =============================================================================================
    # CONTACT AND GRASP QUALITY -- paid by LINKS, not by "this digit touches"
    # =============================================================================================
    link_c = _soft_contact(ctx.link_cup_force)                     # (N,5,3) plain contact
    digit_touch = link_c.amax(dim=-1)                              # (N,5) does this digit touch
    palm_c = _soft_contact(ctx.palm_cup_force)                     # (N,)
    n_touch = digit_touch.sum(dim=-1)                              # (N,) 0..5
    finger_any = digit_touch[:, 1:].amax(dim=-1)                   # (N,) any of the four fingers
    grip = (0.8 * n_touch / 5.0 + 0.2 * palm_c).clamp(min=0.0, max=1.0)

    # WHERE the links touch: a link on a vertical WALL counts fully, one hooked over a rim or lying on
    # a lid counts 0.35 (a touching link is close in by definition). Scoring only: `digit_touch` and
    # `held` stay on the plain force, so geometry noise cannot dissolve a real hold under the wrench.
    link_side = _sidewall(ctx.link_surface_normal[..., 2])                          # (N,5,3)
    link_side_w = SIDE_FLOOR_LINK + (1.0 - SIDE_FLOOR_LINK) * link_side             # (N,5,3)
    link_cw = link_c * link_side_w                                                  # (N,5,3)
    # NUMBER OF LINKS of each digit, convex: 1 wall link -> 0.19, 2 -> 0.54, 3 -> 1.00; one fingertip
    # on the rim -> 0.04.
    digit_links = (link_cw.sum(dim=-1) / 3.0).clamp(min=0.0, max=1.0).pow(1.5)      # (N,5)

    # ---- the hand's OPEN-POSE measure (mean AND worst movable joint) -----------------------------
    dev = (ctx.hand_q_norm - ctx.hand_default_q_norm).abs() * movable               # (N,19)
    dev_mean = dev.sum(dim=-1) / n_movable
    dev_max = dev.amax(dim=-1)
    open_pose = 0.5 * (torch.exp(-dev_mean / OPEN_MEAN_SCALE)
                       + torch.exp(-dev_max / OPEN_MAX_SCALE))                      # (N,) in (0,1]

    # ---- measured per-digit flexion (contact-free) -----------------------------------------------
    head_room = (1.0 - ctx.hand_default_q_norm).clamp(min=0.1)
    rel_flex = ((ctx.hand_q_norm - ctx.hand_default_q_norm) / head_room).clamp(min=0.0, max=1.0)
    closure = (rel_flex.unsqueeze(1) * digit_joint.unsqueeze(0)).sum(-1) / n_digit_joint   # (N,5)
    flex = (closure / 0.5).clamp(max=1.0)        # saturates at half flexion: no crushing drive

    # ---- palm against the object's real surface --------------------------------------------------
    palm_d = ctx.palm_surface_dist.clamp(min=0.0)
    # facing: the palm normal points at the nearest surface point AND against the outward normal
    # there (dot products of context vectors only, nothing tied to a world axis)
    face = (0.5 * (ctx.palm_normal * ctx.palm_surface_dir).sum(-1)
            - 0.5 * (ctx.palm_normal * ctx.palm_surface_normal).sum(-1)).clamp(min=0.0, max=1.0)
    palm_side = _sidewall(ctx.palm_surface_normal[:, 2])           # 1 at a wall, ~0 at a rim/lid
    # side-wall preference only close in: full at 3 cm, 0.37 at 6 cm, 0.05 at 12 cm, floored
    palm_close = torch.exp(-(palm_d - CLOSE_IN).clamp(min=0.0) / CLOSE_IN)
    palm_side_w = 1.0 - palm_close * (1.0 - (SIDE_FLOOR_PALM + (1.0 - SIDE_FLOOR_PALM) * palm_side))
    # steep seating gradient across 0.08 -> 0.02 m (both arms once stalled at a 0.05 m palm gap)
    palm_near = torch.exp(-(palm_d - PALM_DEAD).clamp(min=0.0) / PALM_STEEP)
    seat = torch.maximum(palm_near * (0.3 + 0.7 * face) * palm_side_w, palm_c)
    # contact quality: a first partial wrap is worth forming before the palm has arrived
    cq = 0.35 + 0.65 * seat

    # open-pose factor, rule (c): full weight at 8 cm or more, gone by 2 cm, floored, never zero
    open_gate = ((palm_d - OPEN_CONTACT) / (OPEN_NEAR - OPEN_CONTACT)).clamp(min=0.0, max=1.0)
    open_mul = OPEN_FLOOR + (1.0 - OPEN_FLOOR) * open_pose * open_gate

    # ---- opposition from the surface normals at the touching links ------------------------------
    normals = ctx.link_surface_normal                                        # (N,5,3,3)
    thumb_n = _unit((link_c[:, 0, :].unsqueeze(-1) * normals[:, 0]).sum(dim=1))           # (N,3)
    finger_n = _unit((link_c[:, 1:, :].unsqueeze(-1) * normals[:, 1:]).sum(dim=(1, 2)))   # (N,3)
    thumb_opp = (-(thumb_n * finger_n).sum(-1)).clamp(min=0.0) * digit_touch[:, 0] * finger_any
    palm_opp = ((-(ctx.palm_surface_normal * finger_n).sum(-1)).clamp(min=0.0)
                * palm_c * finger_any)
    opposition = (0.65 * thumb_opp + 0.35 * palm_opp).clamp(min=0.0, max=1.0)             # (N,)

    # "held" = palm plus several OPPOSING digits (smooth, full at four digits). Unchanged: it is the
    # GATE for the reach fade and every cost, so a real four-digit hold is never turned into penalties.
    digit_hold = ((n_touch - 2.0) / 2.0).clamp(min=0.0, max=1.0)
    opp_sat = (opposition / 0.5).clamp(max=1.0)
    held = digit_hold * (0.3 + 0.7 * palm_c) * (0.5 + 0.5 * opp_sat)                      # (N,)
    # the 10/step asks for more than the lift does: palm and opposition enter a SECOND time, so a
    # fingertip touch can never collect the dominant income, even under the wrench
    hold_grip = held * (0.4 + 0.6 * palm_c) * (0.5 + 0.5 * opp_sat)
    loose = (1.0 - held).clamp(min=0.0, max=1.0)

    # =============================================================================================
    # ENVELOPE QUALITY -- ONE smooth multiplier on every downstream income (rule (f)).
    # Symmetric across the digits (sorted / summed / averaged, nothing names a digit), object-agnostic.
    # =============================================================================================
    # per-digit engagement: its soft contact, or at most ENV_NEAR_SHARE of it while its nearest link
    # is within ~1-2 cm of the surface (unchanged from round 16)
    digit_gap = (ctx.link_surface_dist.amin(dim=-1) - LINK_DEAD).clamp(min=0.0)           # (N,5)
    digit_eng = torch.maximum(digit_touch, ENV_NEAR_SHARE * torch.exp(-digit_gap / ENV_NEAR_SCALE))
    # sorted ladder, strongest -> weakest digit: four digits reach 0.25 of it, the FIFTH is the real
    # step (0.75). Sorting makes it symmetric: whichever digit arrives last carries the big rung.
    env_rank = torch.tensor([0.02, 0.02, 0.06, 0.15, 0.75], device=device, dtype=dtype)
    eng_desc, _ = torch.sort(digit_eng, dim=-1, descending=True)
    env_count = (eng_desc * env_rank).sum(dim=-1)                                          # (N,)
    # wrap depth, round 16 again: links in contact per digit, full at two links, mean over five
    count_depth = (link_c.sum(dim=-1) / ENV_DEPTH_LINKS).clamp(max=1.0)                   # (N,5)
    env_depth = count_depth.mean(dim=-1)                                                   # (N,)
    env_base = (ENV_W_COUNT * env_count + ENV_W_DEPTH * env_depth
                + ENV_W_PALM * palm_c + ENV_W_OPP * opp_sat).clamp(min=0.0, max=1.0)       # (N,)

    # ---- ROUND 18 (rule (g)): CURL, a FACTOR of the envelope quality ------------------------------
    # share of each digit's OWN cup force carried by its two inner links (k=0 moved by joint _3,
    # k=1 moved by joint _4) against its fingertip (k=2). Relative, never an absolute force target;
    # pressing harder with the tip lowers it. Same formula for all five digits, no joint angle.
    link_f = ctx.link_cup_force.clamp(min=0.0)                                             # (N,5,3)
    digit_f = link_f.sum(dim=-1)                                                           # (N,5)
    inner_share = ((link_f[..., 0] + link_f[..., 1])
                   / digit_f.clamp(min=1e-6)).clamp(min=0.0, max=1.0)                      # (N,5)
    # smooth trust that the INNER links press: a digit off the object gives exactly 0, never a
    # fallback value, so lifting a finger off the object can never raise the hand's curl; read on
    # the inner force, so extra fingertip force lowers the share and never raises the trust
    inner_f = link_f[..., 0] + link_f[..., 1]                                              # (N,5)
    press = torch.tanh(inner_f / CURL_FORCE_REF)                                           # (N,5)
    digit_curl = (inner_share / CURL_FULL_SHARE).clamp(max=1.0) * press                    # (N,5)
    hand_curl = digit_curl.mean(dim=-1)                                                    # (N,)
    curl_factor = CURL_FLOOR + (1.0 - CURL_FLOOR) * hand_curl                              # [0.5, 1]
    # the ladder, palm and opposition sit INSIDE the factor, so none of them can buy back a digit
    # that moves its grip onto its fingertip
    env_q = env_base * curl_factor                                                         # (N,)
    env_mult = ENV_FLOOR + (1.0 - ENV_FLOOR) * env_q

    # =============================================================================================
    # REACH -- 0.60 total, fading out once the object is really held
    # =============================================================================================
    fade = loose
    reach_far = (1.0 - torch.tanh(palm_d / 0.20)) * (0.4 + 0.6 * face)   # 0.20 m covers 0.15-0.49 m
    r_reach = W_REACH * fade * open_mul * reach_far * (0.45 + 0.55 * grip)

    # palm seating paired with finger engagement (rule (d)): a flat open hand on the wall earns a
    # small fraction; a hand that has started closing keeps the full steep gradient
    engage = (0.5 * flex.mean(dim=-1) + 0.5 * grip).clamp(min=0.0, max=1.0)
    seat_engage = PALMIN_ENGAGE_FLOOR + (1.0 - PALMIN_ENGAGE_FLOOR) * engage
    r_palm_in = (W_PALM_IN * fade * open_mul * palm_near * (0.3 + 0.7 * face) * palm_side_w
                 * seat_engage)

    # closing only pays once the links are near a WALL with the palm seated
    link_near = torch.exp(-(ctx.link_surface_dist - LINK_DEAD).clamp(min=0.0) / 0.02)      # (N,5,3)
    near_digit = (link_near * (0.25 + 0.75 * link_side)).amax(dim=-1)                       # (N,5)
    r_close = W_CLOSE * fade * seat * (flex * near_digit).mean(dim=-1)

    # =============================================================================================
    # KEEP THE GRASP -- 1.20 total. Every digit pays alike; two digits are worth far less than four.
    # =============================================================================================
    # sorted rank weights (sum 1.0): 2 digits 0.18, 3 = 0.42, 4 = 0.80, 5 = 1.00
    rank_w = torch.tensor([0.06, 0.12, 0.24, 0.38, 0.20], device=device, dtype=dtype)
    score_desc, _ = torch.sort(digit_links, dim=-1, descending=True)
    rungs = score_desc * rank_w                                            # (N,5)
    # lower rungs pay WITHOUT the palm (a partial wrap is worth forming); the top rungs need it
    r_count = W_COUNT * cq * (rungs[:, :3].sum(dim=-1)
                              + rungs[:, 3:].sum(dim=-1) * (0.25 + 0.75 * palm_c))
    r_digit = W_DIGIT * cq * digit_links.mean(dim=-1)
    # palm contact is worth having, but only together with digits
    r_palm = W_PALM * palm_c * (0.25 + 0.75 * (n_touch / 3.0).clamp(max=1.0))
    r_oppose = W_OPPOSE * opposition

    # a closed grasp parked on the table becomes unattractive as the episode goes on: the on-table
    # GRASP income decays to 45 % (never the reach terms, rule (b)). episode_progress restarts after
    # a success, so this is time pressure, not a latch.
    late = ((ctx.episode_progress - 0.2) / 0.8).clamp(min=0.0, max=1.0)
    table_decay = 1.0 - 0.55 * late * on_table
    r_close = r_close * table_decay
    r_count = r_count * table_decay
    r_digit = r_digit * table_decay
    r_palm = r_palm * table_decay
    r_oppose = r_oppose * table_decay

    # =============================================================================================
    # LIFT -- 2.50 total, the largest single step up in income and the ONLY door to the goal terms.
    # =============================================================================================
    h_eff = torch.minimum(height, h_need)
    lift_shape = 0.5 * torch.tanh(h_eff / 0.03) + 0.5 * (h_eff / h_need)     # steep first 3 cm
    r_lift = W_LIFT * held * upright * lift_shape * env_mult
    # upward speed pays only until `lifted` latches: the ACT of lifting is never paid twice. This is
    # the only remaining use of the latch, and it gates income, so it can never become a penalty.
    r_rise = (W_RISE * held * upright * (1.0 - lifted) * torch.tanh(vz.clamp(min=0.0) / 0.10)
              * env_mult)

    # =============================================================================================
    # TRANSPORT -- 2.50 total, ALL of it x off_table
    # =============================================================================================
    goal_far = 1.0 - (ctx.goal_dist / 0.30).clamp(max=1.0)           # constant slope over 0-0.30 m
    goal_fine = torch.exp(-ctx.goal_dist / 0.06)
    r_goal = (W_GOAL * held * upright * off_table * (0.55 * goal_far + 0.45 * goal_fine)
              * env_mult)

    # the REDUCTION of goal_dist between steps, read statelessly as the closing speed towards the
    # goal. Signed, so receding costs what approaching pays; faded inside ~3 cm so jitter at rest
    # and the environment's pushes neither pay nor punish.
    to_goal = ctx.goal_pos - ctx.cup_pos
    closing = (ctx.cup_lin_vel * _unit(to_goal)).sum(-1)
    approach_scale = 1.0 - torch.exp(-ctx.goal_dist / 0.03)
    r_progress = (W_PROGRESS * held * upright * off_table * approach_scale
                  * torch.tanh(closing / 0.10) * env_mult)

    # =============================================================================================
    # ARREST AND DEFEND -- 10/step, the dominant income, also x off_table
    # =============================================================================================
    cup_speed = ctx.cup_lin_vel.norm(dim=-1)
    spin = ctx.cup_ang_vel.norm(dim=-1)
    # a flat-topped broad region on three CURRENT tolerances (0.89 at one tolerance, 0.64 at two)
    # plus a fixed 4 cm fine ramp for the last centimetres
    tol_wide = (GOAL_TOL_MULT * ctx.success_tol).clamp(min=GOAL_TOL_MIN)
    near_goal = (0.60 * torch.exp(-(ctx.goal_dist / tol_wide).pow(2))
                 + 0.40 * torch.exp(-ctx.goal_dist / GOAL_FINE))
    still = torch.exp(-cup_speed / 0.05) * torch.exp(-spin / 1.5)
    # floor 0.40: holding station against a real push must stay clearly profitable, while drifting
    # off still loses near_goal (and pays pen_drift below)
    settle = SETTLE_FLOOR + (1.0 - SETTLE_FLOOR) * still
    # dwell grows with the successes already counted in this episode (completed hold windows)
    dwell = DWELL_BASE + (1.0 - DWELL_BASE) * torch.tanh(n_succ / DWELL_SCALE)
    r_hold = W_HOLD * hold_grip * upright * off_table * near_goal * settle * dwell * env_mult

    # the FIRST success of an episode is the big one; later ones are cheap. The factor averages the
    # two possible num_successes conventions, so it decays under either one.
    n_prev = (n_succ - 1.0).clamp(min=0.0)
    first_factor = 0.5 * (torch.exp(-n_prev / 0.35) + torch.exp(-n_succ / 0.35))
    r_success = (success * off_table * (W_SUCCESS_FIRST * first_factor + W_SUCCESS_LATER)
                 * env_mult)

    # =============================================================================================
    # LOSING IT, PUTTING IT DOWN, PARKING, DRIFTING -- short-lived and bounded (rule (e))
    # =============================================================================================
    # DROP: charged only while the object is actually FALLING without a hold (current velocity)
    fall = torch.tanh(((-vz) - FALL_DEAD).clamp(min=0.0) / FALL_SCALE)
    p_drop = P_DROP * loose * fall
    # SET-DOWN: charged only while the object is coming DOWN onto the table (band 5 mm to ~6 cm)
    set_band = torch.tanh(height / SET_LIFTOFF) * torch.exp(-height / SET_BAND)
    p_setdown = P_SETDOWN * set_band * torch.tanh(((-vz) - SET_DEAD).clamp(min=0.0) / SET_SCALE)
    # time pressure, gated on the hand being ON the object; a policy that has not reached it pays 0
    stall_gate = (0.5 * grip + 0.5 * held).clamp(min=0.0, max=1.0)
    p_stall = P_STALL * on_table * late * stall_gate
    # drifting away from the goal once near it (current velocity only, not a latch)
    p_drift = (P_DRIFT * held * off_table * torch.exp(-ctx.goal_dist / 0.12)
               * torch.tanh((-closing).clamp(min=0.0) / 0.08))
    # lowering a held object costs, but only BELOW goal height: descending < still < rising
    descend_gate = (1.0 - height / h_need).clamp(min=0.0, max=1.0)
    p_descend = P_DESCEND * held * descend_gate * torch.tanh((-vz).clamp(min=0.0) / 0.10)

    # =============================================================================================
    # SAFETY AND SMOOTHNESS (all small next to the stage income)
    # =============================================================================================
    p_tilt = P_TILT * torch.tanh(ctx.cup_tilt / (math.pi / 6.0))
    horiz_speed = ctx.cup_lin_vel[:, :2].norm(dim=-1)
    p_shove = P_SHOVE * on_table * 0.5 * (torch.tanh(horiz_speed / 0.05)
                                          + torch.tanh(ctx.cup_tilt / math.radians(10.0)))
    # squeeze unchanged: with rule (g) a harder tip press lowers the curl factor, so this charge no
    # longer competes with the envelope income
    total_force = ctx.link_cup_force.clamp(min=0.0).sum(dim=(1, 2)) + ctx.palm_cup_force.clamp(min=0.0)
    max_link_force = ctx.link_cup_force.clamp(min=0.0).amax(dim=-1).amax(dim=-1)
    p_squeeze = P_SQUEEZE * 0.5 * (torch.tanh((total_force - SQ_TOTAL).clamp(min=0.0) / 30.0)
                                   + torch.tanh((max_link_force - SQ_LINK).clamp(min=0.0) / 10.0))
    # palm half unchanged: the palm's own lowest point (a real surface measurement) within 5 mm
    palm_pen = torch.tanh((0.005 - ctx.palm_clearance).clamp(min=0.0) / 0.01)
    # ROUND 18 (rule (h)): floor half on a band where a finger link resting on the table actually
    # reads (its point sits inside the link): 0 at 2 cm above the top, full at 1 cm, smoothstep.
    # Current height only, bounded; a hand wrapping the side wall (lowest link ~8-9 cm) pays 0.
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
        r_reach + r_palm_in + r_close
        + r_count + r_digit + r_palm + r_oppose
        + r_lift + r_rise
        + r_goal + r_progress
        + r_hold + r_success
        - p_drop - p_setdown - p_stall - p_drift - p_descend
        - p_tilt - p_shove - p_squeeze - p_table
        - p_arm_vel - p_arm_fast - p_cup_vel - p_action
    ) + zero  # keeps shape (N,) even if a term ever broadcasts

    components = {
        "reach": r_reach,
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
        "pen_drop": -p_drop,          # falling-only, no latch
        "pen_setdown": -p_setdown,    # descending-onto-the-table only, no latch
        "pen_stall": -p_stall,
        "pen_drift": -p_drift,
        "pen_descend": -p_descend,
        "pen_tilt": -p_tilt,
        "pen_shove": -p_shove,
        "pen_squeeze": -p_squeeze,
        "pen_table": -p_table,        # round 18: floor band 2 cm -> 1 cm above the table top
        "pen_arm_vel": -p_arm_vel,
        "pen_arm_fast": -p_arm_fast,
        "pen_cup_vel": -p_cup_vel,
        "pen_action": -p_action,
    }
    return reward, components
```
