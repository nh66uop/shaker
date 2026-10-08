"""iter_09(4번째 사이클) 후보의 결정론 성질 검사. iter_08 검사 + [53] 인벨롭 대 입구 쪽 비스듬한 파지(f08·f09 실측, 3·8 cm 들기) + [54] 비스듬한 파지의 들기 이득 상한 + [55] 붓는 동안 수평 조준(주둥이-입구 12 cm → r).
이전 설명: iter_08(4번째 사이클) 후보의 결정론 성질 검사. iter_07 검사 + [50] 붓는 높이(주둥이 대 리시버 입구 21 → 5.6 cm) + [51] 쥔 동안 손목 관절 한계 여유 + [52] 들기 높이 상한(여유 넘는 높이에 수입 없음).
이전 설명: iter_07(4번째 사이클) 후보의 결정론 성질 검사. iter_06 검사 + [45] 엄지 입구 안 걸기(f07 오른손 실측) 대 인벨롭(테이블·3 cm) + [46] 걸기에서 빠져나오는 길에 골이 없음
+ [47]~[49] 위에서 덮는 접근(f07 ep1000·f05 ep500 진입 순간 실측) 대 잡는 높이로 먼저 내린 손, 높이 기울기, 손가락 길이 축 수평.
이전 설명: iter_06(4번째 사이클) 후보의 결정론 성질 검사. iter_05 검사 + [43] 손가락이 늘수록 한 손가락 몫이 커짐(볼록) + [44] 안 닿은 손가락이 벽에 다가가는 기울기.
이전 설명: iter_05(4번째 사이클) 후보의 결정론 성질 검사. iter_04 검사 + [41] 다섯 손가락 파지 > 두 손가락(테이블·3 cm 들기) + [42] 기울이는 동안 컵 기준 파지 자세 유지.
이전 설명: iter_04(4번째 사이클) 후보의 결정론 성질 검사. iter_03 검사 + [38] 좌우 미러 대칭 + [39] 손바닥 위로 굴린 받침 자세(f04 오른손) + [40] 높고 기운 파지(f04 왼손).
이전 설명: iter_03(4번째 사이클) 후보의 결정론 성질 검사. iter_02 검사 + [36] 원장 조건(파지 사다리 1.6 배·세로 집기 5v) + [37] 팔 액션 크기 비용 0(팔 절대 목표).
이전 설명: iter_02(4번째 사이클) 후보의 결정론 성질 검사. iter_01 검사 + [32] 인벨롭 축 + [35] 인벨롭 > 손끝 겨눔 > 수직 손(i24 실측 기하).
★09.28 재편: 관절 제어에서는 손바닥 속도 차분 보상이 액션과 직접 이어지지 않는다(f00 e0~470 접근 −0.001~−0.018, 거리 21~43 cm 왕복).
  → 정지 상태 접근 경사 [15]·[24]·[28] 을 판정으로 되돌리고, 속도 차분 검사 [29]·[30] 은 참고로 둔다. [33] 제어 벌점 상한 추가.
★archive_r25-34/iter_09/check_iter09.py 를 옮겼다: 액션 24 → 26(팔마다 [팔 7, 손 6]), [19] 는 손 채널 포화 비용이 0 이어야 통과,
[20] 재생은 옛 palm·시너지 액션을 싣지 않는다(의미가 다르다). 3번째 사이클 번호 검사는 그대로 둔다.

이번 라운드가 노린 세 가지를 그대로 잰다.
  1. 예산 — 붓는 상태에서 회전·배달 수입이 스텝당 총수입의 절반 이상인가.
     그리고 "두 컵 들고 가만히 있기"(주차)의 스텝당 수입이 붓기보다 확실히 낮은가.
  2. 각도 수입 — 같은 기울기라도 방향이 반대이거나 주둥이가 창 밖이면 지불이 사라지는가.
     기울기 0 에서 70° 로 갈 때 수입이 단조로 늘고, 목표 각을 지나쳐도 벌점이 튀지 않는가(각도 처방 금지).
  3. 관절 여유 — 같은 기울기·같은 위치에서 팔 관절 여유만 다를 때 여유가 큰 쪽이 더 많이 받는가.
  4. 공통 — 유한성, 상태 무보존(같은 입력 두 번 = 같은 값), N·dtype 바뀌어도 동작, 닫힌 경로 차분 합 ≤ 0.
  6. ★되받기 — 부었다가 소스로 되받고 다시 붓는 왕복이, 같은 길이 동안 부어 놓고 유지하는 것보다
     이득이면 안 된다(iter_00 이 이것 때문에 배달을 66 % 잃었다).
  7. ★차단 — 다 붓고 난 뒤에는 같은 자세라도 기울임 수입이 꺼져야 한다.
  8. ★고원 — "붓지 않고 기울인 자세만 유지"(ep3100 의 최적해)가 부어 놓고 유지보다 이득이면 안 되고,
     격차는 2 배 이상이어야 한다.
  9. ★방출각 — 측정상 이 컵은 약 101° 에서 비드가 나온다. 85→105° 에서 회전 수입 경사가 45→65° 경사의
     절반 아래로 평평해지면 FAIL(iter_00 은 86° 에서 포화했다). 45→65° 경사가 없어도 FAIL(DG-5F L1).
 10. ★시작점 — i01 시작 상태(주둥이-입구 10~15r)에서 두 컵이 가까워질수록 수입이 올라야 한다(DG-5F L3).
 11. ★fresh 파지 경로 — 팔마다 멀리 → 깔때기 → 손끝이 벽 → 닫힘 → 한쪽 접촉 → 양쪽 파지 → 들기 순으로 수입이 오른다.
 12. ★리시버 — 소스가 쥐고 있을 때 리시버가 컵으로 가 닿고 잡으면 단계마다 더 번다(rh2_i03 리시버 이탈 방지).
 13. ★파지 전 흘림 — 한 번 건드려 1 알 흘리는 비용이 접근 수입 10 스텝치를 넘지 않는다(iter_01 회피 구조 방지).
 14. ★리시버 중심선 — 자기 컵에서 같은 거리라도 중심선 쪽(소스 쪽)이 바깥쪽보다 이득이거나 차이가 스텝당 0.05 미만이면 FAIL(i02m 왼손 이탈).
 15. ★원거리 접근 — 25→15→6 cm 로 다가갈 때 스텝당 수입이 구간마다 +0.05 이상, 합 +0.15 이상(cup_grasp L1).
 16. ★들기 사다리 — 파지한 채 테이블 < 3 cm(+0.3 이상) < 8 cm (cup_grasp L3).
 17. ★편 손 주차 — 벽 앞 편 손 < 한쪽 접촉 < 양쪽 접촉 (cup_grasp L2).
 18. ★먼 쪽 주차 — 같은 6 cm 라도 컵 먼 쪽(+x)이 로봇 쪽(-x)보다 스텝당 0.1 이상 적게(i03 오른손).
 19. ★명령 포화 — 한 팔 palm 6 채널 ±1 비용 0.3~0.8/스텝, |a|=0.85·손 채널 포화는 0.1 이하. (가설 B 만 해당, A 는 참고)
 20. ★재생 자세 — ep5900 오른손 실패 자세 vs 왼손 좋은 자세 반전: 좋은 쪽이 스텝당 0.3 이상.
 21. ★입구 위 허점 — 옆 2.5r·잡는 높이 자세 총수입이 입구 위보다 스텝당 0.3 이상(iter_05 에서 엄격화).
 22. ★ep850 재생 — 오른손 입구 위 자세를 같은 수평거리에서 잡는 높이로 내리고 손가락을 수평으로 돌리면 스텝당 0.3 이상 이득.
 23. ★경로 원장 — 먼저 내리고 수평 진입 ≥ 대각선, 그리고 수평 먼저·위에서 내림보다 +1.0 이상(60 스텝 할인합).
 25. ★파지 전환 — iter_06 ep950 왼손 손가락 사이 자세에서 닫힘 0.4 +0.1·0.7 +0.3 이상, 먼 손 닫기 +0.05 이하.
 26. ★밀려 들어감 — 손바닥이 축 3 cm 로 밀린 자세가 손가락 사이 자세보다 0.4 이상 손해.
 27. ★파지 전 흘림 — 5 알 한꺼번에 흘려도 스텝당 1.0 이하.
 28. ★편 손 접근 — 컵 10 cm 에서 편 손이 쥔 손(0.7)보다 스텝당 0.05 이상.
 29. ★차분형 접근 — 시작에서 컵 쪽 0.3 m/s +0.05 이상, 멀어짐과 대칭(합 |.|<0.02), 옆 수평 접근 > 위 수직 하강 +0.03.
 [15]·[24] 는 정지 상태 접근 검사라 iter_08 에서는 참고용.
 24. ★시작 높이 기울기 — 시작 자세에서 수평 6 cm·5 cm 내림 각각 스텝당 +0.04 이상(iter_05 실패 반영).
 [18]·[19] 는 참고용(iter_05 에 먼 쪽·포화 벌점 없음).
     (6·7 의 ctx 는 env 의 성공 판정을 담김·흘림·받는 컵 기울기로 흉내낸다.)

사용법: python3 check_iter09.py [후보.py]     기본 compute_reward_c1.py
"""
import math
import sys
import types

import torch

torch.set_default_dtype(torch.float64)
BASE = "/home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/"
CAND = sys.argv[1] if len(sys.argv) > 1 else BASE + "iter_09/compute_reward_c1.py"
R, MZ, BZ, TABLE_Z = 0.02808, 0.053885, -0.0599, 0.205   # RH56F1 shaker x0.65 (env cfg)
DT = 1.0 / 60.0
POUR_AZ = math.radians(68)          # K2: 소스가 부을 수 있는 방위
N = 1024


def load(path):
    ns = {"RewardContext": object}
    exec(open(path).read(), ns)
    return ns


NS = load(CAND)


def reset_cache():
    """후보가 모듈 수준 dict 캐시를 쓰면 비운다(이름 규약 `_..._CACHE` 류)."""
    for k, v in list(NS.items()):
        if k.startswith("_") and not k.startswith("__") and isinstance(v, dict):
            v.clear()


def make_ctx(ps, us, pr, ur, vs=None, ws=None, prog=0.5, margin=0.5, in_t=0.0, d_in=0.0, hand="envelope",
             fingers=None, grasped=True, spill=0.0, d_spill=0.0):
    n = ps.shape[0]
    c = types.SimpleNamespace()
    z0, o = torch.zeros(n), torch.ones(n)
    vs = torch.zeros(n, 3) if vs is None else vs
    ws = torch.zeros(n, 3) if ws is None else ws
    c.src_cup_pos, c.src_cup_up, c.rcv_cup_pos, c.rcv_cup_up = ps, us, pr, ur
    c.src_cup_mouth_pos, c.rcv_cup_mouth_pos = ps + MZ * us, pr + MZ * ur
    c.src_cup_lin_vel, c.rcv_cup_lin_vel = vs, torch.zeros(n, 3)
    c.src_cup_ang_vel, c.rcv_cup_ang_vel = ws, torch.zeros(n, 3)
    c.src_cup_tilt = torch.acos(us[:, 2].clamp(-1, 1))
    c.rcv_cup_tilt = torch.acos(ur[:, 2].clamp(-1, 1))
    c.cup_mouth_z, c.cup_radius, c.cup_bottom_z, c.table_z, c.num_beads = MZ, R, BZ, TABLE_Z, 20
    angs = torch.linspace(0, 2 * math.pi, 6)[:5]
    for side, p, u in (("src", ps, us), ("rcv", pr, ur)):
        setattr(c, f"{side}_cup_spawn_pos",
                torch.stack([p[:, 0], p[:, 1], torch.full((n,), TABLE_Z - BZ)], -1))
        # iter_07: 손끝 고리를 컵 축에 수직인 면에 둔다(손이 컵과 함께 돈다). 세운 컵에서는 이전과 같다(e1 = x, e2 = y).
        #   이전 고리는 월드 수평면이라 70 도 이상 기울면 손끝이 컵 몸통 안(축 0.34 r)에 들어갔다.
        e2 = torch.cross(u, torch.tensor([1.0, 0.0, 0.0]).expand_as(u), dim=-1)
        e2 = e2 / (e2.norm(dim=-1, keepdim=True) + 1e-9)
        e1 = torch.cross(e2, u, dim=-1)
        ring = torch.cos(angs)[None, :, None] * e1[:, None, :] + torch.sin(angs)[None, :, None] * e2[:, None, :]
        setattr(c, f"{side}_tips_pos", p.unsqueeze(1) + 0.3 * MZ * u.unsqueeze(1) + R * ring)
        # iter_09: 기본 손 = K7 인벨롭(세운 컵 기준 손바닥 −1.5 r x·+0.15 입구, 법선·손가락 수평)을 컵과 함께 돌린 것.
        #   이전 기본 손(법선 +x·손가락 아래)은 손가락을 늘어뜨린 파지라, 파지 모양을 보는 보상에서 붓기 검사가 모양 비용까지 쟀다.
        sy = -1.0 if side == "src" else 1.0                 # stage_ledger ENV_AXES["envelope"] 는 리시버 기준, 소스는 y 반전
        n0 = torch.tensor([0.67, -0.742 * sy, 0.0]); f0 = torch.tensor([0.742, 0.67 * sy, 0.0])
        pa0 = torch.tensor([-1.5 * R, 0.0, 0.15 * MZ])
        ez = torch.tensor([0.0, 0.0, 1.0]).expand_as(u)
        kk_ = torch.cross(ez, u, dim=-1); sn = kk_.norm(dim=-1, keepdim=True); cs = (ez * u).sum(-1, keepdim=True)
        kk_ = kk_ / sn.clamp(min=1e-9)
        def _rot(v):
            v = v.expand_as(u)
            return v * cs + torch.cross(kk_, v, dim=-1) * sn + kk_ * (kk_ * v).sum(-1, keepdim=True) * (1.0 - cs)
        setattr(c, f"{side}_palm_pos", p + _rot(pa0))
        setattr(c, f"{side}_palm_axes", torch.cat([_rot(n0), _rot(f0)], -1))
        if hand == "neutral":      # 접근·파지 경로 검사([11]~[35])는 이전 기본 손 그대로(그 검사들이 손 방향을 따로 바꿔 비교한다)
            setattr(c, f"{side}_palm_pos", p - 1.2 * R * u)
            setattr(c, f"{side}_palm_axes", torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]).repeat(n, 1))
        setattr(c, f"{side}_palm_lin_vel", torch.zeros(n, 3))                                           # 정지
        setattr(c, f"{side}_finger_force", (2.0 * torch.ones(n, 5)) if fingers is None else fingers)
        setattr(c, f"{side}_palm_force", z0.clone())
        setattr(c, f"{side}_hand_closure", 0.8 * o)
        setattr(c, f"{side}_grasped", torch.full((n,), bool(grasped)))
        setattr(c, f"{side}_arm_qd", torch.zeros(n, 7))
        setattr(c, f"{side}_arm_qdd", torch.zeros(n, 7))
        m = margin if torch.is_tensor(margin) else margin * torch.ones(n, 7)
        setattr(c, f"{side}_arm_limit_margin", m.clone())
        setattr(c, f"{side}_hand_foreign_force", z0.clone())
    c.bead_in_target_frac = in_t * o if not torch.is_tensor(in_t) else in_t
    c.bead_spill_frac = spill * o if not torch.is_tensor(spill) else spill
    c.bead_in_source_frac = (1.0 - c.bead_in_target_frac - c.bead_spill_frac).clamp(min=0.0)
    c.d_in_target = d_in * o if not torch.is_tensor(d_in) else d_in
    c.d_spill = d_spill * o if not torch.is_tensor(d_spill) else d_spill
    c.cup_cup_force = z0.clone()
    c.episode_progress = prog * o if not torch.is_tensor(prog) else prog
    c.cups_nested = torch.zeros(n, dtype=torch.bool)
    # env 가 계산하는 성공 판정을 흉내낸다: 담김 >= 0.80 · 흘림 <= 0.25 · 받는 컵 20도 이내 ·
    # 두 손 파지 · 포개짐 아님. (20 스텝 유지 조건은 무상태 검사라 재현하지 않는다 —
    # 두 시나리오에 같은 기준을 적용하므로 비교는 공정하다.)
    c.success = ((c.bead_in_target_frac >= 0.80) & (c.bead_spill_frac <= 0.25)
                 & (c.rcv_cup_tilt <= math.radians(20.0)) & bool(grasped))
    c.actions, c.prev_actions = torch.zeros(n, 26), torch.zeros(n, 26)
    return c


def run(c):
    r, comps = NS["compute_reward"](c)
    return r, {k: v for k, v in comps.items()}


def pour_state(n, tilt_deg, az=POUR_AZ, lip_dxy=0.4 * R, lip_dz=2.0 * R, src_z=0.62, lean_az=None):
    """받는 컵 입구 위에 소스 주둥이를 두고 tilt 만큼 기운 상태.

    `az` 는 두 컵의 배치 방위(받는 컵에서 소스 주둥이로 가는 방향)이고,
    `lean_az` 는 컵이 기우는 방위다. 기본은 둘이 같다(= 받는 컵 쪽으로 기움).
    방향 검사는 배치를 그대로 두고 `lean_az` 만 돌린다.
    """
    ang = math.radians(tilt_deg) * torch.ones(n)
    azt = az * torch.ones(n)
    lat = (az if lean_az is None else lean_az) * torch.ones(n)
    us = torch.stack([torch.sin(ang) * torch.cos(lat), torch.sin(ang) * torch.sin(lat), torch.cos(ang)], -1)
    ur = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
    pr = torch.tensor([0.40, 0.0, TABLE_Z - BZ + 0.10]).repeat(n, 1)
    mouth_r = pr + MZ * ur
    lip = mouth_r + torch.stack([lip_dxy * torch.cos(azt), lip_dxy * torch.sin(azt),
                                 lip_dz * torch.ones(n)], -1)
    ps = lip - MZ * us
    ps[:, 2] = src_z - MZ * us[:, 2]
    return ps, us, pr, ur


def state_center(n, tilt_deg, gap=0.052, lean_off=0.0, az=POUR_AZ, dz=3.0 * R, src_z=0.62):
    """컵 **중심**을 받는 컵 입구에서 gap 만큼 떨어진 곳에 고정하고 기울인다.

    실제 팔은 컵 중심을 잡고 있으므로 기울이면 주둥이가 `MZ·sinθ` 만큼 휘돌아 나간다.
    `lean_off` = 0 이면 받는 컵 쪽으로 기울이고(주둥이가 입구 위로 온다), π 면 반대쪽으로 기운다.
    """
    ang = math.radians(tilt_deg) * torch.ones(n)
    u_az = torch.tensor([math.cos(az), math.sin(az)])
    lean = az + math.pi + lean_off                       # 중심 → 입구 방향이 '받는 컵 쪽'
    us = torch.stack([torch.sin(ang) * math.cos(lean), torch.sin(ang) * math.sin(lean), torch.cos(ang)], -1)
    ur = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
    pr = torch.tensor([0.40, 0.0, TABLE_Z - BZ + 0.10]).repeat(n, 1)
    mouth_r = pr + MZ * ur
    ps = mouth_r.clone()
    ps[:, :2] += gap * u_az
    ps[:, 2] = src_z
    return ps, us, pr, ur


def share(comps, keys):
    tot = sum(float(v.mean()) for v in comps.values())
    part = sum(float(comps[k].mean()) for k in comps if any(s in k for s in keys))
    return part, tot


def main() -> int:
    ok = True
    names = None

    # --- 1. 예산: 붓는 상태 대 주차 상태 -------------------------------------------------
    reset_cache()
    ps, us, pr, ur = pour_state(N, 75.0)
    c_pour = make_ctx(ps, us, pr, ur, prog=0.5, in_t=0.35, d_in=1.0 / 20.0)   # 스텝마다 한 알 들어감
    r_pour, comps_pour = run(c_pour)
    names = sorted(comps_pour)
    reset_cache()
    ps0, us0, pr0, ur0 = pour_state(N, 8.0, lip_dxy=3.0 * R, lip_dz=4.0 * R)
    c_park = make_ctx(ps0, us0, pr0, ur0, prog=0.5, in_t=0.0, d_in=0.0)
    r_park, comps_park = run(c_park)
    pour_i, tot_pour = share(comps_pour, ("pour", "tilt", "deliver", "transfer", "rot", "open", "in_target"))
    print(f"[1] 붓는 상태 스텝당 총수입 {float(r_pour.mean()):+.3f} · 회전·배달 몫 {pour_i:+.3f} "
          f"({pour_i / tot_pour * 100 if tot_pour else float('nan'):.0f} %)")
    print(f"    주차(두 컵 들고 정지) 스텝당 총수입 {float(r_park.mean()):+.3f}")
    if not (pour_i > 0.5 * abs(tot_pour)):
        print("    ✗ 회전·배달 수입이 총수입의 절반 미만")
        ok = False
    if not (float(r_pour.mean()) > float(r_park.mean()) + 1.0):
        print("    ✗ 붓기가 주차보다 확실히 크지 않다")
        ok = False

    # --- 2. 각도 수입: 방향·창 조건 -------------------------------------------------------
    rows = []
    for deg in (0, 20, 40, 55, 70, 85, 100):
        reset_cache()
        ps, us, pr, ur = state_center(N, float(deg))
        r, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
        rows.append((deg, float(r.mean())))
    print("[2] 기울기별 스텝당 수입(방향·창 맞음): " + " · ".join(f"{d}°:{v:+.2f}" for d, v in rows))
    rise = [rows[i + 1][1] - rows[i][1] for i in range(len(rows) - 1)]
    print("    구간 증분: " + " · ".join(f"{rows[i][0]}→{rows[i+1][0]}:{rise[i]:+.3f}" for i in range(len(rise))))
    if min(rise[:4]) < -0.05:                         # 0 → 70° 구간은 사실상 단조 증가여야 한다
        print("    ✗ 0~70° 구간에서 수입이 줄어드는 지점이 있다")
        ok = False
    drop_after = rows[4][1] - rows[6][1]              # 70° → 100°
    if drop_after > 0.5 * max(1e-9, rows[4][1]):
        print(f"    ⚠ 70°를 지나면 수입이 {drop_after:+.2f} 줄어든다(각도 상한 처방 의심)")

    reset_cache()
    ps, us, pr, ur = state_center(N, 75.0, lean_off=math.pi)    # 중심 고정, 받는 컵 반대쪽으로 기울임
    r_wrong, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
    reset_cache()
    ps, us, pr, ur = state_center(N, 75.0, gap=8.0 * R)         # 두 컵이 멀다(창 밖)
    r_far, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
    reset_cache()
    ps, us, pr, ur = state_center(N, 75.0)
    r_right, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
    print(f"    같은 75°: 방향 맞음 {float(r_right.mean()):+.3f} · 반대 방향 {float(r_wrong.mean()):+.3f} "
          f"· 창 밖 {float(r_far.mean()):+.3f}")
    if not (float(r_right.mean()) > float(r_wrong.mean()) + 0.2):
        print("    ✗ 방향이 반대인데 수입이 줄지 않는다")
        ok = False
    if not (float(r_right.mean()) > float(r_far.mean()) + 0.2):
        print("    ✗ 창 밖인데 수입이 줄지 않는다")
        ok = False

    # --- 3. 관절 여유 ---------------------------------------------------------------------
    reset_cache()
    ps, us, pr, ur = state_center(N, 70.0)
    r_hi, _ = run(make_ctx(ps, us, pr, ur, prog=0.5, margin=0.6))
    reset_cache()
    m_lo = 0.6 * torch.ones(N, 7)
    m_lo[:, 6] = 0.01            # j7 이 한계에 붙음
    m_lo[:, 1] = 0.05            # j2 도 거의 한계
    r_lo, _ = run(make_ctx(ps, us, pr, ur, prog=0.5, margin=m_lo))
    d = float(r_hi.mean() - r_lo.mean())
    print(f"[3] 같은 70° 자세에서 관절 여유 큼 {float(r_hi.mean()):+.3f} · j7 한계 {float(r_lo.mean()):+.3f} (차 {d:+.3f})")
    if d <= 1e-6:
        print("    ✗ 관절 여유를 구분하지 않는다(이번 라운드 요구 3 미충족)")
        ok = False

    # --- 4. 공통 성질 ---------------------------------------------------------------------
    reset_cache()
    ps, us, pr, ur = pour_state(64, 60.0)
    c = make_ctx(ps, us, pr, ur, prog=0.5)
    r1, _ = run(c)
    r2, _ = run(c)                                    # 같은 입력 두 번
    same = torch.allclose(r1, r2)
    finite = bool(torch.isfinite(r1).all())
    reset_cache()
    ps, us, pr, ur = pour_state(7, 60.0)              # N 이 바뀌어도 동작
    try:
        r3, _ = run(make_ctx(ps, us, pr, ur, prog=0.5))
        nchange = bool(torch.isfinite(r3).all())
    except Exception as e:                            # noqa: BLE001
        nchange, _ = False, print(f"    ✗ N 변경에서 예외: {e}")
    print(f"[4] 유한 {finite} · 같은 입력 재현 {same} · N 변경 {nchange}")
    if not (finite and nchange):
        ok = False
    if not same:
        print("    ⚠ 같은 입력이 다른 값을 낸다(스텝 캐시가 있으면 정상일 수 있다 — 차분 항 확인 필요)")

    # 닫힌 경로: 기울였다 되돌리면 차분 수입 합이 0 이하
    reset_cache()
    cyc = list(range(0, 76, 5)) + list(range(70, -1, -5))
    tot = 0.0
    for i, deg in enumerate(cyc):
        ps, us, pr, ur = state_center(N, float(deg))
        r, _ = run(make_ctx(ps, us, pr, ur, prog=0.3 + i * DT))
        tot += float(r.mean())
    base = 0.0
    reset_cache()
    for i in range(len(cyc)):
        ps, us, pr, ur = state_center(N, 0.0)
        r, _ = run(make_ctx(ps, us, pr, ur, prog=0.3 + i * DT))
        base += float(r.mean())
    print(f"[5] 기울였다 되돌린 닫힌 경로 합 {tot:+.2f} (같은 길이 정지 {base:+.2f})")
    if tot > base + 2.0 * len(cyc):
        print("    ⚠ 되돌림 펌프 의심(왕복만으로 정지보다 스텝당 2 이상 더 번다)")

    # --- 6. 되받기(pour → refill → pour) 가 유지보다 이득이면 안 된다 ------------------
    # 측정(ep1300 trace): env 의 d_in_target 은 "건너간 비드"의 단조 증분이라 되받아도 음수 지급이 없고,
    # 다시 부어도 같은 비드에는 배달 수입이 다시 붙지 않는다. 그래서 왕복의 이득은 회전 수입뿐이다.
    STEPS = 60
    reset_cache()
    cyc_total = 0.0
    for i in range(STEPS):                              # 1차 붓기: 담김 0 → 0.9, 배달 증분 지급
        f = (i + 1) / STEPS
        ps, us, pr, ur = state_center(N, 95.0)
        c = make_ctx(ps, us, pr, ur, prog=0.30 + i * DT, in_t=0.9 * f, d_in=0.9 / STEPS)
        cyc_total += float(run(c)[0].mean())
    for i in range(STEPS):                              # 되받기: 담김 0.9 → 0, 배달 증분 없음
        f = 1.0 - (i + 1) / STEPS
        ps, us, pr, ur = state_center(N, 20.0)
        c = make_ctx(ps, us, pr, ur, prog=0.30 + (STEPS + i) * DT, in_t=0.9 * f, d_in=0.0)
        cyc_total += float(run(c)[0].mean())
    for i in range(STEPS):                              # 2차 붓기: 같은 비드라 배달 증분 0
        f = (i + 1) / STEPS
        ps, us, pr, ur = state_center(N, 95.0)
        c = make_ctx(ps, us, pr, ur, prog=0.30 + (2 * STEPS + i) * DT, in_t=0.9 * f, d_in=0.0)
        cyc_total += float(run(c)[0].mean())
    reset_cache()
    hold_total = 0.0
    for i in range(3 * STEPS):                          # 같은 길이 동안 부어 놓고 유지(직립, 조준 유지)
        ps, us, pr, ur = state_center(N, 8.0)
        c = make_ctx(ps, us, pr, ur, prog=0.30 + i * DT, in_t=0.9, d_in=0.0)
        hold_total += float(run(c)[0].mean())
    print(f"[6] 되받아 다시 붓기 {cyc_total:+.1f} 대 부어 놓고 유지 {hold_total:+.1f} ({3 * STEPS} 스텝)")
    if cyc_total >= hold_total:
        print("    ✗ 되받기가 유지보다 이득이다(iter_00 의 실패 구조)")
        ok = False

    # --- 7. 다 부으면 회전 수입이 꺼지는가 -----------------------------------------------
    # 총수입이 아니라 **회전 수입 항**을 본다(다 부은 뒤에는 유지·성공 수입이 대신 들어오는 것이 정상).
    def rot_income(comps):
        return sum(float(comps[k].mean()) for k in comps
                   if k in ("pour_open", "pour_rot", "pour_rate", "posture_margin"))

    reset_cache()
    ps, us, pr, ur = state_center(N, 95.0)
    _, c_empty = run(make_ctx(ps, us, pr, ur, prog=0.6, in_t=1.0))     # 담김 1.0 (소스가 비었다 = 다 부음)
    reset_cache()
    _, c_full = run(make_ctx(ps, us, pr, ur, prog=0.6, in_t=0.0))      # 담김 0 (아직 안 부음)
    ri_e, ri_f = rot_income(c_empty), rot_income(c_full)
    print(f"[7] 같은 95° 자세 회전 수입: 아직 안 부었을 때 {ri_f:+.3f} · 다 붓고 난 뒤 {ri_e:+.3f}")
    if ri_e >= 0.5 * ri_f:
        print("    ✗ 다 붓고 나서도 회전 수입이 절반 이상 남는다")
        ok = False

    # --- 8. "붓지 않고 기울인 자세만 유지" 가 이득이면 안 된다 (ep3100 에서 확인된 최적해) ---
    reset_cache()
    ps, us, pr, ur = state_center(N, 89.0)
    r_tilt_only, _ = run(make_ctx(ps, us, pr, ur, prog=0.5, in_t=0.0))        # 기울이기만, 배달 0
    reset_cache()
    ps, us, pr, ur = state_center(N, 89.0)
    r_pouring, _ = run(make_ctx(ps, us, pr, ur, prog=0.5, in_t=0.45, d_in=1.0 / 20.0))
    reset_cache()
    ps, us, pr, ur = state_center(N, 8.0)
    r_holding, _ = run(make_ctx(ps, us, pr, ur, prog=0.5, in_t=0.9))          # 부어 놓고 유지
    a, b, c = float(r_tilt_only.mean()), float(r_pouring.mean()), float(r_holding.mean())
    print(f"[8] 스텝당 수입: 붓지 않고 기울이기 {a:+.2f} · 붓는 중 {b:+.2f} · 부어 놓고 유지 {c:+.2f} "
          f"(유지/기울이기 {c / a if a else float('nan'):.2f} 배)")
    if not (c > a and b > a):
        print("    ✗ 붓지 않고 기울이기가 유지·붓기보다 이득이다")
        ok = False
    elif c < 2.0 * a:
        print("    ⚠ 유지가 기울이기의 2 배에 못 미친다 — 기울이기 고원에 눌러앉을 위험")

    # --- 9. 방출 전에 회전 수입이 포화하면 안 된다 (측정: 이 컵은 ~101도에서 방출, iter_00 은 86도에서 포화) ---
    def rot_only(comps):
        return sum(float(comps[k].mean()) for k in comps
                   if any(t in k for t in ("pour_rot", "pour_open", "pour_rate", "posture_margin", "rot")))
    vals = {}
    for deg in (45.0, 65.0, 85.0, 105.0, 115.0):
        reset_cache()
        ps, us, pr, ur = state_center(N, deg)
        _, cc = run(make_ctx(ps, us, pr, ur, prog=0.3, in_t=0.0))
        vals[deg] = rot_only(cc)
    print("[9] 회전 수입(잔량 1.0, 창·방향 맞음): " + " · ".join(f"{int(d)}°:{v:+.2f}" for d, v in vals.items()))
    slope_lo = (vals[65.0] - vals[45.0]) / 20.0
    slope_hi = (vals[105.0] - vals[85.0]) / 20.0
    print(f"    도당 경사 45→65° {slope_lo:+.4f} · 85→105° {slope_hi:+.4f}")
    if not (slope_hi > 0 and slope_hi >= 0.5 * slope_lo):
        print("    ✗ 방출 영역(85→105°)에서 회전 수입이 평평해진다 — 비드가 안 나오는 각에서 포화")
        ok = False
    if slope_lo < 0.02:     # 20° 더 기울일 때 스텝당 +0.4 미만이면 구동이 없는 것과 같다
        print("    ✗ 45→65° 에서 회전 경사가 도당 0.02 미만이다(DG-5F L1: 0° 부터 연속 구동이 필요)")
        ok = False

    # --- 10. 시작점(파지 단계 상태)에서 모이기·조준 경사가 살아 있는가 (DG-5F L3) ---------
    # 측정: i01 시작 상태의 주둥이-입구 수평 거리는 p10/p50/p90 = 9.8r/12.4r/14.7r.
    inc = {}
    for gap_r in (14.0, 10.0, 6.0, 3.0, 1.0):
        reset_cache()
        ps, us, pr, ur = state_center(N, 15.0, gap=gap_r * R)
        r_, _ = run(make_ctx(ps, us, pr, ur, prog=0.1, in_t=0.0))
        inc[gap_r] = float(r_.mean())
    print("[10] 시작점 거리별 스텝당 수입(15° 기울기): " + " · ".join(f"{g:.0f}r:{v:+.2f}" for g, v in inc.items()))
    gaps = list(inc)
    mono = all(inc[gaps[k + 1]] > inc[gaps[k]] for k in range(len(gaps) - 1))
    if not mono:
        print("    ✗ 두 컵이 가까워질수록 수입이 오르지 않는 구간이 있다")
        ok = False
    if inc[10.0] - inc[14.0] < 0.02:
        print("    ✗ 시작 거리(10~14r)에서 모이기 경사가 사실상 0 이다")
        ok = False

    # --- 11. fresh 파지 경로: 손이 컵에 다가가 잡고 드는 동안 수입이 계단처럼 올라가야 한다 ------
    #   (rh2_i01 은 접근 0.93 · 엄지 0 N 에서 주차했다. 단계마다 오르는지, 골짜기가 없는지 본다)
    def hand_state(n, cup_p, cup_u, gap, closure, f_thumb, f_oth, grasped, phi=0.0):
        """한 손의 손끝·손바닥·힘을 만든다. 엄지는 phi 방향, 나머지 네 손가락은 맞은편(phi+pi ±15도)."""
        h = 0.25 * MZ
        def ring(a, rr):
            e = torch.tensor([math.cos(a), math.sin(a), 0.0])
            return cup_p + h * cup_u + rr * e
        rr = R + 0.0005 + gap
        tips = [ring(phi, rr)] + [ring(phi + math.pi + da, rr) for da in (-0.26, -0.09, 0.09, 0.26)]
        tips = torch.stack(tips, 1)                                   # (n,5,3)
        palm = ring(phi + 0.5 * math.pi, rr + 0.03)
        ff = torch.zeros(n, 5); ff[:, 0] = f_thumb; ff[:, 1:] = f_oth / 4.0
        return tips, palm, ff, closure * torch.ones(n), torch.full((n,), bool(grasped))

    def good_axes(palm, cup_p):
        """손가락 길이 방향 = 손바닥에서 컵 축으로 향하는 수평 단위벡터, 법선 = 그에 수직인 수평(왼손 ep5900 자세)."""
        v = cup_p - palm; v[:, 2] = 0.0
        f = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
        nrm = torch.stack([-f[:, 1], f[:, 0], torch.zeros_like(f[:, 0])], -1)
        return torch.cat([nrm, f], dim=-1)

    def grasp_path_ctx(arm, gap, closure, f_t, f_o, grasped, lift, other_grasped=False, other_gap=0.20):
        n = 256
        ps = torch.tensor([0.382, -0.159, TABLE_Z - BZ]).repeat(n, 1)
        pr = torch.tensor([0.378, 0.165, TABLE_Z - BZ]).repeat(n, 1)
        up = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
        if arm == "src":
            ps = ps + torch.tensor([0.0, 0.0, lift])
        else:
            pr = pr + torch.tensor([0.0, 0.0, lift])
        c = make_ctx(ps.clone(), up.clone(), pr.clone(), up.clone(), prog=0.2, hand="neutral")
        for side, cp in (("src", ps), ("rcv", pr)):
            spawn = cp.clone(); spawn[:, 2] = TABLE_Z - BZ
            setattr(c, f"{side}_cup_spawn_pos", spawn)
            if side == arm:
                t, pa, ff, cl, gr = hand_state(n, cp, up, gap, closure, f_t, f_o, grasped)
            else:
                og = other_grasped
                t, pa, ff, cl, gr = hand_state(n, cp, up, 0.0 if og else other_gap, 0.8 if og else 0.0,
                                               1.5 if og else 0.0, 1.5 if og else 0.0, og)
            setattr(c, f"{side}_tips_pos", t); setattr(c, f"{side}_palm_pos", pa)
            setattr(c, f"{side}_finger_force", ff); setattr(c, f"{side}_hand_closure", cl)
            setattr(c, f"{side}_grasped", gr)
            setattr(c, f"{side}_palm_force", torch.zeros(n))
            setattr(c, f"{side}_palm_axes", good_axes(pa, cp))
        return c

    path = [("멀리(20cm)", 0.20, 0.0, 0.0, 0.0, False, 0.0),
            ("깔때기(6cm)", 0.06, 0.0, 0.0, 0.0, False, 0.0),
            ("손끝이 벽(1cm)", 0.01, 0.2, 0.0, 0.0, False, 0.0),
            ("벽에 닿음·닫힘 0.7", 0.0, 0.7, 0.0, 0.0, False, 0.0),
            ("한쪽 접촉 0.5N", 0.0, 0.7, 0.0, 0.5, False, 0.0),
            ("양쪽 1.5N·파지", 0.0, 0.8, 1.5, 1.5, True, 0.0),
            ("파지·5cm 들기", 0.0, 0.8, 1.5, 1.5, True, 0.05)]
    for arm in ("src", "rcv"):
        vals = []
        for lbl, gap, cl, ft, fo, gr, lift in path:
            reset_cache()
            r_, _ = run(grasp_path_ctx(arm, gap, cl, ft, fo, gr, lift))
            vals.append(float(r_.mean()))
        print(f"[11] {arm} 파지 경로: " + " → ".join(f"{p[0]} {v:+.2f}" for p, v in zip(path, vals)))
        steps = [vals[k + 1] - vals[k] for k in range(len(vals) - 1)]
        bad = [path[k + 1][0] for k, d in enumerate(steps) if d <= 0.0]
        if bad:
            print(f"    ✗ {arm}: 다음 단계에서 수입이 오르지 않는다 — {', '.join(bad)}")
            ok = False

    # --- 12. 리시버를 버리지 않게: 소스가 잡고 든 상태에서 리시버가 파지 자세에 있으면 더 벌어야 한다 ----
    reset_cache()
    a, _ = run(grasp_path_ctx("rcv", 0.20, 0.0, 0.0, 0.0, False, 0.0, other_grasped=True))   # 리시버 멀리
    reset_cache()
    b, _ = run(grasp_path_ctx("rcv", 0.0, 0.7, 0.0, 0.5, False, 0.0, other_grasped=True))    # 리시버 한쪽 접촉
    reset_cache()
    cc, _ = run(grasp_path_ctx("rcv", 0.0, 0.8, 1.5, 1.5, True, 0.0, other_grasped=True))    # 리시버 파지
    print(f"[12] 소스 파지 중 리시버: 멀리 {float(a.mean()):+.2f} · 한쪽 접촉 {float(b.mean()):+.2f} · 파지 {float(cc.mean()):+.2f}")
    if not (float(b.mean()) > float(a.mean()) and float(cc.mean()) > float(b.mean())):
        print("    ✗ 리시버가 컵에 가면 손해다(rh2_i03 의 리시버 이탈 구조)")
        ok = False

    # --- 13. 파지 전 한 번 건드려 비드가 쏟아지는 비용 < 컵에 다가가서 얻는 몇 스텝치 수입 -------
    reset_cache()
    far, _ = run(grasp_path_ctx("src", 0.20, 0.0, 0.0, 0.0, False, 0.0))
    reset_cache()
    near_c = grasp_path_ctx("src", 0.0, 0.7, 0.0, 0.0, False, 0.0)
    near, _ = run(near_c)
    reset_cache()
    knock_c = grasp_path_ctx("src", 0.0, 0.7, 0.0, 0.0, False, 0.0)
    knock_c.d_spill = torch.full_like(knock_c.d_spill, 1.0 / 20.0)
    knock_c.bead_spill_frac = torch.full_like(knock_c.bead_spill_frac, 1.0 / 20.0)
    knock, _ = run(knock_c)
    gain = float(near.mean() - far.mean()); cost = float(near.mean() - knock.mean())
    print(f"[13] 파지 전 비드 1 알 흘림 비용 {cost:.2f} 대 컵에 다가가 얻는 스텝당 수입 {gain:+.2f} (비용/수입 {cost / gain if gain > 0 else float('inf'):.1f} 스텝)")
    if gain <= 0 or cost > 10.0 * gain:
        print("    ✗ 파지 전 한 번의 흘림이 10 스텝 이상의 접근 수입을 지운다(iter_01 의 회피 구조)")
        ok = False

    # --- 14~17. 빠른 fresh 파지를 위한 원장(cup_grasp L1~L3 에 대응) -----------------------------
    def offset_ctx(arm, dx, dy, dz, closure=0.0, vel=None, axes=None):
        """arm 손을 자기 컵에서 (dx,dy,dz) 떨어진 가상 중심 둘레에 둔다(손끝이 컵에서 멀다). 다른 팔은 멀리."""
        n = 256
        ps = torch.tensor([0.382, -0.159, TABLE_Z - BZ]).repeat(n, 1)
        pr = torch.tensor([0.378, 0.165, TABLE_Z - BZ]).repeat(n, 1)
        up = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
        c = make_ctx(ps.clone(), up.clone(), pr.clone(), up.clone(), prog=0.2, hand="neutral")
        for side, cp in (("src", ps), ("rcv", pr)):
            spawn = cp.clone(); setattr(c, f"{side}_cup_spawn_pos", spawn)
            if side == arm:
                vc = cp + torch.tensor([dx, dy, dz])
                t, pa, ff, cl, gr = hand_state(n, vc, up, 0.0, closure, 0.0, 0.0, False)
                pa = vc.clone()   # 손바닥을 가상 중심에 둔다: 엄지 +x·네 손가락 -x 라 y 반전에 대칭(12cm 안팎 비교가 공정)
            else:
                t, pa, ff, cl, gr = hand_state(n, cp, up, 0.20, 0.0, 0.0, 0.0, False)
            setattr(c, f"{side}_tips_pos", t); setattr(c, f"{side}_palm_pos", pa)
            setattr(c, f"{side}_finger_force", ff); setattr(c, f"{side}_hand_closure", cl)
            setattr(c, f"{side}_grasped", gr); setattr(c, f"{side}_palm_force", torch.zeros(n))
            if side == arm and vel is not None:
                setattr(c, f"{side}_palm_lin_vel", torch.tensor(vel, dtype=torch.float64).repeat(n, 1))
            if side == arm and axes is not None:
                setattr(c, f"{side}_palm_axes", torch.tensor(axes, dtype=torch.float64).repeat(n, 1))
        return c

    # [14] 리시버: 자기 컵에서 같은 12 cm 라도 바깥쪽(+y)에 있는 것이 중심선 쪽(-y)보다 낫거나 같아야 한다
    reset_cache(); out_r, _ = run(offset_ctx("rcv", -0.03, +0.12, 0.10))
    reset_cache(); in_r, _ = run(offset_ctx("rcv", -0.03, -0.12, 0.10))
    reset_cache(); wall_r, _ = run(grasp_path_ctx("rcv", 0.0, 0.2, 0.0, 0.0, False, 0.0))
    print(f"[14] 리시버 손: 자기 컵 바깥 12cm {float(out_r.mean()):+.3f} · 중심선 쪽 12cm {float(in_r.mean()):+.3f} · 자기 컵 벽 {float(wall_r.mean()):+.3f}")
    if float(in_r.mean()) > float(out_r.mean()) - 0.05:
        print("    ✗ 리시버 손이 중심선 쪽(소스 쪽)으로 가는 것이 더 이득이다(i02m 왼손 이탈 구조)")
        ok = False
    if float(wall_r.mean()) < float(in_r.mean()) + 0.3:
        print("    ✗ 자기 컵 벽에 붙는 수입이 중심선 쪽에 떠 있는 것보다 스텝당 0.3 이상 크지 않다")
        ok = False

    # [15] L1 원거리 접근 경사: 무작위 초기 거리(25 cm)에서 6 cm 까지 다가가면 스텝당 확실히 올라야 한다
    for arm in ("src", "rcv"):
        vals = {}
        for d in (0.25, 0.15, 0.06):
            reset_cache()
            vals[d], _ = run(offset_ctx(arm, 0.0, (-1 if arm == "src" else 1) * d, 0.05))
        g1 = float(vals[0.15].mean() - vals[0.25].mean()); g2 = float(vals[0.06].mean() - vals[0.15].mean())
        print(f"[15] {arm} 원거리 접근: 25→15cm {g1:+.3f} · 15→6cm {g2:+.3f} (스텝당)")
        if g1 < 0.05 or g2 < 0.05 or g1 + g2 < 0.15:
            print(f"    ✗ {arm}: 멀리서 다가갈수록 스텝당 수입이 확실히 오르지 않는다(full-joint 는 상태 경사가 필요)"); ok = False

    # [16] L3 들기 사다리: 파지한 채 테이블 < 3 cm < 8 cm
    for arm in ("src", "rcv"):
        v = []
        for lift in (0.0, 0.03, 0.08):
            reset_cache()
            r_, _ = run(grasp_path_ctx(arm, 0.0, 0.8, 1.5, 1.5, True, lift, other_grasped=True))
            v.append(float(r_.mean()))
        print(f"[16] {arm} 들기 사다리: 테이블 {v[0]:+.2f} · 3cm {v[1]:+.2f} · 8cm {v[2]:+.2f}")
        if not (v[1] > v[0] + 0.3 and v[2] > v[1]):
            print(f"    ✗ {arm}: 테이블 위 파지 대비 들기 이득이 작다(cup_grasp L3)")
            ok = False

    # [17] L2 편 손 주차 < 한 손가락 접촉 < 양쪽 접촉
    for arm in ("src", "rcv"):
        reset_cache(); park, _ = run(grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0))
        reset_cache(); one, _ = run(grasp_path_ctx(arm, 0.0, 0.5, 0.0, 0.5, False, 0.0))
        reset_cache(); two, _ = run(grasp_path_ctx(arm, 0.0, 0.7, 0.6, 0.6, False, 0.0))
        print(f"[17] {arm} 벽 앞 편 손 {float(park.mean()):+.3f} · 한쪽 접촉 {float(one.mean()):+.3f} · 양쪽 접촉 {float(two.mean()):+.3f}")
        if not (float(one.mean()) > float(park.mean()) and float(two.mean()) > float(one.mean())):
            print(f"    ✗ {arm}: 벽 앞 편 손이 접촉보다 이득이다(cup_grasp L2)")
            ok = False

    # [18] 먼 쪽 주차: 같은 거리라도 컵 먼 쪽(+x, 로봇 전방)이 로봇 쪽(-x)보다 스텝당 0.1 이상 적게 벌어야 한다
    for arm in ("src", "rcv"):
        reset_cache(); far_r, _ = run(offset_ctx(arm, +0.06, 0.0, 0.05))
        reset_cache(); near_r, _ = run(offset_ctx(arm, -0.06, 0.0, 0.05))
        print(f"[18] {arm} 컵 먼 쪽 6cm {float(far_r.mean()):+.3f} · 로봇 쪽 6cm {float(near_r.mean()):+.3f}")
        if float(far_r.mean()) > float(near_r.mean()) - 0.1:
            print(f"    (참고) {arm}: 먼 쪽 벌점 없음 — iter_05 는 판정에 쓰지 않는다")

    # [19] ★full-joint: 팔 7 채널 ±1(최대 속도) 비용은 참고, 손 6 채널 ±1(완전히 펴기·쥐기 = 정상 명령) 비용은 0.02 이하여야 한다
    for arm, sl, hs in (("src", slice(0, 7), slice(7, 13)), ("rcv", slice(13, 20), slice(20, 26))):
        base = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        reset_cache(); r0, _ = run(base)
        sat = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        sat.actions = sat.actions.clone(); sat.actions[:, sl] = 1.0; sat.prev_actions = sat.actions.clone()
        reset_cache(); r1, _ = run(sat)
        hand = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        hand.actions = hand.actions.clone(); hand.actions[:, hs] = 1.0; hand.prev_actions = hand.actions.clone()
        reset_cache(); r2, _ = run(hand)
        half = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        half.actions = half.actions.clone(); half.actions[:, sl] = 0.85; half.prev_actions = half.actions.clone()
        reset_cache(); r3, _ = run(half)
        c_sat = float(r0.mean() - r1.mean()); c_hand = float(r0.mean() - r2.mean()); c_half = float(r0.mean() - r3.mean())
        print(f"[19] {arm} 팔 7 채널 ±1 비용 {c_sat:+.3f} · 팔 |a|=0.85 비용 {c_half:+.3f} · 손 6 채널 ±1 비용 {c_hand:+.3f}")
        if c_hand > 0.02:
            print(f"    ✗ {arm}: 손 채널 ±1 에 비용 — 손 절대 목표의 완전 펴기·쥐기를 벌한다(브리프 5-1)"); ok = False

    # [20] ep5900 재생 자세 그대로: 오른손 실패 자세(손목 쪽이 입구 위·손가락 바깥) vs 왼손 좋은 자세를 오른손 컵에 y 반전해 옮긴 것
    import numpy as np
    TR = "/tmp/claude-1000/-home-user-rl-ws/146751a7-8640-4668-9250-a319cc2d947e/scratchpad/i03v/trace_ep5900.npz"
    try:
        zt = np.load(TR)
    except FileNotFoundError:
        zt = None
        print("[20] 재생 파일 없음 — 건너뜀")
        ok = False
    if zt is not None:
        def eul_axes(e):
            y_, p_, r_ = e[..., 0], e[..., 1], e[..., 2]
            cy, sy, cp_, sp, cr, sr = np.cos(y_), np.sin(y_), np.cos(p_), np.sin(p_), np.cos(r_), np.sin(r_)
            c1 = np.stack([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp_ * sr], -1)
            c2 = np.stack([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp_ * cr], -1)
            return np.concatenate([c2, c1], -1)          # env 계약: 앞 3 = 법선(열 2), 뒤 3 = 손가락 길이(열 1)
        ts = list(range(300, 900, 60))
        def grab(side, flip, ts=ts):
            sgn = np.array([1.0, -1.0 if flip else 1.0, 1.0])
            pr_ = zt[f"{side}_palm_rel"][ts] * sgn; tr_ = zt[f"{side}_tips_rel"][ts] * sgn
            ax = eul_axes(zt[f"{side}_palm_pose"][ts][..., 3:]) * np.concatenate([sgn, sgn])
            f = lambda a: torch.tensor(a.reshape(-1, *a.shape[2:]), dtype=torch.float64)
            return dict(palm_rel=f(pr_), tips_rel=f(tr_), axes=f(ax), ff=f(zt[f"{side}_f"][ts]),
                        fp=f(zt[f"{side}_f_palm"][ts]), cl=f(zt[f"{side}_closure"][ts]), gr=f(zt[f"{side}_grasped"][ts]) > 0.5,
                        pa=f(zt["actions"][ts][..., (0 if side == "src" else 12):(6 if side == "src" else 18)]),
                        ha=f(zt["actions"][ts][..., (6 if side == "src" else 18):(12 if side == "src" else 24)]))
        def scene(src_pose, rcv_pose):
            n = src_pose["palm_rel"].shape[0]
            ps = torch.tensor([0.382, -0.159, TABLE_Z - BZ]).repeat(n, 1)
            pr = torch.tensor([0.378, 0.165, TABLE_Z - BZ]).repeat(n, 1)
            up = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
            c = make_ctx(ps.clone(), up.clone(), pr.clone(), up.clone(), prog=0.5)
            acts = torch.zeros(n, 26)
            for side, cp, q, a0 in (("src", ps, src_pose, 0), ("rcv", pr, rcv_pose, 12)):
                setattr(c, f"{side}_cup_spawn_pos", cp.clone())
                setattr(c, f"{side}_palm_pos", cp + q["palm_rel"]); setattr(c, f"{side}_tips_pos", cp.unsqueeze(1) + q["tips_rel"])
                setattr(c, f"{side}_palm_axes", q["axes"]); setattr(c, f"{side}_finger_force", q["ff"])
                setattr(c, f"{side}_palm_force", q["fp"]); setattr(c, f"{side}_hand_closure", q["cl"])
                setattr(c, f"{side}_grasped", q["gr"])
                pass   # ★full-joint: 옛 palm·시너지 액션은 의미가 달라 싣지 않는다(0 액션)
            c.actions, c.prev_actions = acts, acts.clone()
            return c
        src_bad, rcv_good = grab("src", False), grab("rcv", False)
        src_good = grab("rcv", True)        # 왼손 좋은 자세를 오른손 쪽으로 y 반전
        reset_cache(); r_bad, cb = run(scene(src_bad, rcv_good))
        reset_cache(); r_good, cg = run(scene(src_good, rcv_good))
        gap20 = float(r_good.mean() - r_bad.mean())
        keys = [k for k in cb if k.endswith("_src") or k in ("far_cross", "palm_sat")]
        print(f"[20] 재생 자세(오른손 컵): 실패 자세 {float(r_bad.mean()):+.3f} · 왼손 좋은 자세 반전 {float(r_good.mean()):+.3f} (차 {gap20:+.3f}/스텝)")
        print("     성분(실패→좋음): " + " · ".join(f"{k} {float(cb[k].mean()):+.2f}→{float(cg[k].mean()):+.2f}" for k in keys
                                                  if abs(float(cb[k].mean())) + abs(float(cg[k].mean())) > 0.005))
        if gap20 < 0.3:
            print("    ✗ 재생된 실패 자세가 좋은 자세보다 스텝당 0.3 이상 덜 벌지 않는다")
            ok = False

    # [21] 입구 위 허점: 손바닥을 입구 위(축 위, 입구 +1.7 cm)에 둔 접근 수입이 옆(origin 에서 7 cm, 잡는 높이)보다 크면 안 된다
    for arm in ("src", "rcv"):
        cpos = torch.tensor([0.382, -0.159, TABLE_Z - BZ]) if arm == "src" else torch.tensor([0.378, 0.165, TABLE_Z - BZ])
        top = offset_ctx(arm, 0.0, 0.0, MZ + 0.017); side_ = offset_ctx(arm, -2.5 * R, 0.0, 0.25 * MZ)
        reset_cache(); _, ct = run(top)
        reset_cache(); _, cs = run(side_)
        rk = (f"reach_{arm}" if f"reach_{arm}" in ct else f"approach_{arm}" if f"approach_{arm}" in ct else None)  # full-joint: 상태형 접근 항
        rt = float(ct[rk].mean()) if rk else float("nan"); rs = float(cs[rk].mean()) if rk else float("nan")
        print(f"[21] {arm} 접근 수입: 입구 위 {rt:+.3f} · 옆 7cm {rs:+.3f}")
        reset_cache(); rtop, _ = run(top)
        side_low = offset_ctx(arm, -2.5 * R, 0.0, 0.25 * MZ)
        reset_cache(); rlow, _ = run(side_low)
        gap21 = float(rlow.mean() - rtop.mean())
        print(f"     {arm} 총수입: 입구 위(축 위, 입구+1.7cm) {float(rtop.mean()):+.3f} · 옆 2.5r·잡는 높이 {float(rlow.mean()):+.3f} (차 {gap21:+.3f})")
        if gap21 < 0.3:
            print(f"    ✗ {arm}: 옆에서 잡는 높이로 내린 자세가 입구 위보다 스텝당 0.3 이상 이득이 아니다")
            ok = False

    # [22] ep850 재생: 오른손 입구 위 자세 vs 같은 수평거리에서 잡는 높이로 내리고 손가락을 컵 축 수평으로 돌린 자세
    TR2 = "/tmp/claude-1000/-home-user-rl-ws/146751a7-8640-4668-9250-a319cc2d947e/scratchpad/i04v/trace_ep850.npz"
    try:
        z2 = np.load(TR2)
    except Exception:
        z2 = None
        print("[22] 재생 파일 없음"); ok = False
    if z2 is not None:
        zt_bak = zt
        zt = z2
        src_top, rcv_now = grab("src", False), grab("rcv", False)
        # 내린 자세: 손바닥·손끝을 수직으로 내려 손바닥 높이 = 잡는 높이, 손가락 = 컵 축 수평 방향
        low = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in src_top.items()}
        dz = low["palm_rel"][:, 2] - 0.25 * MZ
        low["palm_rel"][:, 2] -= dz; low["tips_rel"][:, :, 2] -= dz.unsqueeze(1)
        hv = -low["palm_rel"].clone(); hv[:, 2] = 0.0; hv = hv / (hv.norm(dim=-1, keepdim=True) + 1e-9)
        nrm = torch.stack([-hv[:, 1], hv[:, 0], torch.zeros_like(hv[:, 0])], -1)
        low["axes"] = torch.cat([nrm, hv], -1)
        reset_cache(); r_top, ctop = run(scene(src_top, rcv_now))
        reset_cache(); r_low, clow = run(scene(low, rcv_now))
        g22 = float(r_low.mean() - r_top.mean())
        keys = [k for k in ctop if k.endswith("_src")]
        print(f"[22] ep850 오른손: 입구 위 {float(r_top.mean()):+.3f} · 같은 수평거리에서 내림·손가락 수평 {float(r_low.mean()):+.3f} (차 {g22:+.3f})")
        print("     성분(위→내림): " + " · ".join(f"{k} {float(ctop[k].mean()):+.2f}→{float(clow[k].mean()):+.2f}" for k in keys
                                              if abs(float(ctop[k].mean())) + abs(float(clow[k].mean())) > 0.005))
        if g22 < 0.3:
            print("    ✗ 내린 자세가 입구 위 자세보다 스텝당 0.3 이상 이득이 아니다"); ok = False
        low2 = dict(low); low2["axes"] = src_top["axes"]          # 손가락 방향은 그대로, 높이만 내림
        reset_cache(); r_low2, _ = run(scene(low2, rcv_now))
        g22b = float(r_low2.mean() - r_top.mean())
        print(f"     높이만 내림(손가락 방향 그대로) {float(r_low2.mean()):+.3f} (차 {g22b:+.3f})")
        if g22b < 0.2:
            print("    ✗ 높이만 내려도 스텝당 0.2 이상 이득이 아니다(내려갈 이유가 약함)"); ok = False
        zt = zt_bak

    # [23] 경로 원장: 시작(수평 15.8·높이 +15.1 cm) → 옆 파지 지점(수평 2.5r·잡는 높이), 60 스텝
    #      대각선 직진 vs 먼저 내리고(30) 수평 진입(30). 손가락은 아래(중립)로 같게, gamma 0.998 할인합.
    def path_sum(pts, arm="src", axes=None):
        tot = 0.0
        for i, (hd, hz) in enumerate(pts):
            reset_cache()
            nx = pts[min(i + 1, len(pts) - 1)]
            vel = [-(nx[0] - hd) / DT, 0.0, (nx[1] - hz) / DT]   # 다음 점으로 가는 속도(60 Hz)
            c = offset_ctx(arm, -hd, 0.0, hz, vel=vel, axes=axes)
            r_, _ = run(c)
            tot += (0.998 ** i) * float(r_.mean())
        return tot
    H0, Z0, H1, Z1 = 0.158, 0.151, 2.5 * R, 0.25 * MZ
    diag = [(H0 + (H1 - H0) * t / 59, Z0 + (Z1 - Z0) * t / 59) for t in range(60)]
    lowf = [(H0, Z0 + (Z1 - Z0) * t / 29) for t in range(30)] + [(H0 + (H1 - H0) * t / 29, Z1) for t in range(30)]
    upf = [(H0 + (H1 - H0) * t / 29, Z0) for t in range(30)] + [(H1, Z0 + (Z1 - Z0) * t / 29) for t in range(30)]
    sd, sl, su = path_sum(diag), path_sum(lowf), path_sum(upf)
    print(f"[23] 60 스텝 할인합: 대각선 {sd:+.2f} · 먼저 내리고 수평 진입 {sl:+.2f} · 수평으로 먼저 가서 위에서 내림 {su:+.2f}")
    if not (sl >= sd and sl > su + 1.0):
        print("    ✗ '먼저 내리고 수평 진입' 경로가 대각선 이상, '위에서 내림'보다 확실히(+1.0) 크지 않다"); ok = False
    # iter_07: 09.29 사용자 "인벨롭 되게 충분히 낮게 내려간 뒤 접근" — 접근 자세 손(소스 인벨롭 축: 법선·손가락 수평, 컵 쪽)으로 같은 세 경로.
    #   먼저 내리는 길이 대각선보다 스텝당 평균 0.25 이상(60 스텝 15). 위 줄은 손가락이 아래인 손(이전 판정 그대로).
    AX_SRC = [0.67, 0.742, 0.0, 0.742, -0.67, 0.0]      # stage_ledger ENV_AXES["envelope"] 의 y 반전(소스)
    sd2, sl2, su2 = path_sum(diag, axes=AX_SRC), path_sum(lowf, axes=AX_SRC), path_sum(upf, axes=AX_SRC)
    print(f"     인벨롭 축 손: 대각선 {sd2:+.2f} · 먼저 내리고 수평 진입 {sl2:+.2f} ({sl2 - sd2:+.2f}) · 위에서 내림 {su2:+.2f}")
    if not (sl2 >= sd2 + 15.0 and sl2 > su2 + 1.0):
        print("    ✗ 인벨롭 축 손: '먼저 내리고 수평 진입'이 대각선보다 15(스텝당 0.25) 이상, '위에서 내림'보다 +1.0 이상 크지 않다"); ok = False

    # [24] 시작 높이 기울기(iter_05 실패): 시작 자세(수평 15.8·높이 +15.1 cm)에서 수평 6 cm 다가감·5 cm 내려감이 각각 스텝당 +0.04 이상
    for arm in ("src", "rcv"):
        vals = {}
        for key, (hd, hz) in {"start": (0.158, 0.151), "closer": (0.098, 0.151), "lower": (0.158, 0.101), "both": (0.098, 0.101)}.items():
            reset_cache(); r_, _ = run(offset_ctx(arm, -hd, 0.0, hz)); vals[key] = float(r_.mean())
        dc, dl, db = vals["closer"] - vals["start"], vals["lower"] - vals["start"], vals["both"] - vals["start"]
        print(f"[24] {arm} 시작 높이: 시작 {vals['start']:+.3f} · 수평 6cm {dc:+.3f} · 5cm 내림 {dl:+.3f} · 둘 다 {db:+.3f}")
        if dc < 0.04 or dl < 0.04:
            print(f"    ✗ {arm}: 시작 자세에서 수평 6 cm·5 cm 내림 각각 +0.04 이상이 아니다(f00 실패 구조)"); ok = False

    # [25]~[27] 파지 전환 (iter_06 ep950 재생, 왼손이 t=60~100 에 손끝을 벽에 대고 컵을 손가락 사이에 둔 자세)
    TR3 = "/tmp/claude-1000/-home-user-rl-ws/146751a7-8640-4668-9250-a319cc2d947e/scratchpad/i06v/trace_ep950.npz"
    try:
        z3 = np.load(TR3)
    except Exception:
        z3 = None; print("[25] 재생 파일 없음"); ok = False
    if z3 is not None:
        zt_bak = zt; zt = z3
        pk = grab("rcv", False, ts=list(range(60, 100, 4)))
        src_far = grab("src", False, ts=list(range(60, 100, 4)))
        def closed(q, cl, pull):
            """닫힘 cl, 네 손가락 끝을 컵 축 쪽으로 pull(m) 당긴 자세(엄지는 그대로)."""
            o = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in q.items()}
            o["cl"] = torch.full_like(o["cl"], cl)
            tr = o["tips_rel"]; rv = tr[:, 1:, :2]; rn = rv.norm(dim=-1, keepdim=True) + 1e-9
            tr[:, 1:, :2] = rv - pull * rv / rn
            return o
        reset_cache(); r0, c0 = run(scene(src_far, pk))
        reset_cache(); r1, c1 = run(scene(src_far, closed(pk, 0.4, 0.005)))
        reset_cache(); r2, c2 = run(scene(src_far, closed(pk, 0.7, 0.012)))
        g1, g2 = float(r1.mean() - r0.mean()), float(r2.mean() - r0.mean())
        print(f"[25] 왼손 손가락 사이 자세에서 닫기: 원래(닫힘 {float(pk['cl'].mean()):.2f}) {float(r0.mean()):+.3f} · 0.4 {g1:+.3f} · 0.7 {g2:+.3f}")
        print("     close_rcv " + " → ".join(f"{float(c['close_rcv'].mean()):+.3f}" for c in (c0, c1, c2)))
        if not (g1 > 0.1 and g2 > 0.3):
            print("    ✗ 컵이 손가락 사이에 있을 때 닫는 기울기가 약하다(0.4 에서 +0.1, 0.7 에서 +0.3 이상)"); ok = False
        # [25b] 같은 자세에서 엄지 끝도 잡는 높이 대역(0~0.5 x 입구) 안으로 내리고 벽 쪽으로 1 cm 당긴 경우 (참고: [25] 재생 자세는 엄지 끝이 +4.7 cm 로 대역 밖)
        def thumb_ok(q):
            o = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in q.items()}
            tr = o["tips_rel"]; tr[:, 0, 2] = 0.25 * MZ
            rv = tr[:, 0, :2]; rn = rv.norm(dim=-1, keepdim=True) + 1e-9; tr[:, 0, :2] = rv - 0.01 * rv / rn
            return o
        pkt = thumb_ok(pk)
        reset_cache(); rt0, _ = run(scene(src_far, pkt))
        reset_cache(); rt2, _ = run(scene(src_far, closed(pkt, 0.7, 0.012)))
        print(f"     [25b] 엄지도 잡는 높이로 내린 자세: 닫힘 0.7 {float(rt2.mean() - rt0.mean()):+.3f} (참고)")
        # 멀리서 닫기는 지급하지 않음: 오른손(컵에서 29 cm) 닫힘 0 → 0.7
        far_c = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in src_far.items()}; far_c["cl"] = torch.full_like(far_c["cl"], 0.7)
        reset_cache(); rf0, _ = run(scene(src_far, pk)); reset_cache(); rf1, _ = run(scene(far_c, pk))
        gf = float(rf1.mean() - rf0.mean())
        print(f"     컵에서 먼 손 닫기 0 → 0.7: {gf:+.3f}")
        if gf > 0.05:
            print("    ✗ 컵에서 먼 손을 닫아도 수입이 오른다"); ok = False
        # [26] 손바닥이 옆 파지 거리 안으로 밀려 들어간 자세(t=400~600) < 손가락 사이 자세(t=60~100)
        pin = grab("rcv", False, ts=list(range(400, 600, 20)))
        reset_cache(); rin, _ = run(scene(src_far, pin))
        g26 = float(r0.mean() - rin.mean())
        print(f"[26] 왼손 밀려 들어간 자세(손바닥 축 3 cm) {float(rin.mean()):+.3f} · 손가락 사이 자세 {float(r0.mean()):+.3f} (차 {g26:+.3f})")
        if g26 < 0.4:
            print("    ✗ 밀려 들어간 자세가 손가락 사이 자세보다 스텝당 0.4 이상 손해가 아니다"); ok = False
        zt = zt_bak
    # [27] 파지 전 흘림: 소스 컵을 쳐서 5 알이 한 스텝에 쏟아져도 비용이 작아야 한다(쥐기 전)
    reset_cache(); cs0 = offset_ctx("src", -0.06, 0.0, 0.02); r_a, _ = run(cs0)
    cs1 = offset_ctx("src", -0.06, 0.0, 0.02); cs1.d_spill = torch.full_like(cs1.d_spill, 5.0 / 20.0)
    reset_cache(); r_b, _ = run(cs1)
    c27 = float(r_a.mean() - r_b.mean())
    print(f"[27] 파지 전 5 알 흘림 비용 {c27:+.3f}/스텝 (iter_06 = 15)")
    if c27 > 1.0:
        print("    ✗ 파지 전 흘림 비용이 크다(소스 컵 회피를 가르칠 수 있음)"); ok = False

    # [29] 차분형 접근(iter_08): 시작 자세에서 컵 쪽 0.3 m/s 이동이 정지보다 스텝당 +0.05 이상, 같은 속도로 멀어지면
    #      같은 크기로 손해(홀함수), 왕복 합 ~0. 입구 바로 위에서 수직 하강은 옆 수평 접근보다 적게.
    for arm in ("src", "rcv"):
        cp = torch.tensor([0.382, -0.159, 0.0]) if arm == "src" else torch.tensor([0.378, 0.165, 0.0])
        hd, hz = 0.158, 0.151
        u = torch.tensor([hd, 0.0, -hz]); u = u / u.norm()          # 손바닥 -> 컵 origin (x+ = 컵 쪽)
        v_in = (0.3 * u).tolist(); v_out = (-0.3 * u).tolist()
        reset_cache(); r0, _ = run(offset_ctx(arm, -hd, 0.0, hz, vel=[0.0, 0.0, 0.0]))
        reset_cache(); r_in, _ = run(offset_ctx(arm, -hd, 0.0, hz, vel=v_in))
        reset_cache(); r_out, _ = run(offset_ctx(arm, -hd, 0.0, hz, vel=v_out))
        gi, go = float(r_in.mean() - r0.mean()), float(r_out.mean() - r0.mean())
        # 옆(잡는 높이, 수평 10 cm)에서 수평 접근 vs 입구 위(축 위 +10 cm)에서 수직 하강, 같은 0.3 m/s
        reset_cache(); rs0, _ = run(offset_ctx(arm, -0.10, 0.0, 0.25 * MZ, vel=[0.0, 0.0, 0.0]))
        reset_cache(); rs1, _ = run(offset_ctx(arm, -0.10, 0.0, 0.25 * MZ, vel=[0.3, 0.0, 0.0]))
        reset_cache(); rt0, _ = run(offset_ctx(arm, 0.0, 0.0, 0.10, vel=[0.0, 0.0, 0.0]))
        reset_cache(); rt1, _ = run(offset_ctx(arm, 0.0, 0.0, 0.10, vel=[0.0, 0.0, -0.3]))
        gs, gt = float(rs1.mean() - rs0.mean()), float(rt1.mean() - rt0.mean())
        print(f"[29] {arm} 차분형 접근: 시작에서 컵 쪽 0.3 m/s {gi:+.3f} · 멀어짐 {go:+.3f} · 옆 수평 접근 {gs:+.3f} · 위에서 수직 하강 {gt:+.3f}")
        if abs(gi + go) > 0.02:
            print(f"    ✗ {arm}: 속도 항이 있다면 다가감과 멀어짐이 대칭이어야 한다(왕복으로 벌 수 있음)"); ok = False
        # (참고) full-joint 재편: 속도 차분은 판정하지 않는다 — 옆 접근 선호는 [21]·[22]·[23] 상태 검사가 본다

    # [28] 다가가는 동안 편 손(iter_08): 컵에서 10 cm·잡는 높이, 손가락 사이에 컵 없음 — 편 손 > 쥔 손.
    #      손가락 사이 자세(재생 [25])에서는 닫는 쪽이 여전히 이득([25] 가 본다).
    for arm in ("src", "rcv"):
        reset_cache(); ro, _ = run(offset_ctx(arm, -0.10, 0.0, 0.25 * MZ, closure=0.0))
        reset_cache(); rc, _ = run(offset_ctx(arm, -0.10, 0.0, 0.25 * MZ, closure=0.7))
        reset_cache(); rfo, _ = run(offset_ctx(arm, -0.25, 0.0, 0.15, closure=0.0))
        reset_cache(); rfc, _ = run(offset_ctx(arm, -0.25, 0.0, 0.15, closure=0.7))
        g28, gfar = float(ro.mean() - rc.mean()), float(rfo.mean() - rfc.mean())
        print(f"[28] {arm} 컵 10 cm 에서 편 손 - 쥔 손(0.7) {g28:+.3f} · 25 cm 밖 {gfar:+.3f}")
        if g28 < 0.05:
            print(f"    ✗ {arm}: 컵 10 cm 앞 편 손이 반쯤 쥔 손(0.7)보다 스텝당 0.05 이상 이득이 아니다(f00 폐쇄도 0.78 유지)"); ok = False

    # [30] (a) 편 손은 차분형: 12 cm 앞 편 손 정지 수입이 쥔 손 정지와 같고(open 0), 편 손으로 다가갈 때만 +
    for arm in ("src", "rcv"):
        # ★iter_00(4번째 사이클): open 이 approach 에 합쳐질 수 있어 항 이름 대신 총수입의 (편 손 − 쥔 손 0.7) 차로 본다
        def _tot(cl, v):
            reset_cache(); r_, _ = run(offset_ctx(arm, -0.12, 0.0, 0.25 * MZ, closure=cl, vel=v)); return float(r_.mean())
        o_still = _tot(0.0, [0.0, 0.0, 0.0]) - _tot(0.7, [0.0, 0.0, 0.0])
        o_move = _tot(0.0, [0.3, 0.0, 0.0]) - _tot(0.7, [0.3, 0.0, 0.0])
        print(f"[30] {arm} 편 손 − 쥔 손: 12 cm 정지 {o_still:+.3f} · 0.3 m/s 접근 중 {o_move:+.3f}")
        # (참고) full-joint 재편: 편 손은 상태 인자로 준다 — 판정은 [28](편 손 > 쥔 손)과 [31](주차 < 손가락 사이)
    # [31] (b) 12 cm 편 손 정지(주차 후보) < 손가락 사이 자세(iter_06 ep950 왼손 t=60~100) — 스텝당 0.5 이상
    if z3 is not None:
        zt_bak = zt; zt = z3
        pk2 = grab("rcv", False, ts=list(range(60, 100, 4)))
        src_far2 = grab("src", False, ts=list(range(60, 100, 4)))
        park = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in pk2.items()}
        hv = park["palm_rel"][:, :2]; hn = hv.norm(dim=-1, keepdim=True) + 1e-9
        shift = torch.zeros_like(park["palm_rel"]); shift[:, :2] = hv / hn * (0.12 - hn)      # 손바닥을 축에서 12 cm 로
        park["palm_rel"] = park["palm_rel"] + shift; park["tips_rel"] = park["tips_rel"] + shift.unsqueeze(1)
        park["cl"] = torch.zeros_like(park["cl"]); park["ff"] = torch.zeros_like(park["ff"])
        reset_cache(); rp, cpk = run(scene(src_far2, park))
        reset_cache(); ri, cin = run(scene(src_far2, pk2))
        g31 = float(ri.mean() - rp.mean())
        print(f"[31] 왼손 12 cm 편 손 정지 {float(rp.mean()):+.3f} · 손가락 사이 자세 {float(ri.mean()):+.3f} (차 {g31:+.3f})"
              f" · {(fk := next(k for k in ('finger_dir_rcv', 'mouth_rcv', 'approach_rcv') if k in cpk))} {float(cpk[fk].mean()):+.3f}→{float(cin[fk].mean()):+.3f}")
        if g31 < 0.5:
            print("    ✗ 손가락 사이 자세가 12 cm 편 손 정지보다 스텝당 0.5 이상 이득이 아니다"); ok = False
        zt = zt_bak
    # [32] (b) 원거리 손가락 배치 기울기: 같은 12 cm 에서 손가락이 컵 축을 향한 배치 > 아래를 향한 배치
    for arm in ("src", "rcv"):
        c_dn = offset_ctx(arm, -0.12, 0.0, 0.25 * MZ)
        c_in = offset_ctx(arm, -0.12, 0.0, 0.25 * MZ)
        fx = torch.tensor([0.67, -0.742, 0.0, 0.742, 0.67, 0.0], dtype=torch.float64).repeat(c_in.src_palm_pos.shape[0], 1)   # ★인벨롭 축(i24 실측)
        setattr(c_in, f"{arm}_palm_axes", fx)
        reset_cache(); r_dn, _ = run(c_dn); reset_cache(); r_in, _ = run(c_in)
        g32 = float(r_in.mean() - r_dn.mean())
        print(f"[32] {arm} 12 cm 에서 인벨롭 축 - 아래 방향 {g32:+.3f}")
        if g32 < 0.03:
            print(f"    ✗ {arm}: 12 cm 에서 손가락을 컵 쪽으로 돌리는 기울기가 없다"); ok = False

    # [33] ★full-joint 제어 벌점 상한(cup_grasp 수준): 무작위 ±1 균등 액션 비용 ≤ 0.06/스텝, 손 채널만 ±1 비용 ≤ 0.02,
    #      팔 관절 0.3 rad/s 로 움직이는 비용 ≤ 0.08 (f00: 팔 제어 벌점 −0.14~−0.19 가 스텝 수입 대부분을 먹었다)
    gen = torch.Generator().manual_seed(0)
    for arm in ("src",):
        base = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        reset_cache(); r0, _ = run(base)
        rnd = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        rnd.actions = torch.rand(rnd.actions.shape, generator=gen) * 2 - 1
        rnd.prev_actions = torch.rand(rnd.actions.shape, generator=gen) * 2 - 1
        reset_cache(); r1, _ = run(rnd)
        mv = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        mv.src_arm_qd = torch.full_like(mv.src_arm_qd, 0.3); mv.rcv_arm_qd = torch.full_like(mv.rcv_arm_qd, 0.3)
        reset_cache(); r2, _ = run(mv)
        c_rnd, c_mv = float(r0.mean() - r1.mean()), float(r0.mean() - r2.mean())
        print(f"[33] 제어 비용: 무작위 ±1 액션 {c_rnd:+.3f} · 팔 관절 0.3 rad/s {c_mv:+.3f} (스텝당)")
        if c_rnd > 0.06:
            print("    ✗ 무작위 탐색 액션 비용이 0.06/스텝을 넘는다(접근 경사를 덮는다)"); ok = False
        if c_mv > 0.08:
            print("    ✗ 팔을 0.3 rad/s 로 움직이는 비용이 0.08/스텝을 넘는다"); ok = False

    # [34] ★09.28 사용자 "3번": 수직 손(손가락이 아래) 컵 근처 자세는 입구 보상을 거의 받지 않아야 한다.
    #   f01 ep1650 재생(trace_f01_ep1650.npz)의 오른손 실제 상태 — 손가락 축 z < −0.7, 손바닥-컵 수평 < 10 cm, 파지 전 —
    #   를 그대로 넣어 mouth_src 평균을 잰다. 기준: 수평 손 '컵이 손가락 사이'(원장 3 과 같은 hand()) 입구 보상의 35 % 이하.
    import numpy as np
    TR34 = "/media/user/DATA/kuku_ai/rl/rl_runs_20260926/pour_bi_rh/trace_f01_ep1650.npz"
    try:
        z4 = np.load(TR34)
    except OSError:
        z4 = None; print("[34] 재생 파일 없음"); ok = False
    if z4 is not None:
        def _R(e):
            y_, p_, r_ = e[..., 0], e[..., 1], e[..., 2]
            cy, sy, cp_, sp, cr, sr = np.cos(y_), np.sin(y_), np.cos(p_), np.sin(p_), np.cos(r_), np.sin(r_)
            return np.stack([np.stack([cy*cp_, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], -1), np.stack([sy*cp_, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], -1),
                             np.stack([-sp, cp_*sr, cp_*cr], -1)], -2)
        Rm = _R(z4["src_palm_pose"][..., 3:]); fz = Rm[..., 2, 1]
        dh = np.linalg.norm(z4["src_palm_rel"][..., :2], axis=-1)
        sel = np.argwhere((fz < -0.7) & (dh < 0.10) & (z4["hold"] < 0.5) & (z4["src_grasped"] < 0.5))
        rng = np.random.default_rng(0); sel = sel[rng.choice(len(sel), size=min(512, len(sel)), replace=False)]
        n4 = len(sel); ti, ni = sel[:, 0], sel[:, 1]
        f64 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
        ps = torch.tensor([0.382, -0.159, TABLE_Z - BZ]).repeat(n4, 1); pr = torch.tensor([0.378, 0.165, TABLE_Z - BZ]).repeat(n4, 1)
        up = torch.tensor([0.0, 0.0, 1.0]).repeat(n4, 1)
        c4 = make_ctx(ps.clone(), up.clone(), pr.clone(), up.clone(), prog=0.3, hand="neutral")
        c4.src_cup_spawn_pos = ps.clone(); c4.rcv_cup_spawn_pos = pr.clone()
        c4.src_palm_pos = ps + f64(z4["src_palm_rel"][ti, ni]); c4.src_tips_pos = ps.unsqueeze(1) + f64(z4["src_tips_rel"][ti, ni])
        c4.src_palm_axes = torch.cat([f64(Rm[ti, ni][:, :, 2]), f64(Rm[ti, ni][:, :, 1])], dim=1)
        c4.src_finger_force = f64(z4["src_f"][ti, ni]); c4.src_palm_force = f64(z4["src_f_palm"][ti, ni])
        c4.src_hand_closure = f64(z4["src_closure"][ti, ni]); c4.src_grasped = torch.zeros(n4, dtype=torch.bool)
        c4.actions, c4.prev_actions = torch.zeros(n4, 26), torch.zeros(n4, 26)
        reset_cache(); _, cv = run(c4)
        # 수평 기준: 원장 3 과 같은 자세(컵이 손가락 사이·손끝 1 cm·손가락 수평) — stage_ledger 의 hand() 로 만든다
        import importlib.util as _iu
        _sp = _iu.spec_from_file_location("_led", BASE + "stage_ledger.py"); _led = _iu.module_from_spec(_sp); _sp.loader.exec_module(_led)
        _m = sys.modules[__name__] if __name__ in sys.modules else None
        base_s = torch.tensor([0.382, -0.159, TABLE_Z - BZ]).repeat(64, 1); base_r = torch.tensor([0.378, 0.165, TABLE_Z - BZ]).repeat(64, 1)
        _led.NUM_ACTIONS = 26
        _mm = types.SimpleNamespace(R=R, MZ=MZ, BZ=BZ, TABLE_Z=TABLE_Z, make_ctx=make_ctx)
        ch = _led.scene(_mm, _led.hand(_mm, base_s, 0.01, 0.2, 0, 0, False), _led.hand(_mm, base_r, 0.01, 0.2, 0, 0, False), base_s, base_r)
        reset_cache(); _, chh = run(ch)
        if "mouth_src" not in cv and "approach_src" in cv:      # ★09.29 iter_03(B): 입구 항 없음 → 들기 전 수입(접근+닫기+파지) 합
            for d_ in (cv, chh):
                d_["pre_src"] = d_["approach_src"] + d_.get("close_src", 0.0) + d_.get("grasp_src", 0.0)
        mk = "mouth_src" if "mouth_src" in cv else ("pre_src" if "pre_src" in cv else None)
        if mk is None:
            print("[34] mouth_src 항 없음 — 이름 확인"); ok = False
        else:
            v_m, h_m = float(cv[mk].mean()), float(chh[mk].mean())
            print(f"[34] 수직 손 재생 자세({n4}개) {mk} {v_m:+.3f} · 수평 손가락 사이 자세 {h_m:+.3f} (비 {v_m / max(h_m, 1e-6):.2f})")
            if v_m > 0.35 * h_m:
                print("    ✗ 수직 손이 입구 보상을 수평 손의 35 % 넘게 받는다(f01 오른손 수직 접근 구조)"); ok = False

    # [35] ★09.29 사용자 "인벨롭 그립 가능하게 테이블 수평으로 접근": 같은 위치·손끝(원장 2)에서 손 방향만 바꿔
    #   인벨롭(i24 실측 축) > 손끝 겨눔 + 0.2, 인벨롭 > 수직 손 + 0.4 (스텝당 총수입, 두 손 합)
    import importlib.util as _iu5
    _sp5 = _iu5.spec_from_file_location("_led5", BASE + "stage_ledger.py"); _l5 = _iu5.module_from_spec(_sp5); _sp5.loader.exec_module(_l5)
    _l5.NUM_ACTIONS = 26
    _m5 = types.SimpleNamespace(R=R, MZ=MZ, BZ=BZ, TABLE_Z=TABLE_Z, make_ctx=make_ctx)
    b_s = torch.tensor([0.382, -0.159, TABLE_Z - BZ]).repeat(64, 1); b_r = torch.tensor([0.378, 0.165, TABLE_Z - BZ]).repeat(64, 1)
    v35 = {}
    for ax in ("envelope", "point", "vertical"):
        c5 = _l5.scene(_m5, _l5.hand(_m5, b_s, 0.05, 0.0, 0, 0, False, axes=ax), _l5.hand(_m5, b_r, 0.05, 0.0, 0, 0, False, axes=ax), b_s, b_r)
        reset_cache(); r5, _ = run(c5); v35[ax] = float(r5.mean())
    print(f"[35] 같은 자리 손 방향: 인벨롭 {v35['envelope']:+.3f} · 손끝 겨눔 {v35['point']:+.3f} · 수직 손 {v35['vertical']:+.3f}")
    if v35["envelope"] < v35["point"] + 0.2:
        print("    ✗ 인벨롭 자세가 손끝 겨눔보다 스텝당 0.2 이상 이득이 아니다"); ok = False
    if v35["envelope"] < v35["vertical"] + 0.4:
        print("    ✗ 인벨롭 자세가 수직 손보다 스텝당 0.4 이상 이득이 아니다"); ok = False

    # [36] ★09.29 iter_03(B) 원장 조건: 단조 0<1<1b<2<3<4<5<6(3 cm), 파지 5 / 벽 4 ≥ 1.6, 5v ≤ 5 − 1.0, 3p ≥ 3 − 0.02, 0p ≥ 0 − 0.02, 1h < 1b
    rows = {}
    for r in _l5.ledger(CAND):
        rows.setdefault(r[0].split(" ")[0], r[1])     # 6 은 첫 행(3 cm)
    chain = ["0", "1", "1b", "2", "3", "4", "5", "6"]
    mono = all(rows[b] > rows[a] for a, b in zip(chain, chain[1:]))
    ratio = rows["5"] / max(rows["4"], 1e-6)
    print(f"[36] 원장 " + " < ".join(f"{k} {rows[k]:.2f}" for k in chain) + f" · 5/4 {ratio:.2f} · 5v {rows['5v']:.2f} · 3p {rows['3p']:.2f} · 0p {rows['0p']:.2f} · 1h {rows['1h']:.2f}")
    if not mono:
        print("    ✗ 원장 단조가 아니다(골)"); ok = False
    if ratio < 1.6:
        print("    ✗ 파지(판정 켜짐)가 벽에 닿고 닫힘보다 1.6 배 미만 — 판정 전 머묾"); ok = False
    if rows["5v"] > rows["5"] - 1.0:
        print("    ✗ 세로 집기 파지가 인벨롭 파지보다 1.0 이상 싸지 않다(f03 왼손)"); ok = False
    if rows["3p"] < rows["3"] - 0.02 or rows["0p"] < rows["0"] - 0.02:
        print("    ✗ 밀린 컵 상태에 비용이 남는다(i09)"); ok = False
    if rows["1h"] >= rows["1b"]:
        print("    ✗ 반쯤 쥔 손이 편 손보다 이득"); ok = False

    # [37] ★09.29 팔 절대 목표: 같은 상태에서 팔 액션을 0 → ±0.8 로 일정하게(변화율 0) 바꿔도 보상이 같아야 한다(크기 비용 = 시작 자세 끌어당김)
    c37 = []
    for arm, sl in (("src", slice(0, 7)), ("rcv", slice(13, 20))):
        for v in (0.8, -0.8):
            b0 = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
            reset_cache(); q0, _ = run(b0)
            b1 = grasp_path_ctx(arm, 0.0, 0.0, 0.0, 0.0, False, 0.0)
            b1.actions = b1.actions.clone(); b1.actions[:, sl] = v; b1.prev_actions = b1.actions.clone()
            reset_cache(); q1, _ = run(b1)
            c37.append(float(q0.mean() - q1.mean()))
    print(f"[37] 팔 액션 ±0.8 일정(변화 0) 비용: " + " · ".join(f"{c:+.4f}" for c in c37))
    if max(abs(c) for c in c37) > 0.002:
        print("    ✗ 팔 액션 크기 비용이 있다 — 절대 목표에서는 시작 자세로 끌어당긴다"); ok = False

    # [38] ★09.29 사용자 "둘 다 대칭": 원장의 들기 전 행(소스 손은 y 미러)에서 팔별 접근·닫기·파지 성분이 같아야 한다
    led38 = _l5.ledger(CAND)
    d38 = 0.0
    for nm, _t, cc in led38:
        if nm.split(" ")[0] in ("0", "1", "1b", "2", "2f", "2v", "3", "4", "5", "5v", "3p", "1h", "0p", "4p", "5p"):
            for a_ in ("approach", "close", "grasp"):
                if a_ + "_src" in cc:
                    d38 = max(d38, abs(cc[a_ + "_src"] - cc[a_ + "_rcv"]))
    print(f"[38] 좌우 미러 자세에서 팔별 들기 전 성분 최대 차 {d38:.4f}")
    if d38 > 1e-3:
        print("    ✗ 같은 자세를 좌우로 미러했는데 두 팔 보상이 다르다"); ok = False

    # [39]·[40] ★09.29 f04 ep2050 트레이스 자세(리시버 기준, 소스는 원장이 미러): 같은 손끝·파지 힘에서 손 자세만 다르게
    import math as _m9
    def _axes(nv, fv):
        return torch.tensor(list(nv) + list(fv))
    nv0, fv0 = _l5.ENV_AXES["envelope"]
    a47, a35 = _m9.radians(47.0), _m9.radians(35.0)
    sup = _axes([_m9.cos(a47) * nv0[0], _m9.cos(a47) * nv0[1], _m9.sin(a47)], fv0)                       # 법선이 위로 47°(f04 오른손 z +0.72)
    tilt = _axes(nv0, [_m9.cos(a35) * fv0[0], _m9.cos(a35) * fv0[1], -_m9.sin(a35)])                     # 손가락 축 아래 35°(f04 왼손 z −0.57)
    def _grasp_scene(ax=None, palm_h=None):
        hs = _l5.hand(_m5, b_s, 0.0, 0.8, 1.5, 1.5, True, palm_h=palm_h)
        hr = _l5.hand(_m5, b_r, 0.0, 0.8, 1.5, 1.5, True, palm_h=palm_h)
        if ax is not None:
            ar = ax.repeat(64, 1); as_ = ar.clone(); as_[:, 1] *= -1; as_[:, 4] *= -1
            hs = (hs[0], hs[1], as_, *hs[3:]); hr = (hr[0], hr[1], ar, *hr[3:])
        return _l5.scene(_m5, hs, hr, b_s, b_r)
    reset_cache(); r_env, _ = run(_grasp_scene())
    reset_cache(); r_sup, _ = run(_grasp_scene(ax=sup))
    reset_cache(); r_hi, _ = run(_grasp_scene(ax=tilt, palm_h=0.034))
    g39 = float(r_env.mean() - r_sup.mean()); g40 = float(r_env.mean() - r_hi.mean())
    print(f"[39] 파지 상태: 인벨롭 {float(r_env.mean()):+.2f} · 손바닥 위로 굴린 받침 {float(r_sup.mean()):+.2f} (차 {g39:+.2f}, 두 손 합)")
    if g39 < 2.0:
        print("    ✗ 손바닥을 위로 굴린 받침 자세가 인벨롭보다 두 손 합 2.0 이상 싸지 않다(f04 오른손 j5 굴림)"); ok = False
    print(f"[40] 파지 상태: 인벨롭(손바닥 원점 +0.8 cm·손가락 수평) {float(r_env.mean()):+.2f} · 높고 기운 파지(+3.4 cm·손가락 35° 아래) {float(r_hi.mean()):+.2f} (차 {g40:+.2f})")
    if g40 < 1.5:
        print("    ✗ 높고 아래로 기운 파지가 낮은 수평 파지보다 두 손 합 1.5 이상 싸지 않다(f04 왼손 영상 판정 '더 낮게')"); ok = False

    # [41] ★09.29 사용자 "인벨롭이라 5지 전부 참여": 같은 자리·방향에서 다섯 손가락(각 1.5 N) 대 엄지·중지만(각 1.5 N), 테이블과 3 cm 들기
    r41 = {r[0].split(" ")[0]: r[1] for r in led38}
    def _two(cp):
        t_, pa_, ax_, ff_, cl_, gr_ = _l5.hand(_m5, cp, 0.0, 0.8, 1.5, 6.0, True)
        ff_ = torch.zeros_like(ff_); ff_[:, 0] = 1.5; ff_[:, 2] = 1.5
        return t_, pa_, ax_, ff_, cl_, gr_
    up3 = torch.tensor([0.0, 0.0, 0.03])
    reset_cache(); r5l, _ = run(_l5.scene(_m5, _l5.hand(_m5, b_s + up3, 0.0, 0.8, 1.5, 6.0, True), _l5.hand(_m5, b_r + up3, 0.0, 0.8, 1.5, 6.0, True), b_s + up3, b_r + up3))
    reset_cache(); r2l, _ = run(_l5.scene(_m5, _two(b_s + up3), _two(b_r + up3), b_s + up3, b_r + up3))
    g41a = r41["5"] - r41["5t"]; g41b = float(r5l.mean() - r2l.mean())
    print(f"[41] 다섯 손가락 - 엄지·중지 두 손가락(두 손 합): 테이블 {g41a:+.2f} · 3 cm 들기 {g41b:+.2f}")
    if g41a < 2.0 or g41b < 2.0:
        print("    ✗ 두 손가락 집기가 다섯 손가락 인벨롭보다 두 손 합 2.0 이상 싸지 않다(f05 오른손)"); ok = False
    # [42] ★09.29 사용자 "틸팅 중에도 그 자세 유지(무게중심 변화)": 소스 45° 에서 손이 컵과 함께 돈 파지 대 세운 채 남은 손
    g42 = r41["9"] if False else None
    rows42 = {r[0]: r[1] for r in led38}
    k45 = next(k for k in rows42 if k.startswith("9 ") and "45" in k); k9r = next(k for k in rows42 if k.startswith("9r"))
    g42 = rows42[k45] - rows42[k9r]
    print(f"[42] 기울기 45°: 손이 컵과 함께 돈 파지 {rows42[k45]:+.2f} · 손이 세운 채 남음 {rows42[k9r]:+.2f} (차 {g42:+.2f})")
    if g42 < 1.5:
        print("    ✗ 기울이는 동안 컵 기준 파지 자세를 유지하는 쪽이 1.5 이상 이득이 아니다"); ok = False

    # [43] ★09.29 사용자 "2,3 개만 붙이고 다른 건 안 붙여도 된다고 생각할 것": 손가락 수별 파지(두 손 합)에서 한 손가락 더 붙일 때의 몫이 늘어야 한다
    r43 = {r[0].split(" ")[0]: r[1] for r in led38}
    seq = [r43["5t"], r43["5_3"], r43["5_4"], r43["5"]]
    inc = [b_ - a_ for a_, b_ in zip(seq, seq[1:])]
    print(f"[43] 손가락 수별 파지 2지 {seq[0]:.2f} · 3지 {seq[1]:.2f} · 4지 {seq[2]:.2f} · 5지 {seq[3]:.2f} (한 손가락 더: {inc[0]:+.2f} · {inc[1]:+.2f} · {inc[2]:+.2f})")
    if not (inc[0] > 0 and inc[1] > inc[0] and inc[2] > inc[1]):
        print("    ✗ 한 손가락 더 붙일 때의 몫이 손가락이 늘수록 커지지 않는다(마지막 손가락을 건너뛰기 쉽다)"); ok = False
    if inc[2] < 2.5 or inc[2] < 1.5 * inc[0]:
        print("    ✗ 5지째(새끼)의 몫이 두 손 합 2.5 미만이거나 3지째 몫의 1.5 배 미만"); ok = False
    # [44] 엄지·검지·중지가 쥔 상태에서 약지·새끼 끝을 벽 바깥으로 d 만큼 — 가까울수록 올라야 한다(닿기 전 기울기)
    import math as _m4
    def _gap(cp, d):
        t_, pa_, ax_, ff_, cl_, gr_ = _l5.hand(_m5, cp, 0.0, 0.8, 1.5, 6.0, True)
        ff_ = torch.zeros_like(ff_); ff_[:, 0] = 1.5; ff_[:, 1] = 1.5; ff_[:, 2] = 1.5
        t_ = t_.clone()
        for i in (3, 4):
            v_ = t_[:, i] - cp; v_[:, 2] = 0.0; v_ = v_ / v_.norm(dim=-1, keepdim=True)
            t_[:, i] = t_[:, i] + d * v_
        return t_, pa_, ax_, ff_, cl_, gr_
    v44 = {}
    for d in (0.04, 0.02, 0.005):
        reset_cache(); r_, _ = run(_l5.scene(_m5, _gap(b_s, d), _gap(b_r, d), b_s, b_r)); v44[d] = float(r_.mean())
    g44a = v44[0.02] - v44[0.04]; g44b = v44[0.005] - v44[0.02]
    print(f"[44] 약지·새끼 끝-벽 40 → 20 mm {g44a:+.3f} · 20 → 5 mm {g44b:+.3f} (두 손 합, 닿기 전)")
    if g44a <= 0.05 or g44b < 0.3:
        print("    ✗ 안 닿은 손가락이 벽에 다가가는 기울기가 약하다(20→5 mm 0.3 미만 또는 40→20 mm 0.05 이하)"); ok = False

    # [45] ★09.29 f07 ep1000 오른손: 엄지를 입구 안(축 0.57 r·4.7 cm·12 N)에 걸고 쥔 파지 — 테이블과 3 cm 들기 모두 다섯 손가락 인벨롭보다 확실히 낮아야
    up3b = torch.tensor([0.0, 0.0, 0.03])
    def _sc(fn, dz=None):
        ps_ = b_s if dz is None else b_s + dz; pr_ = b_r if dz is None else b_r + dz
        reset_cache(); r_, _ = run(_l5.scene(_m5, fn(ps_), fn(pr_), ps_, pr_)); return float(r_.mean())
    env5 = lambda cp: _l5.hand(_m5, cp, 0.0, 0.8, 1.5, 6.0, True)
    hook = lambda cp: _l5.hook_hand(_m5, cp)
    e_t, h_t = _sc(env5), _sc(hook); e_l, h_l = _sc(env5, up3b), _sc(hook, up3b)
    print(f"[45] 인벨롭 - 엄지 입구 안 걸기(두 손 합): 테이블 {e_t - h_t:+.2f} ({e_t:.2f} 대 {h_t:.2f}) · 3 cm 들기 {e_l - h_l:+.2f} ({e_l:.2f} 대 {h_l:.2f})")
    if e_l - h_l < 12.0:
        print("    ✗ 엄지를 입구에 걸고 든 파지가 인벨롭 들기보다 두 손 합 12 이상 낮지 않다(걸고 들기가 매력적)"); ok = False
    # [46] 걸기 → 엄지를 빼서 놓음(판정 꺼짐, 엄지 바깥 1.3 r·2.0 cm·0 N) → 엄지 바깥에서 같은 세 손가락으로 다시 쥠: 골이 없어야 한다
    rel = lambda cp: _l5.hook_hand(_m5, cp, thumb_rad=1.3, thumb_h=0.020, f_thumb=0.0, grasped=False)
    reg = lambda cp: _l5.hook_hand(_m5, cp, thumb_rad=1.3, thumb_h=0.020, f_thumb=1.5, grasped=True)
    v_rel, v_reg = _sc(rel), _sc(reg)
    print(f"[46] 걸기 {h_t:.2f} → 엄지 빼서 놓음 {v_rel:.2f} ({v_rel - h_t:+.2f}) → 엄지 바깥 다시 쥠 {v_reg:.2f} ({v_reg - v_rel:+.2f}) (두 손 합)")
    if v_rel < h_t - 0.05 or v_reg - h_t < 3.0:
        print("    ✗ 엄지 걸기에서 빠져나오는 길에 골이 있거나(놓는 순간 −0.05 초과 손해) 바깥 파지 이득이 3.0 미만"); ok = False

    # [47]~[49] ★09.29 사용자 "옆으로 접근해야 하는데 위에서 아래로 접근" · "palm sensor 로컬 +z 가 중력과 90도" ·
    #   "인벨롭 되게 충분히 낮게 내려간 뒤 접근". f07 ep1000·f05 ep500 결정론 재생에서 손끝이 처음 컵 축 1.3 r 안에 든 순간
    #   (쥐기 전, 64 env x 두 손): 손바닥이 입구보다 3~8 cm 위, 손가락 길이 축 z -0.4~-0.6(아래로 25~36 도), 로컬 +z(법선) 는 거의 수평.
    #   같은 수평 위치에서 (B) 높이만 잡는 높이로 내림, (C) 내리고 손가락 길이 축을 수평으로 돌림(법선 축 둘레 회전, 손끝도 함께).
    EF = np.load(BASE + "iter_09/entry_frames.npz")
    def _eax(e):
        y_, p_, r_ = e[..., 0], e[..., 1], e[..., 2]
        cy, sy, cp_, sp, cr, sr = np.cos(y_), np.sin(y_), np.cos(p_), np.sin(p_), np.cos(r_), np.sin(r_)
        c1 = np.stack([cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp_ * sr], -1)
        c2 = np.stack([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp_ * cr], -1)
        return c2, c1
    def _entry(tag, side, dz_to=None, level=False, n_tilt=0.0):
        g = lambda k: np.array(EF[f"{tag}_{side}_{k}"], dtype=np.float64)
        pr_, tr_ = g("palm_rel"), g("tips_rel")
        nrm, fin = _eax(g("palm_pose")[:, 3:])
        if level:                                    # 법선을 수평으로 눕히고, 손가락 길이 축을 수평으로 돌린다(손끝은 손바닥 둘레로 같이)
            nh = nrm.copy(); nh[:, 2] = 0.0; nh /= np.linalg.norm(nh, axis=-1, keepdims=True)
            fh = np.cross(nh, np.array([0.0, 0.0, 1.0])); fh *= np.sign((fh * fin).sum(-1, keepdims=True) + 1e-12)
            fh /= np.linalg.norm(fh, axis=-1, keepdims=True)
            xo = np.cross(fin, nrm); xn = np.cross(fh, nh)
            Ro = np.stack([xo, fin, nrm], -1); Rn = np.stack([xn, fh, nh], -1)   # 열 = 로컬 x, y(손가락), z(법선)
            Rd = Rn @ np.transpose(Ro, (0, 2, 1))
            tr_ = pr_[:, None, :] + np.einsum("nij,nfj->nfi", Rd, tr_ - pr_[:, None, :])
            nrm, fin = nh, fh
            if n_tilt:                                   # 손가락 길이 축 둘레로 굴려 법선 z = -n_tilt (f07 오른손 파지 -0.18~-0.31)
                xz = np.cross(fin, nrm); sgn = -np.sign(xz[:, 2:3] + 1e-12)
                a = np.arcsin(n_tilt)
                n2 = np.cos(a) * nrm + np.sin(a) * sgn * xz
                x2 = np.cross(fin, n2)
                Ro = np.stack([xz, fin, nrm], -1); Rn = np.stack([x2, fin, n2], -1)
                Rd = Rn @ np.transpose(Ro, (0, 2, 1))
                tr_ = pr_[:, None, :] + np.einsum("nij,nfj->nfi", Rd, tr_ - pr_[:, None, :])
                nrm = n2
        if dz_to is not None:
            dz = pr_[:, 2] - dz_to
            pr_ = pr_.copy(); pr_[:, 2] -= dz; tr_ = tr_ - np.stack([np.zeros_like(dz), np.zeros_like(dz), dz], -1)[:, None, :]
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        return dict(palm_rel=t(pr_), tips_rel=t(tr_), axes=t(np.concatenate([nrm, fin], -1)), ff=t(g("f")),
                    fp=t(g("f_palm")), cl=t(g("closure")), gr=t(g("grasped")) > 0.5)
    def _scene47(sq, rq):
        n = sq["palm_rel"].shape[0]
        ps = torch.tensor([0.382, -0.159, TABLE_Z - BZ]).repeat(n, 1); pr = torch.tensor([0.378, 0.165, TABLE_Z - BZ]).repeat(n, 1)
        up = torch.tensor([0.0, 0.0, 1.0]).repeat(n, 1)
        c = make_ctx(ps.clone(), up.clone(), pr.clone(), up.clone(), prog=0.2, hand="neutral")
        for side, cp, q in (("src", ps, sq), ("rcv", pr, rq)):
            setattr(c, f"{side}_cup_spawn_pos", cp.clone())
            setattr(c, f"{side}_palm_pos", cp + q["palm_rel"]); setattr(c, f"{side}_tips_pos", cp.unsqueeze(1) + q["tips_rel"])
            setattr(c, f"{side}_palm_axes", q["axes"]); setattr(c, f"{side}_finger_force", q["ff"])
            setattr(c, f"{side}_palm_force", q["fp"]); setattr(c, f"{side}_hand_closure", q["cl"]); setattr(c, f"{side}_grasped", q["gr"])
        return c
    H_GRIP = 0.15 * MZ                               # K7 잡는 손바닥 높이(컵 원점 기준)
    for tag in ("f07e1000", "f05e500"):
        vals = {}
        for key, kw in {"A": {}, "B": {"dz_to": H_GRIP}, "C": {"dz_to": H_GRIP, "level": True},
                        "L": {"level": True}, "N": {"dz_to": H_GRIP, "level": True, "n_tilt": 0.3}}.items():
            reset_cache(); r_, _ = run(_scene47(_entry(tag, "src", **kw), _entry(tag, "rcv", **kw))); vals[key] = float(r_.mean())
        hs = {}
        for hz in (0.13, 0.09, 0.05, H_GRIP):
            reset_cache(); r_, _ = run(_scene47(_entry(tag, "src", dz_to=hz), _entry(tag, "rcv", dz_to=hz))); hs[hz] = float(r_.mean())
        hk = list(hs)
        inc = [hs[hk[i + 1]] - hs[hk[i]] for i in range(3)]
        print(f"[47] {tag} 진입 순간(두 손 합): 기록(위) {vals['A']:+.3f} · 잡는 높이로 내림 {vals['B']:+.3f} ({vals['B'] - vals['A']:+.3f}) · "
              f"내리고 손가락 수평 {vals['C']:+.3f} ({vals['C'] - vals['A']:+.3f})")
        if vals["B"] - vals["A"] < 0.4 or vals["C"] - vals["A"] < 0.6:
            print("    ✗ 같은 수평 위치에서 잡는 높이로 내린 손(+손가락 수평)이 위에서 덮는 손보다 두 손 합 0.4(0.6) 이상 이득이 아니다"); ok = False
        print(f"[48] {tag} 손바닥 높이 13 → 9 → 5 cm → 잡는 높이 {100 * H_GRIP:.1f} cm (컵 원점 기준, 두 손 합): "
              + " · ".join(f"{v:+.3f}" for v in inc))
        if min(inc) < 0.10:
            print("    ✗ 같은 수평 위치에서 한 칸(4 cm 안팎) 내려갈 때마다 두 손 합 +0.10 이상 오르지 않는다(높은 곳에서 기울기 없음)"); ok = False
        print(f"[49] {tag} 같은 자리에서 손가락 길이 축만 수평으로: 기록 높이 {vals['L'] - vals['A']:+.3f} · 잡는 높이 {vals['C'] - vals['B']:+.3f} (두 손 합)")
        if vals["C"] - vals["B"] < 0.4:
            print("    ✗ 잡는 높이에서 손가락 길이 축(아래 25~36 도)을 수평으로 돌리는 이득이 두 손 합 0.4 미만"); ok = False
        print(f"     {tag} 로컬 +z(법선) 수평 대 z -0.3(17 도 아래로 굴림), 잡는 높이·손가락 수평: {vals['C'] - vals['N']:+.3f} (두 손 합)")
        if vals["C"] - vals["N"] < 0.2:
            print("    ✗ 로컬 +z 가 중력과 90 도인 손이 17 도 굴린 손보다 두 손 합 0.2 이상 이득이 아니다"); ok = False

    # [50]~[52] ★09.30 사용자 영상 판정(f08 ep11800): "j7 은 과도하게 꺾여 있고 가까이 가지도 않고(높이 듦)".
    #   같은 체크포인트 결정론 재생 64 env, 두 손 쥐고 기울기 > 30° 스텝 평균: 소스 주둥이가 리시버 입구보다 21.3 cm 위(수평 4.6 cm),
    #   소스 컵 30.6 cm 들림(리시버 7.1 cm), 소스팔 j6 여유 1.4°(한계 10° 안 97 %)·j7 여유 2.4°(90 %). 기울이기 전 운반(기울기 < 10°)에도 j2·j4·j6·j7 이 31~41 % 스텝에서 한계 10° 안.
    TILT50 = 54.0
    def _pour_dz(dz_lip, margin=0.5):
        ps, us, pr, ur = pour_state(N, TILT50)
        lip0 = ps + MZ * us     # 주둥이(입구 중심)
        shift = (MZ + ps[:, 2] * 0.0 + (pr[:, 2] + MZ) + dz_lip) - lip0[:, 2] - MZ
        ps = ps.clone(); ps[:, 2] += shift
        reset_cache(); r_, comp = run(make_ctx(ps, us, pr, ur, prog=0.5, margin=margin)); return float(r_.mean()), comp
    dzs = (0.213, 0.15, 0.10, 2.0 * R)
    v50 = [_pour_dz(z)[0] for z in dzs]
    inc50 = [v50[i + 1] - v50[i] for i in range(3)]
    print(f"[50] 기울기 {TILT50:.0f}° 소스 주둥이 - 리시버 입구 높이 21.3 → 15 → 10 → {100 * 2 * R:.1f} cm: " + " · ".join(f"{v:+.2f}" for v in v50)
          + f" (칸별 {inc50[0]:+.3f} · {inc50[1]:+.3f} · {inc50[2]:+.3f}, 합 {v50[-1] - v50[0]:+.2f})")
    if min(inc50) < 0.1 or v50[-1] - v50[0] < 1.0:
        print("    ✗ 높이 든 채 붓는 자세에서 리시버 입구 쪽으로 내려오는 칸마다 +0.1, 합 +1.0 이상 오르지 않는다"); ok = False
    # [51] 쥔 동안 손목(j6·j7) 한계 여유: 붓는 중(54°)과 기울이기 전 운반(0°) 모두
    def _mar(lo_j):
        m_ = 0.5 * torch.ones(N, 7)
        for j in lo_j: m_[:, j] = 0.03       # 여유 0.03 = 90° 관절 2.4° · 45° 관절 1.4° (재생 실측)
        return m_
    r_ok = _pour_dz(2.0 * R, margin=0.5)[0]; r_lim = _pour_dz(2.0 * R, margin=_mar((5, 6)))[0]
    ps0, us0, pr0, ur0 = pour_state(N, 0.0)
    reset_cache(); c_ok, _ = run(make_ctx(ps0, us0, pr0, ur0, prog=0.5, margin=0.5))
    reset_cache(); c_lim, _ = run(make_ctx(ps0, us0, pr0, ur0, prog=0.5, margin=_mar((5, 6))))
    g51a, g51b = r_ok - r_lim, float(c_ok.mean() - c_lim.mean())
    print(f"[51] 손목 j6·j7 여유 0.5 - 0.03: 붓는 중 54° {g51a:+.3f} · 기울이기 전 0° {g51b:+.3f}")
    if g51a < 0.3 or g51b < 0.2:
        print("    ✗ 쥔 동안 손목 관절을 한계에 붙이는 비용이 약하다(붓는 중 0.3, 기울이기 전 0.2 미만)"); ok = False
    # [52] 들기 높이 상한: 두 손 인벨롭 파지, 제자리, 리시버 8 cm. 소스 21 cm(= 리시버 8 + 컵 높이 + r, 리시버 입구 위를 지나갈 여유) 보다 높이 들어도 수입이 늘면 안 된다
    def _lift(h_s, h_r=0.08):
        ps_ = b_s + torch.tensor([0.0, 0.0, h_s]); pr_ = b_r + torch.tensor([0.0, 0.0, h_r])
        env5_ = lambda cp: _l5.hand(_m5, cp, 0.0, 0.8, 1.5, 6.0, True)
        reset_cache(); r_, _ = run(_l5.scene(_m5, env5_(ps_), env5_(pr_), ps_, pr_)); return float(r_.mean())
    h_clear = 0.08 + (MZ - BZ) + R
    l21, l28, l35 = _lift(h_clear), _lift(0.28), _lift(0.355)
    print(f"[52] 소스 들기 {100 * h_clear:.0f} cm {l21:+.2f} · 28 cm {l28:+.2f} ({l28 - l21:+.3f}) · 35.5 cm {l35:+.2f} ({l35 - l21:+.3f}) (두 손 합, 리시버 8 cm)")
    if l28 - l21 > 0.05 or l35 - l21 > 0.0:
        print("    ✗ 리시버 입구를 넘을 여유보다 높이 든 소스가 더 받는다(28 cm +0.05 초과 또는 35.5 cm 이득)"); ok = False

    # [53]~[55] ★09.30 사용자 "지금도 자세가 이상하고 컵도 제대로 파지를 못하고 있음".
    #   f08 ep11800(소스)·f09 ep1600(리시버) 결정론 재생의 쥔·세운·들린 프레임 256 개씩: 손바닥이 입구 높이 근처(+0.82·+0.57 × 입구),
    #   컵 축 2.1~2.5 r, 손가락 축·컵 축 −0.4~−0.7 — 입구 쪽을 비스듬히 위에서 잡은 파지. 붓는 동안 주둥이-리시버 입구 수평 12 cm.
    RF = np.load(BASE + "iter_09/rim_frames.npz")
    NR = 256
    def _rim(side, cp):
        g = lambda k: torch.tensor(np.array(RF[f"{side}_{k}"], dtype=np.float64))
        nrm, fin = _eax(np.array(RF[f"{side}_palm_pose"], dtype=np.float64)[:, 3:])
        ax = torch.tensor(np.concatenate([nrm, fin], -1))
        return (cp.unsqueeze(1) + g("tips_rel"), cp + g("palm_rel"), ax, g("f"), g("closure"), g("grasped") > 0.5)
    def _sc53(kind, lift):
        ps_ = torch.tensor([0.382, -0.159, TABLE_Z - BZ + lift]).repeat(NR, 1); pr_ = torch.tensor([0.378, 0.165, TABLE_Z - BZ + lift]).repeat(NR, 1)
        if kind == "env":
            hs, hr = _l5.hand(_m5, ps_, 0.0, 0.8, 1.5, 6.0, True, n=NR), _l5.hand(_m5, pr_, 0.0, 0.8, 1.5, 6.0, True, n=NR)
        else:
            hs, hr = _rim("src", ps_), _rim("rcv", pr_)
        reset_cache(); r_, _ = run(_l5.scene(_m5, hs, hr, ps_, pr_, n=NR)); return float(r_.mean())
    v53 = {(k, l): _sc53(k, l) for k in ("env", "rim") for l in (0.0, 0.03, 0.08)}
    print(f"[53] 인벨롭 - 입구 쪽 비스듬한 파지(실측, 두 손 합): 테이블 {v53[('env', 0.0)] - v53[('rim', 0.0)]:+.2f} · 3 cm {v53[('env', 0.03)] - v53[('rim', 0.03)]:+.2f} · 8 cm {v53[('env', 0.08)] - v53[('rim', 0.08)]:+.2f}"
          f"  (인벨롭 {v53[('env', 0.03)]:.2f} 대 비스듬 {v53[('rim', 0.03)]:.2f} @3 cm)")
    if min(v53[("env", l)] - v53[("rim", l)] for l in (0.03, 0.08)) < 12.0:
        print("    ✗ 들린 상태에서 인벨롭이 입구 쪽 비스듬한 파지보다 두 손 합 12 이상 크지 않다(iter_08 +9.5~+10.2)"); ok = False
    g_env = v53[("env", 0.08)] - v53[("env", 0.0)]; g_rim = v53[("rim", 0.08)] - v53[("rim", 0.0)]
    print(f"[54] 테이블 → 8 cm 들기 이득: 인벨롭 {g_env:+.2f} · 비스듬한 파지 {g_rim:+.2f} (비 {g_rim / max(g_env, 1e-6):.2f})")
    if g_rim > 0.5 * g_env:
        print("    ✗ 비스듬한 파지로 들어도 인벨롭 들기 이득의 절반 넘게 받는다(뒤 단계 수입이 파지 모양과 무관)"); ok = False
    def _pour_xy(dxy):
        ps, us, pr, ur = pour_state(N, TILT50, lip_dxy=dxy)
        lip0 = ps + MZ * us
        ps = ps.clone(); ps[:, 2] += (pr[:, 2] + MZ + 2.0 * R) - lip0[:, 2]
        reset_cache(); r_, _ = run(make_ctx(ps, us, pr, ur, prog=0.5)); return float(r_.mean())
    x55 = (0.12, 0.08, 0.04, R)
    v55 = [_pour_xy(x) for x in x55]
    inc55 = [v55[i + 1] - v55[i] for i in range(3)]
    print(f"[55] 기울기 {TILT50:.0f}°·높이 차 2 r 에서 주둥이-리시버 입구 수평 12 → 8 → 4 → {100 * R:.1f} cm: " + " · ".join(f"{v:+.2f}" for v in v55)
          + f" (칸별 {inc55[0]:+.3f} · {inc55[1]:+.3f} · {inc55[2]:+.3f}, 합 {v55[-1] - v55[0]:+.2f})")
    if min(inc55) < 0.1 or v55[-1] - v55[0] < 1.5:
        print("    ✗ 붓는 자세에서 주둥이를 리시버 입구 위로 옮기는 칸마다 +0.1, 합 +1.5 이상 오르지 않는다"); ok = False

    print(f"\n항 {len(names)}개: {', '.join(names)}")
    print("판정:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
