"""pour_bi_rh 참 궤적 원장 (10-01 사용자: "왼손이든 오른손이든 보상함수가 테스크 성공을 보장해야 한다 —
에피소드 동안 손 펴고 접근 → 인벨롭 그립 → 리프트 → 이송이라는 참 궤적을 찾을 수 있게").

한 에피소드(900 스텝, 60 Hz) 동안 두 팔이 참 궤적을 따라가는 결정론 상태열을 만들고, 스텝마다 후보 보상을 잰다.
배치 한 행 = 한 시각이라 한 번의 compute_reward 호출로 에피소드 전체를 잰다(float64 CPU).

참 궤적(리시버 기준, 소스는 y 반전 — 두 손 같은 식):
  A   0~150   시작 자세(f10 ep1800 재생 t=0 실측: 컵 원점에서 수평 +4.9·옆 14 cm, 위 15.1 cm, 손 편 채)에서
              손을 편 채(닫힘 0.05 = 실기 편 한계) 잡는 높이(원점 +0.15 x 입구)로 먼저 내리며 손을 테이블과 수평으로 돌린다
  B 150~270   잡는 높이로 컵 둘레 원호(반경 14.8 → 12 cm)를 따라 컵 뒤(로봇 쪽)로 돈다(편 손, 손끝이 컵을 뚫지 않게)
  C 270~350   컵 축 쪽으로 수평 진입 → K7 인벨롭 자리(축 1.5 r)
  D 350~410   닫힘 0.05 → 0.8: 네 손끝은 실측 굽힘 곡선으로 말려 컵 벽에 닿으면 멈추고(접촉 1.5 N), 엄지는 벽 반대쪽으로, 엄지 + 한 손가락 닿으면 파지 판정
  E 410~500   두 컵 8 cm 들기
  F 500~640   소스: 더 들며 리시버 입구 옆으로 이송(리시버는 8 cm 에서 유지)
  G 640~760   소스: 리시버 쪽으로 0 → 105° 기울여 붓기(95° 부터 스텝당 0.5 알, 20 알 전부)
  H 760~900   소스 다시 세우기, 유지(성공)

판정(후보마다):
  [T1] 연속: 참 궤적에서 스텝당 보상(배달 사건 수입 제외)이 한 스텝에 0.05 넘게 떨어지는 곳이 없다(단계 경계 절벽 금지).
  [T2] 전진 최적: H 전 어느 시각 t 에서든 '참 궤적대로 계속' 할인 가치 ≥ '그 자리에 멈춤' + 1.
       gamma 0.99(f00 agent.yaml), 900 뒤는 마지막 상태 유지(value bootstrap). 멈춤은 속도·배달 0.
  [T3] 이탈: 한 팔만 이탈(다른 팔은 참 궤적) 했을 때 분기 시점 할인 가치가 참 궤적보다 낮다(차 ≥ 2.0).
       반쯤 쥔 손 접근 · 위에서 덮는 접근 · 12 cm 편 손 주차 · 10 cm 반쯤 쥔 손 주차(f00) · 쥐고 테이블 유지 ·
       입구 쪽 비스듬한 파지(f08·f09) · 엄지·중지 두 손가락 파지 · 들고 유지 · 이송 뒤 안 붓기 ·
       80 % 붓고(성공 문턱) 80° 로 남은 비드를 안은 채 유지.
  [T4] 편 손: 접근 구간(A~C) 중 잡는 자리 밖(참 궤적 손바닥 축 거리 > 2.6 r) 매 스텝, 편 손(0.05)이 반쯤 쥔 손(0.5, 손끝 위치는 f10 재생 실측)보다 팔당 스텝당 +0.15 이상.
  [T4b] 닫힘 칸 기울기: 접근 구간 중 잡는 자리 밖(손바닥 축 거리 > 2.6 r) 매 스텝 닫힘 0.05 > 0.3 > 0.5 > 0.7 칸마다 팔당 +0.05 이상(f03: 0.5 위 평평).
  [T5] 대칭: 들기까지(0~500) 소스·리시버 팔별 성분(접근·파지; 들기 목표는 역할별) 차가 스텝당 0.02 안(두 컵은 y 거울 자리).

사용법: python3 traj_ledger.py <compute_reward.py> [...]
"""
import importlib.util
import math
import sys

import torch

torch.set_default_dtype(torch.float64)
HERE = "/home/user/rl_ws/hdgp/reward_gen/pour_bi_rh/"
T = 900
DT = 1.0 / 60.0
GAMMA = 0.99
NUM_ACTIONS = 26
PH = {"A": 0, "B": 150, "C": 270, "D": 350, "E": 410, "F": 500, "G": 640, "H": 760}


def load_check(cand):
    sys.argv = ["x", cand]
    spec = importlib.util.spec_from_file_location("chk", HERE + "iter_00/check_iter00.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lerp(a, b, s):
    return a + (b - a) * s


def _smooth(t, t0, t1):
    """t0~t1 에서 0 → 1 (코사인), 밖은 0 또는 1. t: (T,) float."""
    s = ((t - t0) / max(t1 - t0, 1)).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * s)


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _rot_ez_to(u):
    """행마다 ez → u 로 돌리는 Rodrigues 함수."""
    ez = torch.tensor([0.0, 0.0, 1.0]).expand_as(u)
    k = torch.cross(ez, u, dim=-1); sn = k.norm(dim=-1, keepdim=True); cs = (ez * u).sum(-1, keepdim=True)
    k = k / sn.clamp(min=1e-9)
    def rot(v):
        v = v.expand_as(u) if v.dim() == 1 else v
        return v * cs + torch.cross(k, v, dim=-1) * sn + k * (k * v).sum(-1, keepdim=True) * (1.0 - cs)
    return rot


class Geo:
    def __init__(self, m):
        self.R, self.MZ, self.BZ, self.TZ = m.R, m.MZ, m.BZ, m.TABLE_Z
        z0 = m.TABLE_Z - m.BZ
        # f10 ep1800 재생 t=0 컵 원점(스폰 −12 cm) 평균 소스 (0.261, −0.160) · 리시버 (0.258, +0.163) — 대칭 판정을 위해 리시버는 소스의 y 거울로 둔다
        self.cup0 = {"src": torch.tensor([0.261, -0.1595, z0]), "rcv": torch.tensor([0.261, 0.1595, z0])}


def arm_traj(g, role, v):
    """한 팔의 (T,) 상태열. v = 변형 dict. 리시버 기준으로 만들고 소스는 y 반전."""
    R, MZ = g.R, g.MZ
    sy = -1.0 if role == "src" else 1.0
    Y = torch.tensor([1.0, sy, 1.0])
    t = torch.arange(T, dtype=torch.float64)
    cup0 = g.cup0[role]
    h_grip = 0.15 * MZ
    # ---- 손 방향: 시작(손가락 아래, 법선 +x) → 인벨롭(법선·손가락 수평, 손바닥→축 = 0.67 n + 0.73 f = +x)
    n_neu, f_neu = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, -1.0])
    n_env, f_env = torch.tensor([0.67, -0.742, 0.0]) * Y, torch.tensor([0.742, 0.67, 0.0]) * Y
    # ---- 손바닥 경로 (컵 원점 기준, 리시버 좌표 → Y)
    P0 = torch.tensor([0.049, 0.140, 0.151])            # 시작: 옆 바깥 14 cm·앞 4.9 cm·위 15.1 cm
    P1 = torch.tensor([0.049, 0.140, h_grip])            # 먼저 내림
    P2 = torch.tensor([-0.12, 0.0, h_grip])              # 컵 뒤 12 cm(로봇 쪽)
    P3 = torch.tensor([-1.5 * R, 0.0, h_grip])           # K7
    path = v.get("path", "low_first")
    if path == "low_first":
        s1, s2, s3 = _smooth(t, PH["A"], PH["B"]), _smooth(t, PH["B"], PH["C"]), _smooth(t, PH["C"], PH["D"])
        rel = _lerp(P0, P1, s1[:, None])
        # B: 컵 둘레 원호(잡는 높이)로 돈다 — 직선이면 엄지 끝이 컵 몸통(축 0.4 r)을 뚫고 지나간다(10-01 확인).
        a1, a2 = math.atan2(float(P1[1]), float(P1[0])), math.atan2(float(P2[1]), float(P2[0]))
        r1, r2 = float(P1[:2].norm()), float(P2[:2].norm())
        ang = a1 + (a2 - a1) * s2; rad_b = r1 + (r2 - r1) * s2
        arc = torch.stack([rad_b * torch.cos(ang), rad_b * torch.sin(ang), torch.full_like(ang, float(P1[2]))], -1)
        rel = torch.where((t >= PH["B"])[:, None], arc, rel)
        rel = torch.where((t >= PH["C"])[:, None], _lerp(P2, P3, s3[:, None]), rel)
        s_ax = _smooth(t, PH["A"], PH["B"])
    else:                                                # "top": 잡는 높이로 내리지 않고 위에서 K7 위로 간 뒤 내려 덮는다
        Pt = torch.tensor([-1.5 * R, 0.0, 0.151])
        s1, s2 = _smooth(t, PH["A"], PH["C"]), _smooth(t, PH["C"], PH["D"])
        rel = torch.where((t >= PH["C"])[:, None], _lerp(Pt, P3, s2[:, None]), _lerp(P0, Pt, s1[:, None]))
        s_ax = _smooth(t, PH["C"], PH["D"])
    rel = rel * Y
    nv = _unit(_lerp(n_neu, n_env, s_ax[:, None])); fv = _lerp(f_neu, f_env, s_ax[:, None])
    fv = _unit(fv - (fv * nv).sum(-1, keepdim=True) * nv)
    # ---- 손끝: 강체(엄지 = 법선 7 cm, 네 손가락 = 손가락 축 8 cm, 가로로 벌림) → D 에서 컵 벽 고리로
    cl_app = float(v.get("cl_app", 0.05))
    # 네 손가락 끝의 손바닥 기준 위치(손가락 축·법선 성분) = f10 ep1800 재생 실측(쥐기 전 스텝, 닫힘 구간별 평균):
    #   닫힘 <0.1: 10.0·0.3 cm · 0.2~0.3: 9.2·1.3 · 0.4~0.6: 7.1·2.3 · 0.7~0.9: 3.8·4.5 (엄지 끝은 손바닥에서 7.6~9.7 cm)
    CLS, ALONG, NORM = [0.05, 0.25, 0.5, 0.8], [0.100, 0.092, 0.071, 0.038], [0.003, 0.013, 0.023, 0.045]
    def _interp(x, xs, ys):
        for i in range(len(xs) - 1):
            if x <= xs[i + 1]:
                w = (x - xs[i]) / (xs[i + 1] - xs[i]); return ys[i] + w * (ys[i + 1] - ys[i])
        return ys[-1]
    a_f, a_n = _interp(max(cl_app, 0.05), CLS, ALONG), _interp(max(cl_app, 0.05), CLS, NORM)
    def hand_tips(palm_c, nv, fv):
        lat = sy * _unit(torch.cross(fv, nv, dim=-1))   # 외적은 거울에서 부호가 바뀐다 — 소스도 손가락 순서가 리시버의 거울이 되게
        # 10-01 f04 ep1400 오른손 편 손(닫힘 0.15) 실측 손바닥 좌표: 엄지 끝 (검지 쪽 가로 4.6, 손가락 축 4.0, 법선 7.5) cm,
        #   검지·새끼 가로 ±3.3 cm. 예전 엄지(법선 8.5 cm)는 컵 앞쪽이라 닫는 동안 엄지가 컵 둘레를 145° 돌았다.
        return torch.stack([palm_c - 0.046 * lat + 0.040 * fv + 0.075 * nv]
                           + [palm_c + a_f * fv + a_n * nv + d * lat for d in (-0.033, -0.011, 0.011, 0.033)], 1)
    palm_c = cup0 + rel                                  # 컵이 움직이기 전(A~D) 손바닥 월드 위치
    rigid = hand_tips(palm_c, nv, fv)
    grasp = v.get("grasp", "envelope")
    rr, hh = R + 0.0005, 0.25 * MZ
    def ring(a):
        return cup0 + torch.tensor([rr * math.cos(a), sy * rr * math.sin(a), hh])
    ring_t = torch.stack([ring(math.pi / 2)] + [ring(-math.pi / 2 + da) for da in (-0.26, -0.09, 0.09, 0.26)], 0)
    ff_full = torch.tensor([1.5, 1.5, 1.5, 1.5, 1.5])
    if grasp == "rim":                                   # f08·f09 실측: 입구 쪽 위에서 비스듬히(손바닥 축 2.3 r·입구 +0.75 높이, 손가락 축 아래 −0.6)
        P3r = torch.tensor([-2.3 * R, 0.0, 0.75 * MZ]) * Y
        f_r = _unit(torch.tensor([0.6, 0.0, -0.8]) + 0.0 * Y); n_r = _unit(torch.tensor([0.8, 0.0, 0.6]))
        wD = _smooth(t, PH["C"], PH["D"])[:, None]
        rel = torch.where((t >= PH["C"])[:, None], _lerp(P2 * Y, P3r, wD), rel)
        nv = torch.where((t >= PH["C"])[:, None], _unit(_lerp(nv, n_r.expand_as(nv), wD)), nv)
        fv = torch.where((t >= PH["C"])[:, None], _unit(_lerp(fv, f_r.expand_as(fv), wD)), fv)
        palm_c = cup0 + rel
        rigid = hand_tips(palm_c, nv, fv)
        hh = 0.8 * MZ
        ring_t = torch.stack([ring(math.pi / 2)] + [ring(-math.pi / 2 + da) for da in (-0.26, -0.09, 0.09, 0.26)], 0)
        ff_full = torch.tensor([1.5, 1.5, 1.5, 0.0, 0.0])
    if grasp == "tip2":
        ff_full = torch.tensor([1.5, 0.0, 1.5, 0.0, 0.0])
    sw = _smooth(t, PH["D"], PH["E"])[:, None]
    cl = torch.where(t >= PH["D"], _lerp(torch.tensor(cl_app), torch.tensor(0.8), sw[:, 0]), torch.full((T,), cl_app))
    if grasp == "envelope":
        # 10-01 f04: D 구간 네 손끝은 닫힘에 따라 실측 굽힘 곡선(f10 재생)으로 움직이고, 컵 벽(r + 0.5 mm)에 닿으면 거기서 멈춘다(펌웨어 힘 멈춤).
        #   예전처럼 닫힘과 무관하게 벽 고리로 곧장 옮기면 '닫기 전에는 벽에 닿을 수 없는' 실제 순서를 못 본다(f04 오른손 머묾을 놓쳤다).
        #   엄지는 강체 위치에서 벽 고리 엄지 자리로 옮긴다(엄지 굽힘·외전). 접촉한 손가락만 힘 1.5 N, 엄지 + 한 손가락 닿으면 쥠 판정.
        clc = cl.clamp(CLS[0], CLS[-1])
        def _tinterp(x, xs, ys):
            out = torch.full_like(x, ys[-1])
            for i in range(len(xs) - 2, -1, -1):
                w = ((x - xs[i]) / (xs[i + 1] - xs[i])).clamp(0.0, 1.0)
                out = torch.where(x <= xs[i + 1], ys[i] + w * (ys[i + 1] - ys[i]), out)
            return out
        af, an = _tinterp(clc, CLS, ALONG)[:, None], _tinterp(clc, CLS, NORM)[:, None]
        lat_v = sy * _unit(torch.cross(fv, nv, dim=-1))
        fing = torch.stack([palm_c + af * fv + an * nv + d * lat_v for d in (-0.033, -0.011, 0.011, 0.033)], 1)   # (T,4,3)
        rel_f = fing - cup0
        rad_f = rel_f[..., :2].norm(dim=-1, keepdim=True)
        touch_f = (rad_f < rr).squeeze(-1) & (t >= PH["D"])[:, None]
        push = torch.where(touch_f[..., None], rel_f[..., :2] / rad_f.clamp(min=1e-9) * rr, rel_f[..., :2])
        fing = torch.cat([cup0[:2] + push, fing[..., 2:]], -1)
        # 엄지: 컵 축 둘레 원통 좌표(방위·반경·높이)로 옮긴다 — 직선이면 컵 몸통을 뚫는다(축 0.24 r, 10-01 확인)
        r0 = rigid[:, 0] - cup0
        #   엄지 목표 = 다 말린(D 끝) 네 손끝 평균 방위의 반대쪽, 벽(r + 0.5 mm)·잡는 높이 — 고정 고리 자리는 손끝이 닿는 쪽과 어긋났다(cos 0.92)
        fe = fing[PH["E"] - 1].mean(0) - cup0
        a1 = torch.atan2(fe[1], fe[0]) + math.pi
        r1 = torch.stack([rr * torch.cos(a1), rr * torch.sin(a1), torch.tensor(hh)])
        a0 = torch.atan2(r0[:, 1], r0[:, 0])
        da = torch.remainder(a1 - a0 + math.pi, 2 * math.pi) - math.pi
        ang_t = a0 + da * sw[:, 0]
        rad_t = r0[:, :2].norm(dim=-1) + (float(r1[:2].norm()) - r0[:, :2].norm(dim=-1)) * sw[:, 0]
        z_t = r0[:, 2] + (r1[2] - r0[:, 2]) * sw[:, 0]
        thumb = cup0 + torch.stack([rad_t * torch.cos(ang_t), rad_t * torch.sin(ang_t), z_t], -1)
        dtip = torch.cat([thumb[:, None], fing], 1)
        tips = torch.where((t >= PH["D"])[:, None, None], dtip, rigid)
        th_on = _smooth(t, PH["D"] + 20, PH["E"])
        ff = torch.cat([1.5 * th_on[:, None], 1.5 * touch_f.to(th_on.dtype)], 1) * (t >= PH["D"]).to(th_on.dtype)[:, None]
        gr = (th_on > 0.5) & touch_f.any(-1)
    else:
        tips = torch.where((t >= PH["D"])[:, None, None], _lerp(rigid, ring_t.unsqueeze(0), sw[:, :, None]), rigid)
        ff = torch.where((t >= PH["D"])[:, None], ff_full * _smooth(t, PH["D"] + 20, PH["E"])[:, None], torch.zeros(T, 5))
        gr = t >= PH["D"] + 40
    # ---- 컵: E 들기 8 cm, F 소스 이송·더 들기, G 기울이기, H 세우기
    cup = cup0.repeat(T, 1)
    up = torch.tensor([0.0, 0.0, 1.0]).repeat(T, 1)
    lift = 0.08 * _smooth(t, PH["E"], PH["F"])
    cup[:, 2] = cup[:, 2] + lift
    if role == "src":
        rc = g.cup0["rcv"] + torch.tensor([0.0, 0.0, 0.08])
        tgt = rc + torch.tensor([0.0, -0.052, MZ + 3.0 * R])          # 리시버 입구 옆(소스 쪽) 5.2 cm·위 3 r
        sF = _smooth(t, PH["F"], PH["G"])[:, None]
        cup = torch.where((t >= PH["F"])[:, None], _lerp(cup0 + torch.tensor([0.0, 0.0, 0.08]), tgt, sF), cup)
        th = math.radians(105.0) * (_smooth(t, PH["G"], PH["G"] + 90) - _smooth(t, PH["H"], PH["H"] + 60))
        if v.get("pour_hold"):                           # 16 알(0.8 = 성공 문턱)만 붓고 80° 로 되돌려 남은 비드를 안은 채 유지
            t95 = int(torch.nonzero(th >= math.radians(95.0))[0])
            th = torch.where(t >= t95 + 32, torch.full((T,), math.radians(80.0)), th)
        up = torch.stack([torch.zeros(T), torch.sin(th), torch.cos(th)], -1)   # 리시버 쪽(+y)으로 기움
    # ---- 쥔 뒤(E~)는 손이 컵과 함께 움직이고 돈다: D 끝 시점 상대 자세를 컵 좌표로 옮긴다
    iD = PH["E"] - 1
    rot = _rot_ez_to(up)
    after = (t >= PH["E"])
    palm = cup0 + rel
    palm = torch.where(after[:, None], cup + rot((palm[iD] - cup0).expand(T, 3)), palm)
    tips = torch.where(after[:, None, None], cup[:, None, :] + torch.stack([rot((tips[iD, k] - cup0).expand(T, 3)) for k in range(5)], 1), tips)
    nv = torch.where(after[:, None], rot(nv[iD].expand(T, 3)), nv); fv = torch.where(after[:, None], rot(fv[iD].expand(T, 3)), fv)
    # ---- 속도(차분) · 팔 관절 속도(움직이는 구간 0.15 rad/s, f10 재생 평균 0.14~0.16)
    def vel(x):
        d = torch.zeros_like(x); d[1:] = (x[1:] - x[:-1]) / DT; return d
    palm_v, cup_v = vel(palm), vel(cup)
    moving = palm_v.norm(dim=-1) > 1e-4
    qd = 0.15 * moving.to(torch.float64)[:, None].repeat(1, 7)
    st = {"cup": cup, "up": up, "palm": palm, "axes": torch.cat([nv, fv], -1), "tips": tips, "ff": ff, "cl": cl,
          "gr": gr, "palm_v": palm_v, "cup_v": cup_v, "qd": qd, "spawn": cup0.repeat(T, 1)}
    # ---- 멈춤 변형: t_stop 이후 그 시각 상태 그대로(속도 0)
    ts = v.get("stop")
    if ts is not None:
        for k in ("cup", "up", "palm", "axes", "tips", "ff", "cl", "gr"):
            st[k] = torch.where((t >= ts).view(-1, *([1] * (st[k].dim() - 1))), st[k][ts].expand_as(st[k]), st[k])
        for k in ("palm_v", "cup_v", "qd"):
            st[k] = torch.where((t >= ts).view(-1, *([1] * (st[k].dim() - 1))), torch.zeros_like(st[k]), st[k])
        if "stop_cl" in v:
            st["cl"] = torch.where(t >= ts, torch.full((T,), float(v["stop_cl"])), st["cl"])
    return st


def beads(src):
    """소스 기울기에 따른 비드 장부: 95° 부터 스텝당 0.5 알(기대값이라 연속), 20 알을 다 옮길 때까지.
    기울인 채 멈춰도 비드는 계속 나온다(물리) — 멈춤 변형도 같은 식을 쓴다."""
    th = torch.atan2(src["up"][:, 1], src["up"][:, 2])
    out = th >= math.radians(95.0)
    cnt = torch.cumsum(out.to(torch.float64), 0) * 0.5
    n_in = cnt.clamp(max=20.0)
    in_t = n_in / 20.0
    d_in = torch.zeros(T); d_in[1:] = in_t[1:] - in_t[:-1]
    return in_t, d_in


def build_ctx(m, s, r, in_t, d_in):
    c = m.make_ctx(s["cup"].clone(), s["up"].clone(), r["cup"].clone(), r["up"].clone(), prog=0.5, in_t=in_t, d_in=d_in)
    for side, st in (("src", s), ("rcv", r)):
        setattr(c, f"{side}_cup_spawn_pos", st["spawn"]); setattr(c, f"{side}_cup_lin_vel", st["cup_v"])
        setattr(c, f"{side}_palm_pos", st["palm"]); setattr(c, f"{side}_palm_axes", st["axes"])
        setattr(c, f"{side}_tips_pos", st["tips"]); setattr(c, f"{side}_finger_force", st["ff"])
        setattr(c, f"{side}_hand_closure", st["cl"]); setattr(c, f"{side}_grasped", st["gr"])
        setattr(c, f"{side}_palm_lin_vel", st["palm_v"]); setattr(c, f"{side}_arm_qd", st["qd"])
    c.src_cup_mouth_pos = s["cup"] + m.MZ * s["up"]; c.rcv_cup_mouth_pos = r["cup"] + m.MZ * r["up"]
    c.src_cup_tilt = torch.acos(s["up"][:, 2].clamp(-1, 1)); c.rcv_cup_tilt = torch.acos(r["up"][:, 2].clamp(-1, 1))
    c.success = (c.bead_in_target_frac >= 0.80) & (c.bead_spill_frac <= 0.25) & (c.rcv_cup_tilt <= math.radians(20.0)) \
        & s["gr"] & r["gr"]
    c.actions, c.prev_actions = torch.zeros(T, NUM_ACTIONS), torch.zeros(T, NUM_ACTIONS)
    return c


def episode(m, g, v_src=None, v_rcv=None):
    s = arm_traj(g, "src", v_src or {}); r = arm_traj(g, "rcv", v_rcv or {})
    in_t, d_in = beads(s)
    m.reset_cache()
    rew, comps = m.NS["compute_reward"](build_ctx(m, s, r, in_t, d_in))
    return rew, comps


def value(rew, t0):
    """t0 부터 할인 가치, 900 뒤는 마지막 스텝 보상 유지(value bootstrap)."""
    k = torch.arange(T - t0, dtype=torch.float64)
    return float((GAMMA ** k * rew[t0:]).sum() + GAMMA ** (T - t0) * rew[-1] / (1.0 - GAMMA))


def stay_value(m, g, t0):
    """t0 에서 두 팔 모두 멈춤(자세 고정·속도 0). 기울인 채 멈추면 비드는 물리대로 계속 나온다."""
    rew, _ = episode(m, g, {"stop": t0}, {"stop": t0})
    return value(rew, t0)


def run(cand):
    m = load_check(cand)
    g = Geo(m)
    ok = True
    ref, comps = episode(m, g)
    # [T1] 연속
    # 배달 증분(pour_delta)은 비드가 들어가는 스텝에만 붙는 사건 수입이라 다 옮기면 끝난다 — 연속 판정에서 뺀다
    ref_c = ref - comps["pour_delta"] if "pour_delta" in comps else ref
    dr = ref_c[1:] - ref_c[:-1]
    i_min = int(torch.argmin(dr))
    print(f"[T1] 참 궤적 스텝당 보상: " + " · ".join(f"{k}@{PH[k]} {float(ref[PH[k]:PH[k] + 10].mean()):+.2f}" for k in PH)
          + f" · 끝 {float(ref[-1]):+.2f}")
    print(f"     한 스텝 최대 하락 {float(dr[i_min]):+.3f} (t={i_min}→{i_min + 1}, 단계 "
          f"{max(k for k in PH if PH[k] <= i_min + 1)})")
    if float(dr[i_min]) < -0.05:
        worst = sorted(((float(comps[k][i_min + 1] - comps[k][i_min]), k) for k in comps))[:3]
        print("    ✗ 참 궤적에 절벽: " + ", ".join(f"{k} {d:+.3f}" for d, k in worst)); ok = False
    # [T2] 전진 최적
    fails = []
    for t0 in list(range(0, T, 30)) + [PH["D"] + 30, PH["E"] - 1, PH["G"] + 60]:
        vr, vs = value(ref, t0), stay_value(m, g, t0)
        need = 1.0 if t0 < PH["H"] else -1e-3             # H(다 붓고 세운 뒤)는 참 궤적도 멈춰 있어 같으면 통과
        if vr < vs + need:
            fails.append((t0, vr, vs))
    rows = [(t0, value(ref, t0), stay_value(m, g, t0)) for t0 in (0, 100, 200, 300, 380, 450, 560, 700, 800)]
    print("[T2] 계속 − 멈춤 할인 가치: " + " · ".join(f"t{t0} {vr - vs:+.1f}" for t0, vr, vs in rows))
    if fails:
        print("    ✗ 멈추는 편이 나은 시각: " + ", ".join(f"t{t0}({vr - vs:+.1f})" for t0, vr, vs in fails[:8])); ok = False
    # [T3] 이탈(한 팔씩)
    devs = [("반쯤 쥔 손 접근(0.5)", {"cl_app": 0.5}, 0),
            ("위에서 덮는 접근", {"path": "top"}, 0),
            ("12 cm 편 손 주차", {"stop": PH["C"]}, PH["C"]),
            ("10 cm 반쯤 쥔 손 주차(f00)", {"stop": PH["C"] + 20, "stop_cl": 0.5}, PH["C"] + 20),
            ("쥐고 테이블 유지", {"stop": PH["E"]}, PH["E"]),
            ("입구 쪽 비스듬한 파지", {"grasp": "rim"}, PH["C"]),
            ("엄지·중지 두 손가락 파지", {"grasp": "tip2"}, PH["D"])]
    print("[T3] 이탈(분기 시점 할인 가치 차, 참 − 이탈):")
    for arm in ("src", "rcv"):
        cells = []
        extra = [("들고 유지", {"stop": PH["F"]}, PH["F"]), ("이송 뒤 안 붓기", {"stop": PH["G"]}, PH["G"]),
                 ("80 % 붓고 80° 유지", {"pour_hold": True}, PH["G"] + 90)] if arm == "src" else []
        for name, v, tb in devs + extra:
            dv, _ = episode(m, g, v if arm == "src" else None, v if arm == "rcv" else None)
            gap = value(ref, tb) - value(dv, tb)
            cells.append(f"{name} {gap:+.1f}")
            if gap < 2.0:
                ok = False; cells[-1] += " ✗"
        print(f"     {arm}: " + " · ".join(cells))
    # [T4] 편 손
    def _out_site(arm):
        st = arm_traj(g, arm, {}); cp_ = g.cup0[arm][:2]
        return ((st["palm"][:PH["D"], :2] - cp_).norm(dim=-1) > 2.6 * m.R)
    for arm in ("src", "rcv"):
        dv, dc = episode(m, g, {"cl_app": 0.5} if arm == "src" else None, {"cl_app": 0.5} if arm == "rcv" else None)
        d = (ref - dv)[:PH["D"]]
        #   10-01 f04·iter_05: 판정은 잡는 자리 밖(참 궤적 손바닥이 컵 축에서 2.6 r 밖)에서만 — 자리 안에서 닫기는 D 단계(감싸기)의 시작이라
        #   [64](감싸기 직전에서 닫으면 이득)와 같은 자세를 반대로 요구하게 된다. 막으려던 것은 먼 곳에서 미리 쥐기(f00 10 cm·f03 9~10 cm).
        out_site = _out_site(arm)
        d = torch.where(out_site, d, torch.full_like(d, 1e9))
        per = {k: float(d[PH[k]:PH[nk]][d[PH[k]:PH[nk]] < 1e8].mean()) for k, nk in (("A", "B"), ("B", "C"), ("C", "D"))}
        print(f"[T4] {arm} 편 손(0.05) − 반쯤 쥔 손(0.5), 접근 구간 스텝당: 최소 {float(d.min()):+.3f} · "
              + " · ".join(f"{k} {x:+.3f}" for k, x in per.items()))
        i_w = int(torch.argmin(d))
        sd = arm_traj(g, arm, {"cl_app": 0.5}); cp = g.cup0[arm][:2]
        pr_w = float((sd["palm"][i_w, :2] - cp).norm()) / m.R
        tr_w = [round(float((sd["tips"][i_w, k, :2] - cp).norm()) / m.R, 2) for k in range(5)]
        worst_c = sorted(((float(comps[k][i_w] - dc[k][i_w]), k) for k in comps))[:3]
        print(f"     최악 t={i_w}: 손바닥 축 {pr_w:.2f} r · 반쯤 쥔 손 손끝 축 거리(엄지·검지·중지·약지·새끼) {tr_w} r · "
              f"편 손 − 반쯤 쥔 손 성분 " + ", ".join(f"{k} {x:+.3f}" for x, k in worst_c))
        if float(d.min()) < 0.15:
            print(f"    ✗ {arm}: 접근 중 손을 펴는 이득이 스텝당 0.15 미만인 구간이 있다(f00 반쯤 쥔 손 접근)"); ok = False
    # [T4b] 10-01 f03: 편 손 배수가 닫힘 0.5 위에서 평평하면 0.7 로 쥔 손이 펴는 기울기가 0 — 접근 구간(A~C) 매 스텝
    #   닫힘 0.05 > 0.3 > 0.5 > 0.7 이 칸마다 팔당 +0.05 이상 (손끝 위치는 f10 재생 실측 보간)
    for arm in ("src", "rcv"):
        vals, cps = {}, {}
        for clv in (0.05, 0.3, 0.5, 0.7):
            dv, dcp = episode(m, g, {"cl_app": clv} if arm == "src" else None, {"cl_app": clv} if arm == "rcv" else None)
            vals[clv] = dv[:PH["D"]]; cps[clv] = dcp
        steps = [(0.05, 0.3), (0.3, 0.5), (0.5, 0.7)]
        out_site = _out_site(arm)
        for k_ in list(vals):
            vals[k_] = torch.where(out_site, vals[k_], torch.full_like(vals[k_], 1e9 * (1.0 - 0.1 * list(vals).index(k_))))
        mins = {f"{a}>{b}": float((vals[a] - vals[b])[out_site].min()) for a, b in steps}
        print(f"[T4b] {arm} 접근 구간 닫힘 칸별 최소 차(팔당 스텝당): " + " · ".join(f"{k} {v:+.3f}" for k, v in mins.items()))
        a_, b_ = min(steps, key=lambda ab: float((vals[ab[0]] - vals[ab[1]]).min()))
        iw = int(torch.argmin(torch.where(out_site, vals[a_] - vals[b_], torch.full_like(vals[a_], 1e9))))
        sd = arm_traj(g, arm, {"cl_app": b_}); cp = g.cup0[arm][:2]
        wc = sorted(((float(cps[a_][k][iw] - cps[b_][k][iw]), k) for k in cps[a_]))[:3]
        print(f"     최악 칸 {a_}>{b_} t={iw}(단계 {max(k for k in PH if PH[k] <= iw)}): 손바닥 축 {float((sd['palm'][iw, :2] - cp).norm()) / m.R:.2f} r · "
              f"닫힘 {b_} 손끝 축 거리 {[round(float((sd['tips'][iw, k, :2] - cp).norm()) / m.R, 2) for k in range(5)]} r · 성분 차 " + ", ".join(f"{k} {x:+.3f}" for x, k in wc))
        if min(mins.values()) < 0.05:
            print(f"    ✗ {arm}: 접근 중 반쯤 이상 쥔 손을 조금 펴는 기울기가 +0.05 미만인 칸이 있다(f03 닫힘 0.7 머묾)"); ok = False
    # [T5] 대칭
    worst = 0.0; wk = ""
    for k in ("approach", "grasp"):          # 들기 목표는 역할별(소스는 리시버 입구를 넘을 높이)이라 제외
        ks, kr = f"{k}_src", f"{k}_rcv"
        if ks in comps and kr in comps:
            dd = float((comps[ks] - comps[kr])[:PH["F"]].abs().max())
            if dd > worst:
                worst, wk = dd, k
    print(f"[T5] 들기까지 팔별 성분 최대 차 {worst:.3f} ({wk})")
    if worst > 0.02:
        print("    ✗ 두 손이 같은 궤적에서 다른 보상을 받는다"); ok = False
    # 구간별 성분 평균(참고)
    keys = [k for k in comps if float(comps[k].abs().max()) > 1e-6]
    print("     구간 성분 평균: " + "  ".join(k for k in keys))
    for a, b in zip(list(PH), list(PH)[1:] + ["end"]):
        lo, hi = PH[a], (PH[b] if b != "end" else T)
        print(f"     {a:>2} " + " ".join(f"{float(comps[k][lo:hi].mean()):+.2f}" for k in keys))
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    res = [run(c) for c in sys.argv[1:]]
    sys.exit(0 if all(res) else 1)
