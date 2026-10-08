"""pour_fabric 순수 규칙(torch 만, isaaclab 무관) — 09.16 비드 부피 DR · 조준 전 틸트 래치.

env 는 여기 함수를 부르기만 한다. isaaclab 이 없는 로컬에서 테스트되는 유일한 층(`tests/test_pour_rules.py`).

배경(사용자 결정 09.16 "권장 방식으로 진행"):
  - 컵 내부 177 mm × r 41 mm 에 12 mm 비드 20개는 3 % 채움이라 "접근 전 틸트 = 흘림" 이 물리로 안 생긴다.
    비드를 30 mm 로 키우고(스폰 개수 고정) 에피소드마다 활성 개수를 뽑아 부피를 DR 한다. 비활성 비드는
    env 마다 테이블 뒤 지면 위 격자에 눕힌다(한 점에 모으면 브로드페이즈 페어 폭발 — repfalse 이력).
  - 채움 정도 0~1 은 실측(정착 후 활성 비드 평균 높이)으로 정한다 — 개수·크기와 무관한 부피량이라
    실기에서는 사람이 어림잡아 넣는 명령 입력과 뜻이 같다.
  - 조준 전 틸트: 붓는 쪽 림 점이 리시버 입구에서 xy 로 멀리 있는데 소스가 한계 각도를 넘으면 성공 무효
    (i07 계측: 입구 거리 0.236 m 에서 30° 돌파, 실제 붓기 중 입구 거리 중앙값 0.021 m).
    ★09.18 한계 각도는 고정 30° 가 아니라 채움별 유출각 − 여유, 거리는 입구 중심이 아니라 붓는 쪽 림 점 기준.
"""
from __future__ import annotations

import math

import torch

_LAYER_TWIST_RAD = 0.35      # 층마다 비틀어 수직 정렬을 깬다(bead_assets 검증 배치와 같은 값)
_WALL_MARGIN_M = 0.003       # 링 반지름 = 안쪽 반지름 − 비드 반지름 − 여유
_SPAWN_GAP_M = 0.002         # 이웃 비드 사이 여유(소환 겹침 → 벽 관통 사고 08.17 재발 방지)
_PARK_GAP_M = 0.010          # 파킹 격자 간격 여유


def premature_tilt_limit_rad(fill: torch.Tensor, *, release_full_deg: float, release_span_deg: float,
                             margin_deg: float) -> torch.Tensor:
    """조준 없이 허용되는 소스 기울기 상한 [rad] (N,) = 첫 비드 유출 각도(채움 의존) − 여유.

    ★09.18 사용자 지적 "비드 양에 따라 기울이는 각도가 다르고 틸팅을 시도할 거리도 다르다": 고정 30° 를 버린다.
    프로브 실측: 가득 찬 컵은 약 70° 에서, 몇 알 든 컵은 약 90° 에서 첫 비드가 나간다 →
    release = full + span·(1 − fill). 유출각 − margin 까지는 어디서든(접근 중에도) 미리 기울여도 된다.
    """
    release = math.radians(float(release_full_deg)) + math.radians(float(release_span_deg)) * (
        1.0 - fill.clamp(0.0, 1.0))
    return release - math.radians(float(margin_deg))


def pour_dir_update(prev_dir: torch.Tensor, src_cup_xy: torch.Tensor, rcv_mouth_xy: torch.Tensor, *,
                    min_sep_m: float) -> torch.Tensor:
    """붓는 방향 d̂ (N,2) = 소스 컵 원점 → 리시버 입구 수평 단위벡터.

    컵 자세와 무관해서 직립(기울기 0)에서도 정의되고, 컵을 어느 쪽으로 기울였든 같은 쪽을 가리킨다
    (실제 림 최저점은 직립 근처에서 방향이 튀고, 반대로 기울이면 리시버 반대편을 가리킨다 — 사용자 지적 09.18).
    수평거리가 min_sep_m 보다 짧으면 방향이 흔들리므로 직전 값을 유지한다.
    """
    v = rcv_mouth_xy - src_cup_xy
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n >= float(min_sep_m), v / n.clamp(min=1e-6), prev_dir)


def pour_lip(src_mouth: torch.Tensor, src_up: torch.Tensor, pour_dir: torch.Tensor, *,
             rim_radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """붓는 쪽 림 점 (N,3) 과 리시버 쪽 부호 있는 기울기 θ [rad] (N,).

    θ = atan2(up·d̂, up_z) — (d̂, z) 수직면 안의 기울기만 본다(옆으로 기운 성분은 0, 반대로 기울면 음수).
    림 점 = 입구 중심 + r·(cos θ·d̂ − sin θ·ẑ): 직립이면 리시버를 향한 쪽 림, 기울일수록 아래로 내려가고
    90° 를 넘으면 컵 밑으로 말려 들어간다(실제 유출 지점과 같은 거동). 입구 중심 자체가 컵 높이·sin θ 만큼
    d̂ 쪽으로 나가므로, 많이 기울여야 하는(비드 적은) 컵일수록 컵 몸통이 더 멀리서 조준이 성립한다.
    """
    u_d = (src_up[:, :2] * pour_dir).sum(dim=-1)
    theta = torch.atan2(u_d, src_up[:, 2])
    r = float(rim_radius)
    lip = src_mouth.clone()
    lip[:, :2] = lip[:, :2] + (r * torch.cos(theta)).unsqueeze(-1) * pour_dir
    lip[:, 2] = lip[:, 2] - r * torch.sin(theta)
    return lip, theta


def premature_tilt_now(src_tilt: torch.Tensor, tilt_limit: torch.Tensor, lip_dist_xy: torch.Tensor,
                       grasped: torch.Tensor, *, lip_max_m: float) -> torch.Tensor:
    """이 스텝의 순간 판정 (N,) bool — 래치(에피소드 누적)는 env 가 한다.

    **잡은** 소스 컵의 전체 기울기(방향 무관)가 채움별 상한(`premature_tilt_limit_rad`)을 넘었는데
    붓는 쪽 림 점(`pour_lip`)이 리시버 입구 중심에서 xy 로 lip_max_m 보다 멀면 조준 전 틸트.
    방향 무관이라 리시버 반대쪽·옆으로 크게 기울여도 걸린다.

    ★09.17 i08: 파지 조건 없이 걸었더니 첫 epoch 래치 19.8 % 가 전부 파지 0 에서 나왔다(탐색이 컵을 쳐서 넘어뜨림).
    래치는 영구라 정책이 소스 컵을 아예 피했다(파지 0.000, 이물 접촉 15→2 %). 리시버 규칙(파지·들기 이후에만)과
    같은 원칙으로 **잡은 채** 기울인 경우만 잡는다.
    """
    return (src_tilt > tilt_limit) & (lip_dist_xy > float(lip_max_m)) & grasped.to(torch.bool)


def pour_dir_ok(pour_dir_xy: torch.Tensor, *, max_outward: float) -> torch.Tensor:
    """붓는 방향 d̂(소스 컵 원점→리시버 입구, 수평 단위벡터)의 몸 바깥(+x) 성분이 허용치 이하인가 (N,) bool.

    ★09.20 사용자 영상 지적 "컵 입구가 몸통 쪽을 향해야 하는데 바깥을 향한다": i16 은 기울인 구간 d̂ 평균 (0.90, 0.35),
    전 스텝이 +x 였다(리시버를 소스보다 10 cm 앞에 두고 앞으로 부음). 로봇 몸통은 −x 쪽이다.
    """
    return pour_dir_xy[:, 0] <= float(max_outward)


def rcv_side_ok(rcv_cup_y: torch.Tensor, *, side_sign: float, min_side_m: float) -> torch.Tensor:
    """리시버 컵이 자기 쪽(side_sign·y ≥ min_side_m)에 있는가 (N,) bool — 중심선을 넘어 상대 팔 쪽으로 가면 False.

    ★09.20 사용자 영상 지적 "왼팔이 오른팔 쪽으로 과하게 넘어감": i16 붓는 동안 리시버 y 중앙값 −0.13(스폰 +0.16).
    """
    return float(side_sign) * rcv_cup_y >= float(min_side_m)


def wrap_count(f_mid: torch.Tensor, f_dist: torch.Tensor, *, thr: float) -> torch.Tensor:
    """중간 또는 원위 마디가 컵에 닿은 손가락 수 (N,) float — 팁만 닿은 손가락은 세지 않는다(손끝 집기 ≠ 감싸 쥐기).

    ★09.20 사용자 영상 지적 "인벨롭 그립이 아님": i16 리시버 손은 엄지 1개만 마디 접촉, 4지는 팁만(1~4 N).
    """
    return ((f_mid > float(thr)) | (f_dist > float(thr))).float().sum(dim=1)


def cup_hit_now(cup_cup_force: torch.Tensor, hold: torch.Tensor, *, thr: float) -> torch.Tensor:
    """이 스텝에 두 컵이 부딪혔는가 (N,) bool — hold 구간 제외. 래치(에피소드 누적)는 env 가 한다.

    ★09.20 사용자 영상 지적 "파지하고 나서 컵을 서로 부딪힘": i16 은 step ~98 에 두 컵 원점 거리 0.082~0.099 m
    (64 env 전부)까지 테이블 위에서 밀어 붙인 뒤 들었다.
    """
    return (cup_cup_force > float(thr)) & (~hold.to(torch.bool))


def fill_level_from_local_z(z_local: torch.Tensor, active: torch.Tensor, *,
                            bottom_z: float, top_z: float) -> torch.Tensor:
    """활성 비드 평균 높이의 2배를 컵 내부 높이로 나눈 [0,1] (N,).

    균일 기둥이면 평균 높이 = 바닥 + h/2 라 정확하고, 한 알이 튀어 올라도 최고점 방식처럼 흔들리지 않는다.
    활성이 0개면 0.
    """
    a = active.to(z_local.dtype)
    n = a.sum(dim=1)
    mean_z = (z_local * a).sum(dim=1) / n.clamp(min=1.0)
    h = 2.0 * (mean_z - float(bottom_z))
    level = (h / (float(top_z) - float(bottom_z))).clamp(0.0, 1.0)
    return torch.where(n > 0, level, torch.zeros_like(level))


def sample_active_mask(n_envs: int, *, k: int, lo: int, hi: int, device) -> torch.Tensor:
    """에피소드별 활성 비드 수 n ~ U{lo..hi} 를 뽑아 앞쪽 n 개만 True 인 (N,k) bool.

    배치가 아래층부터라 앞쪽 = 낮은 층 → 활성 비드가 컵 바닥에 모인다.
    """
    lo, hi, k = int(lo), int(hi), int(k)
    if not 1 <= lo <= hi <= k:
        raise ValueError(f"활성 범위는 1 ≤ lo ≤ hi ≤ k 여야 한다: lo={lo} hi={hi} k={k}")
    n_active = torch.randint(lo, hi + 1, (int(n_envs),), device=device)
    return torch.arange(k, device=device).unsqueeze(0) < n_active.unsqueeze(1)


def inner_radius_at(z: float, profile: tuple | None, fallback: float) -> float:
    """높이 z 에서의 컵 안쪽 반지름 [m]. profile 이 없으면 fallback(림 반지름)을 쓴다.

    profile 은 (z, r) 오름차순 쌍들 — 컵 .usd 메시 실측(09.23). 범위 밖은 양 끝 값으로 고정한다.
    """
    if not profile:
        return float(fallback)
    zs = [float(a) for a, _ in profile]
    rs = [float(b) for _, b in profile]
    if z <= zs[0]:
        return rs[0]
    if z >= zs[-1]:
        return rs[-1]
    for i in range(1, len(zs)):
        if z <= zs[i]:
            t = (z - zs[i - 1]) / (zs[i] - zs[i - 1])
            return rs[i - 1] + t * (rs[i] - rs[i - 1])
    return rs[-1]


def bead_layout_in_cup(n: int, *, diameter: float, inner_radius: float, bottom_z: float,
                       inner_profile: tuple | None = None) -> list[list[float]]:
    """컵 원점 기준 소환 offset (n,3) — 지름에서 층당 개수·링 반지름·층 간격을 유도한다.

    bead_assets.bead_offsets_in_cup(12 mm 20개 검증 배치)의 일반화. 이웃 중심 거리 ≥ 지름 + 여유를
    보장하고 바닥부터 쌓는다(활성 prefix 규약). 소환 기둥이 림 위로 솟는 것은 허용 — 정착 대기 동안 떨어진다.

    ★09.23 실측: 이 컵은 테이퍼라 안쪽 반지름이 바닥 26.7 mm → 림 42.6 mm 다. 림 반지름 하나로
    링을 잡으면 **바닥 층 비드가 벽 밖에서 소환**되고(지름이 작을수록 링이 커져 중심이 벽을 넘는다)
    정착 중에 컵 밖으로 튕겨 나간다 — 손을 대지 않은 정착만으로 24 mm 0.805 / 27 mm 0.938 / 30 mm 0.998.
    `inner_profile` 을 주면 층 높이마다의 안쪽 반지름으로 링과 층당 개수를 다시 잡는다.
    """
    d = float(diameter)
    dz = d + _SPAWN_GAP_M
    z0 = float(bottom_z) + d / 2.0 + _SPAWN_GAP_M
    out: list[list[float]] = []
    layer = 0
    while len(out) < int(n):
        z = z0 + dz * layer
        r_in = inner_radius_at(z, inner_profile, inner_radius)
        r_ring = r_in - d / 2.0 - _WALL_MARGIN_M
        ratio = (d + _SPAWN_GAP_M) / (2.0 * r_ring) if r_ring > 0.0 else 1.0
        per_layer = 1 if ratio >= 1.0 else max(1, int(math.floor(math.pi / math.asin(ratio))))
        radius = r_ring if per_layer > 1 else 0.0
        for slot in range(per_layer):
            if len(out) >= int(n):
                break
            angle = 2.0 * math.pi * slot / per_layer + _LAYER_TWIST_RAD * layer
            out.append([radius * math.cos(angle), radius * math.sin(angle), z])
        layer += 1
    return out


def park_offsets(k: int, *, diameter: float, origin_xy: tuple[float, float], z: float,
                 per_row: int) -> list[list[float]]:
    """비활성 비드 파킹 격자 (k,3), env-local. 행은 −x 로 물러나며 늘어난다(로봇·테이블은 +x 쪽)."""
    step = float(diameter) + _PARK_GAP_M
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    out: list[list[float]] = []
    for i in range(int(k)):
        row, col = divmod(i, int(per_row))
        out.append([ox - step * row, oy + step * col, float(z)])
    return out


def slew_limit(prev: torch.Tensor, target: torch.Tensor, max_step) -> torch.Tensor:
    """스텝당 변화량 상한(정규화 액션 단위). max_step 은 스칼라 또는 축별 텐서(마지막 차원).
    스칼라 max_step <= 0 이면 끔. 새 텐서를 돌려준다."""
    if not torch.is_tensor(max_step):
        if float(max_step) <= 0.0:
            return target
        return prev + (target - prev).clamp(-float(max_step), float(max_step))
    step = max_step.to(target)
    return prev + torch.maximum(torch.minimum(target - prev, step), -step)


# ---------------------------------------------------------------- ★09.28 fabric 지령 성형(물리 단위)
def palm_vmax_vec(vel_pos_m_s: float, vel_rot_deg_s: float) -> torch.Tensor:
    """palm 6D 최대 속도 벡터 (m/s ×3, rad/s ×3)."""
    return torch.tensor([float(vel_pos_m_s)] * 3 + [math.radians(float(vel_rot_deg_s))] * 3)


def step_palm_increment(target: torch.Tensor, v_norm: torch.Tensor, vmax: torch.Tensor, dt: float,
                        anchor: torch.Tensor, delta_lo: torch.Tensor, delta_hi: torch.Tensor) -> torch.Tensor:
    """★09.28 증분 지령 한 스텝: 목표 += v_norm·vmax·dt, 앵커 기준 델타 박스 [lo, hi] 로 자른다. 새 텐서.

    v_norm 은 [−1, 1] (거른 액션), vmax 는 (6,) 물리 최대 속도, 회전 델타는 wrap 뒤 자른다.
    """
    raw = target + v_norm.clamp(-1.0, 1.0) * vmax.to(target) * float(dt)
    d = raw - anchor
    d_rot = torch.remainder(d[:, 3:] + math.pi, 2.0 * math.pi) - math.pi
    d = torch.cat([d[:, :3], d_rot], dim=1)
    d = torch.maximum(torch.minimum(d, delta_hi.to(d)), delta_lo.to(d))
    return anchor + d


def leash_palm_target(target: torch.Tensor, pose: torch.Tensor, max_pos_m: float,
                      max_rot_deg: float) -> torch.Tensor:
    """palm 6D 목표를 **현재 실제 palm 자세** 근처로 묶는다(목표−실제 간극 상한). 새 텐서.

    위치는 간극 벡터 길이를 max_pos_m 로, 회전은 euler 슬롯별(wrap 후) 차이를 max_rot_deg 로 자른다.
    둘 중 하나가 0 이하면 그 부분은 끈다. target·pose 는 같은 프레임(N,6), 회전은 rad.
    """
    out = target.clone()
    if max_pos_m > 0.0:
        d = target[:, :3] - pose[:, :3]
        n = d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        out[:, :3] = pose[:, :3] + d * torch.clamp(float(max_pos_m) / n, max=1.0)
    if max_rot_deg > 0.0:
        r = torch.remainder(target[:, 3:] - pose[:, 3:] + math.pi, 2.0 * math.pi) - math.pi
        lim = math.radians(float(max_rot_deg))
        out[:, 3:] = pose[:, 3:] + r.clamp(-lim, lim)
    return out


# ---------------------------------------------------------------- ★09.21 학습 중 진단 지표
FILL_BIN_EDGES = (1.0 / 3.0, 2.0 / 3.0)


def action_flip_rate(delta: torch.Tensor, prev_delta: torch.Tensor, *, eps: float) -> torch.Tensor:
    """연속 두 스텝의 액션 변화가 부호를 뒤집은 차원의 비율 (N,) — 떨림 지표. |변화| < eps 인 차원은 세지 않는다.

    ★09.21 사용자 영상 지적 "팔 떨림": i18 trace 에서 소스 회전 3축이 29~49 % 스텝에서 부호 교대.
    """
    real = (delta.abs() > float(eps)) & (prev_delta.abs() > float(eps))
    return (real & (delta * prev_delta < 0.0)).float().mean(dim=1)


def remaining_release_rad(fill: torch.Tensor, in_source_frac: torch.Tensor, *, release_full_deg: float,
                          release_span_deg: float) -> torch.Tensor:
    """지금 컵에 남은 양 기준 첫 유출각 [rad] (N,) — premature_tilt_limit_rad 와 같은 직선, 여유 없음."""
    left = (fill.clamp(0.0, 1.0) * in_source_frac.clamp(0.0, 1.0))
    return math.radians(float(release_full_deg)) + math.radians(float(release_span_deg)) * (1.0 - left)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 가 참인 env 의 평균(스칼라). 해당 env 가 없으면 0 — 같이 기록하는 비율 지표로 구분한다."""
    m = mask.float()
    return (x * m).sum() / m.sum().clamp(min=1.0)


def fill_bins(fill: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """채움 정도 3구간(낮음 < 1/3 ≤ 중간 < 2/3 ≤ 높음) 마스크."""
    lo_edge, hi_edge = FILL_BIN_EDGES
    return fill < lo_edge, (fill >= lo_edge) & (fill < hi_edge), fill >= hi_edge


# ---------------------------------------------------------------- ★09.22 파지 기준(들어 올린 순간)
def force_share(palm_force: torch.Tensor, finger_force: torch.Tensor, *,
                eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """[손바닥, 손가락 F개] 접촉력 비율 (N, F+1) 과 합 (N,). 합이 eps 이하면 비율은 0 벡터.

    09.22 사용자: "특정 힘을 가르치는 건 의미가 없다 — 처음 잡아 올렸을 때의 파지 자세와 힘 분포 양상을 유지".
    절대 크기가 아니라 분포(비율)를 비교하려고 쓴다. 음수 판독은 0 으로 자른다.
    """
    f = torch.cat([palm_force.clamp(min=0.0).unsqueeze(1), finger_force.clamp(min=0.0)], dim=1)
    total = f.sum(dim=1)
    share = torch.where((total > float(eps)).unsqueeze(1), f / total.clamp(min=float(eps)).unsqueeze(1),
                        torch.zeros_like(f))
    return share, total


def capture_grasp_ref(valid: torch.Tensor, capture: torch.Tensor, ref: dict[str, torch.Tensor],
                      now: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """capture 가 참이고 아직 기록 전(~valid)인 env 만 now 값으로 기준을 한 번 기록한다 — 새 텐서를 돌려준다.

    이미 기록된 env 는 에피소드가 끝날 때까지 첫 기록을 유지한다(리셋에서 valid 를 False 로).
    """
    take = capture & ~valid
    out = {}
    for k, v in ref.items():
        m = take.view(-1, *([1] * (v.dim() - 1)))
        out[k] = torch.where(m, now[k].to(v.dtype), v)
    return valid | take, out


def sample_pour_targets(n_act: torch.Tensor, *, frac_range: tuple, schedule: tuple, progress: float,
                        rand_u: torch.Tensor, rand_pick: torch.Tensor) -> torch.Tensor:
    """에피소드별 목표 비드 개수 (N,).

    schedule 이 비면 09.23 규칙: frac_range 에서 연속으로 뽑고, 최소 1 알·전량은 아님.
    ★09.27 사용자 결정(목표량 커리큘럼): schedule = (1.0, 0.8, 0.5, 0.2) 처럼 주면 ADR 진행도로 단계를 연다.
      L 단계 중 1 + floor(progress · L) 개(최대 L)를 열고, 연 단계들 중 rand_pick 으로 하나를 고른다(앞 단계를 섞어 잊지 않게).
      1.0 은 전량(n_act) — 컵이 비면 보상의 need 가 0 이 되어 세우기가 열린다. 그 밖은 최소 1 알·한 알은 남긴다.
    """
    n = n_act.float()
    keep_one = torch.clamp(n - 1.0, min=1.0)
    if not schedule:
        lo_f, hi_f = (float(x) for x in frac_range)
        tgt = torch.round(n * (lo_f + (hi_f - lo_f) * rand_u))
        return torch.minimum(torch.clamp(tgt, min=1.0), keep_one)
    levels = torch.tensor([float(x) for x in schedule], device=n.device)
    n_lv = int(levels.numel())
    unlocked = min(n_lv, 1 + int(math.floor(float(progress) * n_lv)))
    idx = torch.clamp((rand_pick * unlocked).long(), max=unlocked - 1)
    frac = levels[idx]
    partial = torch.minimum(torch.clamp(torch.round(n * frac), min=1.0), keep_one)
    return torch.where(frac >= 1.0, n, partial)
