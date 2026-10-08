"""실기 반응 — 팔·손 명령 지연, 손 펌웨어 힘 멈춤, 편 손 하한. torch 만 쓴다(Isaac 없이 테스트).

09.30 사용자 결정("실제 제어 로직으로 학습" → 1번: 지금 넣고 재기동). 값은 전부 실측이다:
  · 손: 명령 → 첫 움직임 60~80 ms(드라이버 읽기 20 ms 포함)                   sim2real docs/RH56F1_TUNING_0930.md §2
  · 팔: 로봇 자체 150~200 ms(배포 pd FULL — reduced 속도 제한분은 넣지 않는다)   memory cg-l-i01-real-vs-sim-handoff-0928
  · 네 손가락은 0 rad 를 명령해도 레지스터가 벤더 범위(≤ 1740)에서 잘려 2.2~5.2° 굽은 채, 엄지 굽힘 1.2~2.6°   같은 문서 §4
  · 펌웨어 힘 멈춤(forceSet)은 모터 쪽에서 곧바로 선다 — 명령 지연을 거치지 않는다
정책이 보는 목표(관측 arm_q_target · hand_target − q)는 배포처럼 **명령한 값** 그대로다. 여기 함수는 PD 에 들어가는 값만 만든다.
"""

from __future__ import annotations

import torch


class DelayLine:
    """env 별 명령 지연선. push 한 뒤 read(d) 는 d 스텝 전 값(d = 0 이면 방금 넣은 값)."""

    def __init__(self, num_envs: int, width: int, max_delay: int, device) -> None:
        if max_delay < 0:
            raise ValueError(f"max_delay 는 0 이상: {max_delay}")
        self.buf = torch.zeros(num_envs, int(max_delay) + 1, width, device=device)
        self._rows = torch.arange(num_envs, device=device)

    def fill(self, env_ids: torch.Tensor, value: torch.Tensor) -> None:
        """리셋: 지연선 전체를 value (W,) 또는 (len(env_ids), W) 로 채운다 — 이전 에피소드 명령이 새지 않게."""
        v = value if value.ndim == 2 else value.unsqueeze(0).expand(len(env_ids), -1)
        self.buf[env_ids] = v.unsqueeze(1).expand(-1, self.buf.shape[1], -1)

    def push(self, x: torch.Tensor) -> None:
        self.buf = torch.cat([x.unsqueeze(1), self.buf[:, :-1]], dim=1)

    def read(self, delay: torch.Tensor) -> torch.Tensor:
        return self.buf[self._rows, delay.clamp(0, self.buf.shape[1] - 1)]


def sample_delay(n: int, lo_hi: tuple, device) -> torch.Tensor:
    """에피소드마다 정수 지연 U{lo..hi} (n,)."""
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if not 0 <= lo <= hi:
        raise ValueError(f"지연 범위 0 ≤ lo ≤ hi: {lo_hi}")
    return torch.randint(lo, hi + 1, (n,), device=device)


def sample_open_floor(n: int, lo_deg: tuple, hi_deg: tuple, device) -> torch.Tensor:
    """손 관절별 편 쪽 하한 [rad] (n, H) — 에피소드마다 U(lo, hi)."""
    lo = torch.deg2rad(torch.tensor(lo_deg, dtype=torch.float32, device=device))
    hi = torch.deg2rad(torch.tensor(hi_deg, dtype=torch.float32, device=device))
    if bool((lo < 0.0).any()) or bool((hi < lo).any()):
        raise ValueError(f"편 손 하한 0 ≤ lo ≤ hi: {lo_deg} / {hi_deg}")
    return lo.unsqueeze(0) + torch.rand(n, lo.numel(), device=device) * (hi - lo).unsqueeze(0)


def firmware_stop(prev_applied: torch.Tensor, new: torch.Tensor, touch: torch.Tensor,
                  q_open: torch.Tensor, q_grip: torch.Tensor) -> torch.Tensor:
    """닿은 손가락은 닫는 쪽으로 더 가지 않는다(펴기는 그대로). 새 텐서."""
    close_dir = torch.sign(q_grip - q_open).unsqueeze(0)
    closing = ((new - prev_applied) * close_dir) > 0.0
    return torch.where(touch & closing, prev_applied, new)


def open_floor(target: torch.Tensor, q_open: torch.Tensor, q_grip: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """편 쪽으로 floor 보다 더 펴지지 않게 — 닫힌 깊이(open 에서 grip 쪽 거리)를 floor 이상으로. 새 텐서."""
    close_dir = torch.sign(q_grip - q_open).unsqueeze(0)
    depth = torch.maximum((target - q_open.unsqueeze(0)) * close_dir, floor)
    return q_open.unsqueeze(0) + close_dir * depth
