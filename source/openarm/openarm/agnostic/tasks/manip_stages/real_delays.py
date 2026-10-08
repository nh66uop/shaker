"""실기 지연 모사 — env 별 **고정** 지연(에피소드마다 재추첨) + 물체 자세 끊김 (09.28 사용자 "실기 지연도 학습에서 배워둬야").

왜: 부모 `perception_delay.DelayQueue` 는 매 스텝 지연 인덱스를 0..L-1 에서 새로 뽑는다. 실기 지연은 스텝마다 널뛰지 않고
  거의 일정하다(SIME2REAL 09.28 인계, cg_l_i01 실기 대조):
    · 팔 목표 → 실측 300 ms 중 로봇 자체 150~200 ms (sim 40~50 ms). 배포 pd 속도 제한분 ~190 ms 는 넣지 않는다.
    · 손 목표 → 실측 300 ms (sim 70~170 ms).
    · 물체 자세(FP++) 도착 약 0.4 s, 몇 초 끊김 1 회 관측(sim 은 0..9 스텝 무작위, 평균 ~75 ms).
무엇: `LagQueue` 는 `DelayQueue` 와 같은 인터페이스(push(values, flush), reset(ids), max_delay, dim)라 env 가 큐 객체만 바꿔 끼운다.
  · 열 구간마다(팔 7 · 손 19) 따로 [lo, hi] 스텝에서 env 별 정수 지연을 뽑고 reset 에서 다시 뽑는다.
  · 끊김(dropout): 매 스텝 확률 p 로 시작, 길이 [lo, hi] 스텝 동안 직전 출력을 그대로 돌려준다(마지막 값 유지).
정책 1 스텝 = sim dt 1/120 s × decimation 2 = 16.7 ms.
"""

from __future__ import annotations

import torch


class LagQueue:
    """env 별 고정 지연 링버퍼. segments = [(start, stop, lo, hi), ...] 는 열 구간별 지연 범위(스텝, 양끝 포함)."""

    def __init__(self, num_envs: int, dim: int, segments, device, *, dropout_p: float = 0.0,
                 dropout_steps: tuple[int, int] = (0, 0)) -> None:
        if int(num_envs) < 1 or int(dim) < 1:
            raise ValueError(f"num_envs/dim must be >= 1, got {num_envs}/{dim}")
        segs = [(int(a), int(b), int(lo), int(hi)) for a, b, lo, hi in segments]
        cover = sorted((a, b) for a, b, _, _ in segs)
        if not segs or cover[0][0] != 0 or cover[-1][1] != int(dim) or any(p[1] != q[0] for p, q in zip(cover, cover[1:])):
            raise ValueError(f"segments must tile [0, {dim}) without gaps: {segs}")
        if any(lo < 0 or hi < lo for _, _, lo, hi in segs):
            raise ValueError(f"lag range must satisfy 0 <= lo <= hi: {segs}")
        if not (0.0 <= float(dropout_p) < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
        d_lo, d_hi = int(dropout_steps[0]), int(dropout_steps[1])
        if float(dropout_p) > 0.0 and not (1 <= d_lo <= d_hi):
            raise ValueError(f"dropout_steps must satisfy 1 <= lo <= hi, got {dropout_steps}")
        self.num_envs, self.dim, self.device = int(num_envs), int(dim), device
        self.segments = segs
        self.max_delay = max(hi for _, _, _, hi in segs) + 1          # 링버퍼 길이(부모 로그 호환)
        self.dropout_p, self.dropout_steps = float(dropout_p), (d_lo, d_hi)
        self.buf = torch.zeros(self.num_envs, self.max_delay, self.dim, device=device)
        self.lag = torch.zeros(self.num_envs, len(segs), dtype=torch.long, device=device)
        self.last = torch.zeros(self.num_envs, self.dim, device=device)
        self.drop_left = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._arange = torch.arange(self.num_envs, device=device)
        self._sample(self._arange)

    def _sample(self, ids: torch.Tensor) -> None:
        for k, (_, _, lo, hi) in enumerate(self.segments):
            self.lag[ids, k] = torch.randint(lo, hi + 1, (len(ids),), device=self.device)

    def reset(self, ids) -> None:
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
        self.buf[ids] = 0.0
        self.last[ids] = 0.0
        self.drop_left[ids] = 0
        self._sample(ids)

    def push(self, values: torch.Tensor, flush: torch.Tensor) -> torch.Tensor:
        if tuple(values.shape) != (self.num_envs, self.dim):
            raise ValueError(f"values shape {tuple(values.shape)} != {(self.num_envs, self.dim)}")
        if tuple(flush.shape) != (self.num_envs,) or flush.dtype != torch.bool:
            raise ValueError(f"flush must be bool ({self.num_envs},), got {flush.dtype} {tuple(flush.shape)}")
        self.buf[flush] = values[flush].unsqueeze(1).to(self.buf.dtype)
        self.buf = torch.roll(self.buf, shifts=1, dims=1)
        self.buf[:, 0] = values
        out = torch.empty_like(values)
        for k, (a, b, _, _) in enumerate(self.segments):
            out[:, a:b] = self.buf[self._arange, self.lag[:, k], a:b]
        if self.dropout_p > 0.0:
            start = (self.drop_left == 0) & (torch.rand(self.num_envs, device=self.device) < self.dropout_p) & ~flush
            lo, hi = self.dropout_steps
            self.drop_left = torch.where(start, torch.randint(lo, hi + 1, (self.num_envs,), device=self.device),
                                         self.drop_left)
            hold = (self.drop_left > 0) & ~flush
            out = torch.where(hold.unsqueeze(1), self.last, out)
            self.drop_left = (self.drop_left - 1).clamp(min=0)
        self.last = out.clone()
        return out
