"""goal_wrench — 목표 게이트 외란의 게이트·ADR 순수 함수(09.23). Isaac 없이 텐서로 직접 조건을 만든다."""

from __future__ import annotations

import pytest
import torch

from ..goal_wrench import (ADR_LEVEL_STEP, ADR_LOWER_DWELL, ADR_RAISE_DWELL, NEAR_GOAL_RADIUS_M, NO_EVENT,
                           W1_FORCE_SCALE, W1_TORQUE_SCALE, W2_FORCE_SCALE, W2_TORQUE_SCALE,
                           episode_ema, near_goal_mask, next_adr_level, wrench_gate, wrench_scales)

_R = NEAR_GOAL_RADIUS_M


def _lv(x: float) -> torch.Tensor:
    return torch.tensor(float(x))


# ---------------------------------------------------------------- 게이트
def test_gate_fires_only_when_lifted_and_near_goal():
    """네 조합 전부 — 들림 ∧ 목표 근방 에서만 True."""
    lifted = torch.tensor([True, True, False, False])
    goal_dist = torch.tensor([_R - 0.01, _R + 0.01, _R - 0.01, _R + 0.01])
    assert wrench_gate(lifted, goal_dist).tolist() == [True, False, False, False]


def test_gate_is_closed_on_the_boundary_from_outside_and_open_on_it():
    d = torch.tensor([_R, _R + 1e-4])
    assert near_goal_mask(d).tolist() == [True, False]


def test_gate_never_fires_on_a_cup_still_on_the_table_however_close_the_goal_is():
    """높이 래치가 없으면(상판 위) 목표 거리가 0 이어도 발화하지 않는다 — fj_core_cfg 의 경고."""
    gate = wrench_gate(torch.zeros(4, dtype=torch.bool), torch.zeros(4))
    assert not gate.any()


def test_gate_rejects_a_non_boolean_or_mismatched_lift_latch():
    with pytest.raises(ValueError):
        wrench_gate(torch.ones(3), torch.zeros(3))
    with pytest.raises(ValueError):
        wrench_gate(torch.ones(2, dtype=torch.bool), torch.zeros(3))


# ---------------------------------------------------------------- 크기 보간
def test_scales_interpolate_from_w1_at_level_zero_to_w2_at_level_one():
    assert wrench_scales(0.0) == (W1_FORCE_SCALE, W1_TORQUE_SCALE)
    assert wrench_scales(1.0) == (W2_FORCE_SCALE, W2_TORQUE_SCALE)
    f, t = wrench_scales(0.5)
    assert f == pytest.approx(0.5 * (W1_FORCE_SCALE + W2_FORCE_SCALE))
    assert t == pytest.approx(0.5 * (W1_TORQUE_SCALE + W2_TORQUE_SCALE))


def test_scales_clamp_outside_the_unit_interval_and_accept_a_tensor_level():
    assert wrench_scales(-3.0) == (W1_FORCE_SCALE, W1_TORQUE_SCALE)
    assert wrench_scales(7.0) == (W2_FORCE_SCALE, W2_TORQUE_SCALE)
    f, t = wrench_scales(_lv(0.2))
    assert isinstance(f, torch.Tensor) and float(f) == pytest.approx(5.0 + 0.2 * 15.0)
    assert float(t) == pytest.approx(0.5 + 0.2 * 1.5)


# ---------------------------------------------------------------- ADR
def test_level_rises_when_the_policy_dwells_at_the_goal_and_falls_when_it_does_not():
    lo = next_adr_level(_lv(0.4), _lv(ADR_RAISE_DWELL + 0.05))
    assert float(lo) == pytest.approx(0.4 + ADR_LEVEL_STEP)
    hi = next_adr_level(_lv(0.4), _lv(ADR_LOWER_DWELL - 0.01))
    assert float(hi) == pytest.approx(0.4 - ADR_LEVEL_STEP)


def test_level_holds_inside_the_hysteresis_band():
    mid = 0.5 * (ADR_RAISE_DWELL + ADR_LOWER_DWELL)
    assert float(next_adr_level(_lv(0.4), _lv(mid))) == pytest.approx(0.4)


def test_level_clamps_to_the_unit_interval():
    assert float(next_adr_level(_lv(1.0), _lv(1.0))) == pytest.approx(1.0)
    assert float(next_adr_level(_lv(0.0), _lv(0.0))) == pytest.approx(0.0)


def test_level_does_not_move_before_any_episode_has_finished():
    """센티널(음수 EMA)은 강등 조건을 산술로 만족하지만 움직이면 안 된다."""
    assert float(next_adr_level(_lv(0.5), _lv(NO_EVENT))) == pytest.approx(0.5)


def test_level_walks_the_whole_range_in_twenty_updates_each_way():
    lv = _lv(0.0)
    for _ in range(20):
        lv = next_adr_level(lv, _lv(1.0))
    assert float(lv) == pytest.approx(1.0)
    for _ in range(20):
        lv = next_adr_level(lv, _lv(0.0))
    assert float(lv) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- 체류율 EMA
def test_dwell_ema_ignores_steps_where_no_episode_ended():
    prev = _lv(0.3)
    out = episode_ema(prev, torch.tensor([0.9, 0.9]), torch.zeros(2, dtype=torch.bool))
    assert float(out) == pytest.approx(0.3)


def test_dwell_ema_takes_the_first_finished_batch_whole_then_blends():
    first = episode_ema(_lv(NO_EVENT), torch.tensor([0.2, 0.4, 9.0]),
                        torch.tensor([True, True, False]), alpha=0.5)
    assert float(first) == pytest.approx(0.3)                 # 끝난 둘의 평균, 안 끝난 env 는 안 센다
    second = episode_ema(first, torch.tensor([0.5]), torch.tensor([True]), alpha=0.5)
    assert float(second) == pytest.approx(0.4)
