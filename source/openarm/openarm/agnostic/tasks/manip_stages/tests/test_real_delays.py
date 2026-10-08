"""09.28 real_delays.LagQueue: env 별 고정 지연(구간별), reset 재추첨, 끊김 시 마지막 값 유지."""
import pytest
import torch

from openarm.agnostic.tasks.manip_stages.real_delays import LagQueue


def _run(q, T, dim):
    outs = []
    for t in range(T):
        v = torch.full((q.num_envs, dim), float(t))
        outs.append(q.push(v, torch.full((q.num_envs,), t == 0)))
    return torch.stack(outs)            # (T, N, dim)


def test_fixed_lag_per_segment():
    torch.manual_seed(0)
    q = LagQueue(4, 3, [(0, 1, 2, 2), (1, 3, 5, 5)], "cpu")
    o = _run(q, 12, 3)
    assert torch.allclose(o[10, :, 0], torch.full((4,), 8.0))
    assert torch.allclose(o[10, :, 1:], torch.full((4, 2), 5.0))


def test_lag_constant_within_episode_and_in_range():
    torch.manual_seed(1)
    q = LagQueue(64, 1, [(0, 1, 9, 12)], "cpu")
    o = _run(q, 30, 1)[:, :, 0]
    lag = torch.arange(30).unsqueeze(1).float() - o
    assert bool(((lag[15:] >= 9) & (lag[15:] <= 12)).all())
    assert bool((lag[15:] == lag[15]).all())


def test_reset_resamples_and_flush_fills():
    torch.manual_seed(2)
    q = LagQueue(256, 1, [(0, 1, 0, 20)], "cpu")
    before = q.lag.clone()
    q.reset(torch.arange(256))
    assert not torch.equal(before, q.lag)
    out = q.push(torch.full((256, 1), 7.0), torch.ones(256, dtype=torch.bool))
    assert torch.allclose(out, torch.full((256, 1), 7.0))


def test_dropout_holds_last_value():
    torch.manual_seed(3)
    q = LagQueue(1, 1, [(0, 1, 0, 0)], "cpu", dropout_p=0.999, dropout_steps=(5, 5))
    o = _run(q, 12, 1)[:, 0, 0]
    assert o[0] == 0.0
    assert bool((o[1:6] == o[1]).all())


def test_zero_lag_is_identity_and_bad_segments_raise():
    q = LagQueue(2, 2, [(0, 2, 0, 0)], "cpu")
    o = _run(q, 5, 2)
    assert torch.allclose(o[4], torch.full((2, 2), 4.0))
    with pytest.raises(ValueError):
        LagQueue(2, 3, [(0, 1, 0, 0)], "cpu")
    with pytest.raises(ValueError):
        LagQueue(2, 1, [(0, 1, 3, 1)], "cpu")
