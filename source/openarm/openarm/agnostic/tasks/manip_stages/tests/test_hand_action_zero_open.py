"""09.28 hand_action_zero_open: 액션 0 = 리셋(편 손) 자세, 범위·연속성은 부모 사상과 같다."""
import ast
from pathlib import Path

import torch

from openarm.agnostic.tasks.manip_stages.hand_action import zero_open_raw

HERE = Path(__file__).resolve().parents[1]


def _parent_raw(a, lo, hi):
    return lo + 0.5 * (a.clamp(-1, 1) + 1.0) * (hi - lo)


def test_zero_action_gives_reset_pose():
    lo = torch.tensor([0.0, -1.0, 0.0]); hi = torch.tensor([1.5, 1.0, 2.0]); q0 = torch.tensor([0.0, 0.2, 0.5])
    assert torch.allclose(zero_open_raw(torch.zeros(4, 3), lo, hi, q0), q0.expand(4, 3))


def test_endpoints_match_parent_range():
    lo = torch.tensor([0.0, -1.0]); hi = torch.tensor([1.5, 1.0]); q0 = torch.tensor([0.0, 0.3])
    for v in (-1.0, 1.0, -3.0, 3.0):
        a = torch.full((1, 2), v)
        assert torch.allclose(zero_open_raw(a, lo, hi, q0), _parent_raw(a, lo, hi))


def test_monotone_and_continuous():
    lo = torch.tensor([0.0, -1.0]); hi = torch.tensor([1.5, 1.0]); q0 = torch.tensor([0.0, 0.3])
    a = torch.linspace(-1, 1, 2001).unsqueeze(1).expand(-1, 2)
    r = zero_open_raw(a, lo, hi, q0)
    d = r[1:] - r[:-1]
    assert bool((d >= -1e-7).all())
    assert float(d.abs().max()) < 2e-3


def test_cfg_flag_defaults_off_for_replay_compat():
    src = (HERE / "cup_grasp_cfg.py").read_text()
    tree = ast.parse(src)
    vals = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
            and getattr(n.target, "id", "") == "hand_action_zero_open"]
    assert vals == [False, False]
