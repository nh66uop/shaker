"""ObjectRewardContext · 검증기 · t2r_ms 생성기 — Isaac 불요.

    cd hdgp && PYTHONPATH=source/openarm /usr/bin/python3 -m pytest source/openarm/openarm/agnostic/tasks/manip_stages/tests -q
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from openarm.agnostic.tasks.grasp_fj_t2r.t2r import context as base_ctx
from openarm.agnostic.tasks.grasp_fj_t2r.t2r import validator as base_val
from openarm.agnostic.tasks.manip_stages import object_context as C
from openarm.agnostic.tasks.manip_stages import object_validator as OV

_HDGP = Path(__file__).resolve().parents[7]

_NEW = ("palm_surface_dist", "palm_surface_dir", "palm_surface_normal",
        "link_surface_dist", "link_surface_dir", "link_surface_normal")

REWARD_USING_SURFACE = '''\
import torch
import math


def compute_reward(ctx):
    near = torch.exp(-10.0 * ctx.palm_surface_dist)
    wrap = torch.exp(-20.0 * ctx.link_surface_dist).mean(dim=(1, 2))
    facing = -(ctx.link_surface_dir * ctx.link_surface_normal).sum(-1).mean(dim=(1, 2))
    palm_face = -(ctx.palm_surface_dir * ctx.palm_surface_normal).sum(-1)
    lift = ctx.cup_pos[:, 2] - ctx.cup_spawn_pos[:, 2]
    r = near + wrap + 0.1 * facing + 0.1 * palm_face + lift
    return r, {"near": near, "wrap": wrap, "facing": facing, "lift": lift}
'''


def test_context_only_appends_the_surface_fields_to_the_frozen_context():
    base = [f.name for f in fields(base_ctx.RewardContext)]
    ours = [f.name for f in fields(C.ObjectRewardContext)]
    assert ours[:len(base)] == base, "기존 필드·순서는 그대로"
    assert tuple(ours[len(base):]) == _NEW == C.OBJECT_TENSOR_FIELDS
    assert issubclass(C.ObjectRewardContext, base_ctx.RewardContext)


def test_every_new_field_has_a_one_line_english_comment_in_the_prompt_stub():
    for side in ("r", "l"):
        stub = C.context_stub_source(side)
        assert stub.startswith(base_ctx.context_stub_source(side)), "부모 스텁(좌/우 문구) 그대로 + 뒤에 추가"
        for name in _NEW:
            line = next(ln for ln in stub.splitlines() if ln.strip().startswith(f"{name}:"))
            comment = line.split("#", 1)[1]
            assert comment.isascii() and len(comment.split()) >= 5, (name, comment)
        assert "LINK_TOUCH_DIST" not in stub, "자리표시는 실측값으로 치환"
        assert "# ----------" not in stub.split("object surface", 1)[1]


def test_fake_context_has_sensible_surface_values():
    ctx = OV.make_fake_context(32, seed=1)
    assert isinstance(ctx, C.ObjectRewardContext)
    assert ctx.palm_surface_dist.shape == (32,) and ctx.link_surface_dist.shape == (32, 5, 3)
    assert ctx.link_surface_dir.shape == ctx.link_surface_normal.shape == (32, 5, 3, 3)
    for d in (ctx.palm_surface_dist, ctx.link_surface_dist):
        assert float(d.min()) >= 0.0 and float(d.max()) <= 0.3
    for v in (ctx.palm_surface_dir, ctx.palm_surface_normal, ctx.link_surface_dir, ctx.link_surface_normal):
        torch.testing.assert_close(v.norm(dim=-1), torch.ones(v.shape[:-1]), atol=1e-5, rtol=0)
    # 부모 필드는 부모 가짜 ctx 와 같은 값
    base = base_val.make_fake_context(32, seed=1)
    torch.testing.assert_close(ctx.cup_pos, base.cup_pos)


def test_new_validator_accepts_surface_fields_the_frozen_one_rejects(tmp_path):
    p = tmp_path / "compute_reward.py"
    p.write_text(REWARD_USING_SURFACE, encoding="utf-8")
    frozen = base_val.static_check(REWARD_USING_SURFACE)
    assert not frozen.ok and any("palm_surface_dist" in e for e in frozen.errors)
    rep = OV.validate(str(p), devices=("cpu",))
    assert rep.ok, rep.errors
    assert set(rep.term_stats) == {"near", "wrap", "facing", "lift"}
    assert "link_surface_normal" in rep.fields_used
    # 부모 검증기는 호출이 끝나면 원래대로(다른 트랙 무영향)
    assert base_val.static_check is OV._BASE_STATIC_CHECK and base_val.make_fake_context is OV._BASE_FAKE
    assert not base_val.validate(str(p), devices=("cpu",)).ok


def test_new_validator_still_rejects_hallucinated_fields_and_forbidden_code(tmp_path):
    bad = REWARD_USING_SURFACE.replace("ctx.palm_surface_dist", "ctx.palm_surface_distance")
    assert not OV.static_check(bad).ok
    assert not OV.static_check(REWARD_USING_SURFACE.replace("import math", "import os")).ok
    assert not OV.static_check(REWARD_USING_SURFACE + "\n\ndef f(ctx):\n    ctx.link_surface_dist[:] = 0\n").ok


def test_t2r_ms_renders_the_surface_fields_and_ingests_with_the_new_validator(tmp_path):
    sys.path.insert(0, str(_HDGP / "scripts" / "reward_gen"))
    import t2r_fj
    import t2r_ms
    from openarm.agnostic.tasks.grasp_fj_t2r.t2r import prompts as P

    task = tmp_path / "task.txt"
    task.write_text("Grasp any object.", encoding="utf-8")
    orig_stub, orig_v, orig_cmds = P.context_stub_source, t2r_fj.V, (t2r_fj.cmd_render, t2r_fj.cmd_reflect)
    try:
        assert t2r_ms.main(["--root", str(tmp_path), "render", "--track", "ms_t", "--task-file", str(task),
                            "--variant", "rand_left"]) == 0
        d = tmp_path / "ms_t" / "iter_00"
        prompt = (d / "prompt.md").read_text(encoding="utf-8")
        assert "link_surface_normal: torch.Tensor" in prompt and "hand: left Tesollo" in prompt
        assert json.loads((d / "meta.json").read_text())["context"].endswith("object_context.ObjectRewardContext")
        (d / "response.md").write_text("thinking\n```python\n" + REWARD_USING_SURFACE + "```\n", encoding="utf-8")
        assert t2r_ms.main(["--root", str(tmp_path), "ingest", "--iter", str(d)]) == 0
        assert json.loads((d / "validation.json").read_text())["ok"]
    finally:
        P.context_stub_source, t2r_fj.V = orig_stub, orig_v
        t2r_fj.cmd_render, t2r_fj.cmd_reflect = orig_cmds
    # t2r_fj(다른 트랙)는 그대로: 부모 스텁 · 부모 검증기
    assert "palm_surface_dist" not in P.render_prompt(P.PromptSpec(task="x", variant="rand_left"))
    assert t2r_fj.main(["--root", str(tmp_path), "ingest", "--iter", str(d)]) == 1


def test_t2r_fj_source_does_not_know_about_manip_stages():
    src = (_HDGP / "scripts" / "reward_gen" / "t2r_fj.py").read_text(encoding="utf-8")
    assert "manip_stages" not in src and "ObjectRewardContext" not in src


@pytest.mark.parametrize("name", _NEW)
def test_surface_fields_are_tensors(name):
    assert name in C.TENSOR_FIELDS and name not in base_ctx.TENSOR_FIELDS


# ---- 09.27 GraspRewardContext (cup_grasp) ------------------------------------------------------------------------
def test_grasp_context_adds_only_proximal_fields_and_stub_lists_them():
    from openarm.agnostic.tasks.manip_stages import grasp_context as G
    from openarm.agnostic.tasks.manip_stages.object_context import TENSOR_FIELDS
    assert G.GRASP_TENSOR_FIELDS == ("prox_pos", "prox_cup_force", "prox_surface_dist", "prox_surface_dir",
                                     "prox_surface_normal")
    assert not set(G.GRASP_TENSOR_FIELDS) & set(TENSOR_FIELDS)
    stub = G.context_stub_source("r")
    for name in G.GRASP_TENSOR_FIELDS:
        assert f"    {name}:" in stub, name
    assert "PROX_TOUCH_DIST" not in stub
    assert G.CRITIC_CONTACT_DIM == 21


def test_grasp_validator_accepts_proximal_fields_and_rejects_unknown():
    import textwrap
    from openarm.agnostic.tasks.manip_stages import grasp_validator as GV
    ok = textwrap.dedent("""
        import torch
        def compute_reward(ctx):
            f = torch.tanh(ctx.prox_cup_force / 0.15).mean(dim=-1)
            d = torch.exp(-ctx.prox_surface_dist / 0.02).mean(dim=-1)
            r = f + d + 0.0 * ctx.link_surface_dist.mean(dim=(1, 2))
            return r, {"prox": f, "near": d}
    """)
    rep = GV.static_check(ok)
    assert rep.ok, rep.errors
    bad = ok.replace("ctx.prox_surface_dist", "ctx.prox_imaginary")
    assert not GV.static_check(bad).ok
    fake = GV.make_fake_context(8)
    assert tuple(fake.prox_cup_force.shape) == (8, 5) and tuple(fake.prox_surface_normal.shape) == (8, 5, 3)
