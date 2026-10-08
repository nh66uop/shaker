"""태스크 자동 등록 가드 — 원격에서 pull 한 뒤 재기동이 조용히 실패하지 않게 한다.

막는 사고: main 에서 태스크 폴더를 지운 뒤 서버가 pull 하면, 그 폴더를 import 하던
태스크가 `except ImportError: pass` 에 묻혀 등록되지 않고, train.py 는 원인 없이
"task not found" 로만 죽는다. 등록 실패를 모아 두었다가 요청한 태스크가 없을 때
원인과 함께 보여 준다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openarm.tasks import discovery as D

_PKG = Path(D.__file__).resolve().parents[1]           # openarm/ (패키지 루트)
_HDGP = _PKG.parents[2]

#: 2026-09-29 main 에서 지운 레거시 로봇 패키지. 되살리려면 archive/legacy-tasks-20260929 태그.
REMOVED = ("tesollo", "rh56f1", "gripper")
#: 같은 날 지운 실패 agnostic 트랙. cup_pick(manip_stages)의 기반인 grasp_fj·grasp_fj_t2r 는 남겼다.
REMOVED_AGNOSTIC = ("grasp_fj_rh", "grasp_kp", "grasp_s2r", "grasp_ua")


def _make_task(root: Path, rel: str) -> None:
    cfg = root / rel / "config"
    cfg.mkdir(parents=True)
    (cfg / "__init__.py").write_text("")


def test_module_names_follow_robot_side_task_layout(tmp_path):
    _make_task(tmp_path, "agnostic/tasks/rh_aglt_r")
    _make_task(tmp_path, "agnostic/tasks/pour_fabric")
    (tmp_path / "agnostic/tasks/no_config").mkdir(parents=True)

    names = D.task_config_modules(tmp_path)

    assert names == ["openarm.agnostic.tasks.pour_fabric.config", "openarm.agnostic.tasks.rh_aglt_r.config"]


def test_import_failures_are_recorded_not_swallowed():
    def fake_import(name):
        if name.endswith("broken.config"):
            raise ModuleNotFoundError("No module named 'openarm.tesollo'")

    errors = D.import_all(["openarm.a.b.ok.config", "openarm.a.b.broken.config"], fake_import)

    assert list(errors) == ["openarm.a.b.broken.config"]
    assert "openarm.tesollo" in errors["openarm.a.b.broken.config"]


def test_missing_task_error_names_the_failed_imports():
    def spec(_):
        raise KeyError("not registered")

    errors = {"openarm.agnostic.tasks.x.config": "ModuleNotFoundError: No module named 'openarm.tesollo'"}

    with pytest.raises(RuntimeError) as exc:
        D.require_registered("open-rh_r_x-lstm", spec, errors)

    msg = str(exc.value)
    assert "open-rh_r_x-lstm" in msg and "openarm.tesollo" in msg


def test_registered_task_passes():
    D.require_registered("open-ok", lambda _: object(), {"openarm.other.config": "boom"})


def test_missing_task_argument_is_left_to_argparse():
    D.require_registered(None, lambda _: pytest.fail("spec must not be queried"), {})


def test_removed_legacy_packages_stay_removed():
    for name in REMOVED:
        assert not (_PKG / name / "__init__.py").exists(), f"{name} 는 main 에서 지웠다(archive 태그 참고)"
    for name in REMOVED_AGNOSTIC:
        assert not (_PKG / "agnostic" / "tasks" / name / "__init__.py").exists(), f"agnostic/tasks/{name}"


_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(?:openarm\.(?:" + "|".join(REMOVED) + r")\b"
    r"|openarm\.agnostic\.tasks\.(?:" + "|".join(REMOVED_AGNOSTIC) + r")\b"
    r"|\.\.(?:" + "|".join(REMOVED_AGNOSTIC) + r")\b)",
    re.M,
)


@pytest.mark.parametrize("sub", ["source/openarm/openarm", "scripts"])
def test_no_active_code_imports_removed_packages(sub):
    offenders = [
        str(p.relative_to(_HDGP))
        for p in (_HDGP / sub).rglob("*.py")
        if "__pycache__" not in p.parts and _IMPORT_RE.search(p.read_text(errors="ignore"))
    ]
    assert offenders == []
