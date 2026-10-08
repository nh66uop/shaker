"""Gym 태스크 자동 등록과 등록 실패 추적.

전에는 `<robot>/<side>/<task>/config/__init__.py` 를 import 하다 난 ImportError 를
그냥 버렸다. main 에서 태스크 폴더를 지운 뒤 원격이 pull 하면, 그 폴더에 기대던 태스크가
조용히 빠지고 train.py 는 원인 없이 "not found" 로 죽었다(2026-09-29 레거시 정리).
실패는 모아 두고, 요청한 태스크가 없을 때 그 목록을 같이 보여 준다.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path

#: import 실패한 config 모듈 → 오류 한 줄. train.py·play.py 가 require_registered 로 읽는다.
REGISTRATION_ERRORS: dict[str, str] = {}


def task_config_modules(pkg_root: Path) -> list[str]:
    """pkg_root(openarm/) 아래 `<robot>/<side>/<task>/config` 모듈 이름, 정렬."""
    return [
        "openarm." + ".".join(p.parent.relative_to(pkg_root).parts)
        for p in sorted(pkg_root.glob("*/*/*/config/__init__.py"))
    ]


def import_all(modules: Iterable[str], import_fn: Callable[[str], object]) -> dict[str, str]:
    """모듈마다 import_fn 을 부르고, 실패한 것만 {모듈: 오류 한 줄} 로 돌려준다."""
    errors: dict[str, str] = {}
    for name in modules:
        try:
            import_fn(name)
        except ImportError as exc:
            errors[name] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return errors


def require_registered(task: str | None,spec_fn: Callable[[str], object], errors: dict[str, str]) -> None:
    """task 가 gym 에 없으면 등록 실패 목록을 붙여 RuntimeError. task 가 비면 argparse 쪽에 맡긴다."""
    if not task:
        return
    task_key = task.split(":")[-1]
    try:
        spec_fn(task_key)
        return
    except Exception:
        pass
    lines = [f"task '{task_key}' is not registered."]
    if errors:
        lines.append("these task configs failed to import (after a git pull, a removed module is the usual cause):")
        lines += [f"  {mod}: {err}" for mod, err in sorted(errors.items())]
    else:
        lines.append("no task config failed to import; check the task id.")
    raise RuntimeError("\n".join(lines))


def register_all(pkg_root: Path, import_fn: Callable[[str], object]) -> None:
    """모든 태스크 config 를 import 하고 실패를 REGISTRATION_ERRORS 에 모은다."""
    REGISTRATION_ERRORS.clear()
    REGISTRATION_ERRORS.update(import_all(task_config_modules(pkg_root), import_fn))
    if REGISTRATION_ERRORS:
        print(f"[openarm.tasks] {len(REGISTRATION_ERRORS)} task config(s) failed to import: "
              + ", ".join(sorted(REGISTRATION_ERRORS)), file=sys.stderr)
