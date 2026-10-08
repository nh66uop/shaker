#!/usr/bin/env python3
"""text2reward CLI — manip_stages 트랙(물체 모양과 무관한 표면 거리 입력 `ObjectRewardContext`). Isaac 불요.

`t2r_fj.py`(다른 트랙이 쓴다 — 수정 금지)를 **import 해서 그대로** 돌리고 세 가지만 이 프로세스 안에서 바꾼다(09.22 사용자 결정).
  · 프롬프트의 환경 클래스 스텁 = `RewardContext` 스텁 + 표면 필드(`manip_stages.object_context.context_stub_source`)
  · ingest 검증기 = `manip_stages.object_validator`(표면 필드를 아는 정적 검사 + 표면 필드를 채운 가짜 ctx)
  · iter meta.json 의 "context" = `ObjectRewardContext`
하위 명령·인자·디렉터리 규약은 t2r_fj.py 와 같다:

    python3 scripts/reward_gen/t2r_ms.py render --track cup_pick_l_grasp --variant rand_left \
        --task-file scripts/reward_gen/tasks/<과제>.txt
    ../IsaacLab/isaaclab.sh -p scripts/reward_gen/t2r_ms.py ingest --iter reward_gen/cup_pick_l_grasp/iter_00
    python3 scripts/reward_gen/t2r_ms.py reflect --iter reward_gen/cup_pick_l_grasp/iter_00 \
        --description .../observation.md --feedback .../improvement.md

★학습 env 는 `open-short_{l,r}_cup_pick-lstm`(`CupPickEnv`) — 이 컨텍스트를 만드는 env 다. 동결 env 로 돌리면 표면 필드가 없어 죽는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import t2r_fj as F  # noqa: E402  (sys.path 에 openarm 소스도 넣는다)

from openarm.agnostic.tasks.grasp_fj_t2r.t2r import prompts as P  # noqa: E402
from openarm.agnostic.tasks.manip_stages import grasp_context as GC  # noqa: E402
from openarm.agnostic.tasks.manip_stages import grasp_validator as GV  # noqa: E402
from openarm.agnostic.tasks.manip_stages import object_context as C  # noqa: E402
from openarm.agnostic.tasks.manip_stages import object_validator as OV  # noqa: E402

# ★09.27 컨텍스트 두 벌: `object`(cup_pick, 기본) · `grasp`(cup_grasp — 첫마디 필드 추가, `CupGraspEnv`).
#   render 는 `--context grasp` 로 고르고, ingest·reflect 는 `--iter` 폴더 meta.json 의 "context" 로 자동으로 고른다.
_CONTEXTS = {
    "object": (C.context_stub_source, OV, C.ObjectRewardContext),
    "grasp": (GC.context_stub_source, GV, GC.GraspRewardContext),
}
_CTX_NAME = "object"


def _ctx_path(name: str) -> str:
    cls = _CONTEXTS[name][2]
    return f"{cls.__module__}.{cls.__qualname__}"


CONTEXT_PATH = _ctx_path("object")


_BASE_RENDER, _BASE_REFLECT = F.cmd_render, F.cmd_reflect


def _stamp_meta(d: Path) -> None:
    p = d / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["context"] = _ctx_path(_CTX_NAME)
    p.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_render(a) -> int:
    n = a.iter if a.iter is not None else F._next_iter(a.root, a.track)
    a.iter = n
    rc = _BASE_RENDER(a)
    _stamp_meta(a.root / a.track / f"iter_{n:02d}")
    return rc


def cmd_reflect(a) -> int:
    meta = json.loads((Path(a.iter) / "meta.json").read_text(encoding="utf-8"))
    n_next = a.next_iter if a.next_iter is not None else int(meta["iter"]) + 1
    rc = _BASE_REFLECT(a)
    _stamp_meta(a.root / meta["track"] / f"iter_{n_next:02d}")
    return rc


def _install() -> None:
    """이 프로세스의 t2r_fj·prompts 가 ObjectRewardContext 를 쓰게 한다(다른 프로세스·트랙은 무관).

    `F.main` 은 `set_defaults(fn=cmd_render)` 를 호출 시점의 모듈 전역에서 읽으므로 전역을 바꾸면 그 파서가 이쪽을 부른다.
    """
    stub, validator, _cls = _CONTEXTS[_CTX_NAME]
    P.context_stub_source = stub
    F.V = validator
    F.cmd_render, F.cmd_reflect = cmd_render, cmd_reflect


def _pick_context(argv: list[str]) -> tuple[str, list[str]]:
    """`--context <이름>` 을 떼어 내거나, 없으면 `--iter` 폴더의 meta.json "context" 로 고른다(기본 object)."""
    argv = list(argv)
    if "--context" in argv:
        i = argv.index("--context")
        if i + 1 >= len(argv):
            raise SystemExit(f"[t2r_ms] --context 뒤에 이름이 없다 (있는 것: {sorted(_CONTEXTS)})")
        name = argv[i + 1]
        if name not in _CONTEXTS:
            raise SystemExit(f"[t2r_ms] 모르는 --context {name!r} (있는 것: {sorted(_CONTEXTS)})")
        return name, argv[:i] + argv[i + 2:]
    if "--iter" in argv:
        meta = Path(argv[argv.index("--iter") + 1]) / "meta.json"
        if meta.is_file():
            ctx = json.loads(meta.read_text(encoding="utf-8")).get("context", "")
            for name in _CONTEXTS:
                if ctx == _ctx_path(name):
                    return name, argv
    if argv and argv[0] == "render":
        print("[t2r_ms] 경고: render 에 --context 가 없어 object 로 만든다 — cup_grasp 트랙이면 --context grasp", flush=True)
    return "object", argv


def main(argv=None) -> int:
    global _CTX_NAME
    _CTX_NAME, argv = _pick_context(sys.argv[1:] if argv is None else argv)
    print(f"[t2r_ms] 컨텍스트 = {_ctx_path(_CTX_NAME)}", flush=True)
    _install()
    return F.main(argv)


if __name__ == "__main__":
    sys.exit(main())
