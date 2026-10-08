"""gym 등록 — manip_stages(단계별 보상 커리큘럼 프레임워크). 에이전트 yaml 은 `grasp_fj_t2r`(s2r 동결) 것을 그대로 쓴다.

entry point(09.22 사용자 결정): `cup_pick` = `CupPickEnv`(부모 env + 물체 모양과 무관한 표면 거리 **보상 입력**만 추가,
관측·액션·종료·성공은 부모 그대로 — 체크포인트 호환). deprecated `grasp_fj_ab` = 동결 env 그대로.

id 규약: `open-short_{r,l}_<과제>-lstm` → `log/rl_games/open-short/{right,left}/<과제-하이픈>/<label>/`.
★`grasp_fj_ab` id 는 이름을 바꾸기 전에 띄운 좌팔 1단계 런(`fj_abL_a00`)의 재생·영상용으로만 남긴다 — 그 런이 끝나면 지운다.
"""

import gymnasium as gym

from ...grasp_fj.config import agents
from ..cup_grasp_cfg import CupGraspLeftEnvCfg, CupGraspRightEnvCfg
from ..cup_pick_cfg import CupPickLeftEnvCfg, CupPickRightEnvCfg

_ENTRY_FROZEN = "openarm.agnostic.tasks.grasp_fj_t2r.grasp_fj_t2r_env:GraspFJT2REnv"
_ENTRY_CUP_PICK = "openarm.agnostic.tasks.manip_stages.cup_pick_env:CupPickEnv"
# 09.27 사용자 결정: `cup_grasp` = `CupGraspEnv`(cup_pick + 첫마디 보상 입력 + critic 접촉 관측 21). critic 차원이 달라
#   cup_pick 체크포인트와 섞이지 않게 gym id·로그 폴더(`cup-grasp`)를 가른다. actor 관측·액션은 cup_pick 과 같다.
_ENTRY_CUP_GRASP = "openarm.agnostic.tasks.manip_stages.cup_grasp_env:CupGraspEnv"
_ENTRIES = {"cup_pick": _ENTRY_CUP_PICK, "cup_grasp": _ENTRY_CUP_GRASP, "grasp_fj_ab": _ENTRY_FROZEN}


def _play(cls):
    class _Play(cls):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 50
            self.scene.env_spacing = 2.5
            self.tol_eval = self.tol_floor
    _Play.__name__ = cls.__name__ + "_PLAY"
    return _Play


_CFGS = {("short_r", "cup_pick"): CupPickRightEnvCfg,
         ("short_l", "cup_pick"): CupPickLeftEnvCfg,
         ("short_r", "cup_grasp"): CupGraspRightEnvCfg,
         ("short_l", "cup_grasp"): CupGraspLeftEnvCfg,
         # deprecated — 이름 변경 전 런(fj_abL_a00) 재생용. 그 런 종료 후 삭제.
         ("short_r", "grasp_fj_ab"): CupPickRightEnvCfg,
         ("short_l", "grasp_fj_ab"): CupPickLeftEnvCfg}

SKIPPED: dict[str, str] = {}
REGISTERED: list[str] = []

for (_tag, _task), _cls in _CFGS.items():
    _play_cls = _play(_cls)
    globals()[_cls.__name__] = _cls
    globals()[_play_cls.__name__] = _play_cls
    for _suffix, _cfg_name in (("-lstm", _cls.__name__), ("-play-lstm", _play_cls.__name__)):
        gym.register(
            id=f"open-{_tag}_{_task}{_suffix}",
            entry_point=_ENTRIES[_task],
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}:{_cfg_name}",
                    "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml"},
        )
        REGISTERED.append(f"open-{_tag}_{_task}{_suffix}")
