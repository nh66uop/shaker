"""gym 등록 — `open-rh_r_aglt-lstm`(학습) · `open-rh_r_aglt-play-lstm`(재생).

★train.py run_naming 정규식 `^(open-[A-Za-z0-9]+)_([rbl])_(.+?)(?:-play|-lstm|…)*$`
  → 로그 `log/rl_games/open-rh/right/aglt/<label>/`.
★LSTM 만 등록한다(두 형제 트랙 모두 LSTM 1024 로 학습했다 — MLP 판은 쓸 근거가 없다).
"""

import gymnasium as gym

from . import agents
from ..rh_aglt_env_cfg import RhAgltEnvCfg, RhAgltEnvCfg_PLAY

_ENTRY = "openarm.agnostic.tasks.rh_aglt_r.rh_aglt_env:RhAgltEnv"
_YAML = f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml"
REGISTERED: list[str] = []

for _suffix, _cls in (("-lstm", RhAgltEnvCfg), ("-play-lstm", RhAgltEnvCfg_PLAY)):
    _id = f"open-rh_r_aglt{_suffix}"
    gym.register(
        id=_id, entry_point=_ENTRY, disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"openarm.agnostic.tasks.rh_aglt_r.rh_aglt_env_cfg:{_cls.__name__}",
                "rl_games_cfg_entry_point": _YAML},
    )
    REGISTERED.append(_id)
