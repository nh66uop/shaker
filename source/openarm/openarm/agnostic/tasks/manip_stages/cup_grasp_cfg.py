"""cup_grasp cfg — `cup_pick` 과 같은 과제(물체·스폰·목표·외란·종료·성공), critic 관측만 +21 (09.27 사용자 결정).

critic 차원은 부모 `finalize_after_overrides` 가 매번 **대입**으로 다시 계산한다. 여기서는 그 뒤에 접촉 블록 차원을 더한다
(대입 뒤 덧셈이라 여러 번 불려도 누적되지 않는다). actor `observation_space`·`action_space` 는 건드리지 않는다.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .grasp_context import CRITIC_CONTACT_DIM
from .cup_pick_cfg import CupPickLeftEnvCfg, CupPickRightEnvCfg


@configclass
class CupGraspRightEnvCfg(CupPickRightEnvCfg):
    """우팔 — cup_pick 과 같은 과제. critic 에 링크별 접촉력 21."""

    #: 09.28: True 면 손 액션 0 = 리셋(편 손) 자세(`hand_action.zero_open_raw`). 기본 False = 부모 사상(이전 런 재생 호환).
    hand_action_zero_open: bool = False
    #: 09.28 실기 지연(`real_delays.LagQueue`, 정책 1 스텝 = 16.7 ms). (0, 0) 이면 부모 큐 그대로(이전 런 재생 호환).
    #:   팔·손은 **추가** 명령 지연(sim 자체 응답 팔 40~50 ms · 손 70~170 ms 위에 얹는다), 물체는 자세 도착 지연 전체.
    real_arm_lag_steps: tuple = (0, 0)
    real_hand_lag_steps: tuple = (0, 0)
    real_obj_lag_steps: tuple = (0, 0)
    real_obj_dropout_p: float = 0.0            # 스텝당 끊김 시작 확률(마지막 값 유지)
    real_obj_dropout_steps: tuple = (0, 0)     # 끊김 길이 [lo, hi] 스텝
    #: 09.28 긴 유지: >0 이면 이 수만큼 성공한 뒤 목표를 옮기지 않는다(마지막 목표 고정 유지). 조기 종료를 막으려면 goal_max 도 키운다.
    goal_hold_after: int = 0

    def finalize_after_overrides(self) -> None:
        super().finalize_after_overrides()
        self.state_space = self.state_space + CRITIC_CONTACT_DIM


@configclass
class CupGraspLeftEnvCfg(CupPickLeftEnvCfg):
    """좌팔 — cup_pick 과 같은 과제. critic 에 링크별 접촉력 21."""

    #: 09.28: True 면 손 액션 0 = 리셋(편 손) 자세(`hand_action.zero_open_raw`). 기본 False = 부모 사상(이전 런 재생 호환).
    hand_action_zero_open: bool = False
    #: 09.28 실기 지연(`real_delays.LagQueue`, 정책 1 스텝 = 16.7 ms). (0, 0) 이면 부모 큐 그대로(이전 런 재생 호환).
    #:   팔·손은 **추가** 명령 지연(sim 자체 응답 팔 40~50 ms · 손 70~170 ms 위에 얹는다), 물체는 자세 도착 지연 전체.
    real_arm_lag_steps: tuple = (0, 0)
    real_hand_lag_steps: tuple = (0, 0)
    real_obj_lag_steps: tuple = (0, 0)
    real_obj_dropout_p: float = 0.0            # 스텝당 끊김 시작 확률(마지막 값 유지)
    real_obj_dropout_steps: tuple = (0, 0)     # 끊김 길이 [lo, hi] 스텝
    #: 09.28 긴 유지: >0 이면 이 수만큼 성공한 뒤 목표를 옮기지 않는다(마지막 목표 고정 유지). 조기 종료를 막으려면 goal_max 도 키운다.
    goal_hold_after: int = 0

    def finalize_after_overrides(self) -> None:
        super().finalize_after_overrides()
        self.state_space = self.state_space + CRITIC_CONTACT_DIM
