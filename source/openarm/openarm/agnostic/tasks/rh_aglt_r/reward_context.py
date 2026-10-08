"""보상 입력(RewardContext)과 생성 보상 로더 — t2r 생성 코드 `compute_reward(ctx)` 의 계약.

파일 계약(pour_fabric_mimic `t2r/loader.py` 와 같다):
    def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
        return total (N,), {"term_name": (N,), ...}
경로가 비면 영 보상(부팅·무작위 롤아웃용). `exec` 이름공간은 torch·math·RewardContext 뿐이다.

좌표는 전부 env-local [m], 쿼터니언 wxyz, 힘 [N]. 손가락 순서 = (thumb, index, middle, ring, pinky).
★palm_axes 앞 3칸 = 손바닥 법선(손바닥이 향하는 쪽), 뒤 3칸 = 손가락 길이 방향(RH56F1 palm_sensor 열 2·1, 09.15).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class RewardContext:
    # ---- 상수 기하 ----
    table_z: float                 # 상판 윗면 z
    cup_radius: float              # 컵 바깥 반지름(가장 넓은 곳)
    cup_mouth_z: float             # 컵 원점 기준 입구 높이(+)
    cup_bottom_z: float            # 컵 원점 기준 바닥 높이(−)
    lift_latch_m: float            # 들기 래치 높이(스폰 대비)
    goal_tol: float                # 현재 성공 허용오차(키포인트 최대거리)
    goal_max: int                  # 에피소드 목표 수 상한
    # ---- 손·팔 ----
    palm_pos: torch.Tensor         # (N,3)
    palm_lin_vel: torch.Tensor     # (N,3)
    palm_axes: torch.Tensor        # (N,6) [법선, 손가락 길이]
    tips_pos: torch.Tensor         # (N,5,3)
    hand_closure: torch.Tensor     # (N,) 구동 관절 평균 폐쇄도 [0,1]
    hand_q_target_closure: torch.Tensor   # (N,6) 관절별 목표 폐쇄도 [0,1]
    finger_force: torch.Tensor     # (N,5) 손가락별 자기 컵 접촉력 합(첫마디 + 손끝)
    finger_prox_force: torch.Tensor   # (N,5) 첫마디(_1) 컵 접촉력
    finger_tip_force: torch.Tensor    # (N,5) 손끝(_sensor) 컵 접촉력
    palm_force: torch.Tensor       # (N,) 손바닥 패드 컵 접촉력
    hand_foreign_force: torch.Tensor   # (N,) 손이 컵 아닌 것(테이블·몸)에 닿는 힘
    grasped: torch.Tensor          # (N,) bool — 엄지 AND (다른 손가락 하나 이상) 컵 접촉 > 1 N
    hand_min_z: torch.Tensor       # (N,) 손 링크 원점 최저 z
    arm_q: torch.Tensor            # (N,7)
    arm_qd: torch.Tensor           # (N,7)
    arm_limit_margin: torch.Tensor # (N,7) 1 = 범위 중앙, 0 = 한계
    # ---- 컵 ----
    cup_pos: torch.Tensor          # (N,3) 원점
    cup_quat: torch.Tensor         # (N,4)
    cup_up: torch.Tensor           # (N,3) 컵 축(+z)
    cup_tilt: torch.Tensor         # (N,) rad
    cup_lin_vel: torch.Tensor      # (N,3)
    cup_ang_vel: torch.Tensor      # (N,3)
    cup_spawn_pos: torch.Tensor    # (N,3)
    lifted: torch.Tensor           # (N,) bool 래치 — 한 번이라도 lift_latch_m 넘게 들렸다
    # ---- 목표 ----
    goal_pos: torch.Tensor         # (N,3)
    goal_kp_dist: torch.Tensor     # (N,) 키포인트 최대거리(컵 ↔ 목표)
    near_goal: torch.Tensor        # (N,) bool kp_dist ≤ goal_tol
    goal_success: torch.Tensor     # (N,) bool 이 스텝에 목표 하나를 달성(쥔 채)
    goal_successes: torch.Tensor   # (N,) 이번 에피소드 달성 수(이 스텝 포함)
    # ---- 시간·액션 ----
    episode_progress: torch.Tensor # (N,) [0,1]
    actions: torch.Tensor          # (N,13) [팔 7, 손 6]
    prev_actions: torch.Tensor     # (N,13)

    @property
    def num_envs(self) -> int:
        return int(self.palm_pos.shape[0])

    @property
    def device(self) -> torch.device:
        return self.palm_pos.device


RewardFn = Callable[[RewardContext], tuple[torch.Tensor, dict[str, torch.Tensor]]]
ENTRY_NAME = "compute_reward"


def zero_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return torch.zeros(ctx.num_envs, device=ctx.device), {}


def load_reward_fn(path: str | None) -> tuple[RewardFn, str]:
    """(함수, 출처 설명). path 가 비면 zero_reward."""
    if not path:
        return zero_reward, "zero (reward_code_path 비어 있음)"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"reward_code_path 가 없다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    ns: dict = {"torch": torch, "math": math, "RewardContext": RewardContext,
                "__name__": "t2r_generated_reward"}
    exec(compile(src, path, "exec"), ns)   # noqa: S102 — 생성 코드는 ingest 검증을 먼저 거친다
    fn = ns.get(ENTRY_NAME)
    if not callable(fn):
        raise RuntimeError(f"{path}: `{ENTRY_NAME}(ctx)` 가 정의되지 않았다")
    return fn, path


def call_reward_fn(fn: RewardFn, ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """호출 + 반환 형태 강제(조용한 브로드캐스트 금지)."""
    out = fn(ctx)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise RuntimeError(f"{ENTRY_NAME} 는 (reward, terms) 튜플을 돌려줘야 한다: {type(out)}")
    total, terms = out
    n = ctx.num_envs
    if not isinstance(total, torch.Tensor) or tuple(total.shape) != (n,):
        raise RuntimeError(f"{ENTRY_NAME} reward 는 ({n},) 텐서여야 한다: {getattr(total, 'shape', type(total))}")
    if not isinstance(terms, dict):
        raise RuntimeError(f"{ENTRY_NAME} terms 는 dict 여야 한다: {type(terms)}")
    for k, v in terms.items():
        if not isinstance(v, torch.Tensor) or tuple(v.shape) != (n,):
            raise RuntimeError(f"{ENTRY_NAME} terms['{k}'] 는 ({n},) 텐서여야 한다: {getattr(v, 'shape', type(v))}")
    return total, terms
