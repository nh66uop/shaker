"""이송 목표 — 축대칭 키포인트 목표열 · 허용오차 커리큘럼 · 들기 래치. torch 만 쓴다(Isaac 없이 테스트).

수식은 중립 모듈 `modules/keypoint_goal.py`(cup_grasp 가 쓰는 것과 같은 구현)를 그대로 부른다. 이 파일은
env 한 대의 목표 상태를 묶는다:
  · 첫 목표 = 정착한 컵 + xy U(±first_xy) + z U(first_z) — 컵을 제자리에서 들어 올린다
  · 성공 한 번마다 이전 **목표**에서 ±delta 로 다음 목표(박스 clamp) — 이송
  · 성공 = kp 최대거리 ≤ tol 인 스텝이 success_steps 쌓이고, 그 스텝에 **목표 근처에서 쥐고 있을 것**(밀기·던지기·스친 뒤 쥐기 차단)
  · tol 은 interval 프레임마다 직전 에피소드 성공수 평균 ≥ threshold 면 ×factor(하한 floor)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from openarm.agnostic.modules import keypoint_goal as kg


@dataclass(frozen=True)
class GoalParams:
    half_height: float
    seq: kg.GoalSeqCfg
    tol_start: float
    tol_floor: float
    tol_factor: float
    tol_interval: int
    tol_success_threshold: float
    tol_eval: float = 0.0          # > 0 이면 커리큘럼 없이 고정(play)
    lift_latch_m: float = 0.10


class GoalState:
    """env 전체의 목표·추적기. 모든 텐서는 env-local 위치·wxyz 쿼터니언."""

    def __init__(self, num_envs: int, num_tips: int, params: GoalParams, device) -> None:
        self.p = params
        self.n = int(num_envs)
        self.device = device
        self.offsets = kg.keypoint_offsets(float(params.half_height), device)
        self.goal_pos = torch.zeros(self.n, 3, device=device)
        self.goal_quat = torch.zeros(self.n, 4, device=device)
        self.goal_quat[:, 0] = 1.0
        self.trk = kg.GoalTrackers(self.n, num_tips, device)
        self.curr = kg.ToleranceCurriculum(float(params.tol_start), float(params.tol_floor),
                                           float(params.tol_factor), int(params.tol_interval),
                                           float(params.tol_success_threshold))
        self.spawn_z = torch.zeros(self.n, device=device)
        self.lifted = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.kp_dist = torch.zeros(self.n, device=device)

    @property
    def tol(self) -> float:
        return float(self.p.tol_eval) if float(self.p.tol_eval) > 0.0 else self.curr.tol

    def reset(self, ids: torch.Tensor, cup_pos: torch.Tensor, cup_quat: torch.Tensor) -> None:
        """에피소드 리셋: 첫 목표 샘플, 추적기·래치 초기화."""
        pos, quat = kg.sample_first_goal(cup_pos, cup_quat, self.p.seq)
        self.goal_pos[ids] = pos
        self.goal_quat[ids] = quat
        self.spawn_z[ids] = cup_pos[:, 2]
        self.lifted[ids] = False
        self.trk.full_reset(ids)

    def goal_keypoints(self) -> torch.Tensor:
        return kg.keypoints_world(self.goal_pos, self.goal_quat, self.offsets)

    def cup_keypoints(self, cup_pos: torch.Tensor, cup_quat: torch.Tensor) -> torch.Tensor:
        return kg.keypoints_world(cup_pos, cup_quat, self.offsets)

    def step(self, cup_pos: torch.Tensor, cup_quat: torch.Tensor, grasped: torch.Tensor,
             active: torch.Tensor) -> dict[str, torch.Tensor]:
        """한 스텝 갱신. 성공한 env 는 목표를 전진시킨다. 반환값은 이 스텝(전진 전) 기준."""
        self.lifted |= ((cup_pos[:, 2] - self.spawn_z) > float(self.p.lift_latch_m)) & active
        kp_c = self.cup_keypoints(cup_pos, cup_quat)
        self.kp_dist = kg.keypoint_max_dist(kp_c, self.goal_keypoints())
        near, reached = kg.update_near_goal(self.kp_dist, self.tol, self.trk, self.p.seq)
        near = near & active
        # 성공 = 누적이 찼고 **지금** 목표 근처에서 쥐고 있다. 누적은 줄지 않으므로(force_consecutive False) near 를 빼면
        #   목표를 스쳐 지나간 뒤 멀리서 쥐는 순간이 성공이 된다(09.29 리뷰 H1).
        success = near & reached & grasped
        self.trk.near_goal_steps = torch.where(active, self.trk.near_goal_steps,
                                               torch.zeros_like(self.trk.near_goal_steps))
        # 쥐지 않고 누적이 찼으면 성공이 아니다 — 쥘 때까지 누적은 그대로 두고 판정만 미룬다.
        ids = success.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            self.trk.successes[ids] += 1
            pos, quat = kg.sample_delta_goal(self.goal_pos[ids], self.goal_quat[ids], self.p.seq)
            self.goal_pos[ids] = pos
            self.goal_quat[ids] = quat
            self.trk.clear_goal(ids)
        return {"kp_dist": self.kp_dist, "near_goal": near, "success": success}

    def update_curriculum(self) -> bool:
        if float(self.p.tol_eval) > 0.0:
            return False
        return self.curr.update(self.trk.prev_episode_successes)
