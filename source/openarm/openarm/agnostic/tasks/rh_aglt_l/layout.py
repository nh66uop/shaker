"""관측 배치 — 이름과 폭의 단일 출처. cfg 가 차원을 여기서 계산하고 env 가 조립할 때 칸마다 폭을 대조한다.

actor 는 실기에서 얻을 수 있는 값만 둔다(09.14 사용자 규칙 "관측·actor 는 sim2real 가능한 구조로만"):
  관절 각도·속도(엔코더), 손 관절 각도, FK(손바닥·손끝), 컵 pose(FP++ 지각 — 지연·잡음), 손끝 촉각(TouchData1),
  팔 지령 q*(정책 노드 내부 상태), 직전 액션. 손 관절속도는 뺐다(실기 드라이버 velocity ≠ 관절속도, 09.07).
critic 은 actor(참값) 뒤에 시뮬 전용 값을 붙인다.
"""

from __future__ import annotations


def actor_layout(a: int, h: int, f: int) -> tuple[tuple[str, int], ...]:
    return (
        ("arm_q", a), ("arm_qd", a), ("arm_q_target", a),
        ("hand_q", h), ("hand_err", h),
        ("palm_pos", 3), ("palm_rot", 6),
        ("cup_rel_palm", 3), ("cup_up", 3), ("tips_rel_cup", 3 * f),
        ("kp_rel_goal", 12), ("goal_rel_palm", 3),
        ("tactile", f),
        ("prev_action", a + h),
    )


def critic_extra_layout(a: int, h: int, f: int) -> tuple[tuple[str, int], ...]:
    return (
        ("hand_qd", h),
        ("cup_lin_vel", 3), ("cup_ang_vel", 3), ("palm_lin_vel", 3),
        ("contact_prox", f), ("contact_tip", f), ("contact_palm", 1),
        ("lifted", 1), ("grasped", 1), ("progress", 1),
        ("goal_successes", 1), ("kp_dist", 1), ("near_goal_steps", 1),
    )


def layout_width(layout: tuple[tuple[str, int], ...]) -> int:
    return sum(int(w) for _, w in layout)
