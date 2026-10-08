"""manip_stages / cup_pick — 단계별 보상 커리큘럼 프레임워크의 첫 과제(09.20 사용자 결정).

★왜 패키지를 나누나: `grasp_fj_t2r` 의 leaf·로그·체크포인트는 sim2real 대상이라 **동결**한다(사용자 09.20
  "기존 grasping 정책 폴더와 pth 는 냅두고 새 디렉토리"). 여기서는 **env 를 한 줄도 바꾸지 않고** cfg 를 상속만 해
  새 gym id 를 등록한다 — 목적은 로그(`log/rl_games/open-short/<side>/cup-pick/`)와 이력을 기존과 섞지 않는 것뿐이다.
  관측·액션·종료·성공 판정·자산은 전부 `grasp_fj_t2r` 그대로다(액션 26 = 팔 7 증분 + 손 19 절대, 시너지 아님).

★프레임워크 규약(09.20 사용자 "framework 를 기반으로 확장 · 작업에 맞게 디렉토리 이름"): 과제 하나 = 이 패키지의 cfg 한 쌍 +
  `reward_gen/<과제>_<팔>_<단계>` 트랙들 + `log/rl_games/open-short/<팔>/<과제-하이픈>/` + `our_source/policy_zoo/<과제>/<팔>/<단계>/`.
  새 과제(예: cup_pour, shoe_place)는 과제명만 바꿔 같은 골격을 복제한다. 단계 이름은 `approach`(접근·인벨롭) · `manip`(리프트·목표).

★무엇을 실험하나(사용자 09.20 "③ 단일 정책 + 보상 2트랙"): 정책은 **하나**다. 보상만 단계로 갈아 끼우며 이어 받는다.
    1단계 `reward_gen/cup_pick_<side>_approach` — 접근·인벨롭을 가르치는 계열(우 grasp_fj_rand iter_00 = grasp_fj_stage i08).
    2단계 `reward_gen/cup_pick_<side>_manip`       — 리프트·목표를 가르치는 계열(우 grasp_fj_rand iter_01).
  근거(09.20 iter 분석): 우 i00 은 **처음부터** 접근 0.91(e500) → 인벨롭 0.59(e750) 를 가르쳤지만 리프트 0.07 에서 막혔고,
  i01 은 이미 파지하는 정책 위에서 성공 0.85 를 냈지만 좌팔에 처음부터 쓰자 1500 epoch 동안 검지·약지 접촉 0 · 인벨롭 0 이었다.
  단계별 최종 .pth 가 곧 그 단계의 정책이다(사용자: "보상에 따라 최종 pth 가 나오면 그것 자체가 하나의 정책 구분").
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ..grasp_fj_t2r.grasp_fj_t2r_env_cfg import GraspFJT2RRandEnvCfg, GraspFJT2RRandLeftEnvCfg


@configclass
class CupPickRightEnvCfg(GraspFJT2RRandEnvCfg):
    """우팔 — `GraspFJT2RRandEnvCfg` 와 **완전히 같은 과제**. 새 gym id 로 로그만 가른다."""


@configclass
class CupPickLeftEnvCfg(GraspFJT2RRandLeftEnvCfg):
    """좌팔 — `GraspFJT2RRandLeftEnvCfg` 와 완전히 같은 과제. 새 gym id 로 로그만 가른다."""
