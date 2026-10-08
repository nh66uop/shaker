"""rh_aglt_l — RH56F1 왼팔 한 팔: 먼 출발 접근(Approach) → 잡기(Grasp) → 들기(Lift) → 이송(Transport).

`rh_aglt_r` 의 거울상(XZ 평면 반사, y → −y). 좌우가 다른 곳은 `profile.py`(관절·링크 이름·쉬는 팔)와
`rh_aglt_env_cfg.py`(먼 출발 자세·소환·구멍·종료 상자·목표 박스의 y) 두 파일뿐이다. 나머지 본문은 우 패키지와
바이트가 같고 테스트가 잠근다(`test_shared_files_identical_to_right`).
보상은 `reward_gen/rh_aglt_l/iter_NN/compute_reward.py`(좌우 보상 바이트 동일 원칙).
"""
