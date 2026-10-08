"""manip_stages 계약 — 09.20 사용자 "③ 단일 정책 + 보상 2트랙" · "기존 grasping 폴더·pth 는 냅두고 새 디렉토리".

이 패키지는 sim2real 대상인 `grasp_fj_t2r` 의 leaf 를 상속해 gym id 와 로그 폴더를 가른다.
★09.22 사용자 결정으로 "env 코드 없음" 규칙이 바뀌었다: `CupPickEnv` 가 부모 env 를 상속하되 **보상 입력만 더한다**
  (물체 모양과 무관한 표면 거리). 관측·액션·종료·리셋·성공 판정 메서드는 덮지 않고, cfg 클래스는 여전히 필드를 더하지 않는다.
"""

from __future__ import annotations

import ast
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
_HDGP = Path(__file__).resolve().parents[7]
_CFG = (_HERE / "cup_pick_cfg.py").read_text(encoding="utf-8")
_ENV = (_HERE / "cup_pick_env.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")
_ROUND = (_HDGP / "scripts" / "reward_gen" / "t2r_fj_round.py").read_text(encoding="utf-8")


def test_package_env_code_is_one_reward_input_subclass_and_cfgs_add_no_fields():
    files = sorted(p.name for p in _HERE.glob("*.py"))
    # ★09.27 사용자 결정("1,3을 같이"): cup_grasp 네 파일 추가(첫마디 보상 입력 + critic 접촉 관측, 새 gym id).
    # ★09.28 사용자 결정("1,2 모두 필요"): `hand_action.py` 추가(cup_grasp 전용 손 액션 사상, cfg 플래그로만 켠다).
    assert files == ["__init__.py", "cup_grasp_cfg.py", "cup_grasp_env.py", "cup_pick_cfg.py", "cup_pick_env.py",
                     "goal_wrench.py", "grasp_context.py", "grasp_validator.py", "hand_action.py", "object_context.py",
                     "object_surface.py", "object_validator.py", "real_delays.py", "surface_query.py"], files
    tree = ast.parse(_CFG)
    classes = {c.name: [b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", "")
                        for b in c.bases] for c in tree.body if isinstance(c, ast.ClassDef)}
    assert classes == {"CupPickRightEnvCfg": ["GraspFJT2RRandEnvCfg"],
                       "CupPickLeftEnvCfg": ["GraspFJT2RRandLeftEnvCfg"]}
    # 과제는 부모와 같아야 한다 — 필드를 덮으면 s2r leaf 와 다른 과제가 된다
    for c in tree.body:
        if isinstance(c, ast.ClassDef):
            assert not [s for s in c.body if isinstance(s, (ast.AnnAssign, ast.Assign))], c.name
            assert not [s for s in c.body if isinstance(s, ast.FunctionDef)], c.name


def test_cup_pick_env_only_adds_reward_inputs():
    """09.22 사용자 결정 — env 서브클래스는 보상 입력만 더한다. 관측·액션·종료·리셋·성공은 부모(동결) 그대로."""
    tree = ast.parse(_ENV)
    envs = [c for c in tree.body if isinstance(c, ast.ClassDef)]
    assert [(c.name, [getattr(b, "id", "") for b in c.bases]) for c in envs] == [("CupPickEnv", ["GraspFJT2REnv"])]
    methods = {f.name for f in envs[0].body if isinstance(f, ast.FunctionDef)}
    # ★09.23 `_apply_wrench` 추가 — 외란 DR 은 물리 **외란**이지 과제 정의가 아니다(관측·액션·종료·성공 판정 불변,
    #   부모 `grasp_fj` 가 이미 가진 훅을 게이트만 좁혀 덮는다). 리셋·종료·성공은 여전히 덮지 않는다.
    assert methods == {"__init__", "_surface_tensors", "_build_context", "_apply_wrench", "_wrench_gate_now",
                       "_log_fabric_metrics", "_update_wrench_adr"}, methods
    for forbidden in ("_get_observations", "_get_dones", "_get_rewards", "_reset_idx", "_progress_reward",
                      "_pre_physics_step", "_apply_action", "_hand_command", "_setup_scene", "_advance_goals",
                      "_latched", "_success_now"):
        assert forbidden not in methods, forbidden
    # 클래스 속성으로 부모 동작을 바꾸지 않는다(cfg 필드·공간 차원 포함)
    assert not [s for s in envs[0].body if isinstance(s, (ast.Assign, ast.AnnAssign))]
    for tok in ("observation_space", "action_space", "self.cfg.", "cfg.__setattr__", "setattr("):
        assert tok not in _ENV, f"{tok} — cfg·공간을 읽거나 바꾸지 않는다(부팅 전 뱅크 이름만 `cfg.object_bank` 로 읽는다)"
    # 보상 ctx 는 부모 ctx 를 그대로 품는다(부모 → 새 필드 추가)
    bc = _ENV[_ENV.index("def _build_context"):_ENV.index("def _log_fabric_metrics")]
    assert "super()._build_context(kw, out)" in bc and "from_base(base," in bc
    lg = _ENV[_ENV.index("def _log_fabric_metrics"):]
    assert "super()._log_fabric_metrics()" in lg
    assert 'ex["task/palm_surface_dist"]' in lg and 'ex["task/link_surface_dist_min_mean"]' in lg
    # 09.23 목표 게이트 외란 — 게이트는 높이 래치 ∧ 목표 근방, 조용한 no-op 방지 로그가 함께 있어야 한다.
    # ★09.25 회귀 방지: 예전 이 줄은 `wrench_gate(self._lifted, ...)` 를 **요구**했다. 그런데 `self._lifted` 는
    #   `fj_core_env._get_rewards` 에서만 갱신되고 이 계통(`fj_kp_env`)은 그 함수를 덮어 `_latched` 에 둔다 → 게이트가
    #   영원히 닫혀 iter_09~14 내내 외란이 한 번도 안 터졌다(ADR 레벨만 1.0). 발화·ADR·로그가 같은 게이트 함수를 쓰고,
    #   그 함수는 ctx(매 스텝 갱신)의 높이 래치를 읽어야 한다.
    aw = _ENV[_ENV.index("def _apply_wrench"):_ENV.index("def _log_fabric_metrics")]
    assert "self._wrench.step(self._obj_mass, self._wrench_gate_now(ctx))" in aw, aw
    assert "self._lifted" not in _ENV.replace("`self._lifted`", ""), "부모 self._lifted 는 이 계통에서 갱신되지 않는다"
    gn = _ENV[_ENV.index("def _wrench_gate_now"):_ENV.index("def _log_fabric_metrics")]
    assert "wrench_gate(ctx.lifted, ctx.goal_dist)" in gn, gn
    ua = _ENV[_ENV.index("def _update_wrench_adr"):]
    assert "gate = self._wrench_gate_now(ctx)" in ua, "ADR·로그 게이트도 같은 함수여야 한다"
    for tag in ('dr/wrench_fire_frac', 'dr/wrench_f_norm_max', 'dr/wrench_adr_level', 'dr/wrench_force_scale'):
        assert f'ex["{tag}"]' in aw, tag
    for tag in ('task/wrench_near_goal_frac', 'dr/wrench_gate_frac', 'task/wrench_goal_dwell_ep'):
        assert f'ex["{tag}"]' in _ENV, tag


def test_gym_ids_and_entry_points():
    assert 'entry_point=f"open-{_tag}_{_task}{_suffix}"' not in _REG
    assert '_ENTRY_CUP_PICK = "openarm.agnostic.tasks.manip_stages.cup_pick_env:CupPickEnv"' in _REG
    assert '_ENTRY_FROZEN = "openarm.agnostic.tasks.grasp_fj_t2r.grasp_fj_t2r_env:GraspFJT2REnv"' in _REG
    # 09.27 cup_grasp(CupGraspEnv) 추가 — cup_pick 과 동결 entry 는 그대로
    assert '_ENTRIES = {"cup_pick": _ENTRY_CUP_PICK, "cup_grasp": _ENTRY_CUP_GRASP, "grasp_fj_ab": _ENTRY_FROZEN}' in _REG
    assert "entry_point=_ENTRIES[_task]" in _REG
    assert '("short_r", "cup_pick"): CupPickRightEnvCfg' in _REG
    assert '("short_l", "cup_pick"): CupPickLeftEnvCfg' in _REG
    # 이름 변경 전 좌팔 런 재생용 별칭(그 런 종료 후 삭제) — 동결 env 그대로
    assert '("short_l", "grasp_fj_ab"): CupPickLeftEnvCfg' in _REG


def test_both_reward_tracks_share_one_run_folder_apart_from_the_s2r_track():
    import sys
    sys.path.insert(0, str(_HDGP / "scripts" / "reward_gen"))
    import t2r_fj_round as R
    a, m = R.track("cup_pick_l_approach"), R.track("cup_pick_l_manip")
    assert a["task"] == m["task"] == "open-short_l_cup_pick-lstm"
    assert a["server_logdir"] == m["server_logdir"] and a["server_logdir"].endswith("/open-short/left/cup-pick")
    # 우팔도 같은 골격으로 미리 등록돼 있다(구조 확정 후 같은 순서로 학습)
    for side in ("r",):
        for stage in ("approach", "manip"):
            t = R.track(f"cup_pick_{side}_{stage}")
            assert t["task"] == "open-short_r_cup_pick-lstm" and t["server_logdir"].endswith("/open-short/right/cup-pick")
    assert a["local_mirror"] == m["local_mirror"] != R.track("grasp_fj_rand_left")["local_mirror"]
    # s2r 트랙은 건드리지 않는다
    assert R.track("grasp_fj_rand")["server_logdir"].endswith("/open-short/right/grasp-fj-t2r-rand")


def test_stage1_reward_is_the_mirrored_right_round1_and_stage2_the_mirrored_round2():
    root = _HDGP / "reward_gen"
    for stage, right in (("cup_pick_l_approach", "grasp_fj_rand/iter_00"),
                         ("cup_pick_l_manip", "grasp_fj_rand/iter_01")):
        left = (root / stage / "iter_00" / "compute_reward.py").read_text(encoding="utf-8").splitlines()
        r = (root / right / "compute_reward.py").read_text(encoding="utf-8").splitlines()
        diff = [(x, y) for x, y in zip(r, left) if x != y]
        assert len(r) == len(left) and len(diff) == 2, (stage, diff)
        assert diff[1] == ("    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))",
                           "    ang_n = torch.acos(torch.clamp(-n[:, 1], -1.0, 1.0))"), stage


def test_left_rewards_are_exactly_what_the_mirror_generator_produces():
    """손으로 고치면 좌우가 갈라진다 — 좌측 보상은 `mirror_reward.py` 출력과 바이트가 같아야 한다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mirror", _HDGP / "scripts" / "reward_gen" / "mirror_reward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = _HDGP / "reward_gen"
    for stage, right in (("cup_pick_l_approach", "grasp_fj_rand/iter_00"),
                         ("cup_pick_l_manip", "grasp_fj_rand/iter_01")):
        src = (root / right / "compute_reward.py").read_text(encoding="utf-8")
        assert mod.mirror(src, "l") == (root / stage / "iter_00" / "compute_reward.py").read_text(encoding="utf-8"), stage
        assert mod.mirror(mod.mirror(src, "l"), "r") == src, stage      # 왕복


def test_stage_catalog_lists_both_stages_with_their_preconditions():
    cat = (_HDGP / "reward_gen" / "STAGES.md").read_text(encoding="utf-8")
    for tok in ("`approach` 계열", "`manip` 계열", "policy_zoo", "mirror_reward.py",
                "처음부터 쓰면 실패한다", "cup_pick_l_approach", "cup_pick_l_manip"):
        assert tok in cat, tok


_GENV = (_HERE / "cup_grasp_env.py").read_text(encoding="utf-8")
_GCFG = (_HERE / "cup_grasp_cfg.py").read_text(encoding="utf-8")


def test_cup_grasp_env_adds_proximal_inputs_and_critic_contact_only():
    """09.27 사용자 결정 — CupGraspEnv 는 첫마디 보상 입력·critic 접촉 관측·로그만 더한다.
    actor 관측·액션·종료·리셋·성공은 그대로(s2r 배포 계약 actor obs 133 · action 26 불변)."""
    tree = ast.parse(_GENV)
    envs = [c for c in tree.body if isinstance(c, ast.ClassDef)]
    assert [(c.name, [getattr(b, "id", "") for b in c.bases]) for c in envs] == [("CupGraspEnv", ["CupPickEnv"])]
    methods = {f.name for f in envs[0].body if isinstance(f, ast.FunctionDef)}
    # ★09.28 사용자 결정: `_hand_targets` 추가 — 액션 0 = 리셋(편 손) 자세. `cfg.hand_action_zero_open` 이 꺼져 있으면
    #   부모 그대로라 이전 런 재생은 같다. 켜면 액션 의미가 바뀐다(차원 불변, 실기 제어기도 같은 사상 필요).
    # ★09.28 사용자 결정("실기 지연도 학습에서 배워둬야", SIME2REAL 인계): 지연 큐 교체(`__init__`·`_install_real_delays`)와
    #   긴 유지(`_advance_goals`, 목표 고정). 모두 cfg 기본값이면 부모와 같다.
    assert methods == {"_setup_scene", "_prox_body_ids", "_prox_cup_forces", "_build_context",
                       "_privileged_blocks", "_log_fabric_metrics", "_hand_targets",
                       "__init__", "_install_real_delays", "_advance_goals",
                       "_log_joint_flexion"}, methods   # ★09.28 관절별 굽힘 국면 로그(로그 전용)
    assert "return super()._advance_goals(is_success)" in _GENV
    assert "return super()._hand_targets(a_hand)" in _GENV and 'getattr(self.cfg, "hand_action_zero_open", False)' in _GENV
    for forbidden in ("_get_observations", "_get_dones", "_get_rewards", "_reset_idx", "_progress_reward",
                      "_pre_physics_step", "_apply_action", "_hand_command", "_proprio_blocks", "_object_blocks",
                      "_action_obs", "_success_now"):
        assert forbidden not in methods, forbidden
    for tok in ("observation_space", "action_space", "state_space"):
        assert tok not in _GENV, tok
    assert "super()._setup_scene()" in _GENV and "super()._build_context(kw, out)" in _GENV
    assert "super()._privileged_blocks(ob) + [" in _GENV, "critic 블록은 부모 뒤에 덧붙이기만 한다"
    assert "super()._log_fabric_metrics()" in _GENV


def test_cup_grasp_cfg_changes_only_critic_width():
    tree = ast.parse(_GCFG)
    classes = [c for c in tree.body if isinstance(c, ast.ClassDef)]
    assert {c.name: [getattr(b, "id", "") for b in c.bases] for c in classes} == {
        "CupGraspRightEnvCfg": ["CupPickRightEnvCfg"], "CupGraspLeftEnvCfg": ["CupPickLeftEnvCfg"]}
    for c in classes:
        # ★09.28: 필드는 `hand_action_zero_open: bool = False` 하나만 허용(기본 꺼짐 = 부모 사상)
        fields = [s for s in c.body if isinstance(s, (ast.AnnAssign, ast.Assign))]
        assert [(ast.unparse(f.target), ast.unparse(f.value)) for f in fields] == [
            ("hand_action_zero_open", "False"), ("real_arm_lag_steps", "(0, 0)"), ("real_hand_lag_steps", "(0, 0)"),
            ("real_obj_lag_steps", "(0, 0)"), ("real_obj_dropout_p", "0.0"), ("real_obj_dropout_steps", "(0, 0)"),
            ("goal_hold_after", "0")], c.name
        fns = [s for s in c.body if isinstance(s, ast.FunctionDef)]
        assert [f.name for f in fns] == ["finalize_after_overrides"], c.name
        body = ast.unparse(fns[0])
        assert "super().finalize_after_overrides()" in body
        assert "self.state_space = self.state_space + CRITIC_CONTACT_DIM" in body
        assert "observation_space" not in body and "action_space" not in body


def test_cup_grasp_registered_with_its_own_log_folder():
    assert '"cup_grasp": _ENTRY_CUP_GRASP' in _REG
    assert '("short_r", "cup_grasp"): CupGraspRightEnvCfg' in _REG and '("short_l", "cup_grasp"): CupGraspLeftEnvCfg' in _REG
