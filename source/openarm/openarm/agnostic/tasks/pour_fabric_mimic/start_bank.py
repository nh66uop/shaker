"""쥔 상태 시작 뱅크 (09.30 사용자: "잡기 이후부터 스타트", "쥔 상태 80 % + 처음 20 %").

두 가지 역할을 한 파일에 둔다.
- 수집: 정책을 재생하는 동안 두 손이 컵을 쥐고 두 컵이 `start_bank_lift_min` 이상 들렸고 소스 컵이 세워져 있고
  비드가 소스 컵 안에 있는 순간의 상태를 env 별로 모아 npz 로 쓴다(`start_bank_capture_path`).
  같은 에피소드에서 `capture_gap` 스텝 간격으로 최대 `capture_per_ep` 번.
- 복원: 리셋되는 env 중 `start_bank_frac` 비율을 뱅크 상태에서 시작한다. hold 구간은 건너뛴다(뱅크 상태는
  물리 롤아웃의 한 순간이라 위치·속도가 서로 맞다). 나머지는 기존 리셋 그대로.

저장 필드(env-local, 모두 같은 첫 축 B):
  joint_pos, joint_vel (B,J) · src_root, rcv_root (B,13, 위치는 env-local) · beads (B,K,13, 위치는 env-local)
  src_arm_q_target, rcv_arm_q_target (B,7) · src_syn_target, rcv_syn_target (B,n) · src_spawn, rcv_spawn (B,3)
"""
from __future__ import annotations

import os

import numpy as np
import torch

_KEYS = ("joint_pos", "joint_vel", "src_root", "rcv_root", "beads",
         "src_arm_q_target", "rcv_arm_q_target", "src_syn_target", "rcv_syn_target", "src_spawn", "rcv_spawn")


class StartBank:
    def __init__(self, env, capture_gap: int = 60, capture_per_ep: int = 4, upright_cos: float = 0.985,
                 save_every: int = 64) -> None:
        cfg = env.cfg
        self.env = env
        self.frac = float(cfg.start_bank_frac)
        self.path = str(cfg.start_bank_path)
        self.cap_path = str(cfg.start_bank_capture_path)
        self.cap_n = int(cfg.start_bank_capture_n)
        self.lift_min = float(cfg.start_bank_lift_min)
        self.capture_gap, self.capture_per_ep = int(capture_gap), int(capture_per_ep)
        self.upright_cos, self.save_every = float(upright_cos), int(save_every)
        if not 0.0 <= self.frac <= 1.0:
            raise ValueError(f"start_bank_frac 는 [0,1]: {self.frac}")
        if self.frac > 0.0 and not self.path:
            raise ValueError("start_bank_frac > 0 인데 start_bank_path 가 비었다")
        self.bank: dict[str, torch.Tensor] | None = None
        if self.frac > 0.0:
            self.bank = self._load(self.path)
            print(f"[pour_fabric_mimic] 쥔 상태 시작 뱅크 {len(self.bank['joint_pos'])} 개 · 리셋의 "
                  f"{self.frac:.0%} 를 뱅크에서 시작 ({self.path})", flush=True)
        n, dev = env.num_envs, env.device
        self._cap: dict[str, list] = {k: [] for k in _KEYS}
        self._cap_count = 0
        self._last_saved = 0
        self._ep_caps = torch.zeros(n, dtype=torch.long, device=dev)
        self.last_idx = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.seq_next: int | None = None                       # None = 무작위(학습). 정수면 그 위치부터 차례로(프로브)
        self._last_cap_step = torch.full((n,), -10 ** 6, dtype=torch.long, device=dev)
        if self.cap_path:
            print(f"[pour_fabric_mimic] 쥔 상태 수집 → {self.cap_path} (목표 {self.cap_n} 개, 두 컵 {self.lift_min} m 이상)",
                  flush=True)

    @property
    def capturing(self) -> bool:
        return bool(self.cap_path) and self._cap_count < self.cap_n

    def _load(self, path: str) -> dict[str, torch.Tensor]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"start_bank_path 없음: {path}")
        z = np.load(path)
        missing = [k for k in _KEYS if k not in z.files]
        if missing:
            raise ValueError(f"뱅크 {path} 에 필드가 없다: {missing}")
        env = self.env
        bank = {k: torch.as_tensor(z[k], dtype=torch.float32, device=env.device) for k in _KEYS}
        b = len(bank["joint_pos"])
        if b == 0 or any(len(v) != b for v in bank.values()):
            raise ValueError(f"뱅크 {path} 의 첫 축 길이가 맞지 않는다")
        if bank["joint_pos"].shape[1] != env.robot.data.joint_pos.shape[1]:
            raise ValueError(f"뱅크 관절 수 {bank['joint_pos'].shape[1]} ≠ 로봇 {env.robot.data.joint_pos.shape[1]}")
        if bank["beads"].shape[1] != int(env.cfg.bead_count):
            raise ValueError(f"뱅크 비드 수 {bank['beads'].shape[1]} ≠ cfg {env.cfg.bead_count}")
        self._check_names(z, path)
        return bank

    def _check_names(self, z, path: str) -> None:
        """★10.01 다른 env(rh_aglt)에서 모은 뱅크는 관절 이름 배열을 함께 싣는다 — 있으면 순서를 이름으로 대조한다."""
        if not any(k.endswith("_names") for k in z.files):
            return                                             # 같은 env 에서 모은 옛 뱅크 — 대조할 이름 없음
        env = self.env
        jn = list(env.robot.data.joint_names)
        checks = [("joint_names", jn)]
        for tag, rig in (("src", env.src), ("rcv", env.rcv)):
            checks += [(f"{tag}_arm_names", [jn[i] for i in rig.arm_ids]),
                       (f"{tag}_hand_names", [jn[i] for i in rig.syn_ids])]
        for key, want in checks:
            if key not in z.files:
                continue
            got = [str(x) for x in z[key]]
            if got != want:
                raise ValueError(f"뱅크 {path} 의 {key} 순서가 env 와 다르다: {got} ≠ {want}")

    # ------------------------------------------------------------------ 수집
    def on_reset(self, env_ids: torch.Tensor) -> None:
        self._ep_caps[env_ids] = 0
        self._last_cap_step[env_ids] = -10 ** 6

    def maybe_capture(self, in_source_frac: torch.Tensor) -> None:
        if not self.capturing:
            return
        env = self.env
        o = env.scene.env_origins
        step = env.episode_length_buf
        lift_s = (env.source_cup.data.root_pos_w - o)[:, 2] - env._src_spawn[:, 2]
        lift_r = (env.receiver_cup.data.root_pos_w - o)[:, 2] - env._rcv_spawn[:, 2]
        upright = env._cup_up(env.source_cup)[:, 2] > self.upright_cos
        ok = (env._src_grasped & env._rcv_grasped & (lift_s > self.lift_min) & (lift_r > self.lift_min)
              & upright & (in_source_frac >= 0.999) & (~env._hold_mask())
              & (self._ep_caps < self.capture_per_ep) & (step - self._last_cap_step >= self.capture_gap))
        ids = ok.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        ids = ids[: max(self.cap_n - self._cap_count, 0)]
        self._ep_caps[ids] += 1
        self._last_cap_step[ids] = step[ids]

        def _root(obj):
            s = obj.data.root_state_w[ids].clone()
            s[:, :3] -= o[ids]
            return s
        beads = env.beads.data.object_state_w[ids].clone()
        beads[:, :, :3] -= o[ids].unsqueeze(1)
        rows = {"joint_pos": env.robot.data.joint_pos[ids], "joint_vel": env.robot.data.joint_vel[ids],
                "src_root": _root(env.source_cup), "rcv_root": _root(env.receiver_cup), "beads": beads,
                "src_arm_q_target": env.src.arm_q_target[ids], "rcv_arm_q_target": env.rcv.arm_q_target[ids],
                "src_syn_target": env.src.syn_target[ids], "rcv_syn_target": env.rcv.syn_target[ids],
                "src_spawn": env._src_spawn[ids], "rcv_spawn": env._rcv_spawn[ids]}
        for k, v in rows.items():
            self._cap[k].append(v.detach().float().cpu().numpy())
        self._cap_count += len(ids)
        if self._cap_count - self._last_saved >= self.save_every or self._cap_count >= self.cap_n:
            self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.cap_path)) or ".", exist_ok=True)
        out = {k: np.concatenate(v, axis=0) for k, v in self._cap.items()}
        np.savez_compressed(self.cap_path, **out)
        self._last_saved = self._cap_count
        print(f"[pour_fabric_mimic] 쥔 상태 뱅크 저장 {self._cap_count}/{self.cap_n} → {self.cap_path}", flush=True)

    # ------------------------------------------------------------------ 복원
    def restore(self, env_ids: torch.Tensor) -> torch.Tensor:
        """env_ids 중 frac 비율을 뱅크 상태로 덮는다. 덮은 env id 를 돌려준다."""
        if self.bank is None or len(env_ids) == 0:
            return env_ids[:0]
        env = self.env
        dev = env.device
        pick = torch.rand(len(env_ids), device=dev) < self.frac
        ids = env_ids[pick]
        if len(ids) == 0:
            return ids
        nb = len(self.bank["joint_pos"])
        if self.seq_next is None:
            b = torch.randint(0, nb, (len(ids),), device=dev)
        else:                                                   # 검사 프로브 전용: 뱅크를 차례로 한 번씩
            b = (torch.arange(len(ids), device=dev) + int(self.seq_next)) % nb
            self.seq_next = int(self.seq_next) + len(ids)
        self.last_idx[ids] = b                                 # 검사 프로브가 뱅크 항목별 유지율을 잰다
        bk = {k: v[b] for k, v in self.bank.items()}
        o = env.scene.env_origins[ids]

        env.robot.write_joint_state_to_sim(bk["joint_pos"], bk["joint_vel"], env_ids=ids)
        env.robot.set_joint_position_target(bk["joint_pos"], env_ids=ids)
        for rig, tag in ((env.src, "src"), (env.rcv, "rcv")):
            rig.arm_q_target[ids] = bk[f"{tag}_arm_q_target"]
            rig.syn_target[ids] = bk[f"{tag}_syn_target"]
            rig.syn_vel[ids] = 0.0
            span = (rig.syn_grip - rig.syn_open).unsqueeze(0)
            safe = torch.where(span.abs() > 1e-6, span, torch.ones_like(span))
            rig.syn_close[ids] = ((rig.syn_target[ids] - rig.syn_open.unsqueeze(0)) / safe).clamp(0.0, 1.0)
            env._prev_arm_qd[rig.role][ids] = bk["joint_vel"][:, rig.arm_t]
        for cup, tag, store in ((env.source_cup, "src", env._src_spawn), (env.receiver_cup, "rcv", env._rcv_spawn)):
            root = bk[f"{tag}_root"].clone()
            root[:, :3] += o
            cup.write_root_state_to_sim(root, env_ids=ids)
            store[ids] = bk[f"{tag}_spawn"]
        beads = bk["beads"].clone()
        beads[:, :, :3] += o.unsqueeze(1)
        env.beads.write_object_state_to_sim(beads, env_ids=ids)
        env.episode_length_buf[ids] = int(env.cfg.hold_steps)       # hold 건너뜀
        return ids
