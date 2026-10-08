"""쥔 상태 수집 (10.01 사용자: "aglt i10 파지를 기준으로, 파지·붓기 정책을 둘로 나누고 붓기는 100 % 뱅크에서").

aglt 정책을 재생하는 동안 손이 컵을 쥐고(env grasp 플래그) 컵이 스폰보다 `grasp_capture_lift_min` 이상 들렸고
세워져 있고(비드가 있으면 전부 컵 안) hold 가 끝난 순간의 상태를 env 별로 모아 npz 로 쓴다.
같은 에피소드에서 `capture_gap` 스텝 간격으로 최대 `capture_per_ep` 번.

붓기 env(pour_fabric_mimic) 로의 변환은 `scripts/tools/build_pour_handoff_bank.py` 가 한다 — 이 파일은
다른 과제 패키지를 import 하지 않는다(rh_aglt 계약). 위치는 모두 env-local, 이름 배열을 같이 저장해 변환기가
관절 순서를 이름으로 맞춘다.
"""
from __future__ import annotations

import os

import numpy as np
import torch


class GraspCapture:
    def __init__(self, env, capture_gap: int = 60, capture_per_ep: int = 4, upright_cos: float = 0.985,
                 save_every: int = 128) -> None:
        cfg = env.cfg
        self.env = env
        self.path = str(cfg.grasp_capture_path)
        self.cap_n = int(cfg.grasp_capture_n)
        self.lift_min = float(cfg.grasp_capture_lift_min)
        self.capture_gap, self.capture_per_ep = int(capture_gap), int(capture_per_ep)
        self.upright_cos, self.save_every = float(upright_cos), int(save_every)
        if self.cap_n <= 0:
            raise ValueError(f"grasp_capture_n 는 양수: {self.cap_n}")
        n, dev = env.num_envs, env.device
        self._rows: dict[str, list] = {}
        self._count = 0
        self._last_saved = 0
        self._ep_caps = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_step = torch.full((n,), -10 ** 6, dtype=torch.long, device=dev)
        jn = list(env.robot.data.joint_names)
        self._names = {
            "joint_names": np.array(jn),
            "arm_names": np.array([jn[i] for i in env.arm_ids]),
            "hand_names": np.array(list(env.profile.hand_joint_names)),
        }
        print(f"[rh_aglt] 쥔 상태 수집 → {self.path} (목표 {self.cap_n} 개, 들림 ≥ {self.lift_min} m, "
              f"비드 {int(cfg.bead_count)})", flush=True)

    @property
    def done(self) -> bool:
        return self._count >= self.cap_n

    def on_reset(self, env_ids: torch.Tensor) -> None:
        self._ep_caps[env_ids] = 0
        self._last_step[env_ids] = -10 ** 6

    def beads_inside(self) -> torch.Tensor:
        """(N,) 비드가 전부 컵 안(컵 축 기준 반지름 < 외경, 높이 바닥~입구)이면 True. 비드 없으면 True."""
        env = self.env
        if getattr(env, "beads", None) is None:
            return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        cfg = env.cfg
        rel = env.beads.data.object_pos_w - env.cup.data.root_pos_w.unsqueeze(1)          # (N,K,3)
        up = env._cup_up().unsqueeze(1)
        h = (rel * up).sum(dim=-1)
        rad = (rel - h.unsqueeze(-1) * up).norm(dim=-1)
        ok = (rad < float(cfg.cup_radius)) & (h > float(cfg.cup_bottom_z)) & (h < float(cfg.cup_mouth_z))
        return ok.all(dim=1)

    def step(self, grasped: torch.Tensor, active: torch.Tensor) -> None:
        if self.done:
            return
        env = self.env
        lift = env._cup_pos()[:, 2] - env._spawn[:, 2]
        upright = env._cup_up()[:, 2] > self.upright_cos
        t = env.episode_length_buf
        ok = (grasped & active & (lift > self.lift_min) & upright & self.beads_inside()
              & (self._ep_caps < self.capture_per_ep) & (t - self._last_step >= self.capture_gap))
        ids = ok.nonzero(as_tuple=False).squeeze(-1)[: max(self.cap_n - self._count, 0)]
        if len(ids) == 0:
            return
        self._ep_caps[ids] += 1
        self._last_step[ids] = t[ids]
        o = env.scene.env_origins[ids]
        cup = env.cup.data.root_state_w[ids].clone()
        cup[:, :3] -= o
        rows = {"joint_pos": env.robot.data.joint_pos[ids], "joint_vel": env.robot.data.joint_vel[ids],
                "cup_root": cup, "spawn": env._spawn[ids],
                "arm_q_target": env.arm_q_target[ids], "hand_target": env.hand_target[ids],
                "palm_pos": env._palm_pos()[ids], "lift": lift[ids], "step": t[ids].float()}
        if getattr(env, "beads", None) is not None:
            b = env.beads.data.object_state_w[ids].clone()
            b[:, :, :3] -= o.unsqueeze(1)
            rows["beads"] = b
        for k, v in rows.items():
            self._rows.setdefault(k, []).append(v.detach().float().cpu().numpy())
        self._count += len(ids)
        if self._count - self._last_saved >= self.save_every or self.done:
            self.save()

    def save(self) -> None:
        if self._count == 0:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        out = {k: np.concatenate(v, axis=0) for k, v in self._rows.items()}
        np.savez_compressed(self.path, **out, **self._names)
        self._last_saved = self._count
        print(f"[rh_aglt] 쥔 상태 저장 {self._count}/{self.cap_n} → {self.path}", flush=True)
