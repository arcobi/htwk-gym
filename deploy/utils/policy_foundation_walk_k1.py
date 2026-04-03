import json
import os
import glob

import numpy as np
import torch


class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy_path = self._resolve_artifact_path(
                self.cfg["policy"]["policy_path"],
                "*.pt",
            )
            self.metadata_path = self._resolve_artifact_path(
                self.cfg["policy"].get("metadata_path"),
                "*.metadata.json",
                allow_missing=True,
            )
            self.policy = torch.jit.load(self.policy_path)
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._load_metadata()
        self._init_inference_variables()

    def _resolve_artifact_path(self, configured_path, suffix_pattern, allow_missing=False):
        if configured_path and os.path.exists(configured_path):
            return configured_path

        search_roots = [
            configured_path,
            os.path.join("deploy", configured_path) if configured_path else None,
            os.path.join("..", configured_path) if configured_path else None,
        ]
        for candidate in search_roots:
            if candidate and os.path.exists(candidate):
                return candidate

        policy_name = self.cfg.get("policy", {}).get("metadata_path", "")
        policy_name = os.path.splitext(os.path.basename(policy_name))[0].replace(".metadata", "")
        task_glob = os.path.join("..", "logs", "K1", "K1", "Foundation_Walk_K1", "**", suffix_pattern)
        matches = sorted(glob.glob(task_glob, recursive=True), key=os.path.getmtime)
        if policy_name:
            matches = [match for match in matches if policy_name in os.path.basename(match)] or matches
        if matches:
            return matches[-1]

        if allow_missing:
            return configured_path
        raise FileNotFoundError(f"Could not resolve artifact path for {configured_path}")

    def _load_metadata(self):
        self.metadata = {}
        metadata_path = self.metadata_path
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.command_world = np.zeros(4, dtype=np.float32)
        self.smoothed_travel_world = np.zeros(2, dtype=np.float32)
        self.resolved_core = np.zeros(4, dtype=np.float32)
        self.advanced = np.zeros(6, dtype=np.float32)
        self.resolved_advanced = np.zeros(6, dtype=np.float32)
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        self.start_index = self.cfg["common"]["joint_cnt"] - self.cfg["policy"]["num_actions"]
        self.gait_frequency = 0.0
        self.gait_process = 0.0
        self.desired_yaw_rate = 0.0

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _compute_default_drive(self, travel_vx_world, travel_vy_world, desired_yaw_rate, drive):
        if drive is not None:
            return float(np.clip(drive, 0.0, 1.0))
        lin_speed = np.linalg.norm([travel_vx_world, travel_vy_world])
        max_xy = max(
            abs(self.cfg["policy"]["command_ranges"]["travel_vel_x_world"][0]),
            abs(self.cfg["policy"]["command_ranges"]["travel_vel_x_world"][1]),
            abs(self.cfg["policy"]["command_ranges"]["travel_vel_y_world"][0]),
            abs(self.cfg["policy"]["command_ranges"]["travel_vel_y_world"][1]),
            1.0e-6,
        )
        yaw_drive = abs(desired_yaw_rate) / max(self.cfg["policy"]["adapter"]["max_ang_vel_yaw"], 1.0e-6)
        lin_drive = lin_speed / max_xy
        return float(np.clip(max(lin_drive, yaw_drive), 0.0, 1.0))

    def _resolve_advanced_targets(self, local_vx, local_vy, heading_error, drive, advanced_modifiers, advanced_active):
        adapter = self.cfg["policy"]["adapter"]
        self.desired_yaw_rate = float(
            np.clip(
                heading_error * adapter["heading_gain"],
                -adapter["max_ang_vel_yaw"],
                adapter["max_ang_vel_yaw"],
            )
        )
        drive = max(drive, abs(self.desired_yaw_rate) / max(adapter["max_ang_vel_yaw"], 1.0e-6))
        drive = 0.0 if drive < adapter["stand_drive_threshold"] else np.clip(drive, 0.0, 1.0)

        default_pitch = np.clip(
            local_vx * adapter["body_pitch_gain"],
            self.cfg["policy"]["advanced_ranges"]["body_pitch_target"][0],
            self.cfg["policy"]["advanced_ranges"]["body_pitch_target"][1],
        )
        default_roll = np.clip(
            local_vy * adapter["body_roll_gain"],
            self.cfg["policy"]["advanced_ranges"]["body_roll_target"][0],
            self.cfg["policy"]["advanced_ranges"]["body_roll_target"][1],
        )
        default_stance = np.clip(
            adapter["stance_width_nominal"]
            + abs(local_vy) * adapter["stance_width_from_lateral_gain"]
            + abs(heading_error) * adapter["stance_width_from_yaw_gain"],
            self.cfg["policy"]["advanced_ranges"]["stance_width_target"][0],
            self.cfg["policy"]["advanced_ranges"]["stance_width_target"][1],
        )
        default_foot_yaw = np.clip(
            self.desired_yaw_rate * adapter["foot_yaw_from_yaw_gain"],
            adapter["foot_yaw_target_clip"][0],
            adapter["foot_yaw_target_clip"][1],
        )
        gait_frequency = adapter["idle_gait_frequency"] + drive * (adapter["active_gait_frequency"] - adapter["idle_gait_frequency"])
        gait_frequency += advanced_modifiers[0]
        if drive <= 0.0:
            gait_frequency = 0.0
        else:
            gait_frequency = np.clip(gait_frequency, adapter["gait_frequency_clip"][0], adapter["gait_frequency_clip"][1])

        foot_yaw_left = np.clip(default_foot_yaw + advanced_modifiers[1], adapter["foot_yaw_target_clip"][0], adapter["foot_yaw_target_clip"][1])
        foot_yaw_right = np.clip(default_foot_yaw + advanced_modifiers[2], adapter["foot_yaw_target_clip"][0], adapter["foot_yaw_target_clip"][1])

        self.command_world[3] = drive
        return np.array(
            [
                gait_frequency,
                foot_yaw_left,
                foot_yaw_right,
                advanced_modifiers[3] if advanced_active else default_pitch,
                advanced_modifiers[4] if advanced_active else default_roll,
                advanced_modifiers[5] if advanced_active else default_stance,
            ],
            dtype=np.float32,
        )

    def inference(
        self,
        time_now,
        dof_pos,
        dof_vel,
        base_ang_vel,
        projected_gravity,
        base_yaw,
        travel_vx_world,
        travel_vy_world,
        desired_heading_world,
        drive=None,
        advanced_modifiers=None,
    ):
        self.command_world[0] = travel_vx_world
        self.command_world[1] = travel_vy_world
        self.command_world[2] = desired_heading_world
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_travel_world += np.clip(self.command_world[0:2] - self.smoothed_travel_world, *clip_range)

        heading_error = self._wrap_to_pi(desired_heading_world - base_yaw)
        cos_yaw = np.cos(base_yaw)
        sin_yaw = np.sin(base_yaw)
        local_vx = cos_yaw * self.smoothed_travel_world[0] + sin_yaw * self.smoothed_travel_world[1]
        local_vy = -sin_yaw * self.smoothed_travel_world[0] + cos_yaw * self.smoothed_travel_world[1]
        desired_yaw_rate = np.clip(
            heading_error * self.cfg["policy"]["adapter"]["heading_gain"],
            -self.cfg["policy"]["adapter"]["max_ang_vel_yaw"],
            self.cfg["policy"]["adapter"]["max_ang_vel_yaw"],
        )
        default_drive = self._compute_default_drive(
            self.smoothed_travel_world[0],
            self.smoothed_travel_world[1],
            desired_yaw_rate,
            drive,
        )

        advanced_active = advanced_modifiers is not None
        if advanced_modifiers is None:
            self.advanced[:] = 0.0
        else:
            self.advanced[:] = np.asarray(advanced_modifiers, dtype=np.float32)
        self.resolved_advanced[:] = self._resolve_advanced_targets(local_vx, local_vy, heading_error, default_drive, self.advanced, advanced_active)
        self.gait_frequency = self.resolved_advanced[0]
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0) if self.gait_frequency > 1.0e-8 else 0.0

        self.resolved_core[:] = np.array(
            [
                local_vx,
                local_vy,
                np.sin(heading_error),
                np.cos(heading_error),
            ],
            dtype=np.float32,
        )

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6:8] = self.resolved_core[0:2] * self.cfg["policy"]["normalization"]["lin_vel"]
        self.obs[8:10] = self.resolved_core[2:4]
        self.obs[10:16] = self.resolved_advanced * np.array(
            [
                self.cfg["policy"]["normalization"]["gait_frequency"],
                self.cfg["policy"]["normalization"]["foot_yaw"],
                self.cfg["policy"]["normalization"]["foot_yaw"],
                self.cfg["policy"]["normalization"]["body_pitch_target"],
                self.cfg["policy"]["normalization"]["body_roll_target"],
                self.cfg["policy"]["normalization"]["stance_width_target"],
            ],
            dtype=np.float32,
        )
        self.obs[16] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[17] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[18:30] = (dof_pos - self.default_dof_pos)[self.start_index:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[30:42] = dof_vel[self.start_index:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[42:54] = self.actions

        self.actions[:] = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()[0]
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[self.start_index:] += self.cfg["policy"]["control"]["action_scale"] * self.actions
        return self.dof_targets
