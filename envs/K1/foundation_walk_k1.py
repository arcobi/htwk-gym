import os

from isaacgym import gymapi, gymtorch

assert gymtorch

import numpy as np
import torch
from isaacgym.torch_utils import (
    get_axis_params,
    get_euler_xyz,
    quat_from_euler_xyz,
    quat_rotate,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)

from envs.base_task import BaseTask
from utils.utils import apply_randomization


class FoundationWalkK1(BaseTask):

    COMMAND_GROUP_STAND = 0
    COMMAND_GROUP_AXIAL = 1
    COMMAND_GROUP_DIAGONAL = 2
    COMMAND_GROUP_YAW = 3
    COMMAND_GROUP_MIXED = 4

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        self.update_training_curriculum(0)

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        if "effort_limit" in self.cfg["control"]:
            for i in range(self.num_dofs):
                for name in self.cfg["control"]["effort_limit"].keys():
                    if name in self.dof_names[i]:
                        self.torque_limits[i] = self.cfg["control"]["effort_limit"][name]
                        dof_props_asset["effort"][i] = self.cfg["control"]["effort_limit"][name]
                        break

        if "velocity_limit" in self.cfg["control"]:
            for i in range(self.num_dofs):
                for name in self.cfg["control"]["velocity_limit"].keys():
                    if name in self.dof_names[i]:
                        self.dof_vel_limits[i] = self.cfg["control"]["velocity_limit"][name]
                        dof_props_asset["velocity"][i] = self.cfg["control"]["velocity_limit"][name]
                        break

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        if "armature" in self.cfg["control"]:
            for i in range(self.num_dofs):
                for name in self.cfg["control"]["armature"].keys():
                    if name in self.dof_names[i]:
                        dof_props_asset["armature"][i] = self.cfg["control"]["armature"][name]
                        break

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {"rew_terms": {}}

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        self.common_step_counter = 0
        self.debug_termination = bool(self.cfg.get("basic", {}).get("debug_termination", False))
        self.debug_termination_interval = max(1, int(self.cfg.get("basic", {}).get("debug_termination_interval", 100)))
        self.debug_termination_max_envs = max(1, int(self.cfg.get("basic", {}).get("debug_termination_max_envs", 5)))
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

        commands_cfg = self.cfg["commands"]
        self.core_commands = torch.zeros(self.num_envs, commands_cfg["num_core_commands"], dtype=torch.float, device=self.device)
        self.advanced_commands = torch.zeros(self.num_envs, commands_cfg["num_advanced_commands"], dtype=torch.float, device=self.device)
        self.command_targets = torch.zeros(
            self.num_envs,
            commands_cfg["num_core_commands"] + commands_cfg["num_advanced_commands"],
            dtype=torch.float,
            device=self.device,
        )
        self.command_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.advanced_command_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.external_command_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.desired_heading_world = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.desired_yaw_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.foot_yaw_targets = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.body_pitch_target = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.body_roll_target = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stance_width_target = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.command_drive = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()

        self.curriculum_prob = torch.zeros(1, 1, dtype=torch.float, device=self.device)
        self.curriculum_prob[0, 0] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        self.training_phase_index = 0
        self.training_phase_progress = 0.0
        self.current_command_group_weights = torch.ones(5, dtype=torch.float, device=self.device) / 5.0
        self.current_modifier_probability = float(commands_cfg["advanced"]["modifier_probability"])
        self.current_core_ranges = {key: list(value) for key, value in commands_cfg["core"].items()}
        self.current_resampling_time = list(commands_cfg["resampling_time_s"])
        self.current_disturbance_scale = 1.0
        self.eval_cfg = commands_cfg.get("eval", {})
        self.eval_cases = list(self.eval_cfg.get("cases", []))
        self.eval_cycle = bool(self.eval_cfg.get("cycle", True))
        self.eval_enabled_in_play = bool(self.eval_cfg.get("enabled_in_play", False))

        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_pitch = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw_rel = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.last_base_pos = self.base_pos.clone()
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.height_terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics = {
            "heading_error_abs": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "vx_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "vy_tracking_error": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "stance_width_mean": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "feet_x_offset_abs": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "swing_air_ratio": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "reset_height_fraction": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "command_progress": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "stance_slip_mean": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "step_clearance_peak": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "double_support_ratio": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        self.metric_sum_heading_error_abs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sum_vx_tracking_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sum_vy_tracking_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sum_stance_width_mean = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sum_feet_x_offset_abs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sum_swing_air_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sum_command_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sum_stance_slip_mean = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_step_clearance_peak = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_double_support_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_swing_step_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_sample_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

    def _prepare_reward_function(self):
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name in self.reward_scales.keys():
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

    def update_training_curriculum(self, iteration):
        phases = self.cfg["commands"].get("training_phases", [])
        if not phases:
            return
        active_idx = 0
        for i, phase in enumerate(phases):
            if iteration >= int(phase["start_iteration"]):
                active_idx = i
        phase = phases[active_idx]
        weights = torch.tensor(phase["group_weights"], dtype=torch.float, device=self.device)
        weights = torch.clamp(weights, min=0.0)
        if float(weights.sum().item()) <= 0.0:
            weights[:] = 1.0
        self.current_command_group_weights = weights / weights.sum()
        self.current_modifier_probability = float(phase.get("modifier_probability", self.cfg["commands"]["advanced"]["modifier_probability"]))
        self.current_core_ranges = {key: list(value) for key, value in self.cfg["commands"]["core"].items()}
        for key, value in phase.get("core", {}).items():
            self.current_core_ranges[key] = list(value)
        self.current_resampling_time = list(phase.get("resampling_time_s", self.cfg["commands"]["resampling_time_s"]))
        self.current_disturbance_scale = float(phase.get("disturbance_scale", 1.0))
        self.training_phase_index = active_idx
        if active_idx < len(phases) - 1:
            next_iter = int(phases[active_idx + 1]["start_iteration"])
            span = max(next_iter - int(phase["start_iteration"]), 1)
            self.training_phase_progress = float(np.clip((iteration - int(phase["start_iteration"])) / span, 0.0, 1.0))
        else:
            self.training_phase_progress = 1.0

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_commands(force_all=True)
        self.last_actions[:] = 0.0
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_base_pos[:] = self.base_pos
        self._emit_terminal_metrics(torch.zeros(0, dtype=torch.long, device=self.device))
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.command_resample_time[env_ids] = 0
        self.gait_frequency[env_ids] = 0.0
        self.gait_process[env_ids] = 0.0
        self.desired_heading_world[env_ids] = 0.0
        self.heading_error[env_ids] = 0.0
        self.desired_yaw_rate[env_ids] = 0.0
        self.last_base_pos[env_ids] = self.root_states[env_ids, 0:3]
        self.metric_sum_heading_error_abs[env_ids] = 0.0
        self.metric_sum_vx_tracking_error[env_ids] = 0.0
        self.metric_sum_vy_tracking_error[env_ids] = 0.0
        self.metric_sum_stance_width_mean[env_ids] = 0.0
        self.metric_sum_feet_x_offset_abs[env_ids] = 0.0
        self.metric_sum_swing_air_ratio[env_ids] = 0.0
        self.metric_sum_command_progress[env_ids] = 0.0
        self.metric_sum_stance_slip_mean[env_ids] = 0.0
        self.metric_step_clearance_peak[env_ids] = 0.0
        self.metric_double_support_count[env_ids] = 0.0
        self.metric_swing_step_count[env_ids] = 0.0
        self.metric_sample_count[env_ids] = 0.0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        if self._play_eval_enabled():
            self.dof_pos[env_ids] = self.default_dof_pos
        else:
            self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        if not self._play_eval_enabled():
            self.root_states[env_ids, :2] = apply_randomization(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])
        yaw = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        if not self._play_eval_enabled():
            yaw = torch.rand(len(env_ids), device=self.device) * (2 * torch.pi)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            yaw,
        )
        if self._play_eval_enabled():
            self.root_states[env_ids, 7:9] = 0.0
        else:
            self.root_states[env_ids, 7:9] = apply_randomization(
                torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("init_base_lin_vel_xy"),
            )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def set_locomotion_command(self, core_commands, advanced_modifiers=None, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if not torch.is_tensor(core_commands):
            core_commands = torch.tensor(core_commands, dtype=torch.float, device=self.device)
        core_commands = core_commands.to(self.device, dtype=torch.float)
        if core_commands.ndim == 1:
            core_commands = core_commands.unsqueeze(0).expand(len(env_ids), -1)
        if core_commands.shape[-1] == 3:
            self.core_commands[env_ids, 0:3] = core_commands
            self.core_commands[env_ids, 3] = 0.0
        elif core_commands.shape[-1] == 4:
            self.core_commands[env_ids] = core_commands
        else:
            raise ValueError("FoundationWalkK1 expects 3 public core commands or 4 internal core commands")
        self.desired_heading_world[env_ids] = self.core_commands[env_ids, 2]

        if advanced_modifiers is None:
            self.advanced_commands[env_ids] = 0.0
            self.advanced_command_active[env_ids] = False
        else:
            if not torch.is_tensor(advanced_modifiers):
                advanced_modifiers = torch.tensor(advanced_modifiers, dtype=torch.float, device=self.device)
            advanced_modifiers = advanced_modifiers.to(self.device, dtype=torch.float)
            if advanced_modifiers.ndim == 1:
                advanced_modifiers = advanced_modifiers.unsqueeze(0).expand(len(env_ids), -1)
            self.advanced_commands[env_ids] = advanced_modifiers
            self.advanced_command_active[env_ids] = True

        self.external_command_active[env_ids] = True
        self._resolve_commands(env_ids)

    def clear_locomotion_command(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.external_command_active[env_ids] = False
        self.advanced_command_active[env_ids] = False
        self.advanced_commands[env_ids] = 0.0

    def _play_eval_enabled(self):
        return getattr(self, "is_play", False) and self.eval_enabled_in_play and len(self.eval_cases) > 0

    def _sample_range(self, env_count, key):
        cfg = self.current_core_ranges[key]
        return torch_rand_float(cfg[0], cfg[1], (env_count, 1), device=self.device).squeeze(-1)

    def _sample_modifier_range(self, env_count, key):
        cfg = self.cfg["commands"]["advanced"][key]
        return torch_rand_float(cfg[0], cfg[1], (env_count, 1), device=self.device).squeeze(-1)

    def _resample_commands(self, force_all=False):
        if getattr(self, "manual_control", False):
            return
        if self._play_eval_enabled():
            self._resample_play_commands()
            return
        if force_all:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = (self.episode_length_buf == self.command_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        controlled_envs = env_ids[self.external_command_active[env_ids]]
        if len(controlled_envs) > 0:
            self.command_resample_time[controlled_envs] += self._sample_resample_steps(len(controlled_envs))
            self._resolve_commands(controlled_envs)

        env_ids = env_ids[~self.external_command_active[env_ids]]
        if len(env_ids) == 0:
            return

        group_ids = torch.multinomial(self.current_command_group_weights, len(env_ids), replacement=True)
        self.core_commands[env_ids] = 0.0
        self.advanced_commands[env_ids] = 0.0
        self.advanced_command_active[env_ids] = False
        stand_mask = group_ids == self.COMMAND_GROUP_STAND
        axial_mask = group_ids == self.COMMAND_GROUP_AXIAL
        diagonal_mask = group_ids == self.COMMAND_GROUP_DIAGONAL
        yaw_mask = group_ids == self.COMMAND_GROUP_YAW
        mixed_mask = group_ids == self.COMMAND_GROUP_MIXED

        if axial_mask.any():
            ids = env_ids[axial_mask]
            x_span = abs(self.current_core_ranges["travel_vel_x_world"][1] - self.current_core_ranges["travel_vel_x_world"][0])
            y_span = abs(self.current_core_ranges["travel_vel_y_world"][1] - self.current_core_ranges["travel_vel_y_world"][0])
            if x_span <= 1.0e-6 and y_span > 1.0e-6:
                axis_choice = torch.ones(len(ids), dtype=torch.long, device=self.device)
            elif y_span <= 1.0e-6:
                axis_choice = torch.zeros(len(ids), dtype=torch.long, device=self.device)
            else:
                axis_choice = torch.randint(0, 2, (len(ids),), device=self.device)
            lin_x = self._sample_range(len(ids), "travel_vel_x_world")
            lin_y = self._sample_range(len(ids), "travel_vel_y_world")
            self.core_commands[ids, 0] = torch.where(axis_choice == 0, lin_x, torch.zeros_like(lin_x))
            self.core_commands[ids, 1] = torch.where(axis_choice == 1, lin_y, torch.zeros_like(lin_y))
            self.core_commands[ids, 2] = self._sample_heading_world(ids)
            self.core_commands[ids, 3] = torch_rand_float(
                self.current_core_ranges["drive"][0],
                self.current_core_ranges["drive"][1],
                (len(ids), 1),
                device=self.device,
            ).squeeze(-1)

        if diagonal_mask.any():
            ids = env_ids[diagonal_mask]
            self.core_commands[ids, 0] = self._sample_range(len(ids), "travel_vel_x_world")
            self.core_commands[ids, 1] = self._sample_range(len(ids), "travel_vel_y_world")
            self.core_commands[ids, 2] = self._sample_heading_world(ids)
            self.core_commands[ids, 3] = torch_rand_float(
                self.current_core_ranges["drive"][0],
                self.current_core_ranges["drive"][1],
                (len(ids), 1),
                device=self.device,
            ).squeeze(-1)

        if yaw_mask.any():
            ids = env_ids[yaw_mask]
            heading_offset = self._sample_range(len(ids), "heading_offset")
            self.core_commands[ids, 2] = self._wrap_to_pi(self._get_base_yaw()[ids] + heading_offset)
            self.core_commands[ids, 3] = torch_rand_float(
                self.current_core_ranges["turn_drive"][0],
                self.current_core_ranges["turn_drive"][1],
                (len(ids), 1),
                device=self.device,
            ).squeeze(-1)

        if mixed_mask.any():
            ids = env_ids[mixed_mask]
            self.core_commands[ids, 0] = self._sample_range(len(ids), "travel_vel_x_world")
            self.core_commands[ids, 1] = self._sample_range(len(ids), "travel_vel_y_world")
            self.core_commands[ids, 2] = self._sample_heading_world(ids)
            self.core_commands[ids, 3] = torch_rand_float(
                self.current_core_ranges["drive"][0],
                self.current_core_ranges["drive"][1],
                (len(ids), 1),
                device=self.device,
            ).squeeze(-1)

        if stand_mask.any():
            ids = env_ids[stand_mask]
            self.core_commands[ids] = 0.0
            self.core_commands[ids, 2] = self._get_base_yaw()[ids]

        modifier_active = torch.rand(len(env_ids), device=self.device) < self.current_modifier_probability
        modifier_ids = env_ids[modifier_active]
        if len(modifier_ids) > 0:
            self.advanced_command_active[modifier_ids] = True
            self.advanced_commands[modifier_ids, 0] = self._sample_modifier_range(len(modifier_ids), "gait_frequency_bias")
            self.advanced_commands[modifier_ids, 1] = self._sample_modifier_range(len(modifier_ids), "foot_yaw_bias_left")
            self.advanced_commands[modifier_ids, 2] = self._sample_modifier_range(len(modifier_ids), "foot_yaw_bias_right")
            self.advanced_commands[modifier_ids, 3] = self._sample_modifier_range(len(modifier_ids), "body_pitch_target")
            self.advanced_commands[modifier_ids, 4] = self._sample_modifier_range(len(modifier_ids), "body_roll_target")
            self.advanced_commands[modifier_ids, 5] = self._sample_modifier_range(len(modifier_ids), "stance_width_target")

        self.command_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))
        self.desired_heading_world[env_ids] = self.core_commands[env_ids, 2]
        self._resolve_commands(env_ids)

    def _resample_play_commands(self):
        total_duration = 0.0
        for case in self.eval_cases:
            total_duration += max(float(case.get("duration_s", 0.0)), self.dt)
        if total_duration <= 0.0:
            return

        current_time = self.common_step_counter * self.dt
        time_in_cycle = current_time % total_duration if self.eval_cycle else min(current_time, total_duration - self.dt)
        elapsed = 0.0
        active_case = self.eval_cases[-1]
        for case in self.eval_cases:
            elapsed += max(float(case.get("duration_s", 0.0)), self.dt)
            if time_in_cycle < elapsed:
                active_case = case
                break

        core_commands = active_case.get("core", [0.0, 0.0, 0.0])
        advanced_modifiers = active_case.get("advanced")
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.set_locomotion_command(core_commands, advanced_modifiers=advanced_modifiers, env_ids=env_ids)

    def _resolve_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        core = self.core_commands[env_ids]
        advanced = self.advanced_commands[env_ids]
        adapter_cfg = self.cfg["commands"]["adapter"]
        base_yaw = self._get_base_yaw()[env_ids]
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        local_vx = cos_yaw * core[:, 0] + sin_yaw * core[:, 1]
        local_vy = -sin_yaw * core[:, 0] + cos_yaw * core[:, 1]
        heading_error = self._wrap_to_pi(core[:, 2] - base_yaw)
        desired_yaw_rate = torch.clamp(
            heading_error * float(adapter_cfg["heading_gain"]),
            min=-float(adapter_cfg["max_ang_vel_yaw"]),
            max=float(adapter_cfg["max_ang_vel_yaw"]),
        )

        max_xy = max(
            abs(self.current_core_ranges["travel_vel_x_world"][0]),
            abs(self.current_core_ranges["travel_vel_x_world"][1]),
            abs(self.current_core_ranges["travel_vel_y_world"][0]),
            abs(self.current_core_ranges["travel_vel_y_world"][1]),
            1.0e-6,
        )
        lin_speed = torch.norm(core[:, 0:2], dim=-1)
        yaw_drive = torch.abs(desired_yaw_rate) / max(float(adapter_cfg["max_ang_vel_yaw"]), 1.0e-6)
        lin_drive = lin_speed / max(float(max_xy), 1.0e-6)
        drive = torch.max(torch.max(core[:, 3], lin_drive), yaw_drive)
        drive = torch.clamp(drive, 0.0, 1.0)
        drive = torch.where(drive < float(adapter_cfg["stand_drive_threshold"]), torch.zeros_like(drive), drive)

        default_pitch = torch.clamp(
            local_vx * float(adapter_cfg["body_pitch_gain"]),
            min=float(self.cfg["commands"]["advanced"]["body_pitch_target"][0]),
            max=float(self.cfg["commands"]["advanced"]["body_pitch_target"][1]),
        )
        default_roll = torch.clamp(
            local_vy * float(adapter_cfg["body_roll_gain"]),
            min=float(self.cfg["commands"]["advanced"]["body_roll_target"][0]),
            max=float(self.cfg["commands"]["advanced"]["body_roll_target"][1]),
        )
        default_stance = torch.clamp(
            float(adapter_cfg["stance_width_nominal"])
            + torch.abs(local_vy) * float(adapter_cfg["stance_width_from_lateral_gain"])
            + torch.abs(heading_error) * float(adapter_cfg["stance_width_from_yaw_gain"]),
            min=float(self.cfg["commands"]["advanced"]["stance_width_target"][0]),
            max=float(self.cfg["commands"]["advanced"]["stance_width_target"][1]),
        )
        default_foot_yaw = torch.clamp(
            desired_yaw_rate.unsqueeze(-1) * float(adapter_cfg["foot_yaw_from_yaw_gain"]),
            min=float(adapter_cfg["foot_yaw_target_clip"][0]),
            max=float(adapter_cfg["foot_yaw_target_clip"][1]),
        )

        active_mask = self.advanced_command_active[env_ids]
        pitch_target = torch.where(active_mask, advanced[:, 3], default_pitch)
        roll_target = torch.where(active_mask, advanced[:, 4], default_roll)
        stance_target = torch.where(active_mask, advanced[:, 5], default_stance)
        foot_yaw_targets = default_foot_yaw.repeat(1, 2)
        foot_yaw_targets[:, 0] += advanced[:, 1]
        foot_yaw_targets[:, 1] += advanced[:, 2]
        foot_yaw_targets = torch.clamp(
            foot_yaw_targets,
            min=float(adapter_cfg["foot_yaw_target_clip"][0]),
            max=float(adapter_cfg["foot_yaw_target_clip"][1]),
        )

        gait_frequency = (
            float(adapter_cfg["idle_gait_frequency"])
            + drive * (float(adapter_cfg["active_gait_frequency"]) - float(adapter_cfg["idle_gait_frequency"]))
            + advanced[:, 0]
        )
        gait_frequency = torch.where(
            drive > 0.0,
            torch.clamp(
                gait_frequency,
                min=float(adapter_cfg["gait_frequency_clip"][0]),
                max=float(adapter_cfg["gait_frequency_clip"][1]),
            ),
            torch.zeros_like(gait_frequency),
        )

        self.command_drive[env_ids] = drive
        self.gait_frequency[env_ids] = gait_frequency
        self.foot_yaw_targets[env_ids] = foot_yaw_targets
        self.body_pitch_target[env_ids] = pitch_target
        self.body_roll_target[env_ids] = roll_target
        self.stance_width_target[env_ids] = stance_target
        self.desired_heading_world[env_ids] = core[:, 2]
        self.heading_error[env_ids] = heading_error
        self.desired_yaw_rate[env_ids] = desired_yaw_rate

        self.command_targets[env_ids, 0] = local_vx
        self.command_targets[env_ids, 1] = local_vy
        self.command_targets[env_ids, 2] = torch.sin(heading_error)
        self.command_targets[env_ids, 3] = torch.cos(heading_error)
        self.command_targets[env_ids, 4] = gait_frequency
        self.command_targets[env_ids, 5:7] = foot_yaw_targets
        self.command_targets[env_ids, 7] = pitch_target
        self.command_targets[env_ids, 8] = roll_target
        self.command_targets[env_ids, 9] = stance_target

    def step(self, actions):
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()
        self._resolve_commands(torch.arange(self.num_envs, device=self.device))

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()
        self._accumulate_metrics()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._emit_terminal_metrics(env_ids)
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._resample_commands()
        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.last_base_pos[:] = self.base_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            lin_kick = apply_randomization(self.root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            ang_kick = apply_randomization(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.root_states[:, 7:10] = self.root_states[:, 7:10] + (lin_kick - self.root_states[:, 7:10]) * self.current_disturbance_scale
            self.root_states[:, 10:13] = self.root_states[:, 10:13] + (ang_kick - self.root_states[:, 10:13]) * self.current_disturbance_scale
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            ) * self.current_disturbance_scale
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            ) * self.current_disturbance_scale
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, pitch, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_pitch[:] = (pitch.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        self.feet_yaw_rel[:] = (self.feet_yaw - base_yaw.unsqueeze(-1) + torch.pi) % (2 * torch.pi) - torch.pi

        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def _check_termination(self):
        contact_terminate = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        velocity_terminate = self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        height_terminate = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        timeout_terminate = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.height_terminate[:] = height_terminate

        self.reset_buf = contact_terminate | velocity_terminate | height_terminate | timeout_terminate
        self.time_out_buf = timeout_terminate
        self.time_out_buf |= self.episode_length_buf == self.command_resample_time

        if self.debug_termination and (self.common_step_counter % self.debug_termination_interval == 0):
            reset_count = int(self.reset_buf.sum().item())
            if reset_count > 0:
                print(
                    "[termination] "
                    f"step={self.common_step_counter} "
                    f"reset={reset_count} "
                    f"contact={int(contact_terminate.sum().item())} "
                    f"velocity={int(velocity_terminate.sum().item())} "
                    f"height={int(height_terminate.sum().item())} "
                    f"timeout={int(timeout_terminate.sum().item())}"
                )
                reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()[: self.debug_termination_max_envs].tolist()
                print(f"[termination] reset env ids (sample): {reset_env_ids}")

    def _compute_reward(self):
        self.rew_buf[:] = 0.0
        self.extras["rew_terms"] = {}
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        advanced_scale = torch.tensor(
            [
                self.cfg["normalization"]["gait_frequency"],
                self.cfg["normalization"]["foot_yaw"],
                self.cfg["normalization"]["foot_yaw"],
                self.cfg["normalization"]["body_pitch_target"],
                self.cfg["normalization"]["body_roll_target"],
                self.cfg["normalization"]["stance_width_target"],
            ],
            device=self.device,
        )
        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.command_targets[:, 0:2] * self.cfg["normalization"]["lin_vel"],
                self.command_targets[:, 2:4],
                self.command_targets[:, 4:10] * advanced_scale,
                (torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.last_actions,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf
        self.extras["metrics"] = self.metrics

    def _wrap_to_pi(self, angle):
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def _get_base_yaw(self):
        _, _, yaw = get_euler_xyz(self.base_quat)
        return self._wrap_to_pi(yaw)

    def _sample_resample_steps(self, env_count):
        low = int(self.current_resampling_time[0] / self.dt)
        high = int(self.current_resampling_time[1] / self.dt)
        return torch.randint(low, high, (env_count,), device=self.device)

    def _sample_heading_world(self, env_ids):
        heading_offset = self._sample_range(len(env_ids), "heading_offset")
        travel_world = self.core_commands[env_ids, 0:2]
        travel_heading = torch.atan2(travel_world[:, 1], travel_world[:, 0])
        base_yaw = self._get_base_yaw()[env_ids]
        has_travel = torch.norm(travel_world, dim=-1) > 1.0e-6
        return self._wrap_to_pi(torch.where(has_travel, travel_heading, base_yaw) + heading_offset)

    def _get_body_pitch_roll(self):
        roll_all, pitch_all, _ = get_euler_xyz(self.base_quat)
        roll = (roll_all + torch.pi) % (2 * torch.pi) - torch.pi
        pitch = (pitch_all + torch.pi) % (2 * torch.pi) - torch.pi
        return pitch, roll

    def _get_swing_masks(self):
        swing_half_period = 0.5 * self.cfg["rewards"]["swing_period"]
        gait_active = self.gait_frequency > 1.0e-8
        left_swing = (torch.abs(self.gait_process - 0.25) < swing_half_period) & gait_active
        right_swing = (torch.abs(self.gait_process - 0.75) < swing_half_period) & gait_active
        return left_swing, right_swing

    def _body_frame_feet_offsets(self):
        base_yaw = self._get_base_yaw()
        feet_dx = self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]
        feet_dy = self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1]
        feet_x_offset = torch.cos(base_yaw) * feet_dx + torch.sin(base_yaw) * feet_dy
        feet_y_offset = -torch.sin(base_yaw) * feet_dx + torch.cos(base_yaw) * feet_dy
        return feet_x_offset, feet_y_offset

    def _body_frame_feet_distance(self):
        return torch.abs(self._body_frame_feet_offsets()[1])

    def _body_frame_feet_positions(self):
        base_yaw = self._get_base_yaw()
        feet_rel = self.feet_pos[:, :, 0:2] - self.base_pos[:, None, 0:2]
        cos_yaw = torch.cos(base_yaw).unsqueeze(-1)
        sin_yaw = torch.sin(base_yaw).unsqueeze(-1)
        feet_x = cos_yaw * feet_rel[:, :, 0] + sin_yaw * feet_rel[:, :, 1]
        feet_y = -sin_yaw * feet_rel[:, :, 0] + cos_yaw * feet_rel[:, :, 1]
        return feet_x, feet_y

    def _foot_clearance(self):
        flat_feet_pos = self.feet_pos.reshape(-1, 3)
        ground_height = self.terrain.terrain_heights(flat_feet_pos).reshape(self.num_envs, len(self.feet_indices))
        return self.feet_pos[:, :, 2] - ground_height

    def _command_speed_local(self):
        return torch.norm(self.command_targets[:, 0:2], dim=-1)

    def _command_direction_local(self):
        direction = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        speed = self._command_speed_local()
        active = speed > 1.0e-6
        if active.any():
            direction[active] = self.command_targets[active, 0:2] / speed[active].unsqueeze(-1)
        return direction, speed, active

    def _command_progress_speed(self):
        travel_world = self.core_commands[:, 0:2]
        speed = torch.norm(travel_world, dim=-1)
        active = speed > 1.0e-6
        direction = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        if active.any():
            direction[active] = travel_world[active] / speed[active].unsqueeze(-1)
        delta_world = self.base_pos[:, 0:2] - self.last_base_pos[:, 0:2]
        projected_speed = torch.sum(delta_world * direction, dim=-1) / self.dt
        valid = active & (self.episode_length_buf > 1)
        projected_speed = torch.where(valid, projected_speed, torch.zeros_like(projected_speed))
        return projected_speed, valid

    def _command_progress_delta(self):
        travel_world = self.core_commands[:, 0:2]
        speed = torch.norm(travel_world, dim=-1)
        active = speed > 1.0e-6
        direction = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        if active.any():
            direction[active] = travel_world[active] / speed[active].unsqueeze(-1)
        delta_world = self.base_pos[:, 0:2] - self.last_base_pos[:, 0:2]
        progress_delta = torch.sum(delta_world * direction, dim=-1)
        valid = active & (self.episode_length_buf > 1)
        progress_delta = torch.where(valid, progress_delta, torch.zeros_like(progress_delta))
        return progress_delta, valid

    def _stance_slip_speed(self):
        left_swing, right_swing = self._get_swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1)
        stance_mask = self.feet_contact & ~swing_mask
        feet_vel_xy = torch.norm((self.feet_pos[:, :, 0:2] - self.last_feet_pos[:, :, 0:2]) / self.dt, dim=-1)
        stance_count = stance_mask.float().sum(dim=-1)
        stance_slip = torch.sum(feet_vel_xy * stance_mask.float(), dim=-1) / torch.clamp(stance_count, min=1.0)
        valid = (stance_count > 0) & (self.episode_length_buf > 1)
        return torch.where(valid, stance_slip, torch.zeros_like(stance_slip))

    def _reward_survival(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_lin_vel_x(self):
        sigma = self.cfg["rewards"]["tracking_sigma"]
        return torch.exp(-torch.square(self.command_targets[:, 0] - self.filtered_lin_vel[:, 0]) / sigma)

    def _reward_tracking_lin_vel_y(self):
        sigma = self.cfg["rewards"]["tracking_sigma"]
        return torch.exp(-torch.square(self.command_targets[:, 1] - self.filtered_lin_vel[:, 1]) / sigma)

    def _reward_tracking_ang_vel(self):
        sigma = self.cfg["rewards"]["tracking_sigma"]
        return torch.exp(-torch.square(self.desired_yaw_rate - self.filtered_ang_vel[:, 2]) / sigma)

    def _reward_heading(self):
        sigma = max(float(self.cfg["rewards"]["heading_sigma"]), 1.0e-6)
        return torch.exp(-torch.square(self.heading_error) / sigma)

    def _reward_base_height(self):
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_orientation(self):
        pitch, roll = self._get_body_pitch_roll()
        return torch.square(pitch - self.body_pitch_target) + torch.square(roll - self.body_roll_target)

    def _reward_stance_width(self):
        return torch.square(self._body_frame_feet_distance() - self.stance_width_target)

    def _reward_foot_yaw(self):
        return torch.sum(torch.square(self.feet_yaw_rel - self.foot_yaw_targets), dim=-1)

    def _reward_contact_timing(self):
        left_swing, right_swing = self._get_swing_masks()
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()

    def _reward_swing_clearance(self):
        left_swing, right_swing = self._get_swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1)
        if not swing_mask.any():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        clearance = self._foot_clearance()
        target = self.cfg["rewards"]["swing_clearance_target"]
        sigma = self.cfg["rewards"]["swing_clearance_sigma"]
        clearance_reward = torch.exp(-torch.square(clearance - target) / sigma)
        return torch.sum(clearance_reward * swing_mask.float(), dim=-1)

    def _reward_step_reach(self):
        left_swing, right_swing = self._get_swing_masks()
        swing_active = left_swing | right_swing
        if not swing_active.any():
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        feet_x, feet_y = self._body_frame_feet_positions()
        command_direction, command_speed, active = self._command_direction_local()
        projected_positions = feet_x * command_direction[:, None, 0] + feet_y * command_direction[:, None, 1]
        target_mag = torch.clamp(
            self.cfg["rewards"]["step_reach_min"] + command_speed * self.cfg["rewards"]["step_reach_gain"],
            min=self.cfg["rewards"]["step_reach_min"],
            max=self.cfg["rewards"]["step_reach_max"],
        )
        sigma = max(float(self.cfg["rewards"]["step_reach_sigma"]), 1.0e-6)
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if left_swing.any():
            reward[left_swing] = torch.exp(-torch.square(projected_positions[left_swing, 0] - target_mag[left_swing]) / sigma)
        if right_swing.any():
            reward[right_swing] = torch.exp(-torch.square(projected_positions[right_swing, 1] - target_mag[right_swing]) / sigma)
        return reward * active.float()

    def _reward_command_progress(self):
        projected_speed, valid = self._command_progress_speed()
        return torch.clamp(projected_speed, min=0.0) * valid.float()

    def _reward_double_support_drag(self):
        left_swing, right_swing = self._get_swing_masks()
        swing_active = left_swing | right_swing
        both_feet_planted = torch.all(self.feet_contact, dim=-1)
        return (swing_active & both_feet_planted).float() * self.command_drive

    def _reward_scuff_penalty(self):
        left_swing, right_swing = self._get_swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1)
        low_clearance = self._foot_clearance() < self.cfg["rewards"]["scuff_clearance_threshold"]
        return torch.sum((swing_mask & low_clearance).float(), dim=-1)

    def _reward_collision(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_power(self):
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        left_swing, right_swing = self._get_swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1)
        stance_mask = self.feet_contact & ~swing_mask
        feet_vel_xy_sq = torch.square((self.last_feet_pos[:, :, 0:2] - self.feet_pos[:, :, 0:2]) / self.dt).sum(dim=-1)
        return torch.sum(feet_vel_xy_sq * stance_mask.float(), dim=-1) * (self.episode_length_buf > 1).float()

    def _accumulate_metrics(self):
        left_swing, right_swing = self._get_swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1)
        swing_count = swing_mask.float().sum(dim=-1)
        swing_air = ((swing_mask & ~self.feet_contact).float().sum(dim=-1) / torch.clamp(swing_count, min=1.0)) * (swing_count > 0).float()
        feet_x_offset, _ = self._body_frame_feet_offsets()
        progress_delta, _ = self._command_progress_delta()
        stance_slip = self._stance_slip_speed()
        clearance = torch.clamp(self._foot_clearance(), min=0.0)
        swing_clearance_peak = torch.max(clearance * swing_mask.float(), dim=-1).values
        both_feet_planted = torch.all(self.feet_contact, dim=-1)
        double_support_ratio = ((swing_count > 0) & both_feet_planted).float()
        self.metric_sum_heading_error_abs += torch.abs(self.heading_error)
        self.metric_sum_vx_tracking_error += torch.abs(self.command_targets[:, 0] - self.filtered_lin_vel[:, 0])
        self.metric_sum_vy_tracking_error += torch.abs(self.command_targets[:, 1] - self.filtered_lin_vel[:, 1])
        self.metric_sum_stance_width_mean += self._body_frame_feet_distance()
        self.metric_sum_feet_x_offset_abs += torch.abs(feet_x_offset)
        self.metric_sum_swing_air_ratio += swing_air
        self.metric_sum_command_progress += progress_delta
        self.metric_sum_stance_slip_mean += stance_slip
        self.metric_step_clearance_peak = torch.maximum(
            self.metric_step_clearance_peak,
            torch.where(swing_count > 0, swing_clearance_peak, torch.zeros_like(swing_clearance_peak)),
        )
        self.metric_double_support_count += double_support_ratio
        self.metric_swing_step_count += (swing_count > 0).float()
        self.metric_sample_count += 1.0

    def _emit_terminal_metrics(self, done_ids):
        for metric in self.metrics.values():
            metric.zero_()
        if len(done_ids) == 0:
            return
        sample_count = torch.clamp(self.metric_sample_count[done_ids], min=1.0)
        swing_step_count = torch.clamp(self.metric_swing_step_count[done_ids], min=1.0)

        self.metrics["heading_error_abs"][done_ids] = self.metric_sum_heading_error_abs[done_ids] / sample_count
        self.metrics["vx_tracking_error"][done_ids] = self.metric_sum_vx_tracking_error[done_ids] / sample_count
        self.metrics["vy_tracking_error"][done_ids] = self.metric_sum_vy_tracking_error[done_ids] / sample_count
        self.metrics["stance_width_mean"][done_ids] = self.metric_sum_stance_width_mean[done_ids] / sample_count
        self.metrics["feet_x_offset_abs"][done_ids] = self.metric_sum_feet_x_offset_abs[done_ids] / sample_count
        self.metrics["swing_air_ratio"][done_ids] = self.metric_sum_swing_air_ratio[done_ids] / sample_count
        self.metrics["reset_height_fraction"][done_ids] = self.height_terminate[done_ids].float()
        self.metrics["command_progress"][done_ids] = self.metric_sum_command_progress[done_ids]
        self.metrics["stance_slip_mean"][done_ids] = self.metric_sum_stance_slip_mean[done_ids] / sample_count
        self.metrics["step_clearance_peak"][done_ids] = self.metric_step_clearance_peak[done_ids]
        self.metrics["double_support_ratio"][done_ids] = self.metric_double_support_count[done_ids] / swing_step_count
