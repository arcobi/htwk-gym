import os
import csv

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from envs.base_task import BaseTask

from utils.utils import apply_randomization


class BallControlK1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        self._init_csv_logging()

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

        # prepare penalized and termination contact indices
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
        self.ball_handles = []  # Store ball handles
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.ball_radii = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            # Create robot actor
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            # Create ball actor
            ball_radius = float(apply_randomization(self.cfg["ball"]["radius"], self.cfg["randomization"].get("ball_radius")))
            self.ball_radii[i] = ball_radius
            ball_asset = self._create_ball_asset(radius=ball_radius)
            ball_handle = self.gym.create_actor(env_handle, ball_asset, start_pose, "ball", i, True, 0)
            try:
                ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
                for b in range(len(ball_body_props)):
                    ball_body_props[b].mass = apply_randomization(self.cfg["ball"]["mass"], self.cfg["randomization"].get("ball_mass"))
                self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
            except Exception:
                pass

            # Set ball shape properties: restitution/friction (with randomization)
            try:
                ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
                for s in range(len(ball_shape_props)):
                    ball_shape_props[s].restitution = apply_randomization(
                        self.cfg["ball"].get("restitution", 0.1), 
                        self.cfg["randomization"].get("ball_restitution")
                    )
                    ball_shape_props[s].friction = apply_randomization(
                        self.cfg["ball"].get("friction", 1.0),
                        self.cfg["randomization"].get("ball_friction")
                    )
                    ball_shape_props[s].rolling_friction = apply_randomization(
                        self.cfg["ball"].get("rolling_friction", 0.3),
                        self.cfg["randomization"].get("ball_rolling_friction")
                    )
                    ball_shape_props[s].torsion_friction = apply_randomization(
                        self.cfg["ball"].get("torsion_friction", 0.1),
                        self.cfg["randomization"].get("ball_torsion_friction")
                    )
                    ball_shape_props[s].thickness = 0.01
                    ball_shape_props[s].contact_offset = 0.02
                    ball_shape_props[s].rest_offset = 0.0
                self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)
            except Exception as e:
                print(e)
                pass

            # Store handles
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.ball_handles.append(ball_handle)

        # Initialize ball state tensors
        self.ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.ball_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

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

    def _create_ball_asset(self, radius):
        """Create a ball asset with the given radius"""
        ball_options = gymapi.AssetOptions()
        ball_options.fix_base_link = False
        ball_options.density = apply_randomization(
            self.cfg["ball"].get("density", 200),
            self.cfg["randomization"].get("ball_density")
        )
        ball_options.angular_damping = 0.15
        ball_options.linear_damping = 0.38
        ball_options.max_angular_velocity = 1000.0
        ball_options.max_linear_velocity = 20.0
        ball_options.disable_gravity = False
        ball_options.replace_cylinder_with_capsule = False
        ball_options.thickness = 0.01

        ball_asset = self.gym.create_sphere(self.sim, radius, ball_options)
        return ball_asset

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
        self.extras = {}
        self.extras["rew_terms"] = {}
        self.extras["metrics"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        root_states = gymtorch.wrap_tensor(actor_root_state)
        # Reshape root states to separate robot and ball states (2 actors per environment)
        self.root_states = root_states.view(self.num_envs, 2, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies + 1, 13)  # +1 for ball
        
        # Get robot states (index 0) and ball states (index 1)
        self.base_pos = self.root_states[:, 0, 0:3]  # Robot position
        self.base_quat = self.root_states[:, 0, 3:7]  # Robot quaternion
        self.ball_pos = self.root_states[:, 1, 0:3]  # Ball position
        self.ball_rot = self.root_states[:, 1, 3:7]  # Ball quaternion
        self.ball_lin_vel = self.body_states[:, -1, 7:10]  # Ball linear velocity
        self.ball_ang_vel = self.body_states[:, -1, 10:13]  # Ball angular velocity
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.debug_termination = bool(self.cfg.get("basic", {}).get("debug_termination", False))
        self.debug_termination_interval = max(1, int(self.cfg.get("basic", {}).get("debug_termination_interval", 100)))
        self.debug_termination_max_envs = max(1, int(self.cfg.get("basic", {}).get("debug_termination_max_envs", 5)))
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions - 1, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions - 1, dtype=torch.float, device=self.device)
        self.prev_dof_pos = torch.zeros_like(self.dof_pos)
        self.custom_dof_vel = torch.zeros_like(self.dof_vel)
        self.filtered_custom_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_frequency_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.reset_ball_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # Buffer for ball-only resets
        self.last_ball_lin_vel_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)  # World frame
        self.last_relative_ball_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # Last frame ball position in robot frame
        self.last_ball_distance_to_robot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Previous step XY distance robot-ball
        self.episode_start_base_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # Robot XY at episode reset
        self.pass_ref_origin_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # Fixed origin for pass-event checks
        self.pass_ref_dir_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # Fixed shot direction for pass-event checks
        self.pass_ref_dir_xy[:, 0] = 1.0
        self.ball_still_time_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # How long the ball stayed below speed threshold

        # Interception tracking
        self.ball_has_been_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_first_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_first_contact_time = torch.full(
            (self.num_envs,), 0.0, dtype=torch.float, device=self.device
        )
        self.time_since_first_contact = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.ball_max_progress_along_shot = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.prev_ball_max_progress_along_shot = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.ball_initial_progress = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.ball_speed_drop = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.local_ball_vel_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.local_pass_dir_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.local_source_dir_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.intercept_lateral_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.intercept_forward_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.intercept_point_local = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.intercept_time_estimate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.intercept_closest_approach = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.intercept_heading_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_speed_toward_robot = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_has_passed_robot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.clear_miss_time_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.intercept_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.intercept_phase_onehot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.last_intercept_lateral_abs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.intercept_feature_history_len = max(1, int(self.cfg["env"].get("intercept_feature_history_len", 2)))
        self.intercept_feature_history = torch.zeros(
            self.num_envs,
            self.intercept_feature_history_len,
            6,
            dtype=torch.float,
            device=self.device,
        )

        # Ball curriculum state
        curriculum_cfg = self.cfg.get("ball_curriculum", {})
        self.ball_curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        curriculum_window = curriculum_cfg.get("evaluation_window", 200)
        self.ball_curriculum_success_ring = torch.zeros(
            curriculum_window, dtype=torch.float, device=self.device
        )
        self.ball_curriculum_ring_idx = 0
        self.ball_curriculum_global_level = 0

        # Precompute receive mode mask (updated each step before reward computation)
        self.receive_mode_mask = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Ball detection simulation (simulates camera at 30 FPS with jitter)
        # Stores ball positions in ROBOT FRAME - only updates when detection occurs
        self.ball_detection_fps = self.cfg["ball"].get("detection_fps", 30.0)
        self.ball_detection_jitter = self.cfg["ball"].get("detection_fps_jitter", 0.15)
        self.ball_detection_interval = 1.0 / self.ball_detection_fps  # Base interval in seconds
        self.perception_lag_min_s = float(self.cfg["ball"].get("perception_lag_min_s", 0.08))
        self.perception_lag_max_s = float(self.cfg["ball"].get("perception_lag_max_s", 0.18))
        if self.perception_lag_max_s < self.perception_lag_min_s:
            self.perception_lag_min_s, self.perception_lag_max_s = self.perception_lag_max_s, self.perception_lag_min_s
        self.perceived_ball_pos_relative = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # Current detection (in robot frame)
        self.last_perceived_ball_pos_relative = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # Previous detection (in robot frame)
        self.lagged_perceived_ball_pos_relative = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)  # Randomized-lag detection snapshot (in robot frame)
        self.ball_detection_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Time until next detection
        self.lag_snapshot_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Time until lag snapshot refresh on detection
        self.ball_detection_age = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # Time since last detection
        self.ball_pass_event_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # Per-step pass event flag
        self.ball_pass_event_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # Episode-latched pass event flag
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw_rel = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_pitch = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
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
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def _init_csv_logging(self):
        """Initialize CSV logging for reward values"""
        # Check if CSV logging is enabled in config
        self.csv_logging_enabled = self.cfg.get("basic", {}).get("enable_csv_logging", True)

        if not self.csv_logging_enabled:
            print("CSV logging disabled in configuration")
            return

        # Only log for environment 0 (single environment setup)
        self.log_env_id = 0

        # Create logs directory if it doesn't exist
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create CSV file with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.log_dir, f"debug_csv/reward_log_{timestamp}.csv")

        # Prepare CSV headers
        self.csv_headers = [
            "episode_step", "total_reward",
            "ball_pos_x", "ball_pos_y", "ball_pos_z",
            "ball_vel_x", "ball_vel_y", "ball_vel_z",
            "robot_pos_x", "robot_pos_y", "robot_pos_z",
            "robot_lin_vel_x", "robot_lin_vel_y", "robot_lin_vel_z",
            "ball_speed", "ball_distance_to_robot"
        ]
        reward_names = ["reward_" + name for name in self.reward_names]
        self.csv_headers.extend(reward_names)  # Add all individual reward terms

        # Initialize CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_headers)

        print(f"CSV logging initialized: {self.csv_filename}")

    def _log_rewards_to_csv(self):
        """Log current step rewards to CSV file"""
        # Check if CSV logging is enabled
        if not getattr(self, 'csv_logging_enabled', False):
            return

        # Only log for the specified environment (0)
        if hasattr(self, 'episode_length_buf'):
            episode_step = self.episode_length_buf[self.log_env_id].item()
            total_reward = self.rew_buf[self.log_env_id].item()

            # Get ball and robot state information
            ball_pos = self.ball_pos[self.log_env_id].cpu().numpy()
            ball_vel_world = self.root_states[self.log_env_id, 1, 7:10].cpu().numpy()
            robot_pos = self.base_pos[self.log_env_id].cpu().numpy()
            robot_lin_vel = self.base_lin_vel[self.log_env_id].cpu().numpy()

            # Calculate derived metrics
            ball_speed = torch.norm(self.root_states[self.log_env_id, 1, 7:10]).item()
            ball_distance_to_robot = torch.norm(self.ball_pos[self.log_env_id] - self.base_pos[self.log_env_id]).item()

            # Prepare row data
            row_data = [
                episode_step, total_reward,
                ball_pos[0], ball_pos[1], ball_pos[2],
                ball_vel_world[0], ball_vel_world[1], ball_vel_world[2],
                robot_pos[0], robot_pos[1], robot_pos[2],
                robot_lin_vel[0], robot_lin_vel[1], robot_lin_vel[2],
                ball_speed, ball_distance_to_robot
            ]

            # Add individual reward terms
            for reward_name in self.reward_names:
                if reward_name in self.extras["rew_terms"]:
                    reward_value = self.extras["rew_terms"][reward_name][self.log_env_id].item()
                    row_data.append(reward_value)
                else:
                    row_data.append(0.0)  # Default if reward term not found

            # Write to CSV
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_commands()
        target_speed = torch.norm(self.commands[:, 0:2], dim=-1)
        self.receive_mode_mask = (target_speed <= 1e-6).float()
        self.ball_pass_event_now.zero_()
        self._update_intercept_state()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._update_pass_reference(env_ids)
        # Seed initial progress so ball_travel_penalty starts at zero for spawned balls
        rel_ball_origin_reset = self.root_states[env_ids, 1, 0:2] - self.pass_ref_origin_xy[env_ids]
        self.ball_initial_progress[env_ids] = torch.sum(
            rel_ball_origin_reset * self.pass_ref_dir_xy[env_ids], dim=-1
        )
        self.episode_start_base_pos_xy[env_ids] = self.root_states[env_ids, 0, 0:2]
        self.last_ball_distance_to_robot[env_ids] = torch.norm(
            self.ball_pos[env_ids, 0:2] - self.base_pos[env_ids, 0:2], dim=-1
        )

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.prev_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.custom_dof_vel[env_ids] = 0.0
        self.filtered_custom_dof_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        self.last_ball_lin_vel_world[env_ids] = 0.0  # Reset ball velocity tracking
        self.last_relative_ball_pos[env_ids] = 0.0  # Reset last ball position buffer
        self.ball_still_time_buf[env_ids] = 0.0
        self.ball_has_been_contacted[env_ids] = False
        self.ball_first_contact_event[env_ids] = False
        self.ball_first_contact_time[env_ids] = 0.0
        self.time_since_first_contact[env_ids] = 0.0
        self.ball_max_progress_along_shot[env_ids] = 0.0
        self.prev_ball_max_progress_along_shot[env_ids] = 0.0
        self.ball_speed_drop[env_ids] = 0.0
        self.local_ball_vel_xy[env_ids] = 0.0
        self.local_pass_dir_xy[env_ids] = 0.0
        self.local_source_dir_xy[env_ids] = 0.0
        self.intercept_lateral_error[env_ids] = 0.0
        self.intercept_forward_error[env_ids] = 0.0
        self.intercept_point_local[env_ids] = 0.0
        self.intercept_time_estimate[env_ids] = 0.0
        self.intercept_closest_approach[env_ids] = 0.0
        self.intercept_heading_error[env_ids] = 0.0
        self.ball_speed_toward_robot[env_ids] = 0.0
        self.ball_has_passed_robot[env_ids] = False
        self.clear_miss_time_buf[env_ids] = 0.0
        self.intercept_phase[env_ids] = 0
        self.intercept_phase_onehot[env_ids] = 0.0
        self.last_intercept_lateral_abs[env_ids] = 0.0
        self.intercept_feature_history[env_ids] = 0.0
        # NOTE: ball_initial_progress is seeded above (lines ~539-541), do NOT zero it here
        
        # Reset ball detection simulation - randomize initial timer for each env
        self.ball_detection_timer[env_ids] = torch_rand_float(
            0.0, self.ball_detection_interval, (len(env_ids), 1), device=self.device
        ).squeeze(-1)
        # Compute initial ball position in robot frame for reset envs (with detection noise)
        ball_pos_world_frame = self.ball_pos[env_ids] - self.base_pos[env_ids]
        relative_ball_pos = quat_rotate_inverse(self.base_quat[env_ids], ball_pos_world_frame)
        noisy_relative_xy = apply_randomization(relative_ball_pos[:, 0:2], self.cfg["noise"].get("ball_pos"))
        self.perceived_ball_pos_relative[env_ids] = noisy_relative_xy
        self.last_perceived_ball_pos_relative[env_ids] = noisy_relative_xy
        self.lagged_perceived_ball_pos_relative[env_ids] = noisy_relative_xy
        self.ball_detection_age[env_ids] = 0.0
        self.lag_snapshot_timer[env_ids] = self._sample_lag_snapshot_interval(len(env_ids))
        self.ball_pass_event_now[env_ids] = False
        self.ball_pass_event_latched[env_ids] = False

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _sample_lag_snapshot_interval(self, num_envs):
        if num_envs <= 0:
            return torch.zeros(0, dtype=torch.float, device=self.device)
        return torch_rand_float(
            self.perception_lag_min_s,
            self.perception_lag_max_s,
            (num_envs, 1),
            device=self.device,
        ).squeeze(-1)

    def _update_intercept_state(self):
        """Update explicit interception geometry and internal phase state for single-policy learning."""
        current_relative_ball_pos_xy = self.perceived_ball_pos_relative
        past_relative_ball_pos_xy = self.last_perceived_ball_pos_relative
        perceived_motion_xy = current_relative_ball_pos_xy - past_relative_ball_pos_xy
        perceived_motion_norm = torch.norm(perceived_motion_xy, dim=-1, keepdim=True)

        ball_vel_local = quat_rotate_inverse(self.base_quat, self.ball_lin_vel)
        self.local_ball_vel_xy[:] = ball_vel_local[:, 0:2]

        foot_to_ball = self.ball_pos[:, 0:2].unsqueeze(1) - self.feet_pos[:, :, 0:2]
        min_foot_dist = torch.norm(foot_to_ball, dim=-1).min(dim=1).values
        foot_radius = float(self.cfg["rewards"].get("interception_foot_radius", 0.20))
        robot_close = min_foot_dist < foot_radius
        min_speed_drop = float(self.cfg["rewards"].get("interception_min_speed_drop", 0.15))
        significant_impact = self.ball_speed_drop > min_speed_drop
        min_ball_speed = float(self.cfg["rewards"].get("ball_vel_tracking_min_speed", 0.1))
        ball_speed_prev = torch.norm(self.last_ball_lin_vel_world[:, 0:2], dim=-1)
        was_moving = ball_speed_prev > min_ball_speed
        first_contact_now = significant_impact & robot_close & was_moving & (self.receive_mode_mask > 0.5)
        self.ball_first_contact_event[:] = first_contact_now & (~self.ball_has_been_contacted)
        self.ball_has_been_contacted |= self.ball_first_contact_event
        episode_time = self.episode_length_buf.float() * self.dt
        self.ball_first_contact_time[self.ball_first_contact_event] = episode_time[self.ball_first_contact_event]

        motion_min_delta = max(float(self.cfg["rewards"].get("perceived_intercept_min_delta", 0.01)), 1.0e-6)
        perceived_dir = perceived_motion_xy / (perceived_motion_norm + 1.0e-8)
        fallback_dir = -current_relative_ball_pos_xy / (torch.norm(current_relative_ball_pos_xy, dim=-1, keepdim=True) + 1.0e-8)
        use_motion_dir = perceived_motion_norm.squeeze(-1) > motion_min_delta
        self.local_pass_dir_xy[:] = torch.where(use_motion_dir.unsqueeze(-1), perceived_dir, fallback_dir)
        self.local_source_dir_xy[:] = -self.local_pass_dir_xy

        self.intercept_forward_error[:] = torch.sum(current_relative_ball_pos_xy * self.local_source_dir_xy, dim=-1)
        self.intercept_lateral_error[:] = (
            self.local_source_dir_xy[:, 0] * current_relative_ball_pos_xy[:, 1]
            - self.local_source_dir_xy[:, 1] * current_relative_ball_pos_xy[:, 0]
        )
        self.intercept_point_local[:] = self.local_source_dir_xy * torch.clamp(
            self.intercept_forward_error, min=0.0
        ).unsqueeze(-1)
        self.intercept_closest_approach[:] = torch.abs(self.intercept_lateral_error)

        self.ball_speed_toward_robot[:] = torch.sum(self.local_ball_vel_xy * self.local_pass_dir_xy, dim=-1)
        intercept_speed = torch.clamp(self.ball_speed_toward_robot, min=1.0e-3)
        time_clip = float(self.cfg["rewards"].get("intercept_time_clip_s", 3.0))
        self.intercept_time_estimate[:] = torch.clamp(
            self.intercept_forward_error / intercept_speed,
            min=0.0,
            max=time_clip,
        )

        self.intercept_heading_error[:] = torch.atan2(
            self.intercept_point_local[:, 1],
            self.intercept_point_local[:, 0] + 1.0e-6,
        )

        pass_margin = float(self.cfg["rewards"].get("ball_passed_margin_x", 0.03))
        self.ball_has_passed_robot[:] = self.intercept_forward_error < -pass_margin

        self.time_since_first_contact[:] = torch.where(
            self.ball_has_been_contacted,
            torch.clamp(episode_time - self.ball_first_contact_time, min=0.0),
            torch.zeros_like(episode_time),
        )

        approaching_speed_min = float(self.cfg["rewards"].get("intercept_approach_min_speed", 0.08))
        intercept_window_s = float(self.cfg["rewards"].get("intercept_window_time_s", 0.35))
        pre_contact = self.receive_mode_mask > 0.5
        pre_contact &= ~self.ball_has_been_contacted
        pre_contact &= ~self.ball_has_passed_robot
        moving_toward_robot = self.ball_speed_toward_robot > approaching_speed_min

        align_phase = pre_contact & ~((self.intercept_time_estimate <= intercept_window_s) & moving_toward_robot)
        intercept_phase = pre_contact & (self.intercept_time_estimate <= intercept_window_s) & moving_toward_robot
        post_contact_phase = (self.receive_mode_mask > 0.5) & self.ball_has_been_contacted
        miss_phase = (self.receive_mode_mask > 0.5) & self.ball_has_passed_robot & (~self.ball_has_been_contacted)

        self.intercept_phase[:] = 0
        self.intercept_phase[intercept_phase] = 1
        self.intercept_phase[post_contact_phase] = 2
        self.intercept_phase[miss_phase] = 3
        self.intercept_phase_onehot.zero_()
        self.intercept_phase_onehot.scatter_(1, self.intercept_phase.unsqueeze(-1), 1.0)

        miss_moving_gate = self.ball_speed_toward_robot < -float(self.cfg["rewards"].get("miss_moving_away_min", 0.03))
        clear_miss_now = miss_phase & miss_moving_gate
        self.clear_miss_time_buf[:] = torch.where(
            clear_miss_now,
            self.clear_miss_time_buf + self.dt,
            torch.zeros_like(self.clear_miss_time_buf),
        )

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        self.prev_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.custom_dof_vel[env_ids] = 0.0
        self.filtered_custom_dof_vel[env_ids] = 0.0
        # Multiply by 2 because there are 2 actors per environment (robot and ball)
        # This ensures we only update the robot actor's DOFs
        env_ids_int32 = (2 * env_ids).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        # Initialize robot states (index 0)
        self.root_states[env_ids, 0, :] = self.base_init_state
        self.root_states[env_ids, 0, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, 0, :2] = apply_randomization(self.root_states[env_ids, 0, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 0, 2] += self.terrain.terrain_heights(self.root_states[env_ids, 0, :2])
        self.root_states[env_ids, 0, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("init_base_ang")
            ),
        )
        self.root_states[env_ids, 0, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )

        # Reset ball in front of the (newly reset) robot
        self._reset_ball_at_robot_front(env_ids)

        # Update the simulation with new state tensor for both robot and ball
        robot_actor_indices = 2 * env_ids
        ball_actor_indices = 2 * env_ids + 1
        
        actor_indices_to_update = torch.stack((robot_actor_indices, ball_actor_indices), dim=-1).view(-1).to(dtype=torch.int32)
        num_indices = actor_indices_to_update.shape[0]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),  # Full root states buffer
            gymtorch.unwrap_tensor(actor_indices_to_update),  # Indices of actors to update
            num_indices  # Number of actor indices
        )

    def _reset_ball_at_robot_front(self, env_ids_to_reset_ball):
        """Resets the ball in front of the robot for the specified environment IDs."""
        if len(env_ids_to_reset_ball) == 0:
            return

        robot_pos = self.root_states[env_ids_to_reset_ball, 0, 0:3]
        n = len(env_ids_to_reset_ball)

        # --- Curriculum-aware spawn parameters ---
        curriculum_cfg = self.cfg.get("ball_curriculum", {})
        if curriculum_cfg.get("enabled", False):
            level = self.ball_curriculum_global_level
            dist_min = curriculum_cfg["distance_min"][level]
            dist_max = curriculum_cfg["distance_max"][level]
            spd_min  = curriculum_cfg["speed_min"][level]
            spd_max  = curriculum_cfg["speed_max"][level]
            tol      = curriculum_cfg["tolerance"][level]
            y_range  = curriculum_cfg["y_range"][level]

            # Sample distance and lateral offset
            offset_x = torch_rand_float(dist_min, dist_max, (n, 1), device=self.device).squeeze(-1)
            offset_y = torch_rand_float(-y_range, y_range, (n, 1), device=self.device).squeeze(-1)

            ball_target_xy = robot_pos[:, 0:2] + torch.stack((offset_x, offset_y), dim=-1)
            ball_target_z = self.terrain.terrain_heights(ball_target_xy) + self.ball_radii[env_ids_to_reset_ball]

            # Set ball position
            self.root_states[env_ids_to_reset_ball, 1, 0] = ball_target_xy[:, 0]
            self.root_states[env_ids_to_reset_ball, 1, 1] = ball_target_xy[:, 1]
            self.root_states[env_ids_to_reset_ball, 1, 2] = ball_target_z

            # Identity quaternion
            identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(n, -1)
            self.root_states[env_ids_to_reset_ball, 1, 3:7] = identity_quat

            # Aim toward robot with tolerance
            ball_pos_xy = ball_target_xy
            base_pos_xy = robot_pos[:, 0:2]
            to_robot_xy = base_pos_xy - ball_pos_xy
            to_robot_norm = torch.norm(to_robot_xy, dim=-1, keepdim=True).clamp_min(1e-6)
            to_robot_dir = to_robot_xy / to_robot_norm
            perp_dir = torch.stack((-to_robot_dir[:, 1], to_robot_dir[:, 0]), dim=-1)
            lateral_offset = torch_rand_float(-tol, tol, (n, 1), device=self.device).squeeze(-1)
            target_point = base_pos_xy + perp_dir * lateral_offset.unsqueeze(-1)
            vel_dir = target_point - ball_pos_xy
            vel_norm = torch.norm(vel_dir, dim=-1, keepdim=True).clamp_min(1e-6)
            vel_dir = vel_dir / vel_norm
            speed = torch_rand_float(spd_min, spd_max, (n, 1), device=self.device).squeeze(-1)
            self.root_states[env_ids_to_reset_ball, 1, 7:9] = vel_dir * speed.unsqueeze(-1)
            self.root_states[env_ids_to_reset_ball, 1, 9] = 0.0
        else:
            # Original randomization-based spawn (fallback)
            offset_x = apply_randomization(
                torch.zeros(n, dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("ball_init_pos_x"),
            )
            offset_y = apply_randomization(
                torch.zeros(n, dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("ball_init_pos_y"),
            )
            ball_target_xy = robot_pos[:, 0:2] + torch.stack((offset_x, offset_y), dim=-1)
            ball_target_z = self.terrain.terrain_heights(ball_target_xy) + self.ball_radii[env_ids_to_reset_ball]

            self.root_states[env_ids_to_reset_ball, 1, 0] = ball_target_xy[:, 0]
            self.root_states[env_ids_to_reset_ball, 1, 1] = ball_target_xy[:, 1]
            self.root_states[env_ids_to_reset_ball, 1, 2] = ball_target_z

            identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(n, 1)
            self.root_states[env_ids_to_reset_ball, 1, 3:7] = identity_quat

            self.root_states[env_ids_to_reset_ball, 1, 7] = apply_randomization(
                torch.zeros(n, dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("ball_init_lin_vel_x")
            )
            self.root_states[env_ids_to_reset_ball, 1, 8] = apply_randomization(
                torch.zeros(n, dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("ball_init_lin_vel_y")
            )
            self.root_states[env_ids_to_reset_ball, 1, 9] = 0.0

            ball_pos = self.root_states[env_ids_to_reset_ball, 1, 0:3]
            base_pos = self.root_states[env_ids_to_reset_ball, 0, 0:3]

            to_robot_xy = base_pos[:, 0:2] - ball_pos[:, 0:2]
            to_robot_norm = torch.norm(to_robot_xy, dim=-1, keepdim=True).clamp_min(1e-6)
            to_robot_dir_xy = to_robot_xy / to_robot_norm
            perpendicular_dir_xy = torch.stack((-to_robot_dir_xy[:, 1], to_robot_dir_xy[:, 0]), dim=-1)
            ball_tolerance_m = apply_randomization(
                torch.zeros(n, dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("ball_tolerance"),
            )
            target_point_xy = base_pos[:, 0:2] + perpendicular_dir_xy * ball_tolerance_m.unsqueeze(-1)
            final_vel_xy = target_point_xy - ball_pos[:, 0:2]
            final_vel_norm = torch.norm(final_vel_xy, dim=-1, keepdim=True).clamp_min(1e-6)
            final_vel_dir_xy = final_vel_xy / final_vel_norm
            target_speed = apply_randomization(
                torch.zeros(n, dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("ball_target_speed"),
            )
            self.root_states[env_ids_to_reset_ball, 1, 7:9] = final_vel_dir_xy * target_speed.unsqueeze(-1)

        # Angular velocity randomization (same for both paths)
        self.root_states[env_ids_to_reset_ball, 1, 10] = apply_randomization(
            torch.zeros(n, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("ball_init_ang_vel_x")
        )
        self.root_states[env_ids_to_reset_ball, 1, 11] = apply_randomization(
            torch.zeros(n, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("ball_init_ang_vel_y")
        )
        self.root_states[env_ids_to_reset_ball, 1, 12] = apply_randomization(
            torch.zeros(n, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("ball_init_ang_vel_z")
        )

    def _update_pass_reference(self, env_ids):
        """Set per-env fixed origin and shot direction for pass-event rewards."""
        if len(env_ids) == 0:
            return

        self.pass_ref_origin_xy[env_ids] = self.root_states[env_ids, 0, 0:2]

        ball_vel_xy = self.root_states[env_ids, 1, 7:9]
        ball_speed = torch.norm(ball_vel_xy, dim=-1, keepdim=True)
        vel_dir = ball_vel_xy / (ball_speed + 1e-8)

        # Fallback to ball->robot direction when initial ball speed is tiny.
        ball_to_robot = self.root_states[env_ids, 0, 0:2] - self.root_states[env_ids, 1, 0:2]
        ball_to_robot_norm = torch.norm(ball_to_robot, dim=-1, keepdim=True).clamp_min(1e-6)
        fallback_dir = ball_to_robot / ball_to_robot_norm

        use_vel_dir = ball_speed.squeeze(-1) > 1e-4
        self.pass_ref_dir_xy[env_ids] = torch.where(use_vel_dir.unsqueeze(-1), vel_dir, fallback_dir)

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 0, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 0, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        
        # Update robot position
        self.root_states[out_x_min, 0, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 0, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 0, 1] -= self.terrain.env_length + self.terrain.border_size
        
        # Update ball position to follow robot
        self.root_states[out_x_min, 1, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 1, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1, 1] -= self.terrain.env_length + self.terrain.border_size
        
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        if getattr(self, "manual_control", False):
            return
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        
        # Sample ball target direction (unit vector in robot local frame)
        # Sample angle uniformly and convert to unit vector
        target_angles = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        speed = torch.zeros(len(env_ids), 1, dtype=torch.float, device=self.device)
        self.commands[env_ids, 0] = torch.cos(target_angles) * speed.squeeze(1)  # ball_target_dir_x
        self.commands[env_ids, 1] = torch.sin(target_angles) * speed.squeeze(1)  # ball_target_dir_y
            
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def _resample_play_commands(self):
        """Fixed command sequence for play mode: forward, sideways, turning, still (3 sec each)"""
        cycle_duration = 12.0  # Total cycle: 4 phases x 3 seconds
        current_time = self.common_step_counter * self.dt
        phase_time = current_time % cycle_duration
        
        # Determine which phase we're in (0-2)
        phase = int(phase_time // 2.0)
        phase = 0
        
        # Reset commands
        self.commands[:, :] = 0.0
        
        if phase == 0:
            self.commands[:, 0] = 1.0
        elif phase == 1:
            self.commands[:, 0] = 1.0
        elif phase == 2:
            self.commands[:, 1] = 1.0

    def _update_curriculum(self, env_ids):
        """Track interception success and adjust ball spawn difficulty."""
        curriculum_cfg = self.cfg.get("ball_curriculum", {})
        if not curriculum_cfg.get("enabled", False):
            return
        if len(env_ids) == 0:
            return

        # Record success/failure for resetting envs (vectorised)
        successes = self.ball_has_been_contacted[env_ids].float()
        n = len(successes)
        ring_len = len(self.ball_curriculum_success_ring)
        indices = (torch.arange(n, device=self.device) + self.ball_curriculum_ring_idx) % ring_len
        self.ball_curriculum_success_ring[indices] = successes
        self.ball_curriculum_ring_idx = int((self.ball_curriculum_ring_idx + n) % ring_len)

        # Evaluate success rate
        success_rate = self.ball_curriculum_success_ring.mean().item()
        num_levels = curriculum_cfg.get("num_levels", 4)
        advance_thresh = curriculum_cfg.get("advance_threshold", 0.6)
        retreat_thresh = curriculum_cfg.get("retreat_threshold", 0.3)

        if success_rate > advance_thresh and self.ball_curriculum_global_level < num_levels - 1:
            self.ball_curriculum_global_level += 1
            print(f"[Curriculum] Advanced to level {self.ball_curriculum_global_level} "
                  f"(success_rate={success_rate:.2f})")
        elif success_rate < retreat_thresh and self.ball_curriculum_global_level > 0:
            self.ball_curriculum_global_level -= 1
            print(f"[Curriculum] Retreated to level {self.ball_curriculum_global_level} "
                  f"(success_rate={success_rate:.2f})")

        self.ball_curriculum_level[:] = self.ball_curriculum_global_level

    def _resample_curriculum_commands(self, env_ids):
        # Curriculum not used for dribbling task - use standard resampling
        pass

    def step(self, actions):
        # pre physics step
        self.gait_frequency_offset = torch.clamp(actions[:, 12], -0.5, 0.5)
        self.gait_frequency[:] = self.gait_frequency_offset + 2.0
        actions = actions[:, :12]
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = torch.clip(
            self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions,
            min=self.dof_pos_limits[:, 0],
            max=self.dof_pos_limits[:, 1],
        )
        damping_vel_alpha = float(self.cfg["control"].get("damping_velocity_filter_alpha", 0.22))
        sim_dt = float(self.cfg["sim"]["dt"])

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.filtered_custom_dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            self.custom_dof_vel[:] = (self.dof_pos - self.prev_dof_pos) / sim_dt
            self.filtered_custom_dof_vel[:] = (
                (1.0 - damping_vel_alpha) * self.filtered_custom_dof_vel + damping_vel_alpha * self.custom_dof_vel
            )
            self.prev_dof_pos[:] = self.dof_pos
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Update ball state tensors
        self.ball_pos[:] = self.root_states[:, 1, 0:3]
        self.ball_lin_vel[:] = self.body_states[:, -1, 7:10]
        self.ball_ang_vel[:] = self.body_states[:, -1, 10:13]
        
        # Update ball detection simulation (30 FPS camera with jitter)
        self._update_ball_detection()

        # Update robot state tensors
        self.base_pos[:] = self.root_states[:, 0, 0:3]
        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )

        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        self._kick_robots()
        self._push_robots()
        self.ball_pass_event_now[:] = self._compute_ball_pass_event_mask()
        self.ball_pass_event_latched |= self.ball_pass_event_now
        ball_speed_now = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        ball_speed_prev = torch.norm(self.last_ball_lin_vel_world[:, 0:2], dim=-1)
        self.ball_speed_drop[:] = ball_speed_prev - ball_speed_now
        self._update_intercept_state()
        self._check_termination()  # Sets self.reset_buf and potentially self.reset_ball_buf

        # Handle ball-only resets (ball too far, but robot is not resetting)
        ball_only_reset_env_ids = (self.reset_ball_buf & ~self.reset_buf).nonzero(as_tuple=False).flatten()
        if len(ball_only_reset_env_ids) > 0:
            self._reset_ball_at_robot_front(ball_only_reset_env_ids)
            self._update_pass_reference(ball_only_reset_env_ids)
            # Update the ball actors in the simulation
            ball_actor_indices = (2 * ball_only_reset_env_ids + 1).to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(ball_actor_indices),
                len(ball_actor_indices)
            )
            # Update convenience tensors for the reset balls
            self.ball_pos[ball_only_reset_env_ids] = self.root_states[ball_only_reset_env_ids, 1, 0:3]
            self.ball_rot[ball_only_reset_env_ids] = self.root_states[ball_only_reset_env_ids, 1, 3:7]
            self.ball_lin_vel[ball_only_reset_env_ids] = 0.0
            self.ball_ang_vel[ball_only_reset_env_ids] = 0.0
            # Reset ball detection state for ball-only resets (with detection noise)
            ball_pos_world_frame = self.ball_pos[ball_only_reset_env_ids] - self.base_pos[ball_only_reset_env_ids]
            relative_ball_pos = quat_rotate_inverse(self.base_quat[ball_only_reset_env_ids], ball_pos_world_frame)
            noisy_relative_xy = apply_randomization(relative_ball_pos[:, 0:2], self.cfg["noise"].get("ball_pos"))
            self.perceived_ball_pos_relative[ball_only_reset_env_ids] = noisy_relative_xy
            self.last_perceived_ball_pos_relative[ball_only_reset_env_ids] = noisy_relative_xy
            self.lagged_perceived_ball_pos_relative[ball_only_reset_env_ids] = noisy_relative_xy
            self.ball_detection_timer[ball_only_reset_env_ids] = torch_rand_float(
                0.0, self.ball_detection_interval, (len(ball_only_reset_env_ids), 1), device=self.device
            ).squeeze(-1)
            self.lag_snapshot_timer[ball_only_reset_env_ids] = self._sample_lag_snapshot_interval(len(ball_only_reset_env_ids))
            self.ball_detection_age[ball_only_reset_env_ids] = 0.0
            self.ball_pass_event_now[ball_only_reset_env_ids] = False
            self.ball_still_time_buf[ball_only_reset_env_ids] = 0.0
            self.ball_has_been_contacted[ball_only_reset_env_ids] = False
            self.ball_first_contact_event[ball_only_reset_env_ids] = False
            self.ball_first_contact_time[ball_only_reset_env_ids] = 0.0
            self.time_since_first_contact[ball_only_reset_env_ids] = 0.0
            self.ball_max_progress_along_shot[ball_only_reset_env_ids] = 0.0
            self.prev_ball_max_progress_along_shot[ball_only_reset_env_ids] = 0.0
            self.ball_speed_drop[ball_only_reset_env_ids] = 0.0
            self.intercept_lateral_error[ball_only_reset_env_ids] = 0.0
            self.intercept_forward_error[ball_only_reset_env_ids] = 0.0
            self.intercept_point_local[ball_only_reset_env_ids] = 0.0
            self.intercept_time_estimate[ball_only_reset_env_ids] = 0.0
            self.intercept_closest_approach[ball_only_reset_env_ids] = 0.0
            self.intercept_heading_error[ball_only_reset_env_ids] = 0.0
            self.ball_speed_toward_robot[ball_only_reset_env_ids] = 0.0
            self.ball_has_passed_robot[ball_only_reset_env_ids] = False
            self.clear_miss_time_buf[ball_only_reset_env_ids] = 0.0
            self.intercept_phase[ball_only_reset_env_ids] = 0
            self.intercept_phase_onehot[ball_only_reset_env_ids] = 0.0
            self.last_intercept_lateral_abs[ball_only_reset_env_ids] = 0.0
            self.intercept_feature_history[ball_only_reset_env_ids] = 0.0
            # Seed initial progress so penalty starts at zero for freshly spawned balls
            rel_ball_origin_bor = self.ball_pos[ball_only_reset_env_ids, 0:2] - self.pass_ref_origin_xy[ball_only_reset_env_ids]
            self.ball_initial_progress[ball_only_reset_env_ids] = torch.sum(
                rel_ball_origin_bor * self.pass_ref_dir_xy[ball_only_reset_env_ids], dim=-1
            )
            self.last_ball_distance_to_robot[ball_only_reset_env_ids] = torch.norm(
                self.ball_pos[ball_only_reset_env_ids, 0:2] - self.base_pos[ball_only_reset_env_ids, 0:2], dim=-1
            )
            self.reset_ball_buf[ball_only_reset_env_ids] = False

        # Track max ball progress along shot direction (monotonically increasing)
        # Subtract initial progress so freshly spawned balls start at 0
        rel_ball_origin = self.ball_pos[:, 0:2] - self.pass_ref_origin_xy
        progress = torch.sum(rel_ball_origin * self.pass_ref_dir_xy, dim=-1) - self.ball_initial_progress
        self.prev_ball_max_progress_along_shot[:] = self.ball_max_progress_along_shot
        self.ball_max_progress_along_shot = torch.max(
            self.ball_max_progress_along_shot, progress
        )

        # Precompute receive mode mask for reward functions
        target_speed = torch.norm(self.commands[:, 0:2], dim=-1)
        self.receive_mode_mask = (target_speed <= 1e-6).float()

        if self.is_play:
            self._draw_debug_lines()

        self._compute_reward()
        self.last_ball_distance_to_robot[:] = torch.norm(self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1)
        self.last_intercept_lateral_abs[:] = torch.abs(self.intercept_lateral_error)
        self._log_rewards_to_csv()

        # Update last_ball_lin_vel_world before potential full reset
        self.last_ball_lin_vel_world[:] = self.body_states[:, -1, 7:10]

        # --- Capture terminal metrics BEFORE _reset_idx wipes buffers ---
        # Only emit non-zero for done envs so the recorder's per-step
        # accumulation yields the single terminal snapshot value.
        done_mask = self.reset_buf  # bool (N,)
        done_ids = done_mask.nonzero(as_tuple=False).flatten()

        metric_pass_event = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        metric_pass_event[done_ids] = self.ball_pass_event_latched[done_ids].float()

        metric_no_pass_success = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        metric_no_pass_success[done_ids] = (~self.ball_pass_event_latched[done_ids]).float()

        metric_curriculum_level = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        metric_curriculum_level[done_ids] = float(self.ball_curriculum_global_level)

        metric_first_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        metric_first_contact[done_ids] = self.ball_first_contact_time[done_ids]

        metric_first_contact_success = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        metric_first_contact_success[done_ids] = self.ball_has_been_contacted[done_ids].float()

        metric_max_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        metric_max_progress[done_ids] = self.ball_max_progress_along_shot[done_ids]

        metric_clear_miss = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        metric_clear_miss[done_ids] = self.ball_has_passed_robot[done_ids].float() * (~self.ball_has_been_contacted[done_ids]).float()

        metric_post_contact_front = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        ball_rel_local_terminal = quat_rotate_inverse(self.base_quat, self.ball_pos - self.base_pos)
        metric_post_contact_front[done_ids] = (
            self.ball_has_been_contacted[done_ids].float()
            * (ball_rel_local_terminal[done_ids, 0] > 0.0).float()
        )

        env_ids = done_ids
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
            self.reset_ball_buf[env_ids] = False  # Ball reset is handled by full reset
            self.last_ball_lin_vel_world[env_ids] = 0.0

        self._teleport_robot()
        self._resample_commands()
        target_speed = torch.norm(self.commands[:, 0:2], dim=-1)
        self.receive_mode_mask = (target_speed <= 1e-6).float()
        self._update_intercept_state()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.extras["metrics"] = {
            "pass_event": metric_pass_event,
            "no_pass_success_terminal": metric_no_pass_success,
            "ball_curriculum_level": metric_curriculum_level,
            "ball_first_contact_time": metric_first_contact,
            "ball_first_contact_success": metric_first_contact_success,
            "ball_max_progress": metric_max_progress,
            "clear_miss_terminal": metric_clear_miss,
            "post_contact_front_terminal": metric_post_contact_front,
        }

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _draw_debug_lines(self):
        """Draw debug lines for the ball target direction vector."""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
            
        self.gym.clear_lines(self.viewer)
        
        # Length of the debug line (in meters)
        line_length = 1.0
        
        for env_idx in range(self.num_envs):
            # Get ball position in world frame
            ball_pos = self.ball_pos[env_idx].cpu().numpy()
            
            # Get target direction (unit vector) - treating as world frame
            target_dir = np.array([
                self.commands[env_idx, 0].cpu().numpy(),  # ball_target_dir_x
                self.commands[env_idx, 1].cpu().numpy(),  # ball_target_dir_y
                0.0  # Keep on horizontal plane
            ], dtype=np.float32)
            
            # Calculate end point of the line
            end_pos = ball_pos + line_length * target_dir
            
            # Raise the line slightly above ground for visibility
            ball_pos[2] = max(ball_pos[2], 0.15)
            end_pos[2] = max(end_pos[2], 0.15)
            
            # Create vertices array: [p1.x, p1.y, p1.z, p2.x, p2.y, p2.z]
            vertices = np.array([ball_pos[0], ball_pos[1], ball_pos[2],
                                 end_pos[0], end_pos[1], end_pos[2]], dtype=np.float32)
            
            # Color: bright green for target direction
            colors = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            # Draw the line
            self.gym.add_lines(self.viewer, self.envs[env_idx], 1, vertices, colors)
            

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(apply_randomization(0.0, self.cfg["randomization"].get("kick_interval_s")) / self.dt) == 0:
            self.root_states[:, 0, 7:10] = apply_randomization(self.root_states[:, 0, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 0, 10:13] = apply_randomization(self.root_states[:, 0, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(apply_randomization(0.0, self.cfg["randomization"].get("push_interval_s")) / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(apply_randomization(0.0, self.cfg["randomization"].get("push_interval_s")) / self.dt) == np.ceil(
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

    def _update_ball_detection(self):
        """
        Simulates ball detection at ~30 FPS with jitter.
        Updates perceived_ball_pos_relative and last_perceived_ball_pos_relative when detection timer expires.
        Both positions are stored in robot frame and only change when a new detection occurs.
        Detection noise is applied once at detection time (not every policy step).
        """
        # Decrement detection timer
        self.ball_detection_timer -= self.dt
        self.lag_snapshot_timer -= self.dt
        self.ball_detection_age += self.dt
        
        # Find environments where detection should occur (timer expired)
        detect_mask = self.ball_detection_timer <= 0
        
        if detect_mask.any():
            detect_ids = detect_mask.nonzero(as_tuple=False).flatten()
            
            # Compute current ball position in robot frame for detected envs
            ball_pos_world_frame = self.ball_pos[detect_ids] - self.base_pos[detect_ids]
            relative_ball_pos = quat_rotate_inverse(self.base_quat[detect_ids], ball_pos_world_frame)
            current_relative_xy = relative_ball_pos[:, 0:2]
            
            # Apply detection noise (sampled once per detection, stays constant until next detection)
            current_relative_xy = apply_randomization(current_relative_xy, self.cfg["noise"].get("ball_pos"))
            
            # Shift current to last (previous detection becomes "last")
            self.last_perceived_ball_pos_relative[detect_ids] = self.perceived_ball_pos_relative[detect_ids].clone()
            
            # Update current perceived position (in robot frame, with noise baked in)
            self.perceived_ball_pos_relative[detect_ids] = current_relative_xy

            # Refresh randomized lag snapshot only when due, using detected positions.
            snapshot_due = self.lag_snapshot_timer[detect_ids] <= 0.0
            if snapshot_due.any():
                snapshot_ids = detect_ids[snapshot_due]
                self.lagged_perceived_ball_pos_relative[snapshot_ids] = current_relative_xy[snapshot_due]
                self.lag_snapshot_timer[snapshot_ids] = self._sample_lag_snapshot_interval(len(snapshot_ids))

            self.ball_detection_age[detect_ids] = 0.0
            
            # Reset detection timer with randomized interval
            jitter_scale = 1.0 + torch_rand_float(
                -self.ball_detection_jitter, self.ball_detection_jitter, 
                (len(detect_ids), 1), device=self.device
            ).squeeze(-1)
            self.ball_detection_timer[detect_ids] = self.ball_detection_interval * jitter_scale

    def _compute_ball_pass_event_mask(self):
        """Detect if ball passed behind the robot and is moving away along incoming direction."""
        target_speed = torch.norm(self.commands[:, 0:2], dim=-1)
        receive_mode_eps = float(self.cfg["rewards"].get("receive_mode_target_speed_eps", 1e-6))
        receive_mode = target_speed <= receive_mode_eps

        shot_dir = self.pass_ref_dir_xy
        rel_ball_robot = self.ball_pos[:, 0:2] - self.base_pos[:, 0:2]
        progress_along_shot = torch.sum(rel_ball_robot * shot_dir, dim=-1)
        margin_x = float(self.cfg["rewards"].get("ball_passed_margin_x", 0.03))
        passed_behind = progress_along_shot > margin_x

        ball_vel_xy = self.ball_lin_vel[:, 0:2]
        away_dot = torch.sum(ball_vel_xy * shot_dir, dim=-1)
        away_min = float(self.cfg["rewards"].get("ball_passed_away_min", 0.04))
        moving_away = away_dot > away_min

        ball_speed = torch.norm(ball_vel_xy, dim=-1)
        min_speed = float(self.cfg["rewards"].get("ball_passed_speed_threshold", 0.06))
        moving = ball_speed > min_speed

        return receive_mode & passed_behind & moving_away & moving

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        _, pitch, _ = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_pitch[:] = (pitch.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Compute relative yaw to trunk
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        self.feet_yaw_rel = (self.feet_yaw - base_yaw.unsqueeze(-1) + torch.pi) % (2 * torch.pi) - torch.pi
        
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
        """Check if environments need to be reset"""
        contact_terminate = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)

        # Use separate limits for linear and angular velocity to avoid mixing units (m/s and rad/s).
        lin_vel_sq = self.root_states[:, 0, 7:10].square().sum(dim=-1)
        ang_vel_sq = self.root_states[:, 0, 10:13].square().sum(dim=-1)
        rewards_cfg = self.cfg["rewards"]
        if "terminate_lin_vel" in rewards_cfg and "terminate_ang_vel" in rewards_cfg:
            terminate_lin_vel = float(rewards_cfg["terminate_lin_vel"])
            terminate_ang_vel = float(rewards_cfg["terminate_ang_vel"])
            lin_vel_terminate = lin_vel_sq > (terminate_lin_vel * terminate_lin_vel)
            ang_vel_terminate = ang_vel_sq > (terminate_ang_vel * terminate_ang_vel)
            velocity_terminate = lin_vel_terminate | ang_vel_terminate
        else:
            raise KeyError("Missing velocity termination thresholds in rewards config")
        height_terminate = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        timeout_terminate = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        ball_speed_xy = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        ball_still_speed_threshold = self.cfg["rewards"].get("ball_still_speed_threshold", 0.05)
        ball_still_required_time = self.cfg["rewards"].get("ball_still_termination_s", 4.0)
        ball_is_still = ball_speed_xy < ball_still_speed_threshold
        self.ball_still_time_buf = torch.where(
            ball_is_still,
            self.ball_still_time_buf + self.dt,
            torch.zeros_like(self.ball_still_time_buf),
        )
        ball_still_terminate = self.ball_still_time_buf >= ball_still_required_time
        clear_miss_terminate = self.clear_miss_time_buf >= float(self.cfg["rewards"].get("clear_miss_termination_s", 0.35))

        self.reset_buf = contact_terminate | velocity_terminate | height_terminate | timeout_terminate | ball_still_terminate | clear_miss_terminate
        self.time_out_buf = timeout_terminate
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time
        
        # Check if ball is too far from robot (mark for ball-only reset or full reset)
        max_ball_distance = self.cfg["rewards"].get("max_ball_distance", 3.0)
        ball_too_far = torch.norm(self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1) > max_ball_distance
        self.reset_ball_buf = ball_too_far  # Mark ball for reset

        if self.debug_termination and (self.common_step_counter % self.debug_termination_interval == 0):
            reset_count = int(self.reset_buf.sum().item())
            ball_only_reset_mask = self.reset_ball_buf & ~self.reset_buf
            ball_only_reset_count = int(ball_only_reset_mask.sum().item())
            if reset_count > 0 or ball_only_reset_count > 0:
                print(
                    "[termination] "
                    f"step={self.common_step_counter} "
                    f"reset={reset_count} "
                    f"contact={int(contact_terminate.sum().item())} "
                    f"velocity={int(velocity_terminate.sum().item())} "
                    f"velocity_lin={int(lin_vel_terminate.sum().item())} "
                    f"velocity_ang={int(ang_vel_terminate.sum().item())} "
                    f"height={int(height_terminate.sum().item())} "
                    f"timeout={int(timeout_terminate.sum().item())} "
                    f"ball_still={int(ball_still_terminate.sum().item())} "
                    f"clear_miss={int(clear_miss_terminate.sum().item())} "
                    f"ball_only_reset={ball_only_reset_count}"
                )
                if reset_count > 0:
                    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()[: self.debug_termination_max_envs].tolist()
                    print(f"[termination] reset env ids (sample): {reset_env_ids}")
                if ball_only_reset_count > 0:
                    ball_only_env_ids = ball_only_reset_mask.nonzero(as_tuple=False).flatten()[: self.debug_termination_max_envs].tolist()
                    print(f"[termination] ball-only env ids (sample): {ball_only_env_ids}")

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        """Compute explicit interception observations for single-policy pass receive."""
        # Use a trimmed, deployment-oriented observation subset.
        # Ball observations mimic firmware-style perception: current and previous
        # 30 FPS detections in robot frame, without exposing direct ball velocity.
        current_relative_ball_pos_xy = self.perceived_ball_pos_relative
        past_relative_ball_pos_xy = self.last_perceived_ball_pos_relative
        
        # Convert target direction from world frame to robot local frame for observation
        # Commands[:, 1:3] stores target direction in world coordinates
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        world_dir_x = self.commands[:, 0]
        world_dir_y = self.commands[:, 1]
        # Rotate by -yaw to convert to local frame
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        local_target_dir_x = cos_yaw * world_dir_x + sin_yaw * world_dir_y
        local_target_dir_y = -sin_yaw * world_dir_x + cos_yaw * world_dir_y
        
        # Commands scale: gait_frequency (1), ball_target_dir_x (1), ball_target_dir_y (1)
        commands_scale = torch.tensor(
            [
                self.cfg["normalization"]["ball_target_vel"],   # 1: ball_target_dir_x (unit vector, no scaling needed)
                self.cfg["normalization"]["ball_target_vel"],   # 2: ball_target_dir_y (unit vector, no scaling needed)
            ],
            device=self.device,
        )
        
        # Build command observation with local frame target direction
        commands_obs = torch.stack([
            local_target_dir_x,   # target direction x in local frame
            local_target_dir_y,   # target direction y in local frame
        ], dim=-1)

        heading_error_sin = torch.sin(self.intercept_heading_error).unsqueeze(-1)
        heading_error_cos = torch.cos(self.intercept_heading_error).unsqueeze(-1)
        ball_passed_obs = self.ball_has_passed_robot.float().unsqueeze(-1)
        intercept_lateral_obs = self.intercept_lateral_error.unsqueeze(-1)
        intercept_forward_obs = self.intercept_forward_error.unsqueeze(-1)
        self.obs_buf = torch.cat(
            (
                # Proprioceptive observations (same as walking model)
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],  # 3
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],  # 3
                # Commands: gait_frequency + ball target direction (in local frame)
                commands_obs * commands_scale,  # 2 (target_dir_x_local, target_dir_y_local)
                # Ball observations (using perceived positions from 30 FPS detection)
                # Noise is already baked in at detection time, not applied every step
                current_relative_ball_pos_xy * self.cfg["normalization"]["ball_pos"],  # 2
                past_relative_ball_pos_xy * self.cfg["normalization"]["ball_pos"],  # 2
                self.gait_frequency_offset.unsqueeze(-1) * self.cfg["normalization"]["gait_frequency_offset"],  # 1
                intercept_lateral_obs * self.cfg["normalization"].get("ball_pos", 1.0),  # 1
                intercept_forward_obs * self.cfg["normalization"].get("ball_pos", 1.0),  # 1
                self.intercept_point_local * self.cfg["normalization"].get("ball_pos", 1.0),  # 2
                heading_error_sin,  # 1
                heading_error_cos,  # 1
                ball_passed_obs,  # 1
                # Gait process (same as walking model)
                (torch.cos(2 * torch.pi * self.gait_process)).unsqueeze(-1),  # 1
                (torch.sin(2 * torch.pi * self.gait_process)).unsqueeze(-1),  # 1
                # Joint state (same as walking model)
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],  # 12
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],  # 12
            ),
            dim=-1,
        )
        
        # Update last_relative_ball_pos for next frame
        self.last_relative_ball_pos[:] = current_relative_ball_pos_xy
        
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                # Add ball velocity in privileged observations (x, y in world frame)
                self.ball_lin_vel[:, 0:3] * self.cfg["normalization"]["ball_vel"],
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------
    def _upright_posture_term(self):
        # Uprightness from roll/pitch error in world frame.
        roll, pitch, _ = get_euler_xyz(self.base_quat)
        roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
        pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
        sigma = self.cfg["rewards"].get("upright_sigma", 0.10)
        return torch.exp(-(torch.square(roll) + torch.square(pitch)) / sigma)

    def _stability_gate(self):
        # Soft gate to unlock approach/receive rewards once robot is somewhat upright and tall enough.
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        gate_height = self.cfg["rewards"].get("stability_gate_height", 0.46)
        gate_sigma = self.cfg["rewards"].get("stability_gate_sigma", 0.03)
        return torch.sigmoid((base_height - gate_height) / gate_sigma) * self._upright_posture_term()

    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_upright_posture(self):
        return self._upright_posture_term()

    def _reward_standing_stability(self):
        # Encourage low planar linear/angular base velocities for stand-and-balance behavior.
        lin_vel_sigma = self.cfg["rewards"].get("standing_lin_vel_sigma", 0.35)
        ang_vel_sigma = self.cfg["rewards"].get("standing_ang_vel_sigma", 1.0)
        lin_term = torch.exp(-torch.sum(torch.square(self.filtered_lin_vel[:, 0:2]), dim=-1) / lin_vel_sigma)
        ang_term = torch.exp(-torch.sum(torch.square(self.filtered_ang_vel[:, 0:2]), dim=-1) / ang_vel_sigma)
        return lin_term * ang_term

    def _reward_approach_progress(self):
        # Reward only positive progress toward the intercept line in receive mode.
        distance = torch.norm(self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1)
        delta = self.last_ball_distance_to_robot - distance
        clip_value = max(float(self.cfg["rewards"].get("approach_progress_clip", 0.03)), 1.0e-6)
        progress = torch.clamp(delta, min=0.0, max=clip_value) / clip_value
        intercept_delta = self.last_intercept_lateral_abs - torch.abs(self.intercept_lateral_error)
        intercept_progress = torch.clamp(intercept_delta, min=0.0, max=clip_value) / clip_value
        pre_contact_gate = self.receive_mode_mask * (~self.ball_has_been_contacted).float() * (~self.ball_has_passed_robot).float()
        return torch.where(pre_contact_gate > 0.0, intercept_progress, progress)

    def _reward_receive_zone_control(self):
        # Reward controlling the ball near the robot with low ball speed after real contact.
        distance = torch.norm(self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1)
        ball_speed_xy = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        dist_sigma = self.cfg["rewards"].get("receive_dist_sigma", 0.20)
        speed_sigma = self.cfg["rewards"].get("receive_speed_sigma", 0.12)
        r_dist = torch.exp(-torch.square(distance) / dist_sigma)
        r_speed = torch.exp(-torch.square(ball_speed_xy) / speed_sigma)
        ball_rel_local = quat_rotate_inverse(self.base_quat, self.ball_pos - self.base_pos)
        front_min = float(self.cfg["rewards"].get("receive_front_min_x", 0.02))
        front_sigma = max(float(self.cfg["rewards"].get("receive_front_sigma", 0.08)), 1.0e-6)
        front_gate = torch.sigmoid((ball_rel_local[:, 0] - front_min) / front_sigma)
        lateral_sigma = max(float(self.cfg["rewards"].get("receive_lateral_sigma", 0.12)), 1.0e-6)
        lateral_gate = torch.exp(-torch.square(ball_rel_local[:, 1]) / lateral_sigma)
        # Time decay — early ball control is much more valuable
        episode_time = self.episode_length_buf.float() * self.dt
        tau = float(self.cfg["rewards"].get("ball_reward_time_decay_tau", 3.0))
        time_multiplier = torch.exp(-episode_time / tau)
        post_contact_gate = self.receive_mode_mask * self.ball_has_been_contacted.float()
        return r_dist * r_speed * front_gate * lateral_gate * time_multiplier * post_contact_gate

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(-torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(-torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_height_margin(self):
        # Exponential penalty near terminate_height (strong when base is close to the cutoff).
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        margin = torch.clamp(base_height - self.cfg["rewards"]["terminate_height"], min=0.0)
        sigma = self.cfg["rewards"].get("height_margin_sigma", 0.05)
        return torch.exp(-margin / sigma)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        """
        Reward for tracking body pitch and roll targets.
        Computes reward based on:
         - normalized roll angle (minus body_roll_target)
         - normalized pitch angle (minus body_pitch_target)
        Result: orient_reward = roll_error^2 + pitch_error^2
        """
        # Get all Euler angles (roll, pitch, yaw) from base_quat
        roll_all, pitch_all, _ = get_euler_xyz(self.base_quat)

        # Normalize to [-π, +π]
        roll_norm = (roll_all + torch.pi) % (2 * torch.pi) - torch.pi
        pitch_norm = (pitch_all + torch.pi) % (2 * torch.pi) - torch.pi

        # Get target values from commands
        target_pitch = self.commands[:, 6]  # body_pitch_target
        target_roll = self.commands[:, 7]   # body_roll_target

        # Calculate errors
        roll_error = roll_norm - target_roll
        pitch_error = pitch_norm - target_pitch

        # Return quadratic reward (smaller is better)
        orient_reward = torch.square(roll_error) + torch.square(pitch_error)
        return orient_reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 0, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_pitch(self):
        return torch.sum(torch.square(self.feet_pitch), dim=-1)

    def _reward_feet_yaw_diff(self):
        """
        Reward for tracking the commanded difference between left and right foot yaw angles.
        Instead of penalizing asymmetry, this now rewards tracking the commanded difference.
        """
        # Get commanded foot yaw difference
        commanded_diff = self.commands[:, 5] - self.commands[:, 4]  # foot_yaw_R - foot_yaw_L
        
        # Get actual foot yaw difference (relative to trunk)
        actual_diff = self.feet_yaw_rel[:, 1] - self.feet_yaw_rel[:, 0]  # right - left
        
        # Normalize difference to [-π, π]
        diff_error = (actual_diff - commanded_diff + torch.pi) % (2 * torch.pi) - torch.pi
        
        return torch.square(diff_error)

    def _reward_feet_yaw_mean(self):
        """
        Reward for tracking the commanded mean foot yaw angle.
        Instead of penalizing deviation from base yaw, this now rewards tracking the commanded mean.
        """
        # Get commanded foot yaw mean
        commanded_mean = (self.commands[:, 5] + self.commands[:, 4]) * 0.5  # (foot_yaw_R + foot_yaw_L) / 2
        
        # Get actual foot yaw mean (relative to trunk)
        actual_mean = self.feet_yaw_rel.mean(dim=-1)
        
        # Normalize mean to [-π, π]
        mean_error = (actual_mean - commanded_mean + torch.pi) % (2 * torch.pi) - torch.pi
        
        return torch.square(mean_error)

    def _reward_feet_offset_x(self):
        """Reward for tracking feet x-offset target, scaled by forward velocity"""
        # Get feet x-offset using existing helper function
        feet_x_offset, _ = self.get_feet_offset()
        
        # Get target x-offset from commands
        target_x_offset = self.commands[:, 8]  # feet_offset_x_target
        
        # Calculate error
        x_error = feet_x_offset - target_x_offset
        
        # Apply clipping similar to original feet_distance reward
        x_reward = torch.clip(torch.abs(x_error), min=0.0, max=0.1)
        
        # Get forward velocity (x-direction) from commands
        forward_vel = self.commands[:, 0]  # lin_vel_x
        
        # Get maximum forward velocity from config
        max_forward_vel = max(abs(self.cfg["commands"]["lin_vel_x"][0]), abs(self.cfg["commands"]["lin_vel_x"][1]))
        
        # Calculate velocity scaling factor: 1.0 at vel=0, 0.0 at vel=max_vel
        # Use absolute value of velocity for symmetric scaling
        # Quadratic decrease: vel_scale = (1.0 - |velocity| / max_velocity)^2
        vel_scale = torch.clamp((1.0 - torch.abs(forward_vel) / max_forward_vel) ** 2, min=0.0, max=1.0)
        
        # Scale reward by velocity factor
        x_reward = x_reward * vel_scale
        
        return x_reward

    def _reward_feet_offset_y(self):
        """Reward for tracking feet y-offset target, scaled by lateral velocity"""
        # Get feet y-offset using existing helper function
        _, feet_y_offset = self.get_feet_offset()
        
        # Get target y-offset from commands
        target_y_offset = self.commands[:, 9]  # feet_offset_y_target
        
        # Calculate error
        y_error = feet_y_offset - target_y_offset
        
        # Apply clipping similar to original feet_distance reward
        y_reward = torch.clip(torch.abs(y_error), min=0.0, max=0.1)
        
        # Get lateral velocity (y-direction) from commands
        lateral_vel = self.commands[:, 1]  # lin_vel_y
        
        # Get maximum lateral velocity from config
        max_lateral_vel = max(abs(self.cfg["commands"]["lin_vel_y"][0]), abs(self.cfg["commands"]["lin_vel_y"][1]))
        
        # Calculate velocity scaling factor: 1.0 at vel=0, 0.0 at vel=max_vel
        # Use absolute value of velocity for symmetric scaling
        # Quadratic decrease: vel_scale = (1.0 - |velocity| / max_velocity)^2
        vel_scale = torch.clamp((1.0 - torch.abs(lateral_vel) / max_lateral_vel) ** 2, min=0.0, max=1.0)
        
        # Scale reward by velocity factor
        y_reward = y_reward * vel_scale
        
        return y_reward

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()

    def _reward_foot_yaw_L(self):
        """Reward for tracking left foot yaw angle"""
        error = self.feet_yaw_rel[:, 0] - self.commands[:, 4]
        return torch.square(error)

    def _reward_foot_yaw_R(self):
        """Reward for tracking right foot yaw angle"""
        error = self.feet_yaw_rel[:, 1] - self.commands[:, 5]
        return torch.square(error)

    def get_feet_offset(self):
        """
        Helper function to calculate feet offsets in robot coordinates.
        Returns both x and y offsets between right and left feet (right - left).
        For y-offset, the feet_distance_ref is subtracted to get the relative offset.
        """
        # Get base yaw to transform to robot coordinates
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        
        # Calculate feet positions in robot coordinates
        # Transform from world to robot coordinates
        feet_x_offset = (
            torch.cos(base_yaw) * (self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]) +
            torch.sin(base_yaw) * (self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1])
        )
        
        feet_y_offset = (
            -torch.sin(base_yaw) * (self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]) +
            torch.cos(base_yaw) * (self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1])
        )
        
        # Subtract feet_distance_ref from y-offset to get relative offset
        feet_y_offset = feet_y_offset - self.cfg["rewards"]["feet_distance_ref"]
        
        return feet_x_offset, feet_y_offset

    def get_feet_x_offset(self):
        """
        Helper function to calculate feet x-offset in robot coordinates.
        Returns the x-offset between right and left feet (right - left).
        """
        feet_x_offset, _ = self.get_feet_offset()
        return feet_x_offset

    # ------------ ball reward functions ----------------

    def _reward_ball_velocity_tracking(self):
        """
        Rewards ball velocity tracking.
        - For non-zero target velocity: cosine similarity (direction) + magnitude matching.
        - For zero target velocity: rewards low ball speed (stopping behavior).
        
        Output range: -1 to +1
          +1 = perfect direction AND perfect speed
          0  = perpendicular movement or stationary ball
          -1 = opposite direction with matching speed
        
        Config parameters:
          - ball_vel_tracking_sigma: controls sensitivity of speed matching (default: 1.0)
          - ball_vel_tracking_min_speed: minimum ball speed for reward (default: 0.1)
          - ball_vel_tracking_stop_sigma: controls sensitivity for zero-target stopping reward (default: same as ball_vel_tracking_sigma)
        """
        ball_vel_world = self.body_states[:, -1, 7:10]  # Ball velocity in world frame
        actual_vel = ball_vel_world[:, 0:2]  # XY velocity
        target_vel = self.commands[:, 0:2]   # Target XY velocity
        
        # Calculate speeds
        actual_speed = torch.norm(actual_vel, dim=-1)
        target_speed = torch.norm(target_vel, dim=-1)
        
        # Cosine similarity: measures direction alignment
        # Range: -1 (opposite) to +1 (aligned)
        dot_product = torch.sum(actual_vel * target_vel, dim=-1)
        cos_sim = dot_product / (actual_speed * target_speed + 1e-8)
        
        # Speed matching: exponential reward for matching target speed
        # Range: 0 (very different) to 1 (perfect match)
        sigma = self.cfg["rewards"].get("ball_vel_tracking_sigma", 1.0)
        speed_error = torch.abs(actual_speed - target_speed)
        speed_reward = torch.exp(-speed_error / sigma)
        
        # Combined reward: direction * speed_match
        # Range: -1 (opposite dir, good speed) to +1 (aligned, good speed)
        reward = cos_sim * speed_reward
        
        # For non-zero targets, ignore very small actual speeds to avoid noisy direction reward.
        min_speed = self.cfg["rewards"].get("ball_vel_tracking_min_speed", 0.1)
        is_moving = actual_speed > min_speed
        has_target = target_speed > 1e-6

        tracking_reward = reward * is_moving.float()

        # When no target velocity is commanded, reward stopping only if the robot
        # is close enough to plausibly control the ball (avoids passive waiting).
        stop_sigma = self.cfg["rewards"].get("ball_vel_tracking_stop_sigma", sigma)
        stop_reward = torch.exp(-actual_speed / stop_sigma)
        ball_distance = torch.norm(self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1)
        control_dist = self.cfg["rewards"].get("ball_stop_control_distance", 0.45)
        control_dist_sigma = self.cfg["rewards"].get("ball_stop_control_distance_sigma", 0.15)
        distance_gate = torch.exp(-torch.square(torch.clamp(ball_distance - control_dist, min=0.0)) / control_dist_sigma)
        stop_reward = stop_reward * distance_gate

        return torch.where(has_target, tracking_reward, stop_reward)

    def _reward_ball_distance_penalty(self):
        """Penalizes distance from robot to ball"""
        robot_pos = self.base_pos[:, 0:2]  # Robot XY position
        ball_pos = self.ball_pos[:, 0:2]    # Ball XY position

        # For moving-target tasks (non-zero command), target position stays behind the ball
        # in command direction. For ball-receive tasks (zero command), place target ahead
        # of the rolling ball to encourage interception instead of chasing.
        command_norm = torch.norm(self.commands[:, :2], dim=-1, keepdim=True)
        normed_commands = self.commands[:, :2] / (command_norm + 1e-8)
        has_target = command_norm.squeeze(-1) > 1e-6

        ball_vel_xy = self.ball_lin_vel[:, 0:2]
        ball_speed = torch.norm(ball_vel_xy, dim=-1, keepdim=True)
        ball_dir = ball_vel_xy / (ball_speed + 1e-8)
        moving_ball = ball_speed.squeeze(-1) > self.cfg["rewards"].get("ball_vel_tracking_min_speed", 0.1)

        dribble_offset = (0.1 + self.ball_radii).unsqueeze(-1)
        receive_offset = self.cfg["rewards"].get("ball_receive_intercept_offset", 0.25)

        target_pos_dribble = ball_pos - dribble_offset * normed_commands
        target_pos_receive = ball_pos + receive_offset * ball_dir
        target_pos_stop = ball_pos

        target_pos = torch.where(has_target.unsqueeze(-1), target_pos_dribble, target_pos_stop)
        target_pos = torch.where((~has_target & moving_ball).unsqueeze(-1), target_pos_receive, target_pos)

        # visualize target pos via a red line from ball to target pos
        z_coords = np.full(ball_pos.shape[0], 0.2, dtype=np.float32)
        colors = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if self.is_play:
            for env_idx in range(self.num_envs):
                vertices = np.array([ball_pos[env_idx, 0].cpu().numpy(), ball_pos[env_idx, 1].cpu().numpy(), 0.2, target_pos[env_idx, 0].cpu().numpy(), target_pos[env_idx, 1].cpu().numpy(), 0.2], dtype=np.float32)
                self.gym.add_lines(self.viewer, self.envs[env_idx], 1, vertices, colors)
            
        distance = torch.norm(target_pos - robot_pos, dim=-1)
        distance = torch.clamp(distance, min=0.0, max=self.cfg["rewards"].get("max_ball_distance", 3.0))
        
        # Exponential penalty (closer = less penalty)
        sigma = self.cfg["rewards"].get("ball_distance_sigma", 1.0)
        penalty = torch.exp(distance / sigma) - 1.0  # 0 penalty at distance=0
        return penalty

    def _reward_ball_height_penalty(self):
        """Penalizes ball being off the ground (kicked too hard upward)"""
        ball_height = self.ball_pos[:, 2] - self.ball_radii
        
        # Only penalize if ball is significantly above ground
        height_error = torch.clamp(ball_height - 0.02, min=0.0)
        
        return torch.square(height_error)

    def _reward_look_at_ball(self):
        """Rewards the robot for facing towards the ball.
        
        Computes the angular difference between the robot's heading (yaw)
        and the direction to the ball. Returns an exponential reward that
        is maximized when the robot is facing the ball.
        """
        # Get robot yaw angle
        _, _, robot_yaw = get_euler_xyz(self.base_quat)
        
        # Compute direction from robot to ball in world frame
        ball_dir = self.ball_pos[:, 0:2] - self.base_pos[:, 0:2]
        
        # Compute angle to ball
        angle_to_ball = torch.atan2(ball_dir[:, 1], ball_dir[:, 0])
        
        # Compute angular error (normalized to [-pi, pi])
        angle_error = (robot_yaw - angle_to_ball + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Exponential reward (max 1.0 when facing ball, decreases with error)
        sigma = self.cfg["rewards"].get("look_at_ball_sigma", 0.5)
        reward = torch.exp(-torch.square(angle_error) / sigma)
        
        return reward

    def _reward_ball_stop_near_start(self):
        """Reward stopping the ball near the robot start point for receive/control behavior."""
        ball_pos_xy = self.ball_pos[:, 0:2]
        start_pos_xy = self.episode_start_base_pos_xy
        ball_speed = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)

        # Reward proximity of stopped ball to episode start position.
        distance = torch.norm(ball_pos_xy - start_pos_xy, dim=-1)
        dist_sigma = self.cfg["rewards"].get("ball_stop_near_start_sigma", 0.7)
        near_start_reward = torch.exp(-torch.square(distance) / dist_sigma)

        # Gate by low speed so this term mostly matters when ball is being controlled/stopped.
        stop_sigma = self.cfg["rewards"].get("ball_stop_near_start_speed_sigma", 0.2)
        stop_gate = torch.exp(-ball_speed / stop_sigma)

        # Apply this objective only for zero-velocity command mode.
        target_speed = torch.norm(self.commands[:, 0:2], dim=-1)
        no_target = target_speed <= 1e-6

        # Time decay — early stopping is much more valuable
        episode_time = self.episode_length_buf.float() * self.dt
        tau = float(self.cfg["rewards"].get("ball_reward_time_decay_tau", 3.0))
        time_multiplier = torch.exp(-episode_time / tau)
        post_contact_gate = self.ball_has_been_contacted.float()
        return near_start_reward * stop_gate * no_target.float() * time_multiplier * post_contact_gate

    def _reward_perceived_intercept_timing(self):
        """Reward early interception timing using explicit perceived geometry."""
        lateral_sigma = max(float(self.cfg["rewards"].get("perceived_intercept_lateral_sigma", 0.12)), 1e-6)
        forward_min = float(self.cfg["rewards"].get("perceived_intercept_forward_min", 0.05))
        forward_sigma = max(float(self.cfg["rewards"].get("perceived_intercept_forward_sigma", 0.08)), 1e-6)
        tti_target = float(self.cfg["rewards"].get("intercept_arrival_tti_target", 0.18))
        tti_sigma = max(float(self.cfg["rewards"].get("intercept_arrival_tti_sigma", 0.12)), 1e-6)
        speed_sigma = max(float(self.cfg["rewards"].get("intercept_speed_sigma", 0.08)), 1e-6)
        min_toward_speed = float(self.cfg["rewards"].get("intercept_approach_min_speed", 0.08))

        lateral_term = torch.exp(-torch.square(self.intercept_lateral_error) / lateral_sigma)
        forward_term = torch.sigmoid((self.intercept_forward_error - forward_min) / forward_sigma)
        timing_term = torch.exp(-torch.square(self.intercept_time_estimate - tti_target) / tti_sigma)
        speed_term = torch.sigmoid((self.ball_speed_toward_robot - min_toward_speed) / speed_sigma)
        pre_contact_gate = self.receive_mode_mask * (~self.ball_has_been_contacted).float() * (~self.ball_has_passed_robot).float()
        return lateral_term * forward_term * timing_term * speed_term * pre_contact_gate

    def _reward_shot_path_alignment(self):
        """Reward standing on the incoming ball path and slightly ahead of the ball."""
        shot_path_lateral_s = max(float(self.cfg["rewards"].get("shot_path_lateral_sigma", 0.10)), 1e-6)
        r_lateral = torch.exp(-torch.square(self.intercept_lateral_error) / shot_path_lateral_s)
        
        shot_path_forward_s = max(float(self.cfg["rewards"].get("shot_path_forward_sigma", 0.08)), 1e-6)
        shot_path_forward_min = float(self.cfg["rewards"].get("shot_path_forward_min", 0.05))
        r_forward = torch.sigmoid((self.intercept_forward_error - shot_path_forward_min) / shot_path_forward_s)

        reward = r_lateral * r_forward
        min_ball_speed = float(self.cfg["rewards"].get("ball_vel_tracking_min_speed", 0.1))
        moving_ball_gate = torch.norm(self.local_ball_vel_xy, dim=-1) > min_ball_speed
        gate = self.receive_mode_mask * moving_ball_gate.float() * (~self.ball_has_been_contacted).float() * (~self.ball_has_passed_robot).float()
        return reward * gate

    def _reward_incoming_source_position(self):
        """Reward standing on the side the ball originally came from."""
        lateral_sigma = max(float(self.cfg["rewards"].get("incoming_source_lateral_sigma", 0.10)), 1e-6)
        lateral_term = torch.exp(-torch.square(self.intercept_lateral_error) / lateral_sigma)

        # Reward being on the source side of the ball.
        source_min = float(self.cfg["rewards"].get("incoming_source_forward_min", 0.05))
        source_sigma = max(float(self.cfg["rewards"].get("incoming_source_forward_sigma", 0.08)), 1e-6)
        source_term = torch.sigmoid((self.intercept_forward_error - source_min) / source_sigma)

        reward = lateral_term * source_term

        # Apply in receive mode when ball is moving.
        min_ball_speed = float(self.cfg["rewards"].get("ball_vel_tracking_min_speed", 0.1))
        ball_speed = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        moving_ball_gate = ball_speed > min_ball_speed
        gate = self.receive_mode_mask * moving_ball_gate.float() * (~self.ball_has_been_contacted).float() * (~self.ball_has_passed_robot).float()

        return reward * gate

    def _reward_intercept_arrival_timing(self):
        """Reward arriving on the pass line inside a useful interception time window."""
        lateral_sigma = max(float(self.cfg["rewards"].get("intercept_arrival_lateral_sigma", 0.10)), 1.0e-6)
        tti_target = float(self.cfg["rewards"].get("intercept_arrival_tti_target", 0.18))
        tti_sigma = max(float(self.cfg["rewards"].get("intercept_arrival_tti_sigma", 0.12)), 1.0e-6)
        forward_min = float(self.cfg["rewards"].get("intercept_arrival_forward_min", 0.04))
        forward_sigma = max(float(self.cfg["rewards"].get("intercept_arrival_forward_sigma", 0.08)), 1.0e-6)

        lateral_term = torch.exp(-torch.square(self.intercept_lateral_error) / lateral_sigma)
        timing_term = torch.exp(-torch.square(self.intercept_time_estimate - tti_target) / tti_sigma)
        forward_term = torch.sigmoid((self.intercept_forward_error - forward_min) / forward_sigma)
        gate = self.receive_mode_mask * (~self.ball_has_been_contacted).float() * (~self.ball_has_passed_robot).float()
        return lateral_term * timing_term * forward_term * gate

    def _reward_ball_passed_behind_penalty(self):
        """Penalty when ball passed behind robot and is still moving away."""
        target_speed = torch.norm(self.commands[:, 0:2], dim=-1)
        receive_mode_eps = float(self.cfg["rewards"].get("receive_mode_target_speed_eps", 1e-6))
        receive_mode_gate = (target_speed <= receive_mode_eps).float()

        shot_dir = self.pass_ref_dir_xy

        # Use current robot pose so "passed" means passed the robot, not just the reset origin.
        rel_ball_robot = self.ball_pos[:, 0:2] - self.base_pos[:, 0:2]
        progress_along_shot = torch.sum(rel_ball_robot * shot_dir, dim=-1)
        margin_x = float(self.cfg["rewards"].get("ball_passed_margin_x", 0.08))
        depth_cap = max(float(self.cfg["rewards"].get("ball_passed_depth_cap", 0.5)), 1e-6)
        behind_depth = torch.clamp(progress_along_shot - margin_x, min=0.0, max=depth_cap) / depth_cap

        ball_vel_xy = self.ball_lin_vel[:, 0:2]
        away_dot = torch.sum(ball_vel_xy * shot_dir, dim=-1)

        away_min = float(self.cfg["rewards"].get("ball_passed_away_min", 0.04))
        away_sigma = max(float(self.cfg["rewards"].get("ball_passed_away_sigma", 0.10)), 1e-6)
        away_gate = torch.sigmoid((away_dot - away_min) / away_sigma)

        ball_speed = torch.norm(ball_vel_xy, dim=-1)
        min_passed_speed = float(self.cfg["rewards"].get("ball_passed_speed_threshold", 0.12))
        speed_sigma = max(float(self.cfg["rewards"].get("ball_passed_speed_sigma", 0.04)), 1e-6)
        speed_gate = torch.sigmoid((ball_speed - min_passed_speed) / speed_sigma)

        # Reduce penalty when the ball is far off the incoming line.
        lateral_dist = torch.abs(shot_dir[:, 0] * rel_ball_robot[:, 1] - shot_dir[:, 1] * rel_ball_robot[:, 0])
        lateral_sigma = max(float(self.cfg["rewards"].get("ball_passed_lateral_sigma", 0.25)), 1e-6)
        lateral_gate = torch.exp(-torch.square(lateral_dist) / lateral_sigma)

        return behind_depth * away_gate * speed_gate * lateral_gate * receive_mode_gate

    # =========================================================================
    # NEW — Active interception (anti-exploit) rewards
    # =========================================================================

    def _reward_interception_time_bonus(self):
        """Large one-shot bonus when robot first causes ball to decelerate.

        Decays exponentially with episode time.  Earlier interception = much more reward.
        Natural friction is filtered out by a minimum speed-drop threshold.
        Uses a boolean latch (ball_has_been_contacted) instead of float('inf') sentinel.
        Contact detection logic is kept inside the reward for simplicity; it only
        writes to the latch buffer which no other reward reads during the same step.
        """
        episode_time = self.episode_length_buf.float() * self.dt

        # Time-decayed bonus
        tau = float(self.cfg["rewards"].get("interception_time_tau", 2.0))
        time_multiplier = torch.exp(-episode_time / tau)

        # Fire bonus only on the step of first contact
        bonus = self.ball_first_contact_event.float() * time_multiplier

        return bonus * self.receive_mode_mask

    def _reward_ball_speed_reduction(self):
        """Reward ball deceleration CAUSED BY robot contact, not natural friction.

        Only rewards speed drops greater than a per-step threshold that filters
        out friction-induced deceleration.
        """
        ball_speed_now = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        ball_speed_prev = torch.norm(self.last_ball_lin_vel_world[:, 0:2], dim=-1)
        speed_drop = self.ball_speed_drop

        # Filter out natural friction
        min_delta = float(self.cfg["rewards"].get("ball_speed_reduction_min_delta", 0.08))
        impact_speed_drop = torch.clamp(speed_drop - min_delta, min=0.0)

        # Robot foot must be near ball (Gaussian gate aligned with interception_foot_radius)
        foot_to_ball = self.ball_pos[:, 0:2].unsqueeze(1) - self.feet_pos[:, :, 0:2]
        min_foot_dist = torch.norm(foot_to_ball, dim=-1).min(dim=1).values
        proximity_sigma = float(self.cfg["rewards"].get("ball_speed_reduction_proximity_sigma", 0.10))
        proximity_gate = torch.exp(-torch.square(min_foot_dist) / proximity_sigma)

        # Ball was moving
        min_speed = float(self.cfg["rewards"].get("ball_vel_tracking_min_speed", 0.1))
        was_moving = (ball_speed_prev > min_speed).float()

        return impact_speed_drop * proximity_gate * was_moving * self.receive_mode_mask

    def _reward_foot_ball_proximity(self):
        """Reward feet being close to a moving ball — encourages physical blocking."""
        # Closest foot to ball
        foot_to_ball = self.ball_pos[:, 0:2].unsqueeze(1) - self.feet_pos[:, :, 0:2]
        foot_ball_dist = torch.norm(foot_to_ball, dim=-1)           # (N, 2)
        min_dist = foot_ball_dist.min(dim=1).values                 # (N,)

        sigma = float(self.cfg["rewards"].get("foot_ball_proximity_sigma", 0.15))
        proximity_reward = torch.exp(-torch.square(min_dist) / sigma)

        # Only when ball is moving
        ball_speed = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        min_speed = float(self.cfg["rewards"].get("ball_vel_tracking_min_speed", 0.1))
        moving_gate = (ball_speed > min_speed).float()

        return proximity_reward * moving_gate * self.receive_mode_mask

    def _reward_ball_in_front_after_stop(self):
        """Reward ball being in front of the robot (local frame) when ball is slow.

        Teaches the robot to trap the ball on its front side, not let it
        ricochet sideways or behind.  Time-decayed to discourage passive waiting.
        """
        # Ball position in robot local frame
        ball_rel_world = self.ball_pos[:, :3] - self.base_pos[:, :3]
        ball_rel_local = quat_rotate_inverse(self.base_quat, ball_rel_world)

        # Positive local-x = in front of robot
        forward_proj = torch.clamp(ball_rel_local[:, 0], min=0.0, max=0.5) / 0.5
        lateral_sigma = max(float(self.cfg["rewards"].get("ball_in_front_lateral_sigma", 0.12)), 1.0e-6)
        lateral_gate = torch.exp(-torch.square(ball_rel_local[:, 1]) / lateral_sigma)

        # Ball distance gate
        ball_dist = torch.norm(self.ball_pos[:, 0:2] - self.base_pos[:, 0:2], dim=-1)
        dist_sigma = float(self.cfg["rewards"].get("ball_in_front_dist_sigma", 0.30))
        dist_gate = torch.exp(-torch.square(ball_dist) / dist_sigma)

        # Low ball speed gate
        ball_speed = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        speed_sigma = float(self.cfg["rewards"].get("ball_in_front_speed_sigma", 0.30))
        slow_gate = torch.exp(-ball_speed / speed_sigma)

        # Time decay
        episode_time = self.episode_length_buf.float() * self.dt
        tau = float(self.cfg["rewards"].get("ball_reward_time_decay_tau", 3.0))
        time_multiplier = torch.exp(-episode_time / tau)

        post_contact_gate = self.receive_mode_mask * self.ball_has_been_contacted.float()
        return forward_proj * lateral_gate * dist_gate * slow_gate * time_multiplier * post_contact_gate

    def _reward_ball_travel_penalty(self):
        """Penalise how far the ball has progressed along the original shot direction.

        The tracked value (ball_max_progress_along_shot) can only increase,
        so the robot cannot erase it by walking to the ball after it stops.
        Early interception keeps this value small.

        Uses delta formulation: penalises the *increase* in max-progress this step,
        avoiding quadratic accumulation of a monotonically growing value.
        """
        max_dist = float(self.cfg["rewards"].get("ball_travel_penalty_max", 5.0))
        normalized_now = torch.clamp(self.ball_max_progress_along_shot / max_dist, 0.0, 1.0)
        normalized_prev = torch.clamp(self.prev_ball_max_progress_along_shot / max_dist, 0.0, 1.0)
        delta_progress = torch.clamp(normalized_now - normalized_prev, min=0.0)

        return delta_progress * self.receive_mode_mask
