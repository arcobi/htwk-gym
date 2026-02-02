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


class BallInterceptK1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        self._init_csv_logging()
        self.env_resets = 0
        self.env_successes = 0
        self.env_falling = 0
        self.ball_velocities = []

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        # Load robot asset
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
        self.robot_body_names = self.gym.get_asset_rigid_body_names(robot_asset)


        # Continue with robot setup...
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

        # Override effort limits from config if specified (K1-specific values)
        if "effort_limit" in self.cfg["control"]:
            for i in range(self.num_dofs):
                for name in self.cfg["control"]["effort_limit"].keys():
                    if name in self.dof_names[i]:
                        self.torque_limits[i] = self.cfg["control"]["effort_limit"][name]
                        dof_props_asset["effort"][i] = self.cfg["control"]["effort_limit"][name]
                        break

        # Override velocity limits from config if specified (K1-specific values)
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

        # Set per-joint armature values from config (K1-specific)
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

        ball_cfg = self.cfg["ball"]

        # Store ball properties
        self.ball_radius = float(ball_cfg["radius"])
        self.ball_init_pos = to_torch(ball_cfg["init_pos"], device=self.device)
        self.ball_init_rot = to_torch(ball_cfg["init_rot"], device=self.device)
        self.ball_init_lin_vel = to_torch(ball_cfg["init_lin_vel"], device=self.device)
        self.ball_init_ang_vel = to_torch(ball_cfg["init_ang_vel"], device=self.device)

        self._get_env_origins()
        env_lower = gymapi.Vec3(-5, 0.0, -5)
        env_upper = gymapi.Vec3(5, 5, 5)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = []  # Store ball handles
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            #pos = self.env_origins[i].clone()
            #start_pose.p = gymapi.Vec3(*pos)

            # Create robot actor
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                asset_cfg["name"],
                i,
                asset_cfg["self_collisions"],
                0,
            )
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            ball_asset = self._create_ball_asset(radius=apply_randomization(self.cfg["ball"]["radius"], self.cfg["randomization"].get("ball_radius")))
            ball_handle = self.gym.create_actor(env_handle, ball_asset, start_pose, "ball", i, True, 0)
            try:
                ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
                for b in range(len(ball_body_props)):
                    ball_body_props[b].mass = apply_randomization(self.cfg["ball"]["mass"], self.cfg["randomization"].get("ball_mass"))
                self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
            except Exception:
                print("meh")
                pass

            # shape props: restitution/friction
            try:
                ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
                for s in range(len(ball_shape_props)):
                    ball_shape_props[s].restitution = self.cfg["ball"].get("restitution", 0.1)
                    ball_shape_props[s].friction = self.cfg["ball"].get("friction", 1.0)
                    ball_shape_props[s].rolling_friction = self.cfg["ball"].get("rolling_friction", 0.3)
                    ball_shape_props[s].torsion_friction = self.cfg["ball"].get("torsion_friction", 0.1)
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

    def _create_ball_asset(self, radius):
        ball_options = gymapi.AssetOptions()
        ball_options.fix_base_link = False
        ball_options.density = 200
        ball_options.angular_damping = 0.15
        ball_options.linear_damping = 0.38
        ball_options.max_angular_velocity = 1000.0
        ball_options.max_linear_velocity = 20.0
        ball_options.disable_gravity = False
        ball_options.replace_cylinder_with_capsule = False
        ball_options.thickness = 0.01

        ball_asset = self.gym.create_sphere(self.sim, radius, ball_options)

        return ball_asset

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
        

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_ball_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # Buffer for ball-only resets
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.min_ball_vel_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.time_since_ball_is_still_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.time_since_ball_is_moving_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.goalkeeper_target = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.extras = {}
        self.extras["rew_terms"] = {}

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
        # Reshape root states to separate robot and ball states
        self.root_states = root_states.view(self.num_envs, 2, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies + 1, 13)
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
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        #self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        # Only apply gravity to robot's state
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
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
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

        self.last_ball_lin_vel_world = torch.zeros_like(self.body_states[:, -1, 7:10]) # World frame
        self.blocked_ball_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_has_started_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_ball_direction = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        # Store ball position history for velocity estimation (40ms = 2 steps back at 20ms per step)
        self.ball_pos_history = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device)
        
        # Track if ball was shot back (positive X velocity after being touched)
        self.ball_shot_back = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Track time since last contact with ball (for time-windowed return rewards)
        self.time_since_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.had_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        self.reward_scales_ball_rolling = self.cfg["rewards"].get("ball_rolling_scale", {}).copy()

        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        for key in list(self.reward_scales_ball_rolling.keys()):
            scale = self.reward_scales_ball_rolling[key]
            self.reward_scales_ball_rolling[key] *= self.dt

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

    def render(self):
        """Override render to call parent render."""
        super().render()

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        #self._resample_commands()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        self.env_resets += env_ids.shape[0]

        velocity_before_reset = torch.norm(self.root_states[env_ids, 1, 7:10], dim=1)
        
        # only get velocities that are greater than 0.1
        velocity_before_reset = velocity_before_reset[velocity_before_reset > 0.1]
        if len(velocity_before_reset) > 0:
            # append velocities sepeate
            for velocity in velocity_before_reset:
                self.ball_velocities.append(velocity.item())

        # Reset robot
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Continue with existing reset code...
        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.min_ball_vel_buf[env_ids] = 0.0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.time_since_ball_is_still_buf[env_ids] = 0.0
        self.time_since_ball_is_moving_buf[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        self.blocked_ball_buffer[env_ids] = False
        self.ball_has_started_moving[env_ids] = False
        self.initial_ball_direction[env_ids] = 0.0
        
        # Reset actions and last_actions to zero to prevent carrying over old actions
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        
        # Update derived state variables for reset environments
        # These are computed from root_states but need to be explicitly updated here
        # since the normal update happens before the reset in step()
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.root_states[env_ids, 0, 3:7], 
            self.root_states[env_ids, 0, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.root_states[env_ids, 0, 3:7], 
            self.root_states[env_ids, 0, 10:13]
        )
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.root_states[env_ids, 0, 3:7], 
            self.gravity_vec[env_ids]
        )
        
        # Initialize ball_pos_history with current ball position (both timesteps)
        # This must be done AFTER _reset_root_states() has set the new ball position
        current_ball_pos = self.root_states[env_ids, 1, 0:3]
        self.ball_pos_history[env_ids, 0] = current_ball_pos
        self.ball_pos_history[env_ids, 1] = current_ball_pos

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf
        self.last_ball_lin_vel_world[env_ids] = 0.0 # Reset for selected envs
        
        # Reset ball shot back buffer
        self.ball_shot_back[env_ids] = False
        
        # Reset contact tracking
        self.time_since_contact[env_ids] = 0.0
        self.had_contact[env_ids] = False

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
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
        #self.root_states[env_ids, 0, :2] = apply_randomization(self.root_states[env_ids, 0, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        #self.root_states[env_ids, 0, 2] += self.terrain.terrain_heights(self.root_states[env_ids, 0, :2])
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
        # The self.root_states tensor has been updated for both.
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
        robot_quat = self.root_states[env_ids_to_reset_ball, 0, 3:7]

        # Define forward vector in robot's local frame and repeat for each env
        forward_vec_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(len(env_ids_to_reset_ball), 1)
        
        # Rotate forward vector to world frame
        forward_vec_world = quat_rotate(robot_quat, forward_vec_local)

        # Define right vector in world frame
        right_vec_xy = torch.stack([forward_vec_world[:, 1], -forward_vec_world[:, 0]], dim=-1)

        ball_init_pos = torch.zeros_like(forward_vec_world)
        ball_init_pos[:, 0] = apply_randomization(ball_init_pos[:, 0], self.cfg["randomization"].get("ball_init_pos_x"))
        ball_init_pos[:, 1] = apply_randomization(ball_init_pos[:, 1], self.cfg["randomization"].get("ball_init_pos_y"))

        # Calculate ball's target XY position
        ball_target_xy = robot_pos[:, 0:2] + forward_vec_world[:, 0:2] * ball_init_pos[:, 0].unsqueeze(1) + right_vec_xy[:, 0:2] * ball_init_pos[:, 1].unsqueeze(1)
        
        # Calculate ball's target Z position (on the ground + ball radius)
        if hasattr(self, 'terrain'):
            ball_target_z = self.terrain.terrain_heights(ball_target_xy) + self.ball_radius
        else: # Fallback if no terrain, assume ground is at z=0 relative to env_origin
             # This assumes env_origins are at z=0 for the ground level.
            ball_target_z = torch.full_like(ball_target_xy[:, 0], self.ball_radius)


        # Set ball position
        self.root_states[env_ids_to_reset_ball, 1, 0] = ball_target_xy[:, 0]
        self.root_states[env_ids_to_reset_ball, 1, 1] = ball_target_xy[:, 1]
        self.root_states[env_ids_to_reset_ball, 1, 2] = ball_target_z
        
        # Set ball orientation to default (identity quaternion)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(len(env_ids_to_reset_ball), 1)
        self.root_states[env_ids_to_reset_ball, 1, 3:7] = identity_quat

        # Set ball linear and angular velocities to zero
        self.root_states[env_ids_to_reset_ball, 1, 7:13] = 0.0

        # Calculate ball vx and vy with ball_speed and ball_angle_xy randomization
        theta = apply_randomization(torch.full((len(env_ids_to_reset_ball), 1), 0.0, device=self.device), self.cfg["randomization"].get("ball_angle_xy"))
        speed = apply_randomization(self.root_states[env_ids_to_reset_ball, 1, 7], self.cfg["randomization"].get("ball_speed"))
        vx = -speed.unsqueeze(1) * torch.cos(theta)
        vy = speed.unsqueeze(1) * torch.sin(theta)

        # Set ball vx and vy
        self.root_states[env_ids_to_reset_ball, 1, 7] = vx.squeeze(1)
        self.root_states[env_ids_to_reset_ball, 1, 8] = vy.squeeze(1)

        # Update only the ball actors in the simulation
        ball_actor_indices = (2 * env_ids_to_reset_ball + 1).to(dtype=torch.int32)
        if len(ball_actor_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states), # Send full buffer, but only indices for balls are used effectively for this call.
                gymtorch.unwrap_tensor(ball_actor_indices),
                len(ball_actor_indices)
            )

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 0, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 0, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size

        self.root_states[out_x_min, 0, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 0, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 0, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size

        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        if self.cfg["commands"]["curriculum"]:
            self._resample_curriculum_commands(env_ids)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                x = self.env_curriculum_level[env_ids[i], 0] + self.cfg["commands"]["lin_vel_levels"]
                y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _resample_curriculum_commands(self, env_ids):
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["ang_vel_resolution"]

    def step(self, actions):
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        # perform physics step
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

        # Store previous ball velocity in world frame *before* refreshing root states for current step
        prev_ball_lin_vel_world = self.root_states[:, 1, 7:10].clone()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.ball_pos[:] = self.root_states[:, 1, 0:3]
        self.ball_lin_vel[:] = self.body_states[:, -1, 7:10]
        self.ball_ang_vel[:] = self.body_states[:, -1, 10:13]

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

        # Update ball position history for velocity estimation
        self.ball_pos_history[:, 1] = self.ball_pos_history[:, 0].clone()  # Move current to history
        self.ball_pos_history[:, 0] = self.ball_pos.clone()  # Store new current position
        
        # Update time since contact (for time-windowed return rewards)
        # Check if any foot is touching the ball
        foot_positions = self.feet_pos
        ball_positions = self.ball_pos.unsqueeze(1)
        distances = torch.norm(foot_positions - ball_positions, dim=-1)
        is_touching = (distances < self.cfg["rewards"]["touch_ball_distance_threshold"]).any(dim=1)
        
        # Reset time_since_contact on new contact, otherwise increment
        self.time_since_contact = torch.where(
            is_touching & ~self.had_contact,  # New contact (wasn't touching before)
            torch.zeros_like(self.time_since_contact),
            self.time_since_contact + self.dt
        )
        self.had_contact = is_touching

        self.episode_length_buf += 1
        # Check if ball has positive X velocity (moving away from robot/toward goal)
        # Use world frame velocity from root_states for consistency
        ball_moving_toward_goal = self.root_states[:, 1, 7] > 0.1  # Positive X = away from robot
        self.min_ball_vel_buf = torch.where(
            ball_moving_toward_goal,
            self.min_ball_vel_buf + 1.0,
            torch.zeros_like(self.min_ball_vel_buf)
        )
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        # Update time_since_ball_is_still_buf
        ball_speed_threshold = self.cfg["rewards"].get("ball_stationary_speed_threshold", 0.1)  # Configurable threshold
        ball_speed = torch.norm(self.root_states[:, 1, 7:10], dim=-1)
        ball_is_active = ball_speed > ball_speed_threshold
        
        # Only start counting "still" time after ball has been moving at least once
        # This prevents immediate termination if ball spawns with brief zero velocity
        self.time_since_ball_is_still_buf = torch.where(
            ball_is_active, 
            torch.zeros_like(self.time_since_ball_is_still_buf), 
            torch.where(
                self.ball_has_started_moving,
                self.time_since_ball_is_still_buf + self.dt,
                torch.zeros_like(self.time_since_ball_is_still_buf)
            )
        )
        
        # Track when ball first starts moving
        ball_just_started = ball_is_active & ~self.ball_has_started_moving
        self.ball_has_started_moving = self.ball_has_started_moving | ball_just_started
        
        # Only count moving time after ball has started moving
        self.time_since_ball_is_moving_buf = torch.where(
            ~ball_is_active, 
            torch.zeros_like(self.time_since_ball_is_moving_buf), 
            torch.where(
                self.ball_has_started_moving,
                self.time_since_ball_is_moving_buf + self.dt,
                torch.zeros_like(self.time_since_ball_is_moving_buf)
            )
        )

        self._kick_robots()
        self._push_robots()
        self._check_termination()  # Sets self.reset_buf based on termination conditions

        # Handle ball-only resets (ball too far, but robot is not resetting)
        # Note: reset_ball_buf must be set elsewhere if ball-only resets are desired
        ball_only_reset_env_ids = (self.reset_ball_buf & ~self.reset_buf).nonzero(as_tuple=False).flatten()
        if len(ball_only_reset_env_ids) > 0:
            self._reset_ball_at_robot_front(ball_only_reset_env_ids)
            # Update convenience tensors for the reset balls as _compute_observations will use them
            self.ball_pos[ball_only_reset_env_ids] = self.root_states[ball_only_reset_env_ids, 1, 0:3]
            ball_quat_reset = self.root_states[ball_only_reset_env_ids, 1, 3:7]
            self.ball_rot[ball_only_reset_env_ids] = ball_quat_reset
            # Velocities in root_states are world, convert to local for convenience tensors
            world_lin_vel_reset = self.root_states[ball_only_reset_env_ids, 1, 7:10]
            world_ang_vel_reset = self.root_states[ball_only_reset_env_ids, 1, 10:13]
            self.ball_lin_vel[ball_only_reset_env_ids] = quat_rotate_inverse(ball_quat_reset, world_lin_vel_reset)
            self.ball_ang_vel[ball_only_reset_env_ids] = quat_rotate_inverse(ball_quat_reset, world_ang_vel_reset)
            
            self.reset_ball_buf[ball_only_reset_env_ids] = False

        self._compute_reward()

        # Log rewards to CSV file
        self._log_rewards_to_csv()

        # Update last_ball_lin_vel_world *before* potential full reset for next step's calculation
        # For envs that were not reset (neither full nor ball-only), this is their current velocity.
        # For envs that *were* reset (either full or ball-only), their velocity was set to 0.0 during reset,
        # so this correctly reflects their "last" velocity as 0 before the next step.
        self.last_ball_lin_vel_world[:] = self.body_states[:, -1, 7:10]

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids) # This will call _reset_root_states, which handles ball reset too
            self.reset_ball_buf[env_ids] = False # Ball reset is handled by full reset
            # For fully reset environments, ensure their last_ball_lin_vel_world is also 0 for next step
            self.last_ball_lin_vel_world[env_ids] = 0.0

        #self._teleport_robot()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        #print(f"env_resets: {self.env_resets}, env_successes: {self.env_successes}, env_falling: {self.env_falling}")
        #if len(self.ball_velocities) > 0:
        #    print(f"ball_velocities average: {np.mean(self.ball_velocities)}, std: {np.std(self.ball_velocities)}, max: {np.max(self.ball_velocities)}")

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 0, 7:10] = apply_randomization(self.root_states[:, 0, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 0, 10:13] = apply_randomization(self.root_states[:, 0, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
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
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
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
        # Contact termination
        contact_term = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf = contact_term
        
        # Linear velocity termination (separate from angular)
        lin_vel_magnitude = torch.norm(self.root_states[:, 0, 7:10], dim=-1)  # Linear velocity in world frame
        lin_vel_term = lin_vel_magnitude > self.cfg["rewards"]["terminate_lin_vel"]
        self.reset_buf |= lin_vel_term
        
        # Angular velocity termination (separate from linear)
        ang_vel_magnitude = torch.norm(self.root_states[:, 0, 10:13], dim=-1)  # Angular velocity in world frame
        ang_vel_term = ang_vel_magnitude > self.cfg["rewards"]["terminate_ang_vel"]
        self.reset_buf |= ang_vel_term
        
        # Height termination
        height_term = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        self.reset_buf |= height_term
        
        # Episode timeout
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf
        
        # GOALKEEPER FAIL: Ball passed behind robot
        if self.cfg["rewards"].get("ball_passed_terminate", True):
            # Get robot's forward direction in world frame
            forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            forward_world = quat_rotate(self.base_quat, forward_local)
            
            # Vector from robot to ball
            robot_to_ball = self.ball_pos[:, :2] - self.base_pos[:, :2]
            
            # Dot product: negative means ball is behind robot
            ball_behind_robot = torch.sum(robot_to_ball * forward_world[:, :2], dim=-1)
            
            # Threshold from config
            threshold = self.cfg.get("goalkeeper", {}).get("ball_passed_x_threshold", -0.5)
            
            ball_passed_term = ball_behind_robot < threshold
            self.reset_buf |= ball_passed_term
            if torch.any(ball_passed_term):
                self.env_falling += torch.sum(ball_passed_term)  # Count as failure
        
        # SUCCESS: Ball shot back (positive X velocity after being blocked)
        if self.cfg["rewards"].get("shot_back_terminate", True):
            # Ball velocity (world frame)
            ball_vel_x = self.root_states[:, 1, 7]  # X velocity
            ball_speed = torch.norm(self.root_states[:, 1, 7:9], dim=-1)
            
            # Ball shot back if: positive X velocity + fast enough + was blocked
            min_speed = self.cfg["rewards"].get("shot_back_min_speed", 0.8)
            shot_back_term = (ball_vel_x > 0) & (ball_speed > min_speed) & self.blocked_ball_buffer
            self.reset_buf |= shot_back_term
            if torch.any(shot_back_term):
                self.env_successes += torch.sum(shot_back_term)
        
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

        # Ball still for too long
        max_ball_still_time = self.cfg["rewards"].get("max_ball_still_time_s", 3.0)
        ball_still_term = self.time_since_ball_is_still_buf > max_ball_still_time
        self.reset_buf |= ball_still_term

        # Ball moving for too long
        max_ball_moving_time = self.cfg["rewards"].get("max_ball_moving_time_s", 5.0)
        ball_moving_term = self.time_since_ball_is_moving_buf > max_ball_moving_time
        self.reset_buf |= ball_moving_term
        
        self.env_falling += torch.sum(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"])
        self.env_falling += torch.sum(lin_vel_term)
        self.env_falling += torch.sum(ang_vel_term)



    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        # Use world frame velocity for consistent ball movement detection
        ball_speed_threshold = self.cfg["rewards"].get("ball_stationary_speed_threshold", 0.1)
        ball_is_moving = torch.norm(self.root_states[:, 1, 7:10], dim=-1) > ball_speed_threshold

        # --- CALCULATE GOALKEEPER TARGET (Intercept vs Return) ---
        target_x_def = 0.5  # Defense line x-coordinate
        ball_pos = self.ball_pos
        ball_vel = self.root_states[:, 1, 7:10] # World frame
        
        # Calculate time to intercept: t = (target_x - current_x) / vel_x
        # We only care if ball is moving towards the goal (vel_x < 0)
        vel_x = ball_vel[:, 0]
        # Avoid division by zero, though we filter by velocity later anyway
        safe_vel_x = torch.where(torch.abs(vel_x) < 0.01, torch.tensor(-1.0, device=self.device), vel_x)
        
        time_to_intercept = (target_x_def - ball_pos[:, 0]) / safe_vel_x
        pred_y = ball_pos[:, 1] + ball_vel[:, 1] * time_to_intercept
        
        # Logic: 
        # 1. Ball must be in front of defense line (ball_x > 0.5)
        # 2. Ball must be moving towards robot (vel_x < -0.2)
        # 3. Time to intercept must be positive (ball hasn't passed theoretically)
        should_intercept = (ball_pos[:, 0] > target_x_def) & (ball_vel[:, 0] < -0.2) & (time_to_intercept > 0)
        
        # Target position: Intercept Point OR Center (0,0)
        target_x = torch.where(should_intercept, torch.tensor(target_x_def, device=self.device), torch.tensor(0.0, device=self.device))
        target_y = torch.where(should_intercept, torch.clamp(pred_y, -1.3, 1.3), torch.tensor(0.0, device=self.device))
        
        self.goalkeeper_target = torch.stack([target_x, target_y], dim=-1)
        # ---------------------------------------------------------

        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            raw_reward_values = self.reward_functions[i]()  # Shape: (num_envs)
            normal_scale = self.reward_scales[name]

            # Check if a specific scale for "ball moving" scenario exists
            ball_moving_scale = self.reward_scales_ball_rolling.get(name)

            if ball_moving_scale is not None:
                # Use ball_moving_scale when ball is moving, normal_scale otherwise
                effective_scale = torch.where(
                    ball_is_moving,
                    torch.full_like(raw_reward_values, ball_moving_scale),
                    torch.full_like(raw_reward_values, normal_scale)
                )
                rew = raw_reward_values * effective_scale
            else:
                rew = raw_reward_values * normal_scale

            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        """Computes observations"""
        # Calculate ball position relative to the robot
        ball_pos_world_frame = self.ball_pos - self.base_pos
        relative_ball_pos = quat_rotate_inverse(self.base_quat, ball_pos_world_frame)
        
        # Calculate ball position 40ms ago relative to the robot
        # Use stored history position directly (history is updated in step())
        ball_pos_history_world_frame = self.ball_pos_history[:, 1] - self.base_pos
        relative_ball_pos_history = quat_rotate_inverse(self.base_quat, ball_pos_history_world_frame)

        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                # Use relative ball position in observations
                apply_randomization(relative_ball_pos[:, 0:2], self.cfg["noise"].get("ball_pos")) * self.cfg["normalization"]["ball_pos"],
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
                # Ball position history (40ms delayed) - apply same noise as current ball pos for consistency
                apply_randomization(relative_ball_pos_history[:, 0:2], self.cfg["noise"].get("ball_pos")) * self.cfg["normalization"]["ball_pos"],  
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.ball_lin_vel[:, 0:2],
                self.feet_pos[:, 0, 0:2],
                self.feet_pos[:, 1, 0:2],
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
       return torch.exp(-torch.square(0 - self.filtered_lin_vel[:, 0]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(0 - self.filtered_lin_vel[:, 1]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(-torch.square(0 - self.filtered_ang_vel[:, 2]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

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
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

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

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    
    def _reward_left_feet_x(self):
        # Calculate the distance between feet on the x-axis
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_x_distance =  torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
        )
        
        # Get the reference distance from config
        target_distance = self.cfg["rewards"]["feet_distance_ref_x"]
        
        # Normalize the difference between actual and target distance
        normalized_diff = -torch.abs(feet_x_distance - target_distance)
        
        reward = torch.exp(2.0 * (normalized_diff - 1.0) + 2)  # Exponential increase

        #print(f"X-axis: feet_x_distance={feet_x_distance.mean().item():.4f}, target={target_distance:.4f}, normalized_diff={normalized_diff.mean().item():.4f}, reward={reward.mean().item():.4f}")
        
        return reward
    
    def _reward_left_feet_y(self):
        # Calculate the distance between feet on the y-axis
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_y_distance = torch.abs(
            torch.sin(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            + torch.cos(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        
        # Get the reference distance from config
        target_distance = self.cfg["rewards"]["feet_distance_ref_y"]
        
        # Normalize the difference between actual and target distance
        normalized_diff = -torch.abs(feet_y_distance - target_distance)
        
        reward = torch.exp(2.0 * (normalized_diff - 1.0) + 2)  # Exponential increase
        
        #print(f"Y-axis: feet_y_distance={feet_y_distance.mean().item():.4f}, target={target_distance:.4f}, normalized_diff={normalized_diff.mean().item():.4f}, reward={reward.mean().item():.4f}")
        
        return reward

    def _reward_waiting(self):
        """Reward that increases quadratically with time elapsed in the episode."""
        # Get current progress through episode (0 to 1)
        progress = self.episode_length_buf / np.ceil(self.cfg["rewards"]["max_ball_still_time_s"] / self.dt)
        
        # Quadratic reward from 0 to 1 based on progress
        reward = progress * progress
        
        return reward

    def _reward_feet_distance(self):
        """Reward for keeping feet at a target distance apart."""
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        return torch.abs(self.cfg["rewards"]["feet_distance_ref"] - feet_distance)

    def _reward_feet_swing(self):
        """Reward for proper swing phase timing."""
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()

    def _reward_touched_ball(self):
        """Reward for touching the ball with any foot."""
        foot_positions = self.feet_pos  # Shape: (num_envs, num_feet, 3)
        ball_positions = self.ball_pos.unsqueeze(1)  # Shape: (num_envs, 1, 3)
        
        # Calculate distances from each foot to the ball (3D distance)
        distances = torch.norm(foot_positions - ball_positions, dim=-1)  # Shape: (num_envs, num_feet)
        
        # Check if any foot is within the threshold distance to the ball
        touched = (distances < self.cfg["rewards"]["touch_ball_distance_threshold"]).any(dim=1)  # Shape: (num_envs,)
        
        # Optional: Check if ball is in front of robot (using 2D projection to avoid height issues)
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        forward_world = quat_rotate(self.base_quat, forward_local)
        
        # Use only XY plane for direction check
        robot_to_ball_xy = self.ball_pos[:, :2] - self.base_pos[:, :2]
        forward_world_xy = forward_world[:, :2]
        
        # Normalize vectors for proper dot product
        robot_to_ball_xy_norm = robot_to_ball_xy / (torch.norm(robot_to_ball_xy, dim=-1, keepdim=True) + 1e-6)
        forward_world_xy_norm = forward_world_xy / (torch.norm(forward_world_xy, dim=-1, keepdim=True) + 1e-6)
        
        dot_product = torch.sum(forward_world_xy_norm * robot_to_ball_xy_norm, dim=-1)
        is_infront = dot_product > 0.0  # Ball is in front hemisphere (180 degrees)
        
        reward = (touched & is_infront).float()
        
        return reward

    def _reward_blocked_ball(self):
        """Reward for reversing the direction of the ball"""
        # Capture initial ball direction when it first starts moving (use world frame)
        # Note: ball_has_started_moving is now tracked in step() function
        ball_speed_threshold = self.cfg["rewards"].get("ball_stationary_speed_threshold", 0.1)
        ball_speed = torch.norm(self.root_states[:, 1, 7:10], dim=-1)
        ball_just_started = (ball_speed > ball_speed_threshold) & (torch.norm(self.initial_ball_direction, dim=-1) == 0)
        
        # Save initial direction for balls that just started moving (world frame)
        self.initial_ball_direction[ball_just_started] = self.root_states[ball_just_started, 1, 7:10].clone()
        
        # Calculate dot product between current and initial direction for each environment (world frame)
        dot_products = torch.sum(self.root_states[:, 1, 7:10] * self.initial_ball_direction, dim=-1)
        
        # Check which environments have reversed direction (dot product < 0)
        has_reversed = dot_products < 0
        
        # Only reward if reversed AND not yet rewarded AND ball has started moving
        should_reward = has_reversed & ~self.blocked_ball_buffer & self.ball_has_started_moving
        
        # Mark environments that just got rewarded
        self.blocked_ball_buffer[should_reward] = True
        
        # Return float rewards (1.0 for rewarded, 0.0 for others)
        return should_reward.float()
        
    def _reward_ball_distance(self):
        """Reward based on the distance of the ball and robot, near is better"""
        # Calculate 2D distance (ignoring height)
        distance = torch.norm(self.base_pos[:, :2] - self.ball_pos[:, :2], dim=-1)

        # Gaussian-like reward: 1.0 at d=0, quickly dropping off.
        # This provides a strong gradient when closer.
        return torch.exp(-torch.square(distance) / 0.5)

    def _reward_facing_ball(self):
        """Reward for facing the ball (helps robot turn towards target)"""
        # Vector from robot to ball (XY plane)
        to_ball = self.ball_pos[:, :2] - self.base_pos[:, :2]
        to_ball_norm = torch.norm(to_ball, dim=-1, keepdim=True)
        to_ball_dir = to_ball / (to_ball_norm + 1e-6)

        # Robot forward vector in world frame (XY plane)
        # Assuming robot forward is X-axis in local frame
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        forward_world = quat_rotate(self.base_quat, forward_local)
        forward_world_2d = forward_world[:, :2]
        forward_world_2d = forward_world_2d / (torch.norm(forward_world_2d, dim=-1, keepdim=True) + 1e-6)

        # Dot product: 1.0 if perfectly facing, -1.0 if facing away
        facing = torch.sum(forward_world_2d * to_ball_dir, dim=-1)
        
        # Only reward if ball is more than 0.2m away (don't worry about spin when on top of it)
        facing = torch.where(to_ball_norm.squeeze() > 0.2, facing, torch.zeros_like(facing))
        
        return torch.clamp(facing, min=0.0)

    # ------------ GOALKEEPER REWARD FUNCTIONS ----------------
    
    def _reward_ball_passed_robot(self):
        """CRITICAL PENALTY: Ball passed behind the robot (goalkeeper failed)."""
        # Get robot's forward direction in world frame
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        forward_world = quat_rotate(self.base_quat, forward_local)
        
        # Vector from robot to ball
        robot_to_ball = self.ball_pos[:, :2] - self.base_pos[:, :2]
        
        # Dot product: negative means ball is behind robot
        ball_behind_robot = torch.sum(robot_to_ball * forward_world[:, :2], dim=-1)
        
        # Threshold from config (how far behind counts as "passed")
        threshold = self.cfg.get("goalkeeper", {}).get("ball_passed_x_threshold", -0.5)
        
        # Penalty if ball is behind robot
        passed = ball_behind_robot < threshold
        
        return passed.float()
    
    def _reward_ball_shot_back(self):
        """One-time reward for shooting ball back (positive X velocity after blocking)."""
        # Ball velocity (world frame)
        ball_vel_x = self.root_states[:, 1, 7]  # X velocity (positive = forward/away)
        ball_speed = torch.norm(self.root_states[:, 1, 7:9], dim=-1)
        
        # Only reward once when:
        # 1. Ball has positive X velocity (going forward/away from robot)
        # 2. Ball is moving fast enough
        # 3. Ball was touched/blocked first
        # 4. Haven't already rewarded this
        min_speed = self.cfg["rewards"].get("shot_back_min_speed", 0.5)
        
        is_positive_x = ball_vel_x > 0
        is_fast = ball_speed > min_speed
        was_touched = self.blocked_ball_buffer
        not_yet_rewarded = ~self.ball_shot_back
        
        # Check if should reward now
        should_reward = is_positive_x & is_fast & was_touched & not_yet_rewarded
        
        # Mark as rewarded (one-time only)
        self.ball_shot_back = self.ball_shot_back | should_reward
        
        # Return 1.0 for rewarded, 0.0 for others
        return should_reward.float()
    
    def _reward_directional_return(self):
        """Dense reward for post-contact ball velocity toward goal (time-windowed).
        This rewards intentional strikes, not just random bounces."""
        # Only reward within time window after contact (e.g., 0.5s)
        time_window = self.cfg["rewards"].get("return_time_window_s", 2.0)
        within_window = self.time_since_contact < time_window
        had_contact_recently = self.time_since_contact > 0.0  # Some contact happened
        
        # Ball velocity in robot frame
        ball_vel_world = self.root_states[:, 1, 7:10]
        ball_vel_local = quat_rotate_inverse(self.base_quat, ball_vel_world)
        
        # Desired direction: forward (positive X in robot frame)
        vel_forward = ball_vel_local[:, 0]
        
        # Reward positive forward velocity, scaled by speed
        # Clamp to reasonable max (e.g., 3 m/s)
        v_max = 3.0
        reward_raw = torch.clamp(vel_forward, 0.0, v_max) / v_max
        
        # Apply only within time window after contact
        reward = reward_raw * within_window.float() * had_contact_recently.float()
        
        return reward
    
    def _reward_soft_rebound_penalty(self):
        """Penalty for soft/dead rebounds after contact (ball stops or dribbles).
        Uses exponential decay to smooth penalty window over time.
        Continuous softness metric: 0 (fast) -> 1 (still)."""
        # must be negative to be a penalty (or weight it negative outside)
        min_speed = self.cfg["rewards"].get("soft_rebound_min_speed", 0.3)
        tau = self.cfg["rewards"].get("soft_rebound_tau_s", 0.15)

        # "contact happened" mask (better: explicit contact flag if you have one)
        had_contact = (self.time_since_contact > 0.0).float()

        # decay after contact (instead of hard 0.3s window)
        decay = torch.exp(-self.time_since_contact / tau)
        weight = had_contact * decay

        ball_vel_world = self.root_states[:, 1, 7:10]
        ball_speed = torch.norm(ball_vel_world, dim=-1)

        # continuous softness (0 if fast enough, ->1 if stopped)
        softness = torch.clamp((min_speed - ball_speed) / (min_speed + 1e-6), 0.0, 1.0)

        return -(softness * weight)

    def _reward_ball_lateral_vel(self):
        """Penalty for ball moving sideways (Y-axis) instead of forward."""
        
        ball_vel_world = self.root_states[:, 1, 7:10]
        ball_vel_local = quat_rotate_inverse(self.base_quat, ball_vel_world)
        
        # We want to penalize Y velocity (sideways)
        vel_y = torch.abs(ball_vel_local[:, 1])
        
        # Only penalize if ball is moving reasonably fast (don't punish jitter)
        ball_speed = torch.norm(ball_vel_local, dim=-1)
        is_moving = ball_speed > 0.1
        
        return vel_y * is_moving.float()

    def _reward_ball_forward_vel(self):
        """Dense reward for ball moving forward (positive X in local frame)."""
        ball_vel_world = self.root_states[:, 1, 7:10]
        ball_vel_local = quat_rotate_inverse(self.base_quat, ball_vel_world)

        vel_x = ball_vel_local[:, 0]
        return torch.clamp(vel_x, min=0.0)

    def _reward_goalkeeper_pos(self):
        """Reward for being close to the calculated defense target (Intercept or Center)."""
        # Distance to calculated target
        dist = torch.norm(self.base_pos[:, :2] - self.goalkeeper_target, dim=-1)
        
        # Dense reward for getting closer
        return torch.exp(-torch.square(dist) / 0.5)
