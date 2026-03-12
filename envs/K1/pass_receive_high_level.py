import numpy as np
import torch

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    get_euler_xyz,
    quat_rotate,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)

from envs.K1.ball_control_k1 import BallControlK1
from utils.utils import apply_randomization


assert gymtorch


class PassReceiveHighLevel(BallControlK1):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_csv_logging(self):
        self.csv_logging_enabled = False

    def _log_rewards_to_csv(self):
        return

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_ball_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}
        self.extras["metrics"] = {}

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = root_states.view(self.num_envs, 2, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies + 1, 13)

        self.base_pos = self.root_states[:, 0, 0:3]
        self.base_quat = self.root_states[:, 0, 3:7]
        self.ball_pos = self.root_states[:, 1, 0:3]
        self.ball_rot = self.root_states[:, 1, 3:7]
        self.ball_lin_vel = self.body_states[:, -1, 7:10]
        self.ball_ang_vel = self.body_states[:, -1, 10:13]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        self.common_step_counter = 0
        self.debug_termination = bool(self.cfg.get("basic", {}).get("debug_termination", False))
        self.debug_termination_interval = max(1, int(self.cfg.get("basic", {}).get("debug_termination_interval", 100)))
        self.debug_termination_max_envs = max(1, int(self.cfg.get("basic", {}).get("debug_termination_max_envs", 5)))
        self.debug_termination_env_id = min(
            self.num_envs - 1,
            max(0, int(self.cfg.get("basic", {}).get("debug_termination_env_id", 0))),
        )
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_frequency_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.dof_vel_filtered = torch.zeros_like(self.dof_vel)

        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies + 1, 3, dtype=torch.float, device=self.device)
        self.receive_mode_mask = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.env_ids_arange = torch.arange(self.num_envs, device=self.device)

        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw_rel = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_pitch = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.forward_body_vec = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.forward_body_vec[:, 0] = 1.0
        self.feet_edge_relative_pos = to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
        self.feet_edge_xy = self.feet_edge_relative_pos[:, 0:2]
        self.capture_center = torch.tensor(
            self.cfg["receive_geometry"]["capture_zone_center"], device=self.device, dtype=torch.float
        )
        self.capture_sigma = torch.tensor(
            self.cfg["receive_geometry"]["capture_zone_sigma"], device=self.device, dtype=torch.float
        )

        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

        hist_len = int(self.cfg["env"].get("detection_history_len", 6))
        self.ball_detection_history_xy = torch.zeros(self.num_envs, hist_len, 2, dtype=torch.float, device=self.device)
        self.ball_detection_history_age = torch.zeros(self.num_envs, hist_len, dtype=torch.float, device=self.device)
        self.ball_detection_history_valid = torch.zeros(self.num_envs, hist_len, dtype=torch.float, device=self.device)
        self.ball_detection_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_detection_age = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_detection_fps = float(self.cfg["ball"].get("detection_fps", 30.0))
        self.ball_detection_jitter = float(self.cfg["ball"].get("detection_fps_jitter", 0.15))
        self.ball_detection_interval = 1.0 / max(self.ball_detection_fps, 1.0)
        self.ball_detection_dropout_prob = float(self.cfg["ball"].get("detection_dropout_prob", 0.0))

        self.ball_pos_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_vel_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_line_dir_local = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.ball_line_dir_local[:, 0] = -1.0
        self.intercept_point_local = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.intercept_time_estimate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.arrival_confidence = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.estimated_ball_speed = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.receive_side = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.receive_side_onehot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.intercept_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.intercept_phase_onehot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)

        self.feet_pos_local = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device)
        self.chosen_foot_pos_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.chosen_receive_point_local = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.chosen_foot_inner_normal = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.intercept_pose_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.arrival_time_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.heading_alignment = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.side_foot_alignment = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.stance_gap = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stance_gap_target = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tunnel_risk = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tunnel_open_amount = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tunnel_entry_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.tunnel_entry_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.through_legs_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.through_legs_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_tunnel_y = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.right_tunnel_y = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tunnel_x_min = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tunnel_x_max = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.ball_has_been_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_first_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_first_contact_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.time_since_first_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_speed_drop = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_ball_lin_vel_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_has_passed_robot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.clear_miss_time_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.capture_zone_score = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.capture_zone_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.capture_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.capture_success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.controlled_receive_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.controlled_receive_success_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.pass_spawn_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.pass_target_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.pass_ref_dir_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.pass_ref_dir_xy[:, 0] = 1.0
        self.pass_ref_dir_local = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.pass_ref_dir_local[:, 0] = 1.0
        self.pass_distance = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_progress_along_pass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_max_progress_along_pass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_ball_max_progress_along_pass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.robot_progress_along_pass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.robot_lateral_error_to_pass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_lateral_error_to_pass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_progress_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.robot_progress_ratio = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.desired_receive_heading = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.desired_heading_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.desired_heading_vec_local = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.desired_heading_vec_local[:, 0] = 1.0
        self.behind_ball = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.behind_ball_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.late_chase = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.late_chase_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.orbit_behind = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.orbit_behind_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.target_lin_vel_local = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.target_ang_vel_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.target_gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.locomotion_drive = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.step_required = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_required_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.step_event_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.true_step_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.no_step_failure_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.skate_distance_precontact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.skating_indicator = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_base_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.feet_speed_xy = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)

        self.receive_side_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.receive_side_lock_value = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_receive_side = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.receive_side_switch_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.chosen_foot_forward = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.other_foot_pos_local = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.other_foot_forward = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.other_foot_inner_normal = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.contact_type = torch.full((self.num_envs,), 3, dtype=torch.long, device=self.device)
        self.contact_type_onehot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.good_receive_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.good_receive_contact_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wrong_surface_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wrong_surface_contact_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.support_foot_stable_at_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.support_foot_stable_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pass_progress_before_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.metric_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_control = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_tunnel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_through = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_capture = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_conf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_curriculum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_turn_behind = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_late_chase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_step_required = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_true_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_no_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_skate_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_wrong_surface = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_side_switch = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.metric_pass_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        curriculum_cfg = self.cfg.get("curriculum", {})
        self.curriculum_success_ring = torch.zeros(
            int(curriculum_cfg.get("evaluation_window", 256)),
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob = torch.zeros(int(curriculum_cfg.get("num_levels", 1)), dtype=torch.float, device=self.device)
        self.curriculum_prob[0] = 1.0
        self.curriculum_ring_idx = 0
        self.curriculum_global_level = 0
        self.curriculum_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        self.debug_draw_enabled = bool(self.cfg.get("viewer", {}).get("show_debug_viz", True))
        self.debug_video_overlay = bool(self.cfg.get("viewer", {}).get("debug_video_overlay", True))
        self.debug_video_panel_size = int(self.cfg.get("viewer", {}).get("debug_video_panel_size", 220))
        self.debug_video_scale = float(self.cfg.get("viewer", {}).get("debug_video_scale", 120.0))
        self.emit_reward_terms = bool(self.cfg.get("runner", {}).get("log_reward_terms", False))
        self.emit_metrics = bool(self.cfg.get("runner", {}).get("log_env_metrics", False))

        self.contact_type_inner_side = 0
        self.contact_type_front_toe = 1
        self.contact_type_other_foot = 2
        self.contact_type_body_or_none = 3

    def _reset_ball_at_robot_front(self, env_ids_to_reset_ball):
        if len(env_ids_to_reset_ball) == 0:
            return

        level = int(self.curriculum_global_level)
        generator_cfg = self.cfg["pass_generator"]
        n = len(env_ids_to_reset_ball)

        family_weights = torch.tensor(generator_cfg["family_weights"][level], device=self.device, dtype=torch.float)
        family_ids = torch.multinomial(family_weights, n, replacement=True)

        distance_min = float(generator_cfg["distance_min"][level])
        distance_max = float(generator_cfg["distance_max"][level])
        speed_min = float(generator_cfg["speed_min"][level])
        speed_max = float(generator_cfg["speed_max"][level])
        target_x_min = float(generator_cfg["target_x_min"][level])
        target_x_max = float(generator_cfg["target_x_max"][level])
        target_y_max = float(generator_cfg["target_y_max"][level])
        centerline_sigma = float(generator_cfg["centerline_target_sigma"][level])
        shallow_angle_min = np.deg2rad(float(generator_cfg["shallow_angle_deg_min"][level]))
        shallow_angle_max = np.deg2rad(float(generator_cfg["shallow_angle_deg_max"][level]))
        center_angle_std = np.deg2rad(float(generator_cfg["center_angle_deg_std"][level]))
        short_distance_min = float(generator_cfg["short_distance_min"][level])
        short_distance_max = float(generator_cfg["short_distance_max"][level])
        short_speed_min = float(generator_cfg["short_speed_min"][level])
        short_speed_max = float(generator_cfg["short_speed_max"][level])
        foot_target_y = float(generator_cfg["foot_target_y"][level])

        target_local = torch.zeros(n, 2, device=self.device)
        travel_dir = torch.zeros(n, 2, device=self.device)
        pass_distance = torch_rand_float(distance_min, distance_max, (n, 1), device=self.device).squeeze(-1)
        pass_speed = torch_rand_float(speed_min, speed_max, (n, 1), device=self.device).squeeze(-1)
        target_local[:, 0] = torch_rand_float(target_x_min, target_x_max, (n, 1), device=self.device).squeeze(-1)
        target_local[:, 1] = torch_rand_float(-target_y_max, target_y_max, (n, 1), device=self.device).squeeze(-1)

        center_mask = family_ids == 0
        if center_mask.any():
            idx = center_mask.nonzero(as_tuple=False).flatten()
            target_local[idx, 1] = apply_randomization(
                torch.zeros(len(idx), dtype=torch.float, device=self.device),
                {
                    "range": [-centerline_sigma, centerline_sigma],
                    "operation": "additive",
                    "distribution": "gaussian",
                },
            )
            delta = apply_randomization(
                torch.zeros(len(idx), dtype=torch.float, device=self.device),
                {
                    "range": [-center_angle_std, center_angle_std],
                    "operation": "additive",
                    "distribution": "gaussian",
                },
            )
            travel_dir[idx, 0] = -torch.cos(delta)
            travel_dir[idx, 1] = -torch.sin(delta)

        offcenter_mask = family_ids == 1
        if offcenter_mask.any():
            idx = offcenter_mask.nonzero(as_tuple=False).flatten()
            delta = apply_randomization(
                torch.zeros(len(idx), dtype=torch.float, device=self.device),
                {
                    "range": [-1.5 * center_angle_std, 1.5 * center_angle_std],
                    "operation": "additive",
                    "distribution": "gaussian",
                },
            )
            travel_dir[idx, 0] = -torch.cos(delta)
            travel_dir[idx, 1] = -torch.sin(delta)

        fast_center_mask = family_ids == 2
        if fast_center_mask.any():
            idx = fast_center_mask.nonzero(as_tuple=False).flatten()
            target_local[idx, 1] = apply_randomization(
                torch.zeros(len(idx), dtype=torch.float, device=self.device),
                {
                    "range": [-0.5 * centerline_sigma, 0.5 * centerline_sigma],
                    "operation": "additive",
                    "distribution": "gaussian",
                },
            )
            delta = apply_randomization(
                torch.zeros(len(idx), dtype=torch.float, device=self.device),
                {
                    "range": [-0.5 * center_angle_std, 0.5 * center_angle_std],
                    "operation": "additive",
                    "distribution": "gaussian",
                },
            )
            travel_dir[idx, 0] = -torch.cos(delta)
            travel_dir[idx, 1] = -torch.sin(delta)
            pass_speed[idx] = torch_rand_float(
                float(generator_cfg["fast_speed_min"][level]),
                float(generator_cfg["fast_speed_max"][level]),
                (len(idx), 1),
                device=self.device,
            ).squeeze(-1)

        shallow_left_mask = family_ids == 3
        if shallow_left_mask.any():
            idx = shallow_left_mask.nonzero(as_tuple=False).flatten()
            delta = torch_rand_float(shallow_angle_min, shallow_angle_max, (len(idx), 1), device=self.device).squeeze(-1)
            travel_dir[idx, 0] = -torch.cos(delta)
            travel_dir[idx, 1] = -torch.sin(delta)

        shallow_right_mask = family_ids == 4
        if shallow_right_mask.any():
            idx = shallow_right_mask.nonzero(as_tuple=False).flatten()
            delta = torch_rand_float(-shallow_angle_max, -shallow_angle_min, (len(idx), 1), device=self.device).squeeze(-1)
            travel_dir[idx, 0] = -torch.cos(delta)
            travel_dir[idx, 1] = -torch.sin(delta)

        short_left_mask = family_ids == 5
        if short_left_mask.any():
            idx = short_left_mask.nonzero(as_tuple=False).flatten()
            target_local[idx, 1] = torch_rand_float(0.6 * foot_target_y, foot_target_y, (len(idx), 1), device=self.device).squeeze(-1)
            delta = torch_rand_float(0.5 * shallow_angle_min, shallow_angle_min, (len(idx), 1), device=self.device).squeeze(-1)
            travel_dir[idx, 0] = -torch.cos(delta)
            travel_dir[idx, 1] = -torch.sin(delta)
            pass_distance[idx] = torch_rand_float(short_distance_min, short_distance_max, (len(idx), 1), device=self.device).squeeze(-1)
            pass_speed[idx] = torch_rand_float(short_speed_min, short_speed_max, (len(idx), 1), device=self.device).squeeze(-1)

        short_right_mask = family_ids == 6
        if short_right_mask.any():
            idx = short_right_mask.nonzero(as_tuple=False).flatten()
            target_local[idx, 1] = torch_rand_float(-foot_target_y, -0.6 * foot_target_y, (len(idx), 1), device=self.device).squeeze(-1)
            delta = torch_rand_float(-shallow_angle_min, -0.5 * shallow_angle_min, (len(idx), 1), device=self.device).squeeze(-1)
            travel_dir[idx, 0] = -torch.cos(delta)
            travel_dir[idx, 1] = -torch.sin(delta)
            pass_distance[idx] = torch_rand_float(short_distance_min, short_distance_max, (len(idx), 1), device=self.device).squeeze(-1)
            pass_speed[idx] = torch_rand_float(short_speed_min, short_speed_max, (len(idx), 1), device=self.device).squeeze(-1)

        travel_dir = travel_dir / torch.norm(travel_dir, dim=-1, keepdim=True).clamp_min(1e-6)
        spawn_local = target_local - travel_dir * pass_distance.unsqueeze(-1)

        base_quat = self.root_states[env_ids_to_reset_ball, 0, 3:7]
        base_pos = self.root_states[env_ids_to_reset_ball, 0, 0:3]

        spawn_local_3d = torch.zeros(n, 3, device=self.device)
        spawn_local_3d[:, 0:2] = spawn_local
        target_local_3d = torch.zeros(n, 3, device=self.device)
        target_local_3d[:, 0:2] = target_local

        spawn_world_offset = quat_rotate(base_quat, spawn_local_3d)
        target_world_offset = quat_rotate(base_quat, target_local_3d)
        spawn_world = base_pos + spawn_world_offset
        target_world = base_pos + target_world_offset

        self.root_states[env_ids_to_reset_ball, 1, 0:2] = spawn_world[:, 0:2]
        self.root_states[env_ids_to_reset_ball, 1, 2] = self.terrain.terrain_heights(spawn_world[:, 0:2]) + self.ball_radii[env_ids_to_reset_ball]
        self.root_states[env_ids_to_reset_ball, 1, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float
        ).unsqueeze(0).repeat(n, 1)

        pass_vel_world = target_world[:, 0:2] - spawn_world[:, 0:2]
        pass_vel_world = pass_vel_world / torch.norm(pass_vel_world, dim=-1, keepdim=True).clamp_min(1e-6)
        self.root_states[env_ids_to_reset_ball, 1, 7:9] = pass_vel_world * pass_speed.unsqueeze(-1)
        self.root_states[env_ids_to_reset_ball, 1, 9] = 0.0
        self.root_states[env_ids_to_reset_ball, 1, 10] = apply_randomization(
            torch.zeros(n, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("ball_init_ang_vel_x"),
        )
        self.root_states[env_ids_to_reset_ball, 1, 11] = apply_randomization(
            torch.zeros(n, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("ball_init_ang_vel_y"),
        )
        self.root_states[env_ids_to_reset_ball, 1, 12] = apply_randomization(
            torch.zeros(n, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("ball_init_ang_vel_z"),
        )

        pass_dir = target_world[:, 0:2] - spawn_world[:, 0:2]
        pass_dist = torch.norm(pass_dir, dim=-1).clamp_min(1.0e-6)
        self.pass_spawn_xy[env_ids_to_reset_ball] = spawn_world[:, 0:2]
        self.pass_target_xy[env_ids_to_reset_ball] = target_world[:, 0:2]
        self.pass_ref_dir_xy[env_ids_to_reset_ball] = pass_dir / pass_dist.unsqueeze(-1)
        self.pass_distance[env_ids_to_reset_ball] = pass_dist

    def _update_curriculum(self, env_ids):
        if len(env_ids) == 0 or not self.cfg.get("curriculum", {}).get("enabled", False):
            return

        if self.curriculum_prob.numel() > 0:
            self.curriculum_global_level = int(torch.argmax(self.curriculum_prob).item())
            self.curriculum_level[:] = self.curriculum_global_level

        successes = self.controlled_receive_success_latched[env_ids].float()
        ring_len = len(self.curriculum_success_ring)
        start = int(self.curriculum_ring_idx)
        count = int(len(env_ids))

        if count >= ring_len:
            self.curriculum_success_ring[:] = successes[-ring_len:]
        else:
            first_chunk = min(ring_len - start, count)
            self.curriculum_success_ring[start : start + first_chunk] = successes[:first_chunk]
            remaining = count - first_chunk
            if remaining > 0:
                self.curriculum_success_ring[:remaining] = successes[first_chunk:]

        self.curriculum_ring_idx = int((start + count) % ring_len)

        success_rate = self.curriculum_success_ring.mean().item()
        curriculum_cfg = self.cfg["curriculum"]
        max_level = int(curriculum_cfg["num_levels"]) - 1
        if success_rate > float(curriculum_cfg["advance_threshold"]) and self.curriculum_global_level < max_level:
            self.curriculum_global_level += 1
        elif success_rate < float(curriculum_cfg["retreat_threshold"]) and self.curriculum_global_level > 0:
            self.curriculum_global_level -= 1
        self.curriculum_level[:] = self.curriculum_global_level
        self.curriculum_prob.zero_()
        self.curriculum_prob[self.curriculum_global_level] = 1.0
        level_value = float(self.curriculum_global_level)
        self.mean_lin_vel_level = level_value
        self.max_lin_vel_level = level_value
        self.mean_ang_vel_level = 0.0
        self.max_ang_vel_level = 0.0

    def _resample_commands(self):
        return

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._refresh_feet_state()
        self._insert_ball_detections(torch.arange(self.num_envs, device=self.device), reset_timer=True)
        self._update_receive_state()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.dof_vel_filtered[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.gait_frequency_offset[env_ids] = 0.0
        self.gait_frequency[env_ids] = float(self.cfg["commands"]["gait_frequency_base"])
        self.gait_process[env_ids] = torch_rand_float(0.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(-1)
        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)

        self.ball_detection_history_xy[env_ids] = 0.0
        self.ball_detection_history_age[env_ids] = 0.0
        self.ball_detection_history_valid[env_ids] = 0.0
        self.ball_detection_timer[env_ids] = torch_rand_float(0.0, self.ball_detection_interval, (len(env_ids), 1), device=self.device).squeeze(-1)
        self.ball_detection_age[env_ids] = self.cfg["ball"].get("detection_age_clip_s", 0.25)

        self.ball_line_dir_local[env_ids, 0] = -1.0
        self.ball_line_dir_local[env_ids, 1] = 0.0
        self.intercept_point_local[env_ids] = 0.0
        self.intercept_time_estimate[env_ids] = 0.0
        self.arrival_confidence[env_ids] = 0.0
        self.estimated_ball_speed[env_ids] = 0.0
        self.receive_side[env_ids] = 0
        self.receive_side_onehot[env_ids] = 0.0
        self.intercept_phase[env_ids] = 0
        self.intercept_phase_onehot[env_ids] = 0.0
        self.chosen_foot_pos_local[env_ids] = 0.0
        self.chosen_receive_point_local[env_ids] = 0.0
        self.chosen_foot_inner_normal[env_ids] = 0.0
        self.intercept_pose_error[env_ids] = 0.0
        self.arrival_time_error[env_ids] = 0.0
        self.heading_alignment[env_ids] = 0.0
        self.side_foot_alignment[env_ids] = 0.0
        self.stance_gap[env_ids] = 0.0
        self.stance_gap_target[env_ids] = 0.0
        self.tunnel_risk[env_ids] = 0.0
        self.tunnel_open_amount[env_ids] = 0.0
        self.tunnel_entry_event[env_ids] = False
        self.tunnel_entry_latched[env_ids] = False
        self.through_legs_event[env_ids] = False
        self.through_legs_latched[env_ids] = False
        self.left_tunnel_y[env_ids] = 0.0
        self.right_tunnel_y[env_ids] = 0.0
        self.tunnel_x_min[env_ids] = 0.0
        self.tunnel_x_max[env_ids] = 0.0
        self.ball_has_been_contacted[env_ids] = False
        self.ball_first_contact_event[env_ids] = False
        self.ball_first_contact_time[env_ids] = 0.0
        self.time_since_first_contact[env_ids] = 0.0
        self.ball_speed_drop[env_ids] = 0.0
        self.last_ball_lin_vel_world[env_ids] = 0.0
        self.ball_has_passed_robot[env_ids] = False
        self.clear_miss_time_buf[env_ids] = 0.0
        self.capture_zone_score[env_ids] = 0.0
        self.capture_zone_error[env_ids] = 0.0
        self.capture_success[env_ids] = False
        self.capture_success_latched[env_ids] = False
        self.controlled_receive_success[env_ids] = False
        self.controlled_receive_success_latched[env_ids] = False
        self.pass_spawn_xy[env_ids] = 0.0
        self.pass_target_xy[env_ids] = 0.0
        self.pass_ref_dir_xy[env_ids] = 0.0
        self.pass_ref_dir_xy[env_ids, 0] = 1.0
        self.pass_ref_dir_local[env_ids] = 0.0
        self.pass_ref_dir_local[env_ids, 0] = 1.0
        self.pass_distance[env_ids] = 1.0
        self.ball_progress_along_pass[env_ids] = 0.0
        self.ball_max_progress_along_pass[env_ids] = 0.0
        self.prev_ball_max_progress_along_pass[env_ids] = 0.0
        self.robot_progress_along_pass[env_ids] = 0.0
        self.robot_lateral_error_to_pass[env_ids] = 0.0
        self.ball_lateral_error_to_pass[env_ids] = 0.0
        self.ball_progress_ratio[env_ids] = 0.0
        self.robot_progress_ratio[env_ids] = 0.0
        self.desired_receive_heading[env_ids] = 0.0
        self.desired_heading_error[env_ids] = 0.0
        self.desired_heading_vec_local[env_ids] = 0.0
        self.desired_heading_vec_local[env_ids, 0] = 1.0
        self.behind_ball[env_ids] = False
        self.behind_ball_latched[env_ids] = False
        self.late_chase[env_ids] = False
        self.late_chase_latched[env_ids] = False
        self.orbit_behind[env_ids] = False
        self.orbit_behind_latched[env_ids] = False
        self.target_lin_vel_local[env_ids] = 0.0
        self.target_ang_vel_yaw[env_ids] = 0.0
        self.target_gait_frequency[env_ids] = float(self.cfg["locomotion_targets"]["idle_gait_frequency"])
        self.locomotion_drive[env_ids] = 0.0
        self.step_required[env_ids] = False
        self.step_required_latched[env_ids] = False
        self.step_active[env_ids] = False
        self.step_event_latched[env_ids] = False
        self.true_step_latched[env_ids] = False
        self.no_step_failure_latched[env_ids] = False
        self.skate_distance_precontact[env_ids] = 0.0
        self.skating_indicator[env_ids] = 0.0
        self.prev_base_pos_xy[env_ids] = self.root_states[env_ids, 0, 0:2]
        self.feet_speed_xy[env_ids] = 0.0
        self.receive_side_locked[env_ids] = False
        self.receive_side_lock_value[env_ids] = 0
        self.last_receive_side[env_ids] = 0
        self.receive_side_switch_count[env_ids] = 0.0
        self.chosen_foot_forward[env_ids] = 0.0
        self.other_foot_pos_local[env_ids] = 0.0
        self.other_foot_forward[env_ids] = 0.0
        self.other_foot_inner_normal[env_ids] = 0.0
        self.contact_type[env_ids] = self.contact_type_body_or_none
        self.contact_type_onehot[env_ids] = 0.0
        self.contact_type_onehot[env_ids, self.contact_type_body_or_none] = 1.0
        self.good_receive_contact[env_ids] = False
        self.good_receive_contact_latched[env_ids] = False
        self.wrong_surface_contact_event[env_ids] = False
        self.wrong_surface_contact_latched[env_ids] = False
        self.support_foot_stable_at_contact[env_ids] = False
        self.support_foot_stable_latched[env_ids] = False
        self.pass_progress_before_contact[env_ids] = 0.0
        self.curriculum_level[env_ids] = self.curriculum_global_level
        self.extras["time_outs"] = self.time_out_buf

    def _insert_ball_detections(self, detect_ids, reset_timer):
        if len(detect_ids) == 0:
            return
        ball_rel_world = self.ball_pos[detect_ids] - self.base_pos[detect_ids]
        ball_rel_local = quat_rotate_inverse(self.base_quat[detect_ids], ball_rel_world)
        noisy_xy = apply_randomization(ball_rel_local[:, 0:2], self.cfg["noise"].get("ball_pos"))

        self.ball_detection_history_xy[detect_ids, 1:] = self.ball_detection_history_xy[detect_ids, :-1].clone()
        self.ball_detection_history_age[detect_ids, 1:] = self.ball_detection_history_age[detect_ids, :-1].clone()
        self.ball_detection_history_valid[detect_ids, 1:] = self.ball_detection_history_valid[detect_ids, :-1].clone()
        self.ball_detection_history_xy[detect_ids, 0] = noisy_xy
        self.ball_detection_history_age[detect_ids, 0] = 0.0
        self.ball_detection_history_valid[detect_ids, 0] = 1.0
        self.ball_detection_age[detect_ids] = 0.0

        if reset_timer:
            jitter = 1.0 + torch_rand_float(
                -self.ball_detection_jitter,
                self.ball_detection_jitter,
                (len(detect_ids), 1),
                device=self.device,
            ).squeeze(-1)
            self.ball_detection_timer[detect_ids] = self.ball_detection_interval * jitter

    def _update_ball_detection(self, force_reset=False):
        self.ball_detection_timer -= self.dt
        self.ball_detection_age += self.dt
        self.ball_detection_history_age += self.dt * self.ball_detection_history_valid

        if force_reset:
            detect_ids = torch.arange(self.num_envs, device=self.device)
        else:
            detect_ids = (self.ball_detection_timer <= 0.0).nonzero(as_tuple=False).flatten()

        if len(detect_ids) == 0:
            return

        if not force_reset and self.ball_detection_dropout_prob > 0.0:
            keep_mask = torch.rand(len(detect_ids), device=self.device) > self.ball_detection_dropout_prob
            kept_ids = detect_ids[keep_mask]
            dropped_ids = detect_ids[~keep_mask]
            if len(dropped_ids) > 0:
                jitter = 1.0 + torch_rand_float(
                    -self.ball_detection_jitter,
                    self.ball_detection_jitter,
                    (len(dropped_ids), 1),
                    device=self.device,
                ).squeeze(-1)
                self.ball_detection_timer[dropped_ids] = self.ball_detection_interval * jitter
            detect_ids = kept_ids

        self._insert_ball_detections(detect_ids, reset_timer=True)

    def _estimate_intercept_from_history(self):
        times = -self.ball_detection_history_age
        valid = self.ball_detection_history_valid
        weights = valid * torch.exp(-self.ball_detection_history_age / max(float(self.cfg["estimator"]["history_time_scale"]), 1.0e-6))
        weight_sum = weights.sum(dim=1).clamp_min(1.0e-6)

        mean_t = (weights * times).sum(dim=1, keepdim=True) / weight_sum.unsqueeze(-1)
        mean_xy = (weights.unsqueeze(-1) * self.ball_detection_history_xy).sum(dim=1) / weight_sum.unsqueeze(-1)

        centered_t = times - mean_t
        var_t = (weights * centered_t.square()).sum(dim=1).clamp_min(1.0e-6)
        slope_xy = (weights.unsqueeze(-1) * centered_t.unsqueeze(-1) * (self.ball_detection_history_xy - mean_xy.unsqueeze(1))).sum(dim=1) / var_t.unsqueeze(-1)
        intercept_xy = mean_xy - slope_xy * mean_t

        speed = torch.norm(slope_xy, dim=-1)
        dir_xy = slope_xy / speed.unsqueeze(-1).clamp_min(1.0e-6)

        residual = self.ball_detection_history_xy - (intercept_xy.unsqueeze(1) + slope_xy.unsqueeze(1) * times.unsqueeze(-1))
        residual_norm = torch.norm(residual, dim=-1)
        residual_mse = (weights * residual_norm.square()).sum(dim=1) / weight_sum

        valid_count = valid.sum(dim=1)
        count_score = torch.clamp(valid_count / float(valid.shape[1]), 0.0, 1.0)
        age_score = torch.exp(-self.ball_detection_age / max(float(self.cfg["estimator"]["max_recent_age"]), 1.0e-6))
        fit_score = torch.exp(-residual_mse / max(float(self.cfg["estimator"]["fit_residual_sigma"]), 1.0e-6))
        speed_score = torch.sigmoid((speed - float(self.cfg["estimator"]["min_speed_for_fit"])) / max(float(self.cfg["estimator"]["speed_conf_sigma"]), 1.0e-6))
        confidence = count_score * age_score * fit_score * speed_score

        latest_valid = valid[:, 0] > 0.5
        fallback_xy = torch.where(latest_valid.unsqueeze(-1), self.ball_detection_history_xy[:, 0], mean_xy)
        fallback_dir = -fallback_xy / torch.norm(fallback_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)

        use_fit = (valid_count >= 2.0) & (speed > float(self.cfg["estimator"]["min_speed_for_fit"]))
        dir_xy = torch.where(use_fit.unsqueeze(-1), dir_xy, fallback_dir)
        intercept_xy = torch.where(use_fit.unsqueeze(-1), intercept_xy, fallback_xy)
        speed = torch.where(use_fit, speed, torch.zeros_like(speed))

        receive_plane_x = float(self.cfg["estimator"]["receive_plane_x"])
        denom = slope_xy[:, 0]
        valid_plane = use_fit & (torch.abs(denom) > float(self.cfg["estimator"]["min_plane_speed"])) & (intercept_xy[:, 0] > receive_plane_x)
        t_plane = (receive_plane_x - intercept_xy[:, 0]) / denom.clamp(min=-1.0e6, max=1.0e6)
        t_plane = torch.clamp(t_plane, min=0.0, max=float(self.cfg["estimator"]["time_clip_s"]))
        predicted_point = intercept_xy + slope_xy * t_plane.unsqueeze(-1)

        nearest_t = -torch.sum(intercept_xy * dir_xy, dim=-1) / speed.clamp_min(1.0e-6)
        nearest_t = torch.clamp(nearest_t, min=0.0, max=float(self.cfg["estimator"]["time_clip_s"]))
        nearest_point = intercept_xy + slope_xy * nearest_t.unsqueeze(-1)

        use_plane_solution = valid_plane & (t_plane <= float(self.cfg["estimator"]["time_clip_s"]))
        self.intercept_point_local[:] = torch.where(use_plane_solution.unsqueeze(-1), predicted_point, nearest_point)
        self.intercept_time_estimate[:] = torch.where(use_plane_solution, t_plane, nearest_t)
        self.ball_line_dir_local[:] = dir_xy
        self.arrival_confidence[:] = confidence
        self.estimated_ball_speed[:] = speed

    def _compute_local_foot_geometry(self):
        feet_rel_world = self.feet_pos - self.base_pos.unsqueeze(1)
        feet_local = quat_rotate_inverse(
            self.base_quat.unsqueeze(1).expand(-1, len(self.feet_indices), -1).reshape(-1, 4),
            feet_rel_world.reshape(-1, 3),
        ).view(self.num_envs, len(self.feet_indices), 3)
        self.feet_pos_local[:] = feet_local

        foot_y_axis = torch.stack((-torch.sin(self.feet_yaw_rel), torch.cos(self.feet_yaw_rel)), dim=-1)
        left_inner_normal = -foot_y_axis[:, 0]
        right_inner_normal = foot_y_axis[:, 1]

        use_left = self.intercept_point_local[:, 1] >= 0.0
        low_conf = self.arrival_confidence < float(self.cfg["estimator"]["low_confidence_threshold"])
        use_left = torch.where(low_conf, self.ball_pos_local[:, 1] >= 0.0, use_left)
        proposed_side = torch.where(use_left, torch.zeros_like(self.receive_side), torch.ones_like(self.receive_side))
        pre_contact = ~self.ball_has_been_contacted
        lock_threshold = float(self.cfg["contact_classifier"]["side_lock_conf_threshold"])
        new_lock = (self.arrival_confidence >= lock_threshold) & pre_contact & (~self.receive_side_locked)
        self.receive_side_lock_value[new_lock] = proposed_side[new_lock]
        self.receive_side_locked |= new_lock

        current_side = torch.where(self.receive_side_locked, self.receive_side_lock_value, proposed_side)
        switched = pre_contact & (self.episode_length_buf > 0) & (current_side != self.last_receive_side)
        self.receive_side_switch_count += switched.float()
        self.last_receive_side[:] = current_side
        self.receive_side[:] = current_side
        self.receive_side_onehot.zero_()
        self.receive_side_onehot.scatter_(1, self.receive_side.unsqueeze(-1), 1.0)

        chosen_idx = self.receive_side
        other_idx = 1 - chosen_idx
        self.chosen_foot_pos_local[:] = feet_local[self.env_ids_arange, chosen_idx]
        self.other_foot_pos_local[:] = feet_local[self.env_ids_arange, other_idx]
        chosen_is_left = (self.receive_side == 0).unsqueeze(-1)
        self.chosen_foot_inner_normal[:] = torch.where(chosen_is_left, left_inner_normal, right_inner_normal)

        foot_forward = torch.stack((torch.cos(self.feet_yaw_rel), torch.sin(self.feet_yaw_rel)), dim=-1)
        chosen_forward = foot_forward[self.env_ids_arange, chosen_idx]
        other_forward = foot_forward[self.env_ids_arange, other_idx]
        self.chosen_foot_forward[:] = chosen_forward
        self.other_foot_forward[:] = other_forward
        self.other_foot_inner_normal[:] = torch.where(chosen_is_left, right_inner_normal, left_inner_normal)
        forward_offset = float(self.cfg["receive_geometry"]["foot_forward_contact_offset"])
        side_offset = float(self.cfg["receive_geometry"]["foot_inner_contact_offset"])
        self.chosen_receive_point_local[:] = (
            self.chosen_foot_pos_local[:, 0:2]
            + chosen_forward * forward_offset
            + self.chosen_foot_inner_normal * side_offset
        )

        incoming_dir = -self.ball_line_dir_local
        incoming_dir = incoming_dir / torch.norm(incoming_dir, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.side_foot_alignment[:] = torch.clamp(torch.sum(self.chosen_foot_inner_normal * incoming_dir, dim=-1), min=0.0, max=1.0)

        self.heading_alignment[:] = torch.clamp(torch.sum(self.forward_body_vec * incoming_dir, dim=-1), min=0.0, max=1.0)
        self.intercept_pose_error[:] = torch.norm(self.chosen_receive_point_local - self.intercept_point_local, dim=-1)

        plan_speed = torch.maximum(
            torch.norm(self.filtered_lin_vel[:, 0:2], dim=-1),
            torch.full((self.num_envs,), float(self.cfg["receive_geometry"]["robot_nominal_reposition_speed"]), device=self.device),
        )
        estimated_arrival = self.intercept_pose_error / plan_speed.clamp_min(1.0e-6)
        self.arrival_time_error[:] = estimated_arrival - self.intercept_time_estimate

    def _update_pass_frame_state(self):
        pass_dir = self.pass_ref_dir_xy / torch.norm(self.pass_ref_dir_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        pass_perp = torch.stack((-pass_dir[:, 1], pass_dir[:, 0]), dim=-1)

        ball_from_spawn = self.ball_pos[:, 0:2] - self.pass_spawn_xy
        robot_from_spawn = self.base_pos[:, 0:2] - self.pass_spawn_xy

        self.ball_progress_along_pass[:] = torch.sum(ball_from_spawn * pass_dir, dim=-1)
        self.prev_ball_max_progress_along_pass[:] = self.ball_max_progress_along_pass
        self.ball_max_progress_along_pass[:] = torch.maximum(self.ball_max_progress_along_pass, self.ball_progress_along_pass)
        self.robot_progress_along_pass[:] = torch.sum(robot_from_spawn * pass_dir, dim=-1)
        self.ball_lateral_error_to_pass[:] = torch.sum(ball_from_spawn * pass_perp, dim=-1)
        self.robot_lateral_error_to_pass[:] = torch.sum(robot_from_spawn * pass_perp, dim=-1)

        pass_distance = self.pass_distance.clamp_min(1.0e-6)
        self.ball_progress_ratio[:] = torch.clamp(self.ball_progress_along_pass / pass_distance, min=-0.25, max=2.0)
        self.robot_progress_ratio[:] = torch.clamp(self.robot_progress_along_pass / pass_distance, min=-0.25, max=2.0)

        _, _, base_yaw = get_euler_xyz(self.base_quat)
        base_yaw = (base_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        self.pass_ref_dir_local[:, 0] = cos_yaw * pass_dir[:, 0] + sin_yaw * pass_dir[:, 1]
        self.pass_ref_dir_local[:, 1] = -sin_yaw * pass_dir[:, 0] + cos_yaw * pass_dir[:, 1]

        side_sign = torch.where(self.receive_side == 0, 1.0, -1.0)
        heading_bias = float(self.cfg["turn_guard"]["receive_heading_side_bias"])
        desired_forward_world = pass_dir + side_sign.unsqueeze(-1) * heading_bias * pass_perp
        desired_forward_world = desired_forward_world / torch.norm(desired_forward_world, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.desired_receive_heading[:] = torch.atan2(desired_forward_world[:, 1], desired_forward_world[:, 0])
        self.desired_heading_error[:] = (self.desired_receive_heading - base_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        self.desired_heading_vec_local[:, 0] = cos_yaw * desired_forward_world[:, 0] + sin_yaw * desired_forward_world[:, 1]
        self.desired_heading_vec_local[:, 1] = -sin_yaw * desired_forward_world[:, 0] + cos_yaw * desired_forward_world[:, 1]

        pre_contact = ~self.ball_has_been_contacted
        behind_margin = float(self.cfg["turn_guard"]["behind_ball_x_margin"])
        self.behind_ball[:] = pre_contact & (self.ball_pos_local[:, 0] < -behind_margin)
        self.behind_ball_latched |= self.behind_ball

        late_threshold = float(self.cfg["turn_guard"]["late_chase_progress_ratio"])
        late_grace_steps = np.ceil(float(self.cfg["turn_guard"].get("late_chase_grace_s", 0.35)) / self.dt)
        late_progress_gap = float(self.cfg["turn_guard"].get("late_chase_robot_lag_margin", 0.20))
        late_ball_x_threshold = float(self.cfg["turn_guard"].get("late_chase_ball_x_threshold", 0.0))
        self.late_chase[:] = (
            pre_contact
            & (self.episode_length_buf > late_grace_steps)
            & (self.ball_progress_ratio > late_threshold)
            & ((self.ball_progress_ratio - self.robot_progress_ratio) > late_progress_gap)
            & (self.ball_pos_local[:, 0] < late_ball_x_threshold)
        )
        self.late_chase_latched |= self.late_chase

        orbit_heading = float(self.cfg["turn_guard"]["orbit_heading_threshold"])
        orbit_distance = float(self.cfg["turn_guard"]["orbit_ball_distance"])
        orbit_progress = float(self.cfg["turn_guard"]["orbit_progress_threshold"])
        ball_dist_local = torch.norm(self.ball_pos_local[:, 0:2], dim=-1)
        self.orbit_behind[:] = (
            pre_contact
            & (torch.abs(self.desired_heading_error) > orbit_heading)
            & (ball_dist_local < orbit_distance)
            & (self.ball_progress_ratio > orbit_progress)
        )
        self.orbit_behind_latched |= self.orbit_behind

        self.pass_progress_before_contact[:] = torch.where(
            self.ball_has_been_contacted,
            self.pass_progress_before_contact,
            self.ball_max_progress_along_pass,
        )

    def _update_locomotion_targets(self):
        cfg = self.cfg["locomotion_targets"]
        reposition_delta = self.intercept_point_local - self.chosen_receive_point_local
        forward_clip = float(cfg.get("forward_clip", cfg["step_required_distance"]))
        forward_error = torch.clamp(reposition_delta[:, 0], min=-forward_clip, max=forward_clip)
        lateral_error = reposition_delta[:, 1]

        target_lin_x = forward_error * float(cfg["position_gain_x"])
        target_lin_y = lateral_error * float(cfg["position_gain_y"])
        self.target_lin_vel_local[:, 0] = torch.clamp(
            target_lin_x,
            min=-float(cfg["max_lin_vel_x"]),
            max=float(cfg["max_lin_vel_x"]),
        )
        self.target_lin_vel_local[:, 1] = torch.clamp(
            target_lin_y,
            min=-float(cfg["max_lin_vel_y"]),
            max=float(cfg["max_lin_vel_y"]),
        )
        self.target_ang_vel_yaw[:] = torch.clamp(
            self.desired_heading_error * float(cfg["heading_gain"]),
            min=-float(cfg["max_ang_vel_yaw"]),
            max=float(cfg["max_ang_vel_yaw"]),
        )

        weighted_pose = torch.stack(
            (
                forward_error * float(cfg.get("forward_error_weight", 1.0)),
                lateral_error * float(cfg.get("lateral_error_weight", 1.0)),
            ),
            dim=-1,
        )
        pose_error = torch.norm(weighted_pose, dim=-1)
        near_dist = float(cfg["near_distance"])
        drive_sigma = max(float(cfg["drive_sigma"]), 1.0e-6)
        pre_contact = (~self.ball_has_been_contacted).float()
        confidence = torch.clamp(self.arrival_confidence, 0.2, 1.0)
        self.locomotion_drive[:] = torch.sigmoid((pose_error - near_dist) / drive_sigma) * confidence * pre_contact

        idle_gait = float(cfg["idle_gait_frequency"])
        active_gait = float(cfg["active_gait_frequency"])
        self.target_gait_frequency[:] = idle_gait + (active_gait - idle_gait) * self.locomotion_drive

        step_threshold = float(cfg["step_required_distance"])
        self.step_required[:] = pose_error > step_threshold
        self.step_required_latched |= self.step_required

        feet_delta = self.feet_pos[:, :, 0:2] - self.last_feet_pos[:, :, 0:2]
        self.feet_speed_xy[:] = torch.norm(feet_delta / max(self.dt, 1.0e-6), dim=-1)

        swing_gate = self.target_gait_frequency > float(cfg["step_frequency_threshold"])
        self.step_active[:] = ((~self.feet_contact).any(dim=-1)) & swing_gate & self.step_required & (~self.ball_has_been_contacted)
        self.step_event_latched |= self.step_active
        self.true_step_latched |= self.step_active & self.step_required

        base_delta_xy = self.base_pos[:, 0:2] - self.prev_base_pos_xy
        both_feet_contact = self.feet_contact.all(dim=-1)
        target_speed = torch.norm(self.target_lin_vel_local, dim=-1)
        base_speed_xy = torch.norm(base_delta_xy, dim=-1) / max(self.dt, 1.0e-6)
        skate_speed_thresh = float(cfg["skate_speed_threshold"])
        locomoting = target_speed > float(cfg["locomotion_speed_threshold"])
        skate_excess = torch.clamp(base_speed_xy - skate_speed_thresh, min=0.0)
        self.skating_indicator[:] = skate_excess * both_feet_contact.float() * locomoting.float() * pre_contact
        self.skate_distance_precontact += torch.norm(base_delta_xy, dim=-1) * both_feet_contact.float() * locomoting.float() * pre_contact

        self.no_step_failure_latched |= (
            self.step_required_latched
            & (~self.step_event_latched)
            & ((self.late_chase_latched | self.ball_has_passed_robot | self.ball_has_been_contacted))
        )

    def _classify_first_contact(self, first_contact_now):
        env_ids = first_contact_now.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        cfg = self.cfg["contact_classifier"]
        chosen_to_ball = self.ball_pos_local[env_ids, 0:2] - self.chosen_foot_pos_local[env_ids, 0:2]
        rel_to_receive = self.ball_pos_local[env_ids, 0:2] - self.chosen_receive_point_local[env_ids]
        chosen_forward = self.chosen_foot_forward[env_ids]
        chosen_inner = self.chosen_foot_inner_normal[env_ids]

        forward_err = torch.sum(rel_to_receive * chosen_forward, dim=-1)
        side_err = torch.sum(rel_to_receive * chosen_inner, dim=-1)
        forward_proj = torch.sum(chosen_to_ball * chosen_forward, dim=-1)
        inner_proj = torch.sum(chosen_to_ball * chosen_inner, dim=-1)
        other_foot_dist = torch.norm(self.ball_pos_local[env_ids, 0:2] - self.other_foot_pos_local[env_ids, 0:2], dim=-1)
        support_idx = 1 - self.receive_side[env_ids]
        support_speed = self.feet_speed_xy[env_ids, support_idx]
        support_stable = support_speed < float(cfg["support_foot_speed_max"])
        body_contact = torch.any(
            torch.norm(self.contact_forces[env_ids][:, self.penalized_contact_indices, :], dim=-1)
            > float(cfg["body_contact_force_threshold"]),
            dim=-1,
        ) if len(self.penalized_contact_indices) > 0 else torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        inner_side = (
            (torch.abs(forward_err) <= float(cfg["inner_side_forward_tolerance"]))
            & (torch.abs(side_err) <= float(cfg["inner_side_lateral_tolerance"]))
            & (inner_proj > float(cfg["inner_side_min_projection"]))
        )
        front_toe = (
            (forward_proj > float(cfg["toe_forward_min"]))
            & (torch.abs(inner_proj) < float(cfg["toe_lateral_max"]))
            & (~inner_side)
        )
        other_foot = (other_foot_dist < float(cfg["other_foot_radius"])) & (~inner_side) & (~front_toe)

        contact_type = torch.full((len(env_ids),), self.contact_type_body_or_none, dtype=torch.long, device=self.device)
        contact_type[front_toe] = self.contact_type_front_toe
        contact_type[other_foot] = self.contact_type_other_foot
        contact_type[inner_side] = self.contact_type_inner_side
        contact_type[body_contact & (~inner_side)] = self.contact_type_body_or_none

        self.contact_type[env_ids] = contact_type
        onehot = torch.zeros(len(env_ids), 4, dtype=torch.float, device=self.device)
        onehot.scatter_(1, contact_type.unsqueeze(-1), 1.0)
        self.contact_type_onehot[env_ids] = onehot
        self.support_foot_stable_at_contact[env_ids] = support_stable
        self.support_foot_stable_latched[env_ids] = support_stable

        good_contact = inner_side & support_stable & (~body_contact)
        self.good_receive_contact[env_ids] = good_contact
        self.good_receive_contact_latched[env_ids] |= good_contact
        wrong_surface = ~good_contact
        self.wrong_surface_contact_event[env_ids] = wrong_surface
        self.wrong_surface_contact_latched[env_ids] |= wrong_surface
        self.pass_progress_before_contact[env_ids] = self.ball_max_progress_along_pass[env_ids]

    def _compute_stance_tunnel(self):
        edge_xy = self.feet_edge_xy.view(1, 1, -1, 2)
        foot_cos = torch.cos(self.feet_yaw_rel).unsqueeze(-1)
        foot_sin = torch.sin(self.feet_yaw_rel).unsqueeze(-1)
        edge_x = edge_xy[..., 0]
        edge_y = edge_xy[..., 1]
        rot_x = foot_cos * edge_x - foot_sin * edge_y
        rot_y = foot_sin * edge_x + foot_cos * edge_y
        edge_local_xy = self.feet_pos_local[:, :, None, 0:2]
        edge_local_xy = torch.stack(
            (
                edge_local_xy[..., 0] + rot_x,
                edge_local_xy[..., 1] + rot_y,
            ),
            dim=-1,
        )

        left_edges = edge_local_xy[:, 0]
        right_edges = edge_local_xy[:, 1]
        left_inner_y = left_edges[:, :, 1].min(dim=1).values
        right_inner_y = right_edges[:, :, 1].max(dim=1).values
        left_x_min = left_edges[:, :, 0].min(dim=1).values
        left_x_max = left_edges[:, :, 0].max(dim=1).values
        right_x_min = right_edges[:, :, 0].min(dim=1).values
        right_x_max = right_edges[:, :, 0].max(dim=1).values

        corridor_x_min = torch.maximum(left_x_min, right_x_min)
        corridor_x_max = torch.minimum(left_x_max, right_x_max)
        corridor_valid = corridor_x_max > corridor_x_min
        self.left_tunnel_y[:] = left_inner_y
        self.right_tunnel_y[:] = right_inner_y
        self.tunnel_x_min[:] = corridor_x_min
        self.tunnel_x_max[:] = corridor_x_max

        self.stance_gap[:] = torch.clamp(left_inner_y - right_inner_y - 2.0 * self.ball_radii, min=0.0)

        centerline_gate = torch.exp(
            -torch.square(self.ball_lateral_error_to_pass) / max(float(self.cfg["rewards"]["centerline_sigma"]), 1.0e-6)
        )
        foot_open_gate = 1.0 - self.side_foot_alignment
        safe_gap = torch.full_like(self.stance_gap, float(self.cfg["receive_geometry"]["safe_gap_margin"]))
        swing_gate = self.step_active.float()
        modest_gap = torch.full_like(self.stance_gap, float(self.cfg["receive_geometry"]["wide_gap_target"]))
        swing_open_gate = swing_gate * centerline_gate * (0.5 + 0.5 * foot_open_gate)
        self.stance_gap_target[:] = safe_gap + swing_open_gate * (modest_gap - safe_gap)

        gap_sigma = max(float(self.cfg["rewards"]["stance_gap_sigma"]), 1.0e-6)
        gap_open_gate = torch.sigmoid((self.stance_gap - self.stance_gap_target) / gap_sigma)
        approach_gate = torch.sigmoid(
            (self.estimated_ball_speed - float(self.cfg["rewards"]["min_approach_speed_for_tunnel"])) /
            max(float(self.cfg["rewards"]["approach_speed_sigma"]), 1.0e-6)
        )
        confidence_gate = torch.clamp(self.arrival_confidence, 0.0, 1.0)
        self.tunnel_risk[:] = gap_open_gate * centerline_gate * approach_gate * confidence_gate
        self.tunnel_open_amount[:] = torch.clamp(self.stance_gap - self.stance_gap_target, min=0.0)

        ball_x = self.ball_pos_local[:, 0]
        ball_y = self.ball_pos_local[:, 1]
        inside_x = corridor_valid & (ball_x >= corridor_x_min) & (ball_x <= corridor_x_max)
        inside_y = (ball_y <= left_inner_y) & (ball_y >= right_inner_y)
        approaching = self.ball_vel_local[:, 0] < -float(self.cfg["rewards"]["min_ball_x_velocity_for_entry"])

        self.tunnel_entry_event[:] = inside_x & inside_y & approaching
        self.tunnel_entry_latched |= self.tunnel_entry_event

        through_margin = float(self.cfg["rewards"]["through_legs_x_margin"])
        self.through_legs_event[:] = self.tunnel_entry_latched & (ball_x < -through_margin)
        self.through_legs_latched |= self.through_legs_event

    def _update_contact_state(self):
        self.wrong_surface_contact_event[:] = False
        foot_to_ball = self.ball_pos[:, 0:2].unsqueeze(1) - self.feet_pos[:, :, 0:2]
        min_foot_dist = torch.norm(foot_to_ball, dim=-1).min(dim=1).values
        ball_speed_now = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        ball_speed_prev = torch.norm(self.last_ball_lin_vel_world[:, 0:2], dim=-1)
        self.ball_speed_drop[:] = ball_speed_prev - ball_speed_now

        speed_drop_thresh = float(self.cfg["rewards"]["contact_speed_drop_threshold"])
        contact_radius = float(self.cfg["rewards"]["contact_foot_radius"])
        moving_thresh = float(self.cfg["rewards"]["contact_ball_speed_threshold"])
        first_contact_now = (
            (self.ball_speed_drop > speed_drop_thresh)
            & (min_foot_dist < contact_radius)
            & (ball_speed_prev > moving_thresh)
            & (~self.ball_has_been_contacted)
        )
        self.ball_first_contact_event[:] = first_contact_now
        self.ball_has_been_contacted |= first_contact_now
        self._classify_first_contact(first_contact_now)

        episode_time = self.episode_length_buf.float() * self.dt
        self.ball_first_contact_time[first_contact_now] = episode_time[first_contact_now]
        self.time_since_first_contact[:] = torch.where(
            self.ball_has_been_contacted,
            torch.clamp(episode_time - self.ball_first_contact_time, min=0.0),
            torch.zeros_like(episode_time),
        )

    def _update_capture_state(self):
        ball_rel = self.ball_pos_local[:, 0:2]
        capture_err = ((ball_rel - self.capture_center) / self.capture_sigma).square().sum(dim=-1)
        self.capture_zone_score[:] = torch.exp(-capture_err)
        self.capture_zone_error[:] = torch.norm(ball_rel - self.capture_center, dim=-1)

        slow_ball = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1) < float(self.cfg["receive_geometry"]["capture_speed_threshold"])
        in_front = ball_rel[:, 0] > float(self.cfg["receive_geometry"]["capture_front_min"])
        in_width = torch.abs(ball_rel[:, 1]) < float(self.cfg["receive_geometry"]["capture_lateral_max"])
        self.capture_success[:] = slow_ball & in_front & in_width & self.ball_has_been_contacted
        self.capture_success_latched |= self.capture_success
        self.controlled_receive_success[:] = (
            self.capture_success
            & (~self.tunnel_entry_latched)
            & (~self.through_legs_latched)
            & self.good_receive_contact_latched
            & (~self.orbit_behind_latched)
        )
        self.controlled_receive_success_latched |= self.controlled_receive_success

    def _update_receive_state(self):
        self.ball_pos_local[:] = quat_rotate_inverse(self.base_quat, self.ball_pos - self.base_pos)
        self.ball_vel_local[:] = quat_rotate_inverse(self.base_quat, self.ball_lin_vel)
        self._estimate_intercept_from_history()
        self._compute_local_foot_geometry()
        self._update_pass_frame_state()
        self._update_locomotion_targets()
        self._update_contact_state()
        self._compute_stance_tunnel()
        self._update_capture_state()

        self.ball_has_passed_robot[:] = self.ball_pos_local[:, 0] < -float(self.cfg["rewards"]["clear_miss_x_margin"])
        clear_miss_now = self.ball_has_passed_robot & (~self.ball_has_been_contacted)
        self.clear_miss_time_buf[:] = torch.where(
            clear_miss_now,
            self.clear_miss_time_buf + self.dt,
            torch.zeros_like(self.clear_miss_time_buf),
        )
        self.no_step_failure_latched |= (
            self.step_required_latched
            & (~self.step_event_latched)
            & (clear_miss_now | self.late_chase_latched | self.ball_first_contact_event)
        )

        phase = torch.zeros_like(self.intercept_phase)
        intercept_window = float(self.cfg["rewards"]["intercept_window_s"])
        phase[(self.arrival_confidence >= float(self.cfg["estimator"]["low_confidence_threshold"])) & (~self.ball_has_been_contacted)] = 1
        phase[(self.intercept_time_estimate <= intercept_window) & (~self.ball_has_been_contacted)] = 2
        phase[self.ball_has_been_contacted] = 3
        phase[self.ball_has_passed_robot & (~self.ball_has_been_contacted)] = 0
        self.intercept_phase[:] = phase
        self.intercept_phase_onehot.zero_()
        self.intercept_phase_onehot.scatter_(1, self.intercept_phase.unsqueeze(-1).clamp(max=3), 1.0)

    def step(self, actions):
        self.actions[:] = torch.clamp(actions, -float(self.cfg["normalization"]["clip_actions"]), float(self.cfg["normalization"]["clip_actions"]))
        self.gait_frequency_offset[:] = torch.clamp(
            self.actions[:, self.num_dofs],
            -float(self.cfg["commands"]["gait_frequency_offset_clip"]),
            float(self.cfg["commands"]["gait_frequency_offset_clip"]),
        )
        self.gait_frequency[:] = self.gait_frequency_offset + float(self.cfg["commands"]["gait_frequency_base"])
        joint_actions = self.actions[:, : self.num_dofs]
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * joint_actions

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

        self.ball_pos[:] = self.root_states[:, 1, 0:3]
        self.ball_lin_vel[:] = self.body_states[:, -1, 7:10]
        self.ball_ang_vel[:] = self.body_states[:, -1, 10:13]

        self.base_pos[:] = self.root_states[:, 0, 0:3]
        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        filter_weight = float(self.cfg["normalization"]["filter_weight"])
        self.filtered_lin_vel[:] = self.base_lin_vel * filter_weight + self.filtered_lin_vel * (1.0 - filter_weight)
        self.filtered_ang_vel[:] = self.base_ang_vel * filter_weight + self.filtered_ang_vel * (1.0 - filter_weight)
        dof_vel_alpha = float(self.cfg["normalization"]["dof_vel_filter_alpha"])
        self.dof_vel_filtered[:] = dof_vel_alpha * self.dof_vel + (1.0 - dof_vel_alpha) * self.dof_vel_filtered

        self._refresh_feet_state()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        self._kick_robots()
        self._push_robots()
        self._update_ball_detection()
        self._update_receive_state()
        self._check_termination()

        self._compute_reward()
        self.last_ball_lin_vel_world[:] = self.body_states[:, -1, 7:10]

        done_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.emit_metrics:
            self.metric_contact.zero_()
            self.metric_control.zero_()
            self.metric_tunnel.zero_()
            self.metric_through.zero_()
            self.metric_capture.zero_()
            self.metric_left.zero_()
            self.metric_conf.zero_()
            self.metric_curriculum.zero_()
            self.metric_turn_behind.zero_()
            self.metric_late_chase.zero_()
            self.metric_step_required.zero_()
            self.metric_true_step.zero_()
            self.metric_no_step.zero_()
            self.metric_skate_dist.zero_()
            self.metric_wrong_surface.zero_()
            self.metric_side_switch.zero_()
            self.metric_pass_progress.zero_()
            self.metric_contact[done_ids] = self.ball_has_been_contacted[done_ids].float()
            self.metric_control[done_ids] = self.controlled_receive_success_latched[done_ids].float()
            self.metric_tunnel[done_ids] = self.tunnel_entry_latched[done_ids].float()
            self.metric_through[done_ids] = self.through_legs_latched[done_ids].float()
            self.metric_capture[done_ids] = self.capture_success_latched[done_ids].float()
            self.metric_left[done_ids] = (self.receive_side[done_ids] == 0).float()
            self.metric_conf[done_ids] = self.arrival_confidence[done_ids]
            self.metric_curriculum[done_ids] = float(self.curriculum_global_level)
            self.metric_turn_behind[done_ids] = (self.orbit_behind_latched[done_ids] | self.behind_ball_latched[done_ids]).float()
            self.metric_late_chase[done_ids] = self.late_chase_latched[done_ids].float()
            self.metric_step_required[done_ids] = self.step_required_latched[done_ids].float()
            self.metric_true_step[done_ids] = self.true_step_latched[done_ids].float()
            self.metric_no_step[done_ids] = self.no_step_failure_latched[done_ids].float()
            self.metric_skate_dist[done_ids] = self.skate_distance_precontact[done_ids]
            self.metric_wrong_surface[done_ids] = self.wrong_surface_contact_latched[done_ids].float()
            self.metric_side_switch[done_ids] = self.receive_side_switch_count[done_ids]
            self.metric_pass_progress[done_ids] = self.pass_progress_before_contact[done_ids]

        if len(done_ids) > 0:
            self._reset_idx(done_ids)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self._refresh_feet_state()
            self._insert_ball_detections(done_ids, reset_timer=False)
            self.last_ball_lin_vel_world[done_ids] = 0.0

        self._teleport_robot()
        if len(done_ids) > 0 or self.terrain.type != "plane":
            self._update_receive_state()
        self._compute_observations()
        self._draw_debug_visuals()
        self._overlay_debug_on_latest_frame()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos
        self.prev_base_pos_xy[:] = self.base_pos[:, 0:2]
        if self.emit_metrics:
            self.extras["metrics"] = {
                "receive_contact_success_terminal": self.metric_contact,
                "controlled_receive_success_terminal": self.metric_control,
                "tunnel_entry_terminal": self.metric_tunnel,
                "through_legs_terminal": self.metric_through,
                "capture_success_terminal": self.metric_capture,
                "receive_left_terminal": self.metric_left,
                "arrival_confidence_terminal": self.metric_conf,
                "curriculum_level_terminal": self.metric_curriculum,
                "turn_behind_terminal": self.metric_turn_behind,
                "late_chase_terminal": self.metric_late_chase,
                "step_required_terminal": self.metric_step_required,
                "true_step_terminal": self.metric_true_step,
                "no_step_failure_terminal": self.metric_no_step,
                "skate_distance_precontact_terminal": self.metric_skate_dist,
                "wrong_surface_first_contact_terminal": self.metric_wrong_surface,
                "chosen_side_switch_count_terminal": self.metric_side_switch,
                "pass_progress_before_contact_terminal": self.metric_pass_progress,
            }
        else:
            self.extras["metrics"] = {}

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _check_termination(self):
        contact_terminate = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        lin_vel_sq = self.root_states[:, 0, 7:10].square().sum(dim=-1)
        ang_vel_sq = self.root_states[:, 0, 10:13].square().sum(dim=-1)
        lin_vel_terminate = lin_vel_sq > float(self.cfg["rewards"]["terminate_lin_vel"]) ** 2
        ang_vel_terminate = ang_vel_sq > float(self.cfg["rewards"]["terminate_ang_vel"]) ** 2
        base_height_above_ground = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        height_terminate = base_height_above_ground < float(self.cfg["rewards"]["terminate_height"])
        timeout_terminate = self.episode_length_buf > np.ceil(float(self.cfg["rewards"]["episode_length_s"]) / self.dt)
        clear_miss_terminate = self.clear_miss_time_buf >= float(self.cfg["rewards"]["clear_miss_termination_s"])
        late_chase_terminate = self.late_chase_latched & (~self.ball_has_been_contacted)
        orbit_terminate = self.orbit_behind_latched & (~self.ball_has_been_contacted)
        through_legs_terminate = self.through_legs_latched
        success_terminate = self.controlled_receive_success_latched

        self.reset_buf = (
            contact_terminate
            | lin_vel_terminate
            | ang_vel_terminate
            | height_terminate
            | timeout_terminate
            | clear_miss_terminate
            | late_chase_terminate
            | orbit_terminate
            | through_legs_terminate
            | success_terminate
        )
        self.time_out_buf = timeout_terminate

        if self.debug_termination:
            env_id = self.debug_termination_env_id
            if bool(self.reset_buf[env_id].item()):
                contact_force_norm = torch.norm(self.contact_forces[env_id, self.termination_contact_indices, :], dim=-1)
                max_contact_force = float(contact_force_norm.max().item()) if contact_force_norm.numel() > 0 else 0.0
                lin_vel = float(torch.sqrt(lin_vel_sq[env_id]).item())
                ang_vel = float(torch.sqrt(ang_vel_sq[env_id]).item())
                base_height = float(base_height_above_ground[env_id].item())
                heading_error = float(torch.abs(self.desired_heading_error[env_id]).item())
                ball_dist_local = float(torch.norm(self.ball_pos_local[env_id, 0:2]).item())
                episode_time = float(self.episode_length_buf[env_id].item()) * self.dt
                reasons = []
                if bool(contact_terminate[env_id].item()):
                    reasons.append("contact")
                if bool(lin_vel_terminate[env_id].item()):
                    reasons.append("lin_vel")
                if bool(ang_vel_terminate[env_id].item()):
                    reasons.append("ang_vel")
                if bool(height_terminate[env_id].item()):
                    reasons.append("height")
                if bool(timeout_terminate[env_id].item()):
                    reasons.append("timeout")
                if bool(clear_miss_terminate[env_id].item()):
                    reasons.append("clear_miss")
                if bool(late_chase_terminate[env_id].item()):
                    reasons.append("late_chase")
                if bool(orbit_terminate[env_id].item()):
                    reasons.append("orbit")
                if bool(through_legs_terminate[env_id].item()):
                    reasons.append("through_legs")
                if bool(success_terminate[env_id].item()):
                    reasons.append("success")

                print(
                    "[pass_receive termination env] "
                    f"step={self.common_step_counter} "
                    f"env={env_id} "
                    f"reasons={','.join(reasons) if reasons else 'unknown'} "
                    f"episode_time={episode_time:.3f}s "
                    f"ball_progress={float(self.ball_progress_ratio[env_id].item()):.3f} "
                    f"robot_progress={float(self.robot_progress_ratio[env_id].item()):.3f} "
                    f"progress_gap={float((self.ball_progress_ratio[env_id] - self.robot_progress_ratio[env_id]).item()):.3f} "
                    f"ball_x_local={float(self.ball_pos_local[env_id, 0].item()):.3f} "
                    f"heading_err={heading_error:.3f}/{float(self.cfg['turn_guard']['orbit_heading_threshold']):.3f} "
                    f"ball_dist={ball_dist_local:.3f}/{float(self.cfg['turn_guard']['orbit_ball_distance']):.3f} "
                    f"lin_vel={lin_vel:.3f}/{float(self.cfg['rewards']['terminate_lin_vel']):.3f} "
                    f"ang_vel={ang_vel:.3f}/{float(self.cfg['rewards']['terminate_ang_vel']):.3f} "
                    f"base_height={base_height:.3f}/{float(self.cfg['rewards']['terminate_height']):.3f} "
                    f"clear_miss={float(self.clear_miss_time_buf[env_id].item()):.3f}/{float(self.cfg['rewards']['clear_miss_termination_s']):.3f} "
                    f"max_contact_force={max_contact_force:.3f} "
                    f"ball_contacted={int(self.ball_has_been_contacted[env_id].item())} "
                    f"late_chase={int(self.late_chase_latched[env_id].item())} "
                    f"orbit={int(self.orbit_behind_latched[env_id].item())} "
                    f"through={int(self.through_legs_latched[env_id].item())} "
                    f"success={int(self.controlled_receive_success_latched[env_id].item())}"
                )

        if self.debug_termination and (self.common_step_counter % self.debug_termination_interval == 0):
            reset_count = int(self.reset_buf.sum().item())
            if reset_count > 0:
                reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()[: self.debug_termination_max_envs].tolist()
                print(
                    "[pass_receive termination] "
                    f"step={self.common_step_counter} "
                    f"reset={reset_count} "
                    f"contact={int(contact_terminate.sum().item())} "
                    f"lin={int(lin_vel_terminate.sum().item())} "
                    f"ang={int(ang_vel_terminate.sum().item())} "
                    f"height={int(height_terminate.sum().item())} "
                    f"timeout={int(timeout_terminate.sum().item())} "
                    f"miss={int(clear_miss_terminate.sum().item())} "
                    f"late={int(late_chase_terminate.sum().item())} "
                    f"orbit={int(orbit_terminate.sum().item())} "
                    f"through={int(through_legs_terminate.sum().item())} "
                    f"success={int(success_terminate.sum().item())} "
                    f"sample={reset_ids}"
                )

    def _kick_robots(self):
        interval_s = float(self.cfg["randomization"].get("kick_interval_s", 0.0))
        if interval_s <= 0.0:
            return
        interval_steps = max(1, int(np.ceil(interval_s / self.dt)))
        if self.common_step_counter % interval_steps != 0:
            return
        self.root_states[:, 0, 7:10] = apply_randomization(
            self.root_states[:, 0, 7:10], self.cfg["randomization"].get("kick_lin_vel")
        )
        self.root_states[:, 0, 10:13] = apply_randomization(
            self.root_states[:, 0, 10:13], self.cfg["randomization"].get("kick_ang_vel")
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        interval_s = float(self.cfg["randomization"].get("push_interval_s", 0.0))
        if interval_s <= 0.0:
            return
        interval_steps = max(1, int(np.ceil(interval_s / self.dt)))
        duration_steps = max(1, int(np.ceil(float(self.cfg["randomization"].get("push_duration_s", 0.0)) / self.dt)))

        if self.common_step_counter % interval_steps == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, self.base_indice, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, self.base_indice, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % interval_steps == duration_steps:
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _compute_reward(self):
        self.rew_buf[:] = 0.0
        self.extras["rew_terms"] = {}
        for i, reward_fn in enumerate(self.reward_functions):
            name = self.reward_names[i]
            rew = reward_fn() * self.reward_scales[name]
            self.rew_buf += rew
            if self.emit_reward_terms:
                self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"].get("only_positive_rewards", False):
            self.rew_buf[:] = torch.clamp(self.rew_buf, min=0.0)

    def _compute_observations(self):
        history_age_clip = float(self.cfg["ball"].get("detection_age_clip_s", 0.25))
        hist_age = torch.clamp(self.ball_detection_history_age, 0.0, history_age_clip)
        history_obs = torch.cat(
            (
                self.ball_detection_history_xy * self.cfg["normalization"]["ball_pos"],
                hist_age.unsqueeze(-1) * self.cfg["normalization"]["ball_detection_age"],
                self.ball_detection_history_valid.unsqueeze(-1),
            ),
            dim=-1,
        ).reshape(self.num_envs, -1)

        estimator_obs = torch.cat(
            (
                self.intercept_point_local * self.cfg["normalization"]["ball_pos"],
                self.intercept_time_estimate.unsqueeze(-1) * self.cfg["normalization"]["intercept_time"],
                self.ball_line_dir_local,
                self.arrival_confidence.unsqueeze(-1),
            ),
            dim=-1,
        )
        locomotion_obs = torch.cat(
            (
                self.target_lin_vel_local * self.cfg["normalization"]["lin_vel"],
                self.target_ang_vel_yaw.unsqueeze(-1) * self.cfg["normalization"]["ang_vel"],
                self.target_gait_frequency.unsqueeze(-1) * self.cfg["normalization"]["gait_frequency"],
                self.locomotion_drive.unsqueeze(-1),
                self.feet_contact.float(),
                torch.sin(self.desired_heading_error).unsqueeze(-1),
                torch.cos(self.desired_heading_error).unsqueeze(-1),
                self.ball_progress_ratio.unsqueeze(-1),
                (self.robot_lateral_error_to_pass * self.cfg["normalization"]["ball_pos"]).unsqueeze(-1),
                self.receive_side_locked.float().unsqueeze(-1),
                self.receive_side_onehot,
            ),
            dim=-1,
        )

        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                torch.cos(2.0 * torch.pi * self.gait_process).unsqueeze(-1),
                torch.sin(2.0 * torch.pi * self.gait_process).unsqueeze(-1),
                (self.intercept_phase.float() / 3.0).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel_filtered, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.last_actions,
                history_obs,
                estimator_obs,
                locomotion_obs,
            ),
            dim=-1,
        )
        privileged_pass_obs = torch.cat(
            (
                self.pass_ref_dir_local,
                torch.sin(self.desired_heading_error).unsqueeze(-1),
                torch.cos(self.desired_heading_error).unsqueeze(-1),
                self.ball_progress_ratio.unsqueeze(-1),
                self.robot_progress_ratio.unsqueeze(-1),
                (self.robot_lateral_error_to_pass * self.cfg["normalization"]["ball_pos"]).unsqueeze(-1),
                self.behind_ball.float().unsqueeze(-1),
                self.late_chase.float().unsqueeze(-1),
                self.orbit_behind.float().unsqueeze(-1),
                self.target_lin_vel_local * self.cfg["normalization"]["lin_vel"],
                self.target_ang_vel_yaw.unsqueeze(-1) * self.cfg["normalization"]["ang_vel"],
                self.target_gait_frequency.unsqueeze(-1) * self.cfg["normalization"]["gait_frequency"],
                self.skate_distance_precontact.unsqueeze(-1),
                self.contact_type_onehot,
                self.wrong_surface_contact_latched.float().unsqueeze(-1),
                self.step_required_latched.float().unsqueeze(-1),
                self.true_step_latched.float().unsqueeze(-1),
            ),
            dim=-1,
        )

        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                self.ball_pos_local,
                self.ball_vel_local,
                self.ball_ang_vel,
                self.feet_pos_local.reshape(self.num_envs, -1),
                self.feet_yaw_rel,
                self.chosen_foot_pos_local,
                self.stance_gap.unsqueeze(-1),
                self.stance_gap_target.unsqueeze(-1),
                self.tunnel_risk.unsqueeze(-1),
                self.receive_side_onehot,
                self.ball_has_been_contacted.float().unsqueeze(-1),
                privileged_pass_obs,
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    def _upright_posture_term(self):
        roll, pitch, _ = get_euler_xyz(self.base_quat)
        roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
        pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
        sigma = float(self.cfg["rewards"].get("upright_sigma", 0.10))
        return torch.exp(-(roll.square() + pitch.square()) / sigma)

    def _get_swing_masks(self):
        swing_half_period = 0.5 * float(self.cfg["rewards"]["swing_period"])
        gait_active = self.gait_frequency > 1.0e-8
        left_swing = (torch.abs(self.gait_process - 0.25) < swing_half_period) & gait_active
        right_swing = (torch.abs(self.gait_process - 0.75) < swing_half_period) & gait_active
        return left_swing, right_swing

    def _body_frame_feet_distance(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        return torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )

    def _foot_clearance(self):
        flat_feet_pos = self.feet_pos.reshape(-1, 3)
        ground_height = self.terrain.terrain_heights(flat_feet_pos).reshape(self.num_envs, len(self.feet_indices))
        return self.feet_pos[:, :, 2] - ground_height

    def _reward_survival(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_stability(self):
        upright = self._upright_posture_term()
        lin_sigma = float(self.cfg["rewards"]["stability_lin_sigma"])
        ang_sigma = float(self.cfg["rewards"]["stability_ang_sigma"])
        lin_term = torch.exp(-torch.sum(self.filtered_lin_vel[:, 0:2].square(), dim=-1) / lin_sigma)
        ang_term = torch.exp(-torch.sum(self.filtered_ang_vel[:, 0:2].square(), dim=-1) / ang_sigma)
        return upright * lin_term * ang_term

    def _reward_action_smoothness(self):
        return torch.sum((self.actions - self.last_actions).square(), dim=-1)

    def _reward_intercept_pos(self):
        sigma = float(self.cfg["rewards"]["intercept_pos_sigma"])
        pre_contact = (~self.ball_has_been_contacted).float()
        return torch.exp(-self.intercept_pose_error.square() / sigma) * self.arrival_confidence * pre_contact

    def _reward_intercept_time(self):
        sigma = float(self.cfg["rewards"]["intercept_time_sigma"])
        pre_contact = (~self.ball_has_been_contacted).float()
        return torch.exp(-self.arrival_time_error.square() / sigma) * self.arrival_confidence * pre_contact

    def _reward_heading(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        sigma = max(float(self.cfg["turn_guard"]["heading_reward_sigma"]), 1.0e-6)
        return torch.exp(-self.desired_heading_error.square() / sigma) * self.arrival_confidence * pre_contact

    def _reward_side_foot_yaw(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        return self.side_foot_alignment * self.arrival_confidence * pre_contact

    def _reward_stance_narrow_when_needed(self):
        sigma = float(self.cfg["rewards"]["stance_gap_sigma"])
        center_gate = torch.exp(
            -torch.square(self.ball_lateral_error_to_pass) / max(float(self.cfg["rewards"]["centerline_sigma"]), 1.0e-6)
        )
        gap_err = self.stance_gap - self.stance_gap_target
        pre_contact = (~self.ball_has_been_contacted).float()
        return torch.exp(-gap_err.square() / sigma) * center_gate * pre_contact

    def _reward_no_tunnel(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        return (1.0 - self.tunnel_risk) * pre_contact

    def _reward_contact_quality(self):
        speed_sigma = float(self.cfg["rewards"]["contact_quality_speed_sigma"])
        stability = self._reward_stability()
        post_speed = torch.norm(self.ball_lin_vel[:, 0:2], dim=-1)
        return (
            self.ball_first_contact_event.float()
            * torch.exp(-post_speed.square() / speed_sigma)
            * self.good_receive_contact.float()
            * stability
        )

    def _reward_capture_zone(self):
        post_contact = self.ball_has_been_contacted.float()
        slow_gate = torch.exp(
            -torch.norm(self.ball_lin_vel[:, 0:2], dim=-1) / max(float(self.cfg["rewards"]["capture_speed_sigma"]), 1.0e-6)
        )
        return self.capture_zone_score * slow_gate * post_contact

    def _reward_through_legs_penalty(self):
        return self.through_legs_latched.float() + self.tunnel_entry_event.float()

    def _reward_clear_miss_penalty(self):
        return self.ball_has_passed_robot.float() * (~self.ball_has_been_contacted).float()

    def _reward_tracking_lin_vel_x(self):
        sigma = max(float(self.cfg["rewards"]["tracking_sigma"]), 1.0e-6)
        return (
            torch.exp(-torch.square(self.target_lin_vel_local[:, 0] - self.filtered_lin_vel[:, 0]) / sigma)
            * self.locomotion_drive
            * (~self.ball_has_been_contacted).float()
        )

    def _reward_tracking_lin_vel_y(self):
        sigma = max(float(self.cfg["rewards"]["tracking_sigma"]), 1.0e-6)
        return (
            torch.exp(-torch.square(self.target_lin_vel_local[:, 1] - self.filtered_lin_vel[:, 1]) / sigma)
            * self.locomotion_drive
            * (~self.ball_has_been_contacted).float()
        )

    def _reward_tracking_ang_vel(self):
        sigma = max(float(self.cfg["rewards"]["tracking_sigma"]), 1.0e-6)
        return (
            torch.exp(-torch.square(self.target_ang_vel_yaw - self.filtered_ang_vel[:, 2]) / sigma)
            * self.locomotion_drive
            * (~self.ball_has_been_contacted).float()
        )

    def _reward_gait_frequency_tracking(self):
        sigma = max(float(self.cfg["rewards"]["gait_frequency_sigma"]), 1.0e-6)
        return (
            torch.exp(-torch.square(self.target_gait_frequency - self.gait_frequency) / sigma)
            * self.locomotion_drive
            * (~self.ball_has_been_contacted).float()
        )

    def _reward_base_height(self):
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - float(self.cfg["rewards"]["base_height_target"]))

    def _reward_orientation(self):
        roll, pitch, _ = get_euler_xyz(self.base_quat)
        roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
        pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.square(roll) + torch.square(pitch)

    def _reward_feet_slip(self):
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
            * (~self.ball_has_been_contacted).float()
        )

    def _reward_feet_distance(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        feet_distance = self._body_frame_feet_distance()
        return torch.clamp(feet_distance - float(self.cfg["rewards"]["feet_distance_ref"]), min=0.0, max=0.12) * pre_contact

    def _reward_feet_swing(self):
        left_swing, right_swing = self._get_swing_masks()
        return (
            (left_swing & ~self.feet_contact[:, 0]).float()
            + (right_swing & ~self.feet_contact[:, 1]).float()
        ) * self.locomotion_drive * (~self.ball_has_been_contacted).float()

    def _reward_swing_clearance(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        left_swing, right_swing = self._get_swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        airborne_mask = (~self.feet_contact).float()
        clearance = self._foot_clearance()
        clearance_target = float(self.cfg["rewards"]["swing_clearance_target"])
        clearance_sigma = max(float(self.cfg["rewards"]["swing_clearance_sigma"]), 1.0e-6)
        clearance_error = torch.clamp(clearance_target - clearance, min=0.0)
        clearance_reward = torch.exp(-clearance_error.square() / clearance_sigma) * swing_mask * airborne_mask
        swing_count = swing_mask.sum(dim=-1).clamp_min(1.0)
        return clearance_reward.sum(dim=-1) / swing_count * self.step_required.float() * self.locomotion_drive * pre_contact

    def _reward_step_lateral_progress(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        left_swing, right_swing = self._get_swing_masks()
        chosen_swing = (
            ((self.receive_side == 0) & left_swing)
            | ((self.receive_side == 1) & right_swing)
        ).float()
        side_sign = torch.where(self.receive_side == 0, 1.0, -1.0)
        target_lateral = side_sign * float(self.cfg["rewards"]["step_lateral_target"])
        sigma = max(float(self.cfg["rewards"]["step_lateral_target"]) ** 2, 1.0e-6)
        lateral_error = self.chosen_foot_pos_local[:, 1] - target_lateral
        return (
            torch.exp(-lateral_error.square() / sigma)
            * chosen_swing
            * self.step_required.float()
            * self.locomotion_drive
            * pre_contact
        )

    def _reward_skate_penalty(self):
        return self.skating_indicator

    def _reward_double_support_shuffle_penalty(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        base_delta_xy = self.base_pos[:, 0:2] - self.prev_base_pos_xy
        base_speed_xy = torch.norm(base_delta_xy, dim=-1) / max(self.dt, 1.0e-6)
        return (
            base_speed_xy
            * self.feet_contact.all(dim=-1).float()
            * self.step_required.float()
            * self.locomotion_drive
            * pre_contact
        )

    def _reward_stance_width_hard_penalty(self):
        pre_contact = (~self.ball_has_been_contacted).float()
        excess_width = torch.clamp(
            self._body_frame_feet_distance() - float(self.cfg["rewards"]["max_keeper_stance_width"]),
            min=0.0,
        )
        return excess_width * pre_contact

    def _reward_turn_guard_penalty(self):
        hard_turn = torch.clamp(
            torch.abs(self.desired_heading_error) - float(self.cfg["turn_guard"]["hard_heading_error"]),
            min=0.0,
        ) / torch.pi
        pre_contact = (~self.ball_has_been_contacted).float()
        return pre_contact * (
            hard_turn
            + self.behind_ball.float()
            + self.late_chase.float()
            + self.orbit_behind.float()
        )

    def _reward_pass_progress_penalty(self):
        max_dist = max(float(self.cfg["turn_guard"]["progress_penalty_distance"]), 1.0e-6)
        normalized_now = torch.clamp(self.ball_max_progress_along_pass / max_dist, 0.0, 1.0)
        normalized_prev = torch.clamp(self.prev_ball_max_progress_along_pass / max_dist, 0.0, 1.0)
        delta_progress = torch.clamp(normalized_now - normalized_prev, min=0.0)
        return delta_progress * (~self.ball_has_been_contacted).float()

    def _reward_wrong_surface_penalty(self):
        return self.wrong_surface_contact_event.float() + 0.25 * (
            self.wrong_surface_contact_latched.float() * self.ball_has_been_contacted.float()
        )

    def _draw_debug_visuals(self):
        if not self.debug_draw_enabled or self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)
        env_count = min(self.num_envs, int(self.cfg.get("viewer", {}).get("debug_env_count", 1)))
        for env_idx in range(env_count):
            env_handle = self.envs[env_idx]
            base = self.base_pos[env_idx].cpu().numpy()
            base_z = max(float(base[2]), 0.25)

            ball_local = self.ball_pos_local[env_idx].cpu()
            intercept_local = self.intercept_point_local[env_idx].cpu()
            receive_local = self.chosen_receive_point_local[env_idx].cpu()
            pass_dir_local = self.pass_ref_dir_local[env_idx].cpu()
            desired_heading_local = self.desired_heading_vec_local[env_idx].cpu()
            tunnel_x_min = float(self.tunnel_x_min[env_idx].item())
            tunnel_x_max = float(self.tunnel_x_max[env_idx].item())
            left_y = float(self.left_tunnel_y[env_idx].item())
            right_y = float(self.right_tunnel_y[env_idx].item())

            def local_to_world(local_x, local_y, z_offset):
                local = torch.tensor([[local_x, local_y, 0.0]], device=self.device)
                world = self.base_pos[env_idx : env_idx + 1] + quat_rotate(self.base_quat[env_idx : env_idx + 1], local)
                out = world[0].cpu().numpy()
                out[2] = base_z + z_offset
                return out

            ball_world = local_to_world(float(ball_local[0].item()), float(ball_local[1].item()), 0.06)
            intercept_world = local_to_world(float(intercept_local[0].item()), float(intercept_local[1].item()), 0.08)
            receive_world = local_to_world(float(receive_local[0].item()), float(receive_local[1].item()), 0.08)

            ball_line = np.array(
                [ball_world[0], ball_world[1], ball_world[2], intercept_world[0], intercept_world[1], intercept_world[2]],
                dtype=np.float32,
            )
            self.gym.add_lines(self.viewer, env_handle, 1, ball_line, np.array([1.0, 0.9, 0.0], dtype=np.float32))

            receive_line = np.array(
                [intercept_world[0], intercept_world[1], intercept_world[2], receive_world[0], receive_world[1], receive_world[2]],
                dtype=np.float32,
            )
            self.gym.add_lines(self.viewer, env_handle, 1, receive_line, np.array([0.0, 1.0, 1.0], dtype=np.float32))

            pass_line_end = local_to_world(float(pass_dir_local[0].item()) * 0.4, float(pass_dir_local[1].item()) * 0.4, 0.10)
            pass_line = np.array(
                [base[0], base[1], base_z + 0.10, pass_line_end[0], pass_line_end[1], pass_line_end[2]],
                dtype=np.float32,
            )
            self.gym.add_lines(self.viewer, env_handle, 1, pass_line, np.array([0.9, 0.5, 0.1], dtype=np.float32))

            heading_end = local_to_world(float(desired_heading_local[0].item()) * 0.35, float(desired_heading_local[1].item()) * 0.35, 0.14)
            heading_line = np.array(
                [base[0], base[1], base_z + 0.14, heading_end[0], heading_end[1], heading_end[2]],
                dtype=np.float32,
            )
            self.gym.add_lines(self.viewer, env_handle, 1, heading_line, np.array([0.2, 0.9, 0.2], dtype=np.float32))

            f_tol = float(self.cfg["contact_classifier"]["inner_side_forward_tolerance"])
            s_tol = float(self.cfg["contact_classifier"]["inner_side_lateral_tolerance"])
            zone_center = receive_local.numpy()
            zone_forward = self.chosen_foot_forward[env_idx].cpu().numpy()
            zone_inner = self.chosen_foot_inner_normal[env_idx].cpu().numpy()
            corner0 = zone_center + zone_forward * f_tol + zone_inner * s_tol
            corner1 = zone_center + zone_forward * f_tol - zone_inner * s_tol
            corner2 = zone_center - zone_forward * f_tol - zone_inner * s_tol
            corner3 = zone_center - zone_forward * f_tol + zone_inner * s_tol
            rect_pts = [corner0, corner1, corner2, corner3, corner0]
            rect_world = [local_to_world(float(p[0]), float(p[1]), 0.05) for p in rect_pts]
            verts = []
            for p0, p1 in zip(rect_world[:-1], rect_world[1:]):
                verts.extend([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]])
            self.gym.add_lines(
                self.viewer,
                env_handle,
                4,
                np.array(verts, dtype=np.float32),
                np.array([0.2, 1.0, 0.2] * 4, dtype=np.float32),
            )

            if tunnel_x_max > tunnel_x_min:
                left_start = local_to_world(tunnel_x_min, left_y, 0.03)
                left_end = local_to_world(tunnel_x_max, left_y, 0.03)
                right_start = local_to_world(tunnel_x_min, right_y, 0.03)
                right_end = local_to_world(tunnel_x_max, right_y, 0.03)
                verts = np.array(
                    [
                        left_start[0], left_start[1], left_start[2], left_end[0], left_end[1], left_end[2],
                        right_start[0], right_start[1], right_start[2], right_end[0], right_end[1], right_end[2],
                    ],
                    dtype=np.float32,
                )
                colors = np.array([1.0, 1.0, 1.0, 1.0, 0.4, 0.4], dtype=np.float32)
                self.gym.add_lines(self.viewer, env_handle, 2, verts, colors)

    def _overlay_debug_on_latest_frame(self):
        if not self.debug_video_overlay or not hasattr(self, "camera_frames") or len(self.camera_frames) == 0:
            return

        frame = self.camera_frames[-1]
        if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return

        panel = min(self.debug_video_panel_size, frame.shape[0], frame.shape[1])
        if panel < 64:
            return

        frame[:panel, :panel, 0:3] = (0.15 * frame[:panel, :panel, 0:3]).astype(frame.dtype)
        frame[:panel, :panel, 3] = 255

        env_idx = int(self.cfg["viewer"].get("record_env_idx", 0))
        scale = self.debug_video_scale
        center_x = panel // 2
        center_y = int(panel * 0.78)

        def draw_circle(x, y, radius, color):
            yy, xx = np.ogrid[:panel, :panel]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            frame[:panel, :panel, 0][mask] = color[0]
            frame[:panel, :panel, 1][mask] = color[1]
            frame[:panel, :panel, 2][mask] = color[2]
            frame[:panel, :panel, 3][mask] = 255

        def draw_line(x0, y0, x1, y1, color, thickness=1):
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            xs = np.linspace(x0, x1, steps + 1).astype(np.int32)
            ys = np.linspace(y0, y1, steps + 1).astype(np.int32)
            for x, y in zip(xs, ys):
                x0c = max(0, x - thickness)
                x1c = min(panel, x + thickness + 1)
                y0c = max(0, y - thickness)
                y1c = min(panel, y + thickness + 1)
                frame[y0c:y1c, x0c:x1c, 0] = color[0]
                frame[y0c:y1c, x0c:x1c, 1] = color[1]
                frame[y0c:y1c, x0c:x1c, 2] = color[2]
                frame[y0c:y1c, x0c:x1c, 3] = 255

        def local_to_panel(local_x, local_y):
            px = int(np.clip(center_x - local_y * scale, 0, panel - 1))
            py = int(np.clip(center_y - local_x * scale, 0, panel - 1))
            return px, py

        robot_px = local_to_panel(0.0, 0.0)
        ball_px = local_to_panel(float(self.ball_pos_local[env_idx, 0].item()), float(self.ball_pos_local[env_idx, 1].item()))
        intercept_px = local_to_panel(float(self.intercept_point_local[env_idx, 0].item()), float(self.intercept_point_local[env_idx, 1].item()))
        receive_px = local_to_panel(float(self.chosen_receive_point_local[env_idx, 0].item()), float(self.chosen_receive_point_local[env_idx, 1].item()))
        foot_px = local_to_panel(float(self.chosen_foot_pos_local[env_idx, 0].item()), float(self.chosen_foot_pos_local[env_idx, 1].item()))

        draw_circle(*robot_px, 5, (220, 80, 80))
        draw_circle(*ball_px, 4, (40, 40, 220))
        draw_circle(*intercept_px, 4, (40, 220, 220))
        draw_circle(*receive_px, 4, (40, 220, 40))
        draw_circle(*foot_px, 4, (220, 40, 220))
        draw_line(ball_px[0], ball_px[1], intercept_px[0], intercept_px[1], (60, 200, 240), thickness=1)
        draw_line(intercept_px[0], intercept_px[1], receive_px[0], receive_px[1], (60, 240, 60), thickness=1)

        pass_dir_end = local_to_panel(
            float(self.pass_ref_dir_local[env_idx, 0].item()) * 0.35,
            float(self.pass_ref_dir_local[env_idx, 1].item()) * 0.35,
        )
        desired_heading_end = local_to_panel(
            float(self.desired_heading_vec_local[env_idx, 0].item()) * 0.30,
            float(self.desired_heading_vec_local[env_idx, 1].item()) * 0.30,
        )
        draw_line(robot_px[0], robot_px[1], pass_dir_end[0], pass_dir_end[1], (230, 150, 40), thickness=1)
        draw_line(robot_px[0], robot_px[1], desired_heading_end[0], desired_heading_end[1], (40, 230, 40), thickness=1)

        tunnel_x_min = float(self.tunnel_x_min[env_idx].item())
        tunnel_x_max = float(self.tunnel_x_max[env_idx].item())
        if tunnel_x_max > tunnel_x_min:
            left0 = local_to_panel(tunnel_x_min, float(self.left_tunnel_y[env_idx].item()))
            left1 = local_to_panel(tunnel_x_max, float(self.left_tunnel_y[env_idx].item()))
            right0 = local_to_panel(tunnel_x_min, float(self.right_tunnel_y[env_idx].item()))
            right1 = local_to_panel(tunnel_x_max, float(self.right_tunnel_y[env_idx].item()))
            draw_line(left0[0], left0[1], left1[0], left1[1], (230, 230, 230), thickness=1)
            draw_line(right0[0], right0[1], right1[0], right1[1], (230, 120, 120), thickness=1)

        f_tol = float(self.cfg["contact_classifier"]["inner_side_forward_tolerance"])
        s_tol = float(self.cfg["contact_classifier"]["inner_side_lateral_tolerance"])
        zone_center = np.array(
            [
                float(self.chosen_receive_point_local[env_idx, 0].item()),
                float(self.chosen_receive_point_local[env_idx, 1].item()),
            ],
            dtype=np.float32,
        )
        zone_forward = np.array(
            [
                float(self.chosen_foot_forward[env_idx, 0].item()),
                float(self.chosen_foot_forward[env_idx, 1].item()),
            ],
            dtype=np.float32,
        )
        zone_inner = np.array(
            [
                float(self.chosen_foot_inner_normal[env_idx, 0].item()),
                float(self.chosen_foot_inner_normal[env_idx, 1].item()),
            ],
            dtype=np.float32,
        )
        corners = [
            zone_center + zone_forward * f_tol + zone_inner * s_tol,
            zone_center + zone_forward * f_tol - zone_inner * s_tol,
            zone_center - zone_forward * f_tol - zone_inner * s_tol,
            zone_center - zone_forward * f_tol + zone_inner * s_tol,
        ]
        panel_corners = [local_to_panel(float(p[0]), float(p[1])) for p in corners]
        for p0, p1 in zip(panel_corners, panel_corners[1:] + panel_corners[:1]):
            draw_line(p0[0], p0[1], p1[0], p1[1], (60, 240, 60), thickness=1)

        bar_x0 = 12
        bar_y0 = 12
        bar_h = 10
        bar_gap = 8
        bar_w = panel - 24
        debug_bars = [
            (float(self.arrival_confidence[env_idx].item()), (40, 220, 40)),
            (float(self.tunnel_risk[env_idx].item()), (40, 40, 220)),
            (min(float(self.stance_gap[env_idx].item()) / max(float(self.stance_gap_target[env_idx].item()), 1.0e-6), 1.5), (220, 220, 40)),
            (min(float(self.skating_indicator[env_idx].item()), 1.0), (220, 60, 60)),
        ]
        for idx, (value, color) in enumerate(debug_bars):
            y0 = bar_y0 + idx * (bar_h + bar_gap)
            frame[y0 : y0 + bar_h, bar_x0 : bar_x0 + bar_w, 0:3] = 30
            fill = int(np.clip(value, 0.0, 1.0) * bar_w)
            frame[y0 : y0 + bar_h, bar_x0 : bar_x0 + fill, 0] = color[0]
            frame[y0 : y0 + bar_h, bar_x0 : bar_x0 + fill, 1] = color[1]
            frame[y0 : y0 + bar_h, bar_x0 : bar_x0 + fill, 2] = color[2]
