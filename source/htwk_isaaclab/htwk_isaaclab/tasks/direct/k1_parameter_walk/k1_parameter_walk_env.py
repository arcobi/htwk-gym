from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, sample_uniform


REPO_ROOT = Path(__file__).resolve().parents[6]
K1_USD_PATH = REPO_ROOT / "resources" / "isaaclab" / "K1_locomotion.usd"


@configclass
class K1ParameterWalkEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 10
    action_scale = 1.0
    action_space = 12
    observation_space = 54
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(K1_USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.58),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*Hip_Pitch": -0.2,
                ".*Knee_Pitch": 0.4,
                ".*Ankle_Pitch": -0.25,
                ".*Hip_Roll": 0.0,
                ".*Hip_Yaw": 0.0,
                ".*Ankle_Roll": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=300.0,
                velocity_limit_sim=100.0,
                stiffness={
                    ".*Hip.*": 100.0,
                    ".*Knee.*": 100.0,
                    ".*Ankle.*": 50.0,
                },
                damping={
                    ".*Hip.*": 2.0,
                    ".*Knee.*": 2.0,
                    ".*Ankle.*": 1.0,
                },
            ),
        },
    )

    command_ranges = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_yaw": (-1.6, 1.6),
        "gait_frequency": (1.5, 2.4),
        "foot_yaw_l": (-0.7, 0.7),
        "foot_yaw_r": (-0.7, 0.7),
        "body_pitch": (-0.1, 0.3),
        "body_roll": (-0.1, 0.1),
        "feet_offset_x": (-0.15, 0.15),
        "feet_offset_y": (-0.08, 0.15),
    }
    resampling_time_s = (3.0, 8.0)
    termination_height = 0.25
    dof_vel_scale = 0.1
    action_rate_scale = -0.01
    torque_scale = -2.0e-5


class K1ParameterWalkEnv(DirectRLEnv):
    cfg: K1ParameterWalkEnvCfg

    def __init__(self, cfg: K1ParameterWalkEnvCfg, render_mode: str | None = None, **kwargs):
        if not K1_USD_PATH.exists():
            raise FileNotFoundError(
                f"Missing Isaac Lab USD asset: {K1_USD_PATH}. "
                "Run `scripts/isaaclab_convert_assets.sh` from the repository root first."
            )
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, self._joint_names = self.robot.find_joints(".*")
        self._base_body_id = self.robot.find_bodies("Trunk")[0][0]
        self._feet_body_ids = self.robot.find_bodies(".*foot.*")[0]
        self.commands = torch.zeros(self.num_envs, 10, device=self.device)
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.gravity_vec = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        target = self.robot.data.default_joint_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(target, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        root_quat = self.robot.data.root_quat_w
        base_ang_vel = quat_apply_inverse(root_quat, self.robot.data.root_ang_vel_w)
        projected_gravity = quat_apply_inverse(root_quat, self.gravity_vec)
        self.gait_process[:] = torch.remainder(self.gait_process + self.step_dt * self.commands[:, 3], 1.0)
        obs = torch.cat(
            (
                self.commands,
                base_ang_vel,
                projected_gravity,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel * self.cfg.dof_vel_scale,
                self.actions,
                torch.cos(2.0 * torch.pi * self.gait_process).unsqueeze(-1),
                torch.sin(2.0 * torch.pi * self.gait_process).unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_quat = self.robot.data.root_quat_w
        base_lin_vel = quat_apply_inverse(root_quat, self.robot.data.root_lin_vel_w)
        base_ang_vel = quat_apply_inverse(root_quat, self.robot.data.root_ang_vel_w)
        lin_x = torch.exp(-torch.square(self.commands[:, 0] - base_lin_vel[:, 0]) / 0.25)
        lin_y = torch.exp(-torch.square(self.commands[:, 1] - base_lin_vel[:, 1]) / 0.25)
        yaw = torch.exp(-torch.square(self.commands[:, 2] - base_ang_vel[:, 2]) / 0.25)
        alive = torch.ones_like(lin_x)
        action_rate = torch.sum(torch.square(self.actions - self.previous_actions), dim=-1)
        torque_penalty = torch.sum(torch.square(self.robot.data.applied_torque), dim=-1)
        return alive + lin_x + lin_y + yaw + self.cfg.action_rate_scale * action_rate + self.cfg.torque_scale * torque_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.command_time_left -= self.step_dt
        resample_ids = torch.nonzero(self.command_time_left <= 0.0, as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fell = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        return fell, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos += sample_uniform(-0.05, 0.05, joint_pos.shape, joint_pos.device)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.gait_process[env_ids] = 0.0
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        ranges = self.cfg.command_ranges
        keys = tuple(ranges.keys())
        for command_id, key in enumerate(keys):
            low, high = ranges[key]
            self.commands[env_ids, command_id] = sample_uniform(low, high, (len(env_ids),), self.device)
        still_count = int(0.1 * len(env_ids))
        if still_count > 0:
            still_ids = env_ids[torch.randperm(len(env_ids), device=self.device)[:still_count]]
            self.commands[still_ids, :3] = 0.0
        low, high = self.cfg.resampling_time_s
        self.command_time_left[env_ids] = sample_uniform(low, high, (len(env_ids),), self.device)


class K1ParameterWalkEnvCfg_PLAY(K1ParameterWalkEnvCfg):
    def __post_init__(self):
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.0
