import numpy as np
import torch
from utils.observation_controller import get_controller


class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()
        # Initialize observation controller for live control
        self.obs_controller = get_controller()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        self.gait_frequency = 2.0
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)

        start_index = self.cfg["common"]["joint_cnt"] - self.cfg["policy"]["num_actions"]
        
        # Use live-controlled walk commands from observation controller
        # Fallback to remote control service if not available
        try:
            self.commands[0] = self.obs_controller.get_vx_cmd()
            self.commands[1] = self.obs_controller.get_vy_cmd()
            self.commands[2] = self.obs_controller.get_vyaw_cmd()
        except:
            # Fallback to remote control service values
            self.commands[0] = vx
            self.commands[1] = vy
            self.commands[2] = vyaw
            
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6] = 0.2
        self.obs[7] = 0
        self.obs[8] = 0
        
        self.obs[9] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[10] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[11:23] = (dof_pos - self.default_dof_pos)[start_index:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[23:35] = dof_vel[start_index:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[35:47] = self.actions
        self.obs[47] = 2.0

        output = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()[0]
        self.actions[:] = output[0:12]
        frequency = output[12]
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[start_index:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
