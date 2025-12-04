import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import wandb
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Recorder:

    def __init__(self, cfg):
        self.cfg = cfg
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # Create logs in robot-type/task-name hierarchy
        task_name = self.cfg["basic"]["task"]
        
        # Determine robot type from task name
        robot_type = self._get_robot_type(task_name)
        
        # Create hierarchical structure: logs/robot_type/task_name/timestamp
        robot_log_dir = os.path.join("logs", robot_type)
        task_log_dir = os.path.join(robot_log_dir, task_name)
        self.dir = os.path.join(task_log_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        if self.cfg["runner"]["use_wandb"]:
            # Sanitize project name for wandb (remove invalid characters)
            project_name = self._sanitize_project_name(self.cfg["basic"]["task"])
            wandb.init(
                project=project_name,
                dir=self.dir,
                name=name,
                notes=self.cfg["basic"]["description"],
                config=self.cfg,
            )

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

    def _sanitize_project_name(self, task_name):
        """Sanitize task name for wandb project name by removing invalid characters."""
        # Replace invalid characters with underscores
        invalid_chars = ['/', '\\', '#', '?', '%', ':']
        sanitized = task_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized

    def _get_robot_type(self, task_name):
        """Determine robot type from task name."""
        # Check if task name starts with K1 or T1
        if task_name.startswith("K1"):
            return "K1"
        elif task_name.startswith("T1"):
            return "T1"
        else:
            # Default fallback - could be extended for other robot types
            return "Unknown"

    def record_episode_statistics(self, done, ep_info, it, write_record=False):
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=int)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        if write_record:
            for key in self.last_episode.keys():
                path = ("" if key == "steps" or key == "reward" else "episode/") + key
                value = self._mean(self.last_episode[key])
                self.writer.add_scalar(path, value, it)
                if self.cfg["runner"]["use_wandb"]:
                    wandb.log({path: value}, step=it)
                self.last_episode[key].clear()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self.writer.add_scalar(key, float(value), it)
            if self.cfg["runner"]["use_wandb"]:
                wandb.log({key: float(value)}, step=it)

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("Saving model to {}".format(path))
        torch.save(model_dict, path)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)

    def log_video(self, frames, it, dt):
        """Log video frames to wandb.
        
        Args:
            frames: List of camera frames in RGBA format (height, width, 4)
            it: Current iteration step
            dt: Simulation timestep for calculating FPS
        """
        if not self.cfg["runner"]["use_wandb"]:
            return
        
        if len(frames) == 0:
            return
        
        import numpy as np
        
        # Convert frames from RGBA to RGB format
        # Isaac Gym returns images in BGRA format, so we need to convert
        video_array = []
        for frame in frames:
            # Handle different frame shapes
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    # Isaac Gym returns BGRA format, convert to RGB
                    # Take BGR channels and reverse to RGB: [B, G, R, A] -> [R, G, B]
                    rgb_frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
                elif frame.shape[2] == 3:
                    rgb_frame = frame
                else:
                    # Unexpected format, try to take first 3 channels
                    rgb_frame = frame[:, :, :3]
            elif len(frame.shape) == 2:
                # Grayscale, convert to RGB by repeating channels
                rgb_frame = np.stack([frame, frame, frame], axis=2)
            else:
                # Try to reshape if it's a flattened image
                # Assuming it's (height*width*4,) or similar
                if frame.size % 4 == 0:
                    # Try to reshape as RGBA
                    h = int(np.sqrt(frame.size // 4))
                    w = frame.size // (4 * h)
                    frame_reshaped = frame.reshape(h, w, 4)
                    rgb_frame = frame_reshaped[:, :, [2, 1, 0]]  # BGR -> RGB
                else:
                    continue  # Skip invalid frames
            
            # Ensure uint8 format (0-255 range)
            if rgb_frame.dtype != np.uint8:
                # Handle float values in [0, 1] range
                if rgb_frame.max() <= 1.0:
                    rgb_frame = (rgb_frame * 255).astype(np.uint8)
                else:
                    rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
            
            video_array.append(rgb_frame)
        
        if len(video_array) == 0:
            print(f"Warning: No valid frames to log at iteration {it}")
            return
        
        # Stack frames: (time, height, width, channels)
        video_tensor = np.stack(video_array, axis=0)

        # wandb.Video expects (time, channels, width, height) format
        # Transpose from (time, height, width, channels) to (time, channels, width, height)
        video_tensor = np.transpose(video_tensor, (0, 3, 1, 2))
        
        # Calculate FPS from simulation timestep (same as in play mode)
        fps = int(1.0 / dt)
        
        # Log to wandb
        wandb.log({"video/training": wandb.Video(video_tensor, fps=fps, format="mp4")}, step=it)

    def log_video_rewards(self, total_reward, separated_reward, it):
        """Log rewards collected during video capture as time-series graphs.
        
        Args:
            total_reward: List of total reward values per step (list of floats)
            separated_reward: Dictionary of reward term names to lists of floats
            it: Current iteration step
        """
        matplotlib.use('Agg')  # Use non-interactive backend

        if len(total_reward) == 0:
            print(f"Warning: No rewards to log at iteration {it}")
            return
        
        if not self.cfg["runner"]["use_wandb"]:
            print(f"Warning: wandb is disabled, skipping reward trajectory logging")
            return
            
        
        # Convert total reward to numpy array (should already be list of floats)
        total_reward_np = np.array(total_reward)
        
        print(f"Logging video rewards at iteration {it}: {len(total_reward_np)} frames, {len(separated_reward)} reward terms")
        
        # Calculate statistics for total reward
        mean_total_reward = float(np.mean(total_reward_np))
        sum_total_reward = float(np.sum(total_reward_np))
        
        # Log summary statistics
        self.writer.add_scalar("video/mean_reward", mean_total_reward, it)
        self.writer.add_scalar("video/sum_reward", sum_total_reward, it)
        
        timesteps = np.arange(len(total_reward_np))
        
        # Log to wandb
        if self.cfg["runner"]["use_wandb"]:
            
            # Create and log figure for total reward
            fig_total = plt.figure(figsize=(12, 4))
            plt.plot(timesteps, total_reward_np, linewidth=2, color='blue')
            plt.title(f'Total Reward (Mean: {mean_total_reward:.3f}, Sum: {sum_total_reward:.3f})', fontsize=12, fontweight='bold')
            plt.xlabel('Frame')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.tight_layout()
            wandb.log({"video_plots/total_reward_trajectory": wandb.Image(fig_total)}, step=it)
            print(f"Logged total reward trajectory at iteration {it}")
            plt.close(fig_total)
            
            # Create and log figure for each reward term
            for key, values in separated_reward.items():
                if len(values) == 0:
                    continue
                
                values_np = np.array(values)
                mean_value = float(np.mean(values_np))
                sum_value = float(np.sum(values_np))
                
                # Create figure for this reward term
                fig_term = plt.figure(figsize=(12, 4))
                plt.plot(timesteps, values_np, linewidth=2)
                plt.title(f'{key} (Mean: {mean_value:.3f}, Sum: {sum_value:.3f})', fontsize=12)
                plt.xlabel('Frame')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.tight_layout()
                wandb.log({f"video_plots/reward_trajectories/{key}": wandb.Image(fig_term)}, step=it)
                plt.close(fig_term)
            
            
