import numpy as np
from isaacgym import gymapi, terrain_utils
import torch

from utils.utils import apply_randomization


class Terrain:

    def __init__(self, gym, sim, device, terrain_cfg):
        self.terrain_cfg = terrain_cfg
        self.gym = gym
        self.sim = sim
        self.device = device
        self.type = self.terrain_cfg["type"]
        self.friction_map = None

        if self.type == "plane":
            self._create_ground_plane()
        elif self.type == "trimesh":
            self._create_trimesh()
        else:
            raise ValueError(f"Invalid terrain type: {self.type}")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.terrain_cfg["static_friction"]
        plane_params.dynamic_friction = self.terrain_cfg["dynamic_friction"]
        plane_params.restitution = self.terrain_cfg["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.env_width = self.terrain_cfg["num_terrains"] * self.terrain_cfg["terrain_width"]
        self.env_length = self.terrain_cfg["terrain_length"]
        self.border_size = self.terrain_cfg["border_size"]
        self.horizontal_scale = self.terrain_cfg["horizontal_scale"]
        self.vertical_scale = self.terrain_cfg["vertical_scale"]
        self.border_pixels = int(self.border_size / self.horizontal_scale)
        terrain_width_pixels = int(self.terrain_cfg["terrain_width"] / self.horizontal_scale)
        terrain_length_pixels = int(self.terrain_cfg["terrain_length"] / self.horizontal_scale)
        self.height_field_raw = np.zeros(
            (
                self.terrain_cfg["num_terrains"] * terrain_width_pixels + 2 * self.border_pixels,
                terrain_length_pixels + 2 * self.border_pixels,
            ),
            dtype=np.int16,
        )
        proportions = [
            self.terrain_cfg["num_terrains"]
            * np.sum(self.terrain_cfg["terrain_proportions"][: i + 1])
            / np.sum(self.terrain_cfg["terrain_proportions"])
            for i in range(len(self.terrain_cfg["terrain_proportions"]))
        ]
        for i in range(self.terrain_cfg["num_terrains"]):
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=terrain_width_pixels,
                length=terrain_length_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )
            if i < proportions[0]:
                pass
            elif i < proportions[1]:
                terrain_utils.pyramid_sloped_terrain(terrain, slope=self.terrain_cfg["slope"], platform_size=3.0)
            elif i < proportions[2]:
                terrain_utils.random_uniform_terrain(
                    terrain,
                    min_height=-0.5 * self.terrain_cfg["random_height"],
                    max_height=0.5 * self.terrain_cfg["random_height"],
                    step=0.005,
                    downsampled_scale=0.2,
                )
            else:
                terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    max_height=self.terrain_cfg["discrete_height"],
                    min_size=1.0,
                    max_size=2.0,
                    num_rects=20,
                    platform_size=3.0,
                )
            start_x = self.border_pixels + i * terrain_width_pixels
            end_x = self.border_pixels + (i + 1) * terrain_width_pixels
            start_y = self.border_pixels
            end_y = self.border_pixels + terrain_length_pixels
            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.height_field_raw, self.horizontal_scale, self.vertical_scale, self.terrain_cfg["slope_threshold"]
        )

        self.height_field_torch = torch.from_numpy(self.height_field_raw).to(self.device)

        # IsaacGym uses terrain friction incorrectly for triangle meshes, so we split
        # the mesh into patches and assign each patch its own randomized friction.
        friction_cfg = self.terrain_cfg.get("friction")
        patch_size = float(self.terrain_cfg.get("patch_size", 0.0))
        if friction_cfg and patch_size > 0.0:
            verts = vertices.reshape(-1, 3)
            tris = triangles.reshape(-1, 3)

            tri_verts = verts[tris]
            centroids = tri_verts.mean(axis=1)

            px = np.floor(centroids[:, 0] / patch_size).astype(np.int32)
            py = np.floor(centroids[:, 1] / patch_size).astype(np.int32)
            patch_ids = np.stack([px, py], axis=1)

            patch_keys, inverse = np.unique(patch_ids, axis=0, return_inverse=True)
            self.friction_map = torch.zeros(len(patch_keys), 3, dtype=torch.float, device=self.device)

            for patch_index, (patch_x, patch_y) in enumerate(patch_keys):
                tris_in_patch = tris[inverse == patch_index]
                unique_verts, new_indices = np.unique(tris_in_patch.flatten(), return_inverse=True)
                patch_vertices = verts[unique_verts].astype(np.float32, copy=False)
                patch_triangles_local = new_indices.reshape(-1, 3).astype(np.uint32, copy=False)

                tm_params = gymapi.TriangleMeshParams()
                tm_params.nb_vertices = patch_vertices.shape[0]
                tm_params.nb_triangles = patch_triangles_local.shape[0]
                tm_params.transform.p.x = -self.border_size
                tm_params.transform.p.y = -self.border_size
                tm_params.transform.p.z = 0.0

                friction_val = float(apply_randomization(0.0, friction_cfg))
                tm_params.static_friction = friction_val
                tm_params.dynamic_friction = friction_val
                tm_params.restitution = self.terrain_cfg["restitution"]
                self.gym.add_triangle_mesh(self.sim, patch_vertices.flatten(order="C"), patch_triangles_local.flatten(order="C"), tm_params)

                self.friction_map[patch_index, 0] = (float(patch_x) + 0.5) * patch_size - self.border_size
                self.friction_map[patch_index, 1] = (float(patch_y) + 0.5) * patch_size - self.border_size
                self.friction_map[patch_index, 2] = friction_val
        else:
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = vertices.shape[0]
            tm_params.nb_triangles = triangles.shape[0]
            tm_params.transform.p.x = -self.border_size
            tm_params.transform.p.y = -self.border_size
            tm_params.transform.p.z = 0.0
            tm_params.static_friction = self.terrain_cfg["static_friction"]
            tm_params.dynamic_friction = self.terrain_cfg["dynamic_friction"]
            tm_params.restitution = self.terrain_cfg["restitution"]
            self.gym.add_triangle_mesh(self.sim, vertices.flatten(order="C"), triangles.flatten(order="C"), tm_params)

    def terrain_heights(self, base_pos):
        if self.type == "plane":
            return torch.zeros(len(base_pos), dtype=torch.float, device=self.device)
        else:
            x = self.border_pixels + base_pos[:, 0].cpu().numpy() / self.horizontal_scale
            y = self.border_pixels + base_pos[:, 1].cpu().numpy() / self.horizontal_scale
            x1 = np.floor(x).astype(int)
            x2 = x1 + 1
            y1 = np.floor(y).astype(int)
            y2 = y1 + 1
            x1 = np.clip(x1, 0, self.height_field_raw.shape[0] - 2)
            x2 = np.clip(x2, 1, self.height_field_raw.shape[0] - 1)
            y1 = np.clip(y1, 0, self.height_field_raw.shape[1] - 2)
            y2 = np.clip(y2, 1, self.height_field_raw.shape[1] - 1)
            return torch.tensor(
                (
                    (x2 - x) * (y2 - y) * self.height_field_raw[x1, y1]
                    + (x - x1) * (y2 - y) * self.height_field_raw[x2, y1]
                    + (x2 - x) * (y - y1) * self.height_field_raw[x1, y2]
                    + (x - x1) * (y - y1) * self.height_field_raw[x2, y2]
                )
                * self.vertical_scale,
                dtype=torch.float,
                device=self.device,
            )
