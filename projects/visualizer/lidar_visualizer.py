import os
import cv2
import torch
import numpy as np
from mmdet3d.registry import VISUALIZERS


def points_to_bev(points,
                  x_range=(-50.0, 50.0),
                  y_range=(-50.0, 50.0),
                  z_range=(-3.0, 5.0),
                  resolution=0.5,
                  background_color=(20, 20, 20)):
    """
        Convert 3D Lidar points to a 2D BEV image.
        points: (N, 3+) array of x, y, z, ...
    """
    assert points.shape[1] >= 3
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    mask = (
        (points[:,0] >= x_min) & (points[:,0] < x_max) &
        (points[:,1] >= y_min) & (points[:,1] < y_max) &
        (points[:,2] >= z_min) & (points[:,2] < z_max)
    )
    pts = points[mask]
    if pts.shape[0] == 0:
        H = int((x_max - x_min) / resolution)
        W = int((y_max - y_min) / resolution)
        bev = np.full((H, W, 3), background_color, dtype=np.uint8)
        return bev

    H = int(np.ceil((x_max - x_min) / resolution))
    W = int(np.ceil((y_max - y_min) / resolution))

    xs = ((pts[:,0] - x_min) / resolution).astype(np.int32)
    ys = ((pts[:,1] - y_min) / resolution).astype(np.int32)

    rows = H - 1 - xs
    cols = ys

    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    rows, cols = rows[valid], cols[valid]
    zs = pts[valid, 2]

    occ = np.zeros((H, W), dtype=np.uint16)
    np.add.at(occ, (rows, cols), 1)

    height = np.full((H, W), -1e9, dtype=np.float32)
    for r, c, z in zip(rows, cols, zs):
        if z > height[r, c]:
            height[r, c] = z
    height[height < -1e8] = z_min
    height_norm = (height - z_min) / max(1e-6, (z_max - z_min))
    height_img = (height_norm * 255).clip(0, 255).astype(np.uint8)

    height_color = cv2.applyColorMap(height_img, cv2.COLORMAP_VIRIDIS)

    occ_mask = (occ > 0).astype(np.uint8)[..., None]
    bg_color_array = np.array(background_color, dtype=np.uint8)
    bev_color = height_color * occ_mask + (1 - occ_mask) * bg_color_array

    # rotate bev to align with occupancy visualization
    bev_color = cv2.rotate(bev_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bev_color = cv2.rotate(bev_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bev_color = cv2.flip(bev_color, 1)

    return bev_color


@VISUALIZERS.register_module()
class LidarBEVVisualizer:
    def __init__(self,
                 save_dir='work_dirs/vis_lidar',
                 x_range=(-50.0, 50.0),
                 y_range=(-50.0, 50.0),
                 z_range=(-3.0, 5.0),
                 resolution=0.2,
                 background_color=(20, 20, 20)):
        """
        Args:
            save_dir (str): Directory to save the visualized lidar BEV maps.
            x_range (tuple): (min, max) range in meters for x axis.
            y_range (tuple): (min, max) range in meters for y axis.
            z_range (tuple): (min, max) range in meters for z axis.
            resolution (float): grid resolution in meters.
            background_color (tuple): rgb/bgr background.
        """
        self.save_dir = save_dir
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        self.background_color = background_color
        self.step = 0
        os.makedirs(save_dir, exist_ok=True)


    def draw_lidar_bev_compare(self, pred_points, targ_points, name_prefix='compare_lidar', save_name=None):
        """
        Visualize prediction and target 3D Lidar points side by side.
        Handles both single frame and multi-frame sequences.
        """
        def format_points(points):
            # returns List[List[np.ndarray]] representing B x F x (N, 3)
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            if isinstance(points, np.ndarray):
                if points.ndim == 2:
                    return [[points]]
                elif points.ndim == 3:
                    return [[p] for p in points] # B x (N, 3), F=1
                elif points.ndim == 4:
                    return [[f for f in b] for b in points] # B x F x (N, 3)
            elif isinstance(points, list):
                if len(points) == 0:
                    return [[]]
                first = points[0]
                if isinstance(first, list):
                    # List[List] -> B x F
                    return [[p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p) for p in b] for b in points]
                elif hasattr(first, 'ndim') and getattr(first, 'ndim') == 2:
                    # List of 2D tensors -> Assume B=1, F=len(points)
                    converted = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in points]
                    return [converted]
                elif hasattr(first, 'ndim') and getattr(first, 'ndim') == 3:
                    # List of 3D tensors -> Assume B=len(points), F=len(first)
                    converted = [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p for p in points]
                    return [[f for f in b] for b in converted]
            
            raise ValueError(f"Unsupported points type or shape: {type(points)}")

        pred_batches = format_points(pred_points)
        targ_batches = format_points(targ_points)

        for b, (pred_seq, targ_seq) in enumerate(zip(pred_batches, targ_batches)):
            frame_imgs = []
            for pred, targ in zip(pred_seq, targ_seq):
                pred_img = points_to_bev(pred, x_range=self.x_range, y_range=self.y_range, z_range=self.z_range, resolution=self.resolution, background_color=self.background_color)
                targ_img = points_to_bev(targ, x_range=self.x_range, y_range=self.y_range, z_range=self.z_range, resolution=self.resolution, background_color=self.background_color)
                
                compare_img = np.concatenate((pred_img, targ_img), axis=0)
                    
                frame_imgs.append(compare_img)
                
            final_img = np.concatenate(frame_imgs, axis=1) if len(frame_imgs) > 1 else frame_imgs[0]
            
            if save_name is not None:
                save_path = os.path.join(self.save_dir, f'{save_name}.png')
            else:
                save_path = os.path.join(self.save_dir, f'{name_prefix}_step{self.step}_b{b}.png')
            cv2.imwrite(save_path, final_img)
            
        self.step += 1
