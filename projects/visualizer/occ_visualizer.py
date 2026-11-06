import os
import cv2
import torch
import numpy as np
from mmdet3d.registry import VISUALIZERS

from tools.visualization.colors import get_occ3d_color_map

@VISUALIZERS.register_module()
class OccBEVVisualizer:
    def __init__(self, save_dir='work_dirs/vis_occ', empty_cls=17, ignore_index=255):
        """
        Args:
            save_dir (str): Directory to save the visualized occupancy maps.
            empty_cls (int): the category index for free/empty space.
            ignore_index (int): the category index for ignored regions.
        """
        self.save_dir = save_dir
        self.empty_cls = empty_cls
        self.ignore_index = ignore_index
        self.color_map = get_occ3d_color_map()
        self.step = 0

        # if save_dir exits, remove it and create a new one
        os.makedirs(save_dir, exist_ok=True)


    def draw_occ_bev_compare(self, pred_semantics, targ_semantics, name_prefix='compare_occ', save_name=None):
        """
        Visualize prediction and target Occupancy grids side by side.
        Handles both 3D [Y, X, Z] and 4D [F, Y, X, Z] shapes.
        """
        def format_semantics(voxel_semantics):
            if isinstance(voxel_semantics, torch.Tensor):
                voxel_semantics = voxel_semantics.detach().cpu().numpy()
            elif isinstance(voxel_semantics, list):
                voxel_semantics = np.stack([
                    v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                    for v in voxel_semantics
                ])
            if len(voxel_semantics.shape) == 3:
                # [Y, X, Z] -> [1, 1, Y, X, Z]
                voxel_semantics = np.expand_dims(voxel_semantics, axis=(0, 1))
            elif len(voxel_semantics.shape) == 4:
                # [F, Y, X, Z] -> [1, F, Y, X, Z]
                voxel_semantics = np.expand_dims(voxel_semantics, axis=0)
            return voxel_semantics

        pred_batch = format_semantics(pred_semantics)
        targ_batch = format_semantics(targ_semantics)
        
        B, F, bev_y, bev_x, Z = pred_batch.shape

        for b in range(B):
            frame_imgs = []
            for f in range(F):
                def render_bev(semantics_b_f):
                    bev_map = np.full((bev_y, bev_x), self.empty_cls, dtype=np.uint8)
                    for z in range(Z - 1, -1, -1):
                        layer = semantics_b_f[..., z]
                        valid_mask = (layer != self.empty_cls) & (layer != self.ignore_index) & (layer >= 0) & (layer < 18)
                        update_mask = (bev_map == self.empty_cls) & valid_mask
                        bev_map[update_mask] = layer[update_mask]
                    return self.color_map[bev_map].astype(np.uint8)
    
                pred_img = render_bev(pred_batch[b, f])
                targ_img = render_bev(targ_batch[b, f])
    
                compare_img = np.concatenate((pred_img, targ_img), axis=0)
                    
                frame_imgs.append(compare_img)
                
            # Concatenate all frames vertically
            final_img = np.concatenate(frame_imgs, axis=1) if len(frame_imgs) > 1 else frame_imgs[0]

            if save_name is not None:
                save_path = os.path.join(self.save_dir, f'{save_name}.png')
            else:
                save_path = os.path.join(self.save_dir, f'{name_prefix}_step{self.step}_b{b}.png')
            cv2.imwrite(save_path, final_img)
            
        self.step += 1
