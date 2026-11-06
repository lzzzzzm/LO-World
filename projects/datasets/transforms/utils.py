import numpy as np

def pts2range_np(        
        pts, 
        ego2lidar=None,  
        lidar2ego=None,
        pc_range=None,
        pc_range_coord: str = 'ego',
        H: int = 32,
        W: int = 1024,
        min_depth: float = 1.45,
        max_depth: float = 80.0,
        fov_up: float = 10.0,
        fov_down: float = -30.0,
        filter_out_of_fov: bool = False,):
    """
        pts: np.ndarray (N,5) -> (x,y,z, intensity, ring_id) or (N,3) -> (x,y,z)
        change pts to range image, only return the depth channel of the range image [H, W, 1]
        ego2lidar: (4,4), used when input points are in ego coordinates
        lidar2ego: (4,4), used to filter lidar-coordinate points by ego pc_range
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    xyz = np.asarray(pts[:, :3], dtype=np.float32)

    if pc_range is not None:
        pc_range = np.asarray(pc_range, dtype=np.float32)
        if pc_range.shape != (6,):
            raise ValueError(f'Expected pc_range with shape (6,), got {pc_range.shape}')
        if pc_range_coord not in ('ego', 'lidar'):
            raise ValueError(
                f'pc_range_coord must be "ego" or "lidar", got {pc_range_coord}')

        xyz_for_range = xyz
        if pc_range_coord == 'ego':
            if ego2lidar is not None:
                # Input points are already in ego coordinates.
                xyz_for_range = xyz
            elif lidar2ego is not None:
                xyz_hom = np.concatenate(
                    [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)],
                    axis=1)
                xyz_for_range = (lidar2ego @ xyz_hom.T).T[:, :3]
            else:
                raise ValueError(
                    'lidar2ego is required when filtering lidar-coordinate '
                    'points by ego-coordinate pc_range.')
        elif ego2lidar is not None:
            xyz_hom = np.concatenate(
                [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
            xyz_for_range = (ego2lidar @ xyz_hom.T).T[:, :3]

        x_min, y_min, z_min, x_max, y_max, z_max = pc_range
        mask = ((xyz_for_range[:, 0] >= x_min) &
                (xyz_for_range[:, 0] < x_max) &
                (xyz_for_range[:, 1] >= y_min) &
                (xyz_for_range[:, 1] < y_max) &
                (xyz_for_range[:, 2] >= z_min) &
                (xyz_for_range[:, 2] < z_max))
        xyz = xyz[mask]

    if ego2lidar is not None:
        # from ego to lidar
        xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)  # (N,4)
        xyz = (ego2lidar @ xyz_hom.T).T[:, :3]  # (N,3)

    depth = np.linalg.norm(xyz, axis=1)
    
    mask = (depth >= min_depth) & (depth <= max_depth)
    xyz = xyz[mask]
    depth = depth[mask]
    
    # 获取 x, y, z
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    yaw = np.arctan2(y, x)
    # pitch 的范围是 [-pi/2, pi/2]
    pitch = np.arcsin(z / depth)
    
    fov_up_rad = fov_up / 180.0 * np.pi
    fov_down_rad = fov_down / 180.0 * np.pi
    fov_rad = fov_up_rad - fov_down_rad

    if filter_out_of_fov:
        fov_mask = (pitch >= fov_down_rad) & (pitch <= fov_up_rad)
        yaw = yaw[fov_mask]
        pitch = pitch[fov_mask]
        depth = depth[fov_mask]
    
    proj_x = 0.5 * (1.0 - yaw / np.pi) * W
    proj_y = (1.0 - (pitch - fov_down_rad) / fov_rad) * H
    
    proj_x = np.floor(proj_x).astype(np.int64)
    proj_y = np.floor(proj_y).astype(np.int64)
    
    proj_x = np.clip(proj_x, 0, W - 1)
    proj_y = np.clip(proj_y, 0, H - 1)
    
    if depth.size == 0:
        return np.zeros((H, W, 1), dtype=np.float32)

    # Keep the nearest point for each range-image pixel without sorting every
    # point in dataloader workers.
    flat_index = proj_y * W + proj_x
    flat_range = np.full(H * W, np.inf, dtype=np.float32)
    np.minimum.at(flat_range, flat_index, depth.astype(np.float32, copy=False))
    flat_range[~np.isfinite(flat_range)] = 0.0
    range_img = flat_range.reshape(H, W)

    return np.expand_dims(range_img, axis=-1)  # 返回形状为 [H, W, 1]
