"""Shared visualization palettes, independent of model registration."""

import numpy as np


def get_occ3d_color_map() -> np.ndarray:
    """
    Get the OCC3D standard color map for occupancy visualization.

    Returns:
        Color map array with shape (18, 3) in BGR format for Open3D compatibility

    Classes:
        0: others (black)
        1: barrier (orange)
        2: bicycle (pink)
        3: bus (yellow)
        4: car (blue)
        5: construction_vehicle (cyan)
        6: motorcycle (dark orange)
        7: pedestrian (red)
        8: traffic_cone (light yellow)
        9: trailer (brown)
        10: truck (purple)
        11: driveable_surface (dark pink)
        12: other_flat (dark red)
        13: sidewalk (dark purple)
        14: terrain (light green)
        15: manmade (white)
        16: vegetation (green)
        17: free (white)
    """
    occ3d_colors_map = np.array([
        [0, 0, 0],          # others               black
        [255, 120, 50],     # barrier              orange
        [255, 192, 203],    # bicycle              pink
        [255, 255, 0],      # bus                  yellow
        [0, 150, 245],      # car                  blue
        [0, 255, 255],      # construction_vehicle cyan
        [255, 127, 0],      # motorcycle           dark orange
        [255, 0, 0],        # pedestrian           red
        [255, 240, 150],    # traffic_cone         light yellow
        [135, 60, 0],       # trailer              brown
        [160, 32, 240],     # truck                purple
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dark purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # free                 white
    ])
    # Convert RGB to BGR for Open3D compatibility
    return occ3d_colors_map[:, ::-1]

