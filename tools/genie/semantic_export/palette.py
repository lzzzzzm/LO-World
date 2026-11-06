"""Official GenieDrive render labels and camera ordering, without model imports."""

import numpy as np

CAM_NAMES = (
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
)

# Identical to GenieDrive process_utils.color_id; values are RGB.
GENIE_COLOR_ID = np.array([
    [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [255, 127, 0], [255, 0, 0], [255, 240, 150],
    [135, 60, 0], [160, 32, 240], [255, 0, 255], [139, 137, 137],
    [75, 0, 75], [150, 240, 80], [230, 230, 250], [0, 175, 0],
    [0, 255, 127], [222, 155, 161], [140, 62, 69], [227, 164, 30],
], dtype=np.uint8)
