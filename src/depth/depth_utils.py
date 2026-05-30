import numpy as np

def filter_invalid_depth(depth_values):
    return depth_values[(depth_values > 0)]

def get_center_depth(depth_frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if 0 <= cy < depth_frame.shape[0] and 0 <= cx < depth_frame.shape[1]:
        return depth_frame[cy, cx]
    return 0

def get_median_depth_in_bbox(depth_frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_frame.shape[1], x2), min(depth_frame.shape[0], y2)
    roi = depth_frame[y1:y2, x1:x2]
    valid_depths = filter_invalid_depth(roi)
    if len(valid_depths) > 0:
        return np.median(valid_depths)
    return 0

def mm_to_m(depth_mm):
    return depth_mm / 1000.0
