from .depth_utils import get_median_depth_in_bbox, mm_to_m

def estimate_distances(detections, depth_frame):
    """
    detections: list of dicts with 'box', 'class_name', 'confidence'
    Returns updated detections with 'distance_m' and updated labels.
    """
    for det in detections:
        depth_mm = get_median_depth_in_bbox(depth_frame, det['box'])
        distance_m = mm_to_m(depth_mm)
        det['distance_m'] = distance_m
        
        # Format label: class_name confidence | distance_m
        base_label = f"{det['class_name']} {det['confidence']:.2f}"
        if distance_m > 0:
            det['label'] = f"{base_label} | {distance_m:.1f}m"
        else:
            det['label'] = f"{base_label} | N/A"
    return detections
