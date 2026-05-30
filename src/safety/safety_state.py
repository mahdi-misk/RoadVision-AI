def evaluate_safety(class_name: str, confidence: float, distance_m: float = None) -> str:
    """
    Evaluates safety state based on distance and class.
    distance < 1.0m = DANGER
    distance between 1.0m and 2.0m = WARNING
    distance > 2.0m = SAFE
    """
    base_state = "SAFE"
    
    if distance_m is not None:
        if distance_m < 1.0:
            base_state = "DANGER"
        elif 1.0 <= distance_m <= 2.0:
            base_state = "WARNING"
            
    # Increase risk level for specific classes
    if class_name in ["severe_pothole", "descending_stairs"]:
        if base_state == "SAFE":
            base_state = "WARNING"
        elif base_state == "WARNING":
            base_state = "DANGER"
            
    return base_state
