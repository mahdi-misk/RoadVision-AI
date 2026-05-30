from .depth_camera import DepthCamera
import numpy as np
import cv2

class RealSenseCamera(DepthCamera):
    def __init__(self):
        self.pipeline = None
        self.align = None
        try:
            import pyrealsense2 as rs
            self.rs = rs
        except ImportError:
            self.rs = None
            print("pyrealsense2 is not installed. Depth features will be unavailable.")

    def start(self):
        if self.rs is None:
            return False
            
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        config.enable_stream(self.rs.stream.depth, 640, 480, self.rs.format.z16, 30)
        config.enable_stream(self.rs.stream.color, 640, 480, self.rs.format.bgr8, 30)
        
        self.pipeline.start(config)
        align_to = self.rs.stream.color
        self.align = self.rs.align(align_to)
        return True

    def read(self):
        if self.pipeline is None:
            return False, None, None
            
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return False, None, None
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return True, color_image, depth_image

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()
