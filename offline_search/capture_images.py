import cv2
import numpy as np
import os
import pyrealsense2.pyrealsense2 as rs

# Create directories for saving images
depth_dir = 'depth'
color_dir = 'color'
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(color_dir, exist_ok=True)

# Configure Capture fps and duration
fps = 15
duration = 3 # minutes

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)

# Start streaming
pipeline.start(config)

# Initialize frame count
frame_count = 0

try:
    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # # Convert depth frame to a numpy array
        # depth_image = np.asanyarray(depth_frame.get_data())

        # Convert color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Save color image
        color_filename = os.path.join(color_dir, f'frame_{frame_count:04d}.png')
        if frame_count >= 30: cv2.imwrite(color_filename, color_image)

        # Increment frame count
        frame_count += 1

        # Break 
        if frame_count >= fps*60*duration:
            break

finally:
    # Stop streaming
    pipeline.stop()