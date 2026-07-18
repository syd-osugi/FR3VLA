"""
Camera Pixel Coordinate Utilities
---------------------------------
Converts a 2D pixel and its aligned RealSense depth reading into a 3D point in
the camera optical frame.

Robot-frame conversion happens later in robot/trajectory.py; this file only
does camera intrinsics/deprojection math.
"""

def _load_rs():
    """Lazy load pyrealsense2 only if this specific math function is called."""
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyrealsense2 is required to deproject RealSense depth pixels.") from exc
    return rs

def pixel_to_xyz(u, v, depth_rs_frame, depth_scale):
    """
    Converts a 2D LLM pixel guess into 3D real-world meters.
    
    Args:
        u (int): Pixel x coordinate from the LLM
        v (int): Pixel y coordinate from the LLM
        depth_rs_frame (rs.frame): The aligned RealSense depth frame object
        depth_scale (float): Kept for compatibility. get_distance() already returns meters.
    
    Returns:
        dict: {'x': float, 'y': float, 'z': float, 'valid': bool}
    """
    if depth_rs_frame is None:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'valid': False}

    # Grab the specific camera lens distortion parameters needed for the math
    intrinsics = depth_rs_frame.profile.as_video_stream_profile().intrinsics
    
    # Query the distance at the exact pixel. We wrap this in a try/except because 
    # if the LLM hallucinates a massive number or a string, this will throw an error.
    try:
        pixel = [int(u), int(v)]
        z_raw = depth_rs_frame.get_distance(pixel[0], pixel[1])
    except (TypeError, ValueError, RuntimeError):
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'valid': False}
    
    # WHY CHECK FOR 0.0?
    # Depth cameras cannot see transparent objects, shiny objects, or the inside of 
    # holes (like a donut). They return 0.0 for these. We must flag this as 'invalid' 
    # so the LLM knows it guessed wrong and needs to pick a new pixel.
    if z_raw <= 0.0:
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'valid': False}

    # The magic math: 2D pixel + Depth distance -> 3D vector
    rs = _load_rs()
    xyz = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, z_raw)
    
    return {
        'x': xyz[0],
        'y': xyz[1], 
        'z': xyz[2],
        'valid': True
    }
