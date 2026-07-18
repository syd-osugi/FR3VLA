"""
RealSense Camera Hardware Wrapper
---------------------------------
Owns live D435/D405 camera streams during runtime.

Each RealSense object starts RGB and depth streams, aligns depth to RGB, keeps
the newest frames updated in a background thread, and can return timestamp-
synchronized frame pairs for LLM image and depth tools.
"""

import threading
import time

import config as cfg

def _load_realsense_dependencies():
    """
    LAZY LOADING PATTERN:
    We don't import pyrealsense2 at the top of the file. We import it inside this function.
    Why? If someone imports this file on a laptop that doesn't have a RealSense camera 
    installed, the whole program would instantly crash with "ModuleNotFoundError".
    By loading it here, the crash only happens if they actually try to start the camera.
    """
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for RealSense frame conversion.") from exc

    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyrealsense2 is required to use Intel RealSense cameras.") from exc

    return np, rs

class RealSense:
    def __init__(self, serial_number=None, resolution=None, fps=None):
        # Load the heavy C++ bindings only when we actually instantiate the camera
        self._np, self._rs = _load_realsense_dependencies()
        if resolution is None:
            resolution = cfg.D435_RESOLUTION
        if fps is None:
            fps = cfg.CAMERA_FPS
        self.resolution = resolution
        self._pipeline_started = False
        
        # WHY A LOCK? 
        # A background thread constantly pulls frames from the camera. If your main code 
        # tries to read the image at the exact millisecond the background thread is writing 
        # a new image, you get a torn/glitched image (half old frame, half new frame). 
        # The lock acts as a traffic light to prevent this.
        self._lock = threading.Lock()
        
        # These hold the actual image data
        self._rgb_frame = None
        self._depth_frame = None
        self._depth_rs_frame = None
        self._timestamp_ms = None
        self.running = False

        self.pipeline = self._rs.pipeline()
        config = self._rs.config()

        # WHY SERIAL NUMBERS?
        # If you have a D435 and a D405 plugged in, Linux sees them as identical generic 
        # USB devices. If you don't force the serial number, the code might open the D435 
        # twice and completely ignore the D405.
        if serial_number:
            config.enable_device(serial_number)

        # Tell the camera to output specific streams at specific sizes
        config.enable_stream(
            self._rs.stream.color, resolution[0], resolution[1], self._rs.format.bgr8, fps
        )
        config.enable_stream(
            self._rs.stream.depth, resolution[0], resolution[1], self._rs.format.z16, fps
        )

        self.profile = self.pipeline.start(config)
        self._pipeline_started = True
        
        # WHY DEPTH SCALE?
        # The camera outputs raw integers (e.g., a pixel reads '1250'). This number is 
        # meaningless until you multiply it by the depth scale (e.g., 0.001) to get 
        # real-world meters (1.25 meters).
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        
        # WHY ALIGNMENT? (CRITICAL FOR LLM PIXEL COORDINATES)
        # The RGB lens and the Depth IR sensor are physically spaced about 2cm apart 
        # on the camera hardware. Because of this, pixel (100,100) in the RGB image does 
        # NOT point to the same physical spot as pixel (100,100) in the Depth image.
        # If the LLM guesses a cup is at pixel (100,100) in RGB, and we look up (100,100) 
        # in unaligned depth, we get the depth of empty space next to the cup.
        # This mathematically warps the depth image to perfectly overlay the RGB image.
        self.align = self._rs.align(self._rs.stream.color)
        
        # Pre-allocate the pointcloud object. Creating a new one every time we need 3D 
        # math is very slow, so we create it once here and reuse it.
        self.pc = self._rs.pointcloud()

        # Start pulling frames in the background so the main robot code never freezes
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        """This runs continuously in the background thread."""
        while self.running:
            try:
                # Wait for the camera hardware to send a new set of frames (blocks this thread)
                frames = self.pipeline.wait_for_frames(timeout_ms=cfg.CAMERA_FRAME_TIMEOUT_MS)
            except Exception:
                if not self.running:
                    break
                continue

            # Apply the alignment math we configured above
            timestamp_ms = frames.get_timestamp()
            frames = self.align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # If the camera glitched and dropped a frame, skip this loop
            if not depth_frame or not color_frame:
                continue

            with self._lock:
                # WHY .copy()?
                # pyrealsense2 gives us a pointer to the camera's internal memory buffer.
                # If we don't copy it, the next time the camera captures a frame, it will 
                # overwrite the data, and our main code will see the image morphing/glitching.
                self._rgb_frame = self._np.asanyarray(color_frame.get_data()).copy()
                self._depth_frame = self._np.asanyarray(depth_frame.get_data()).copy()
                
                # WHY .keep()?
                # When you extract a numpy array from a depth frame, the camera assumes you 
                # are done with it and schedules the underlying C++ memory for deletion.
                # However, our tools.py needs to query this exact depth frame LATER to get 
                # the 3D coordinates of a pixel. .keep() tells the camera "Do not delete 
                # this memory yet, I am going to need it later."
                depth_frame.keep()
                self._depth_rs_frame = depth_frame
                self._timestamp_ms = timestamp_ms

    def get_frames(self):
        """
        Safe getter function. Returns a tuple of the RGB image, raw depth image, and the 
        raw RealSense depth object needed for 3D math.
        """
        with self._lock:
            rgb = self._rgb_frame.copy() if self._rgb_frame is not None else None
            depth = self._depth_frame.copy() if self._depth_frame is not None else None
            # We pass the reference to depth_rs, not a copy, because .keep() made it safe
            depth_rs = self._depth_rs_frame
        return rgb, depth, depth_rs

    def _get_snapshot_with_timestamp(self):
        with self._lock:
            if self._rgb_frame is None or self._depth_frame is None or self._depth_rs_frame is None:
                return None
            return (
                self._rgb_frame.copy(),
                self._depth_frame.copy(),
                self._depth_rs_frame,
                self._timestamp_ms,
            )

    def stop(self):
        """Safely shuts down the hardware thread and releases the USB camera."""
        self.running = False
        if hasattr(self, "_capture_thread") and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._pipeline_started:
            self.pipeline.stop()
            self._pipeline_started = False

    def grab_synced_snapshot(self, other_camera, max_delta_ms=None):
        """
        SOFTWARE TRIGGER:
        Used when the LLM requests frames. Instead of taking whatever is in the background 
        buffer, we actively wait for the next available frame on both cameras, check their 
        internal hardware timestamps, and only accept them if they happened within 
        `max_delta_ms` of each other. 
        
        Args:
            other_camera: The second RealSense object (e.g., passing d405 if this is d435).
            max_delta_ms: Maximum allowed timestamp difference in milliseconds.
            
        Returns:
            tuple: (self_rgb, self_depth, self_rs, other_rgb, other_depth, other_rs)
                   Returns None if it failed to sync within a few tries.
        """
        if max_delta_ms is None:
            max_delta_ms = cfg.CAMERA_SYNC_TOLERANCE_MS

        tries = 0
        max_tries = cfg.CAMERA_SYNC_MAX_TRIES
        
        while tries < max_tries:
            tries += 1

            self_snapshot = self._get_snapshot_with_timestamp()
            other_snapshot = other_camera._get_snapshot_with_timestamp()

            if self_snapshot is None or other_snapshot is None:
                time.sleep(cfg.CAMERA_SYNC_RETRY_SLEEP_S)
                continue

            rgb_self, depth_self_arr, depth_self, time_self = self_snapshot
            rgb_other, depth_other_arr, depth_other, time_other = other_snapshot

            if time_self is None or time_other is None:
                time.sleep(cfg.CAMERA_SYNC_RETRY_SLEEP_S)
                continue

            delta = abs(time_self - time_other)

            if delta <= max_delta_ms:
                return (
                    rgb_self, depth_self_arr, depth_self,
                    rgb_other, depth_other_arr, depth_other,
                )

            time.sleep(cfg.CAMERA_SYNC_RETRY_SLEEP_S)

        return None
