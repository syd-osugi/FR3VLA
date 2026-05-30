# Image encoding utilities (reusable)

"""
Image Encoding Utilities
------------------------
Functions for converting camera images to formats the LLM can understand.

WHY A SEPARATE FILE?
====================
Image encoding is a utility that could be used by:
- The tool handlers (sending images to LLM)
- A debug UI (showing images to human operators)
- A logging system (saving images with timestamps)

By separating it, we avoid duplicating this code.

THE ENCODING PIPELINE:
======================
Camera Output (BGR numpy) 
    → Convert to RGB (PIL expects RGB, OpenCV uses BGR)
    → Create PIL Image
    → Encode to JPEG (compress to reduce size)
    → Convert to base64 string
    → Wrap in data URL format
    → Return as LLM message dict

WHY JPEG AND NOT PNG?
=====================
JPEG at quality 75 is typically 10-50x smaller than PNG.
The LLM doesn't need pixel-perfect images - it just needs to see shapes and colors.
Smaller images = faster API responses = lower costs.
"""

import base64
import config as cfg


def _load_cv2():
    """
    Lazy-loads OpenCV.
    
    WHY LAZY LOADING?
    ==================
    If someone imports this file on a machine without OpenCV installed,
    we don't want the import to crash immediately. The crash should only
    happen when someone actually tries to encode an image.
    
    Returns:
        cv2 module
    """
    try:
        import cv2
        return cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "opencv-python is required for image encoding. "
            "Install with: pip install opencv-python"
        ) from exc


def _load_pil():
    """
    Lazy-loads Pillow (PIL).
    
    Returns:
        PIL.Image module
    """
    try:
        from PIL import Image
        return Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for image encoding. "
            "Install with: pip install Pillow"
        ) from exc


def encode_image_for_llm(bgr_image, quality=None):
    """
    Converts an OpenCV BGR image into a base64 string for the LLM API.
    
    THE DATA FLOW:
    ===============
    1. Input: numpy array in BGR format (OpenCV's default)
    2. Convert BGR → RGB (PIL expects RGB)
    3. Create PIL Image from numpy array
    4. Save to in-memory buffer as JPEG
    5. Encode buffer contents as base64 string
    6. Wrap in the data URL format the LLM API expects
    
    WHY BASE64?
    ===========
    The LLM API is text-based (JSON). We can't send binary image data.
    Base64 converts binary data to a text-safe alphabet (A-Z, a-z, 0-9, +, /).
    This increases size by ~33% but allows sending images through text APIs.
    
    Args:
        bgr_image: OpenCV image as numpy array (shape: HxWx3, dtype: uint8)
        quality: JPEG quality (1-100). If None, uses config.LLM_IMAGE_JPEG_QUALITY
        
    Returns:
        dict: Message dict in the format expected by OpenAI-compatible APIs:
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,<base64_string>"}
                    }
                ]
            }
            
    Raises:
        RuntimeError: If cv2 or PIL not installed
        ValueError: If image is invalid
    """
    # Load dependencies (lazy)
    cv2 = _load_cv2()
    Image = _load_pil()
    
    # Use configured quality if not specified
    if quality is None:
        quality = cfg.LLM_IMAGE_JPEG_QUALITY
    
    # Validate input
    if bgr_image is None:
        raise ValueError("Image is None")
    
    if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
        raise ValueError(f"Expected BGR image (HxWx3), got shape {bgr_image.shape}")
    
    # Step 1: Convert BGR to RGB
    # OpenCV uses BGR order, PIL uses RGB order
    # If we skip this, red and blue channels will be swapped!
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # Step 2: Create PIL Image from numpy array
    pil_image = Image.fromarray(rgb_image)
    
    # Step 3: Encode to JPEG in memory
    # We use BytesIO instead of a file to avoid disk I/O
    # JPEG quality affects file size dramatically:
    #   Quality 95: ~500KB for 640x480
    #   Quality 75: ~50KB for 640x480
    #   Quality 30: ~15KB for 640x480
    from io import BytesIO
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=quality)
    
    # Step 4: Convert to base64 string
    # .getvalue() returns the raw JPEG bytes
    # base64.b64encode() returns bytes, so we decode to string
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Step 5: Wrap in the LLM API format
    # The "data:image/jpeg;base64," prefix tells the API this is a JPEG image
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_str}"
                }
            }
        ]
    }


def get_image_size_kb(bgr_image, quality=None):
    """
    Calculates what the encoded image size would be in KB.
    
    Useful for debugging - if images are too large, you know to lower quality.
    
    Args:
        bgr_image: OpenCV BGR image
        quality: JPEG quality (1-100)
        
    Returns:
        float: Approximate size in kilobytes
    """
    from io import BytesIO

    cv2 = _load_cv2()
    Image = _load_pil()
    
    if quality is None:
        quality = cfg.LLM_IMAGE_JPEG_QUALITY
    
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    
    return len(buffer.getvalue()) / 1024  # Bytes to KB
