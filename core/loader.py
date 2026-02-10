import tifffile
import numpy as np
import cv2
from typing import Optional, Tuple

def load_tiff_image(file_path: str) -> Optional[np.ndarray]:
    """
    Loads a TIFF image from the specified path.
    
    Args:
        file_path: Absolute path to the TIFF file.
        
    Returns:
        np.ndarray: The image data as a numpy array, or None if loading failed.
    """
    try:
        # tifffile handles multi-page tiffs better than cv2
        image = tifffile.imread(file_path)
        print(f"DEBUG: Image loaded. Shape: {image.shape}, Dtype: {image.dtype}")
        return image
    except Exception as e:
        print(f"ERROR: Failed to load image {file_path}. Reason: {e}")
        return None

def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """
    Converts any image (16-bit, float, etc.) to a standard 8-bit image 
    for GUI display (0-255 range).
    """
    if image is None:
        return None

    # If already 8-bit, return as is
    if image.dtype == np.uint8:
        return image

    # Normalize 16-bit or others to 0-255
    # cv2.normalize handles min-max scaling efficiently
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)