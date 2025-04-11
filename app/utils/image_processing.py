# app/utils/image_processing.py
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format (BGR)
    
    Args:
        pil_image: PIL Image
        
    Returns:
        numpy.ndarray: OpenCV image (BGR)
    """
    # Convert PIL image to RGB numpy array
    rgb_image = np.array(pil_image.convert('RGB'))
    
    # Convert RGB to BGR (OpenCV format)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    return bgr_image

def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image to PIL Image
    
    Args:
        cv_image: OpenCV image (BGR)
        
    Returns:
        PIL.Image: PIL Image (RGB)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def resize_with_aspect_ratio(
    image: np.ndarray, 
    width: Optional[int] = None, 
    height: Optional[int] = None, 
    max_size: Optional[int] = None
) -> np.ndarray:
    """
    Resize image maintaining aspect ratio
    
    Args:
        image: Input image (OpenCV format)
        width: Target width
        height: Target height
        max_size: Maximum dimension
        
    Returns:
        numpy.ndarray: Resized image
    """
    h, w = image.shape[:2]
    
    if max_size:
        # Calculate the largest dimension
        max_dim = max(h, w)
        
        # Calculate scaling factor
        if max_dim > max_size:
            scale = max_size / max_dim
            width = int(w * scale)
            height = int(h * scale)
        else:
            return image
    
    # If only one dimension is specified, calculate the other 
    # to maintain aspect ratio
    if width is None and height is not None:
        width = int(w * height / h)
    elif height is None and width is not None:
        height = int(h * width / w)
    elif width is None and height is None:
        return image
    
    # Resize the image
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized

def extract_masks_from_detections(
    image: np.ndarray, 
    detections: List[Dict[str, Any]]
) -> List[np.ndarray]:
    """
    Extract binary masks from bounding box detections
    
    Args:
        image: Input image (OpenCV format)
        detections: List of detection dictionaries with 'box' key
        
    Returns:
        list: List of binary masks for each detection
    """
    h, w = image.shape[:2]
    masks = []
    
    for det in detections:
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Get bounding box
        x1, y1, x2, y2 = det["box"]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Fill bounding box region with 255 (white)
        mask[y1:y2, x1:x2] = 255
        
        masks.append(mask)
    
    return masks

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an image
    
    Args:
        image: Input image (OpenCV format)
        mask: Binary mask
        
    Returns:
        numpy.ndarray: Masked image
    """
    # Ensure mask is binary
    mask_binary = mask.astype(bool)
    
    # Create output image (all black)
    masked_image = np.zeros_like(image)
    
    # Apply mask
    if len(image.shape) == 3:  # Color image
        for c in range(image.shape[2]):
            masked_image[:, :, c] = image[:, :, c] * mask_binary
    else:  # Grayscale image
        masked_image = image * mask_binary
    
    return masked_image

def extract_dominant_colors(
    image: np.ndarray, 
    mask: Optional[np.ndarray] = None, 
    n_colors: int = 5
) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from an image or masked region
    
    Args:
        image: Input image (OpenCV format)
        mask: Optional binary mask
        n_colors: Number of dominant colors to extract
        
    Returns:
        list: List of (B, G, R) color tuples
    """
    # Apply mask if provided
    if mask is not None:
        image = apply_mask_to_image(image, mask)
    
    # Reshape image to a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Remove black pixels (from mask background)
    non_black_pixels = pixels[np.sum(pixels, axis=1) > 0]
    
    # If no valid pixels, return empty list
    if len(non_black_pixels) == 0:
        return []
    
    # Use K-means to find dominant colors
    from sklearn.cluster import KMeans
    
    # Limit clusters to the number of non-black pixels
    k = min(n_colors, len(non_black_pixels))
    
    if k == 0:
        return []
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(non_black_pixels)
    
    # Get colors and convert to integer tuples
    colors = [(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
    
    return colors