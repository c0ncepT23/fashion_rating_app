# app/models/item_detector.py
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import asyncio
import torch
import os
import cv2

from app.config import settings
from app.models.fashion_clip_classifier import FashionCLIPClassifier
from segment_anything import SamPredictor

class ItemDetector:
    """
    SAM+CLIP detector for clothing items in images (no YOLOv8)
    """
    
    def __init__(self, fashion_clip=None, sam_predictor=None):
        """
        Initialize the clothing item detector with SAM and FashionCLIP
        
        Args:
            fashion_clip: Optional existing FashionCLIP instance
            sam_predictor: Optional existing SAM predictor instance
        """
        # Use provided FashionCLIP instance or create a new one
        self.fashion_clip = fashion_clip or FashionCLIPClassifier(
            weights_path=os.path.join(settings.MODEL_DIR, 'openfashionclip.pt')
        )
        
        # Store SAM predictor
        self.sam_predictor = sam_predictor
        
        # Define predefined clothing categories
        self.category_mapping = {
            # Tops
            "shirt": "top",
            "blouse": "top",
            "t-shirt": "top", 
            "top": "top",
            "tank top": "top",
            "crop top": "top",
            "hoodie": "top",
            "sweater": "top",
            "cardigan": "top",
            "tank": "top",
            "tee": "top",
            "tunic": "top",
            
            # Bottoms
            "jeans": "bottom",
            "pants": "bottom",
            "trousers": "bottom",
            "slacks": "bottom", 
            "shorts": "bottom",
            "skirt": "bottom",
            "mini skirt": "bottom",
            "midi skirt": "bottom",
            "maxi skirt": "bottom",
            "leggings": "bottom",
            
            # Dresses
            "dress": "dress",
            "mini dress": "dress",
            "midi dress": "dress", 
            "maxi dress": "dress",
            "gown": "dress",
            "sun dress": "dress",
            "slip dress": "dress",
            
            # Outerwear
            "jacket": "outerwear",
            "blazer": "outerwear",
            "coat": "outerwear",
            
            # Others (add more as needed)
            "jumpsuit": "jumpsuit",
            "romper": "jumpsuit"
        }
        
    def apply_mask(self, image, mask):
        """Apply a binary mask to an image"""
        return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8)*255)
        
    async def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect clothing items using SAM for segmentation and CLIP for classification
        
        Args:
            image: PIL Image object
            
        Returns:
            list: Detected clothing items with metadata
        """
        # Convert PIL image to OpenCV format for SAM
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
        
        # Placeholder for async behavior compatibility
        await asyncio.sleep(0)
        
        # Check if SAM predictor is available
        if self.sam_predictor is None:
            # Run CLIP on the whole image as fallback
            return self._analyze_whole_image(image)
        
        # Set the image in SAM predictor
        self.sam_predictor.set_image(cv_image)
        
        # Use SAM to segment the person in the image
        # First, we'll try automatic segmentation with a center point
        h, w = cv_image.shape[:2]
        center_point = np.array([[w//2, h//2]])  # Center of the image
        input_label = np.array([1])  # Foreground
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=center_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Take the highest scoring mask
        best_mask_idx = np.argmax(scores)
        person_mask = masks[best_mask_idx]
        
        # Apply the mask to get just the person
        masked_image = self.apply_mask(cv_image, person_mask)
        
        # Convert back to PIL for CLIP
        masked_pil = Image.fromarray(masked_image[:, :, ::-1])  # BGR to RGB
        
        # Now segment into body regions (upper, lower, full)
        upper_mask = np.zeros_like(person_mask)
        lower_mask = np.zeros_like(person_mask)
        
        # Simple division by height for upper and lower body
        upper_mask = person_mask.copy()
        upper_mask[h//2:, :] = 0  # Keep only top half
        
        lower_mask = person_mask.copy()
        lower_mask[:h//2, :] = 0  # Keep only bottom half
        
        # Create masked images for each region
        upper_masked = self.apply_mask(cv_image, upper_mask)
        lower_masked = self.apply_mask(cv_image, lower_mask)
        
        # Convert to PIL
        upper_pil = Image.fromarray(upper_masked[:, :, ::-1])
        lower_pil = Image.fromarray(lower_masked[:, :, ::-1])
        
        # Analyze each region with CLIP
        full_result = self.fashion_clip.classify_garment(masked_pil)
        upper_result = self.fashion_clip.classify_garment(upper_pil)
        lower_result = self.fashion_clip.classify_garment(lower_pil)
        
        # Check if full body detection is a dress or jumpsuit
        full_type = full_result["garment"]["best_match"]["type"]
        full_confidence = full_result["garment"]["best_match"]["confidence"]
        
        is_dress = any(dress_term in full_type.lower() for dress_term in ["dress", "gown", "tunic"])
        is_jumpsuit = any(jump_term in full_type.lower() for jump_term in ["jumpsuit", "romper", "overall"])
        
        clothing_items = []
        
        # If we detected a dress or jumpsuit with good confidence, prioritize that
        if (is_dress or is_jumpsuit) and full_confidence > 0.45:
            # Create dress/jumpsuit item
            category = "dress" if is_dress else "jumpsuit"
            
            # Calculate box coordinates from mask
            y_indices, x_indices = np.where(person_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x1, x2 = np.min(x_indices), np.max(x_indices)
                y1, y2 = np.min(y_indices), np.max(y_indices)
                
                # Convert to relative positions
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                area = width * height
            else:
                # Default position if mask is empty
                center_x, center_y = 0.5, 0.5
                width, height = 0.6, 0.8
                area = width * height
            
            # Create the item
            item = {
                "type": full_type,
                "category": category,
                "confidence": float(full_confidence),
                "position": {
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "width": float(width),
                    "height": float(height),
                    "area": float(area)
                },
                "clip_analysis": full_result,
                "fit": self._determine_fit(full_type),
                "sam_mask": person_mask  # Save mask for color analysis
            }
            
            clothing_items.append(item)
        else:
            # Add top from upper body analysis
            upper_type = upper_result["garment"]["best_match"]["type"]
            upper_confidence = upper_result["garment"]["best_match"]["confidence"]
            
            if upper_confidence > 0.3:
                # Calculate box coordinates from upper mask
                y_indices, x_indices = np.where(upper_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, x2 = np.min(x_indices), np.max(x_indices)
                    y1, y2 = np.min(y_indices), np.max(y_indices)
                    
                    # Convert to relative positions
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    area = width * height
                else:
                    # Default position if mask is empty
                    center_x, center_y = 0.5, 0.3
                    width, height = 0.6, 0.3
                    area = width * height
                
                # Create the top item
                category = self._determine_category(upper_type)
                item = {
                    "type": upper_type,
                    "category": category,
                    "confidence": float(upper_confidence),
                    "position": {
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "width": float(width),
                        "height": float(height),
                        "area": float(area)
                    },
                    "clip_analysis": upper_result,
                    "fit": self._determine_fit(upper_type),
                    "sam_mask": upper_mask  # Save mask for color analysis
                }
                
                clothing_items.append(item)
            
            # Add bottom from lower body analysis
            lower_type = lower_result["garment"]["best_match"]["type"]
            lower_confidence = lower_result["garment"]["best_match"]["confidence"]
            
            if lower_confidence > 0.3:
                # Calculate box coordinates from lower mask
                y_indices, x_indices = np.where(lower_mask)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, x2 = np.min(x_indices), np.max(x_indices)
                    y1, y2 = np.min(y_indices), np.max(y_indices)
                    
                    # Convert to relative positions
                    center_x = (x1 + x2) / 2 / w
                    center_y = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    area = width * height
                else:
                    # Default position if mask is empty
                    center_x, center_y = 0.5, 0.7
                    width, height = 0.5, 0.4
                    area = width * height
                
                # Create the bottom item
                category = self._determine_category(lower_type)
                item = {
                    "type": lower_type,
                    "category": category,
                    "confidence": float(lower_confidence),
                    "position": {
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "width": float(width),
                        "height": float(height),
                        "area": float(area)
                    },
                    "clip_analysis": lower_result,
                    "fit": self._determine_fit(lower_type),
                    "sam_mask": lower_mask  # Save mask for color analysis
                }
                
                clothing_items.append(item)
        
        # If we failed to detect any clothing items, fall back to analyzing the whole image
        if not clothing_items:
            clothing_items = self._analyze_whole_image(image)
        
        return clothing_items
    
    def _analyze_whole_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Analyze whole image as fallback when SAM segmentation fails"""
        # Run CLIP on the whole image
        clip_result = self.fashion_clip.classify_garment(image)
        
        # Extract garment type and confidence
        garment_type = clip_result["garment"]["best_match"]["type"]
        confidence = clip_result["garment"]["best_match"]["confidence"]
        
        # Determine garment category based on type
        category = self._determine_category(garment_type)
        
        # Create item dictionary
        item = {
            "type": garment_type,
            "category": category,
            "confidence": float(confidence),
            "position": {
                "center_x": 0.5,
                "center_y": 0.5,
                "width": 0.6,
                "height": 0.7,
                "area": 0.42
            },
            "clip_analysis": clip_result,
            "fit": self._determine_fit(garment_type)
        }
        
        return [item]
    
    def _determine_category(self, garment_type: str) -> str:
        """Determine the category of a garment type"""
        # Try direct mapping first
        if garment_type in self.category_mapping:
            return self.category_mapping[garment_type]
        
        # Check for partial matches
        for key, category in self.category_mapping.items():
            if key in garment_type.lower():
                return category
        
        # Default to "unknown" if no match found
        return "unknown"
    
    def _determine_fit(self, garment_type: str) -> str:
        """Determine the fit type based on garment type"""
        garment_lower = garment_type.lower()
        
        # Check for specific fits from garment type
        if "wide" in garment_lower or "flowy" in garment_lower:
            return "loose_flowy"
        elif "skinny" in garment_lower or "slim" in garment_lower or "fitted" in garment_lower:
            return "fitted_slim" 
        elif "baggy" in garment_lower or "oversized" in garment_lower:
            return "loose_flowy"
        elif "structured" in garment_lower or "tailored" in garment_lower:
            return "structured_contemporary"
        
        # Default fits based on category
        category = self._determine_category(garment_type)
        if category == "dress":
            return "elegant_formal" if "gown" in garment_lower or "formal" in garment_lower else "relaxed_casual"
        elif category == "outerwear":
            return "structured_contemporary"
        elif category == "top":
            return "relaxed_casual"
        elif category == "bottom":
            if "skirt" in garment_lower:
                return "loose_flowy" if "maxi" in garment_lower else "balanced_silhouette"
            else:
                return "balanced_silhouette"
        
        # Default
        return "balanced_silhouette"
    
    def determine_top_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the top type from detected items
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Description of top type
        """
        # Filter for top and outerwear items
        top_items = [item for item in clothing_items if item["category"] in ["top", "outerwear"]]
        
        # Also check for dresses
        dress_items = [item for item in clothing_items if item["category"] == "dress"]
        if dress_items:
            # Prioritize dress over top
            return dress_items[0]["type"].replace("_", " ")
        
        if not top_items:
            return "unknown_top"
        
        # If there are multiple tops (layering), describe the combination
        if len(top_items) > 1:
            # Sort by confidence
            top_items.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Get the top two items
            primary = top_items[0]["type"].replace("_", " ")
            secondary = top_items[1]["type"].replace("_", " ")
            
            return f"{primary}_with_{secondary}"
        else:
            # Single top item
            return top_items[0]["type"].replace("_", " ")
    
    def determine_pants_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the pants/bottoms type
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Pants/bottoms type description
        """
        # Check for dresses first
        dress_items = [item for item in clothing_items if item["category"] == "dress"]
        if dress_items:
            # No separate bottom for dresses
            return "none"
            
        # Filter for bottom items
        bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
        
        if not bottom_items:
            return "unknown_bottom"
        
        # Sort by confidence and take the highest
        bottom = max(bottom_items, key=lambda x: x["confidence"])
        
        # Format the bottom type
        bottom_type = bottom["type"].replace("_", " ")
        
        # Add fit information if available
        if "fit" in bottom:
            if bottom["fit"] == "loose_flowy":
                if "jean" in bottom_type:
                    return "wide_leg_jeans"
                elif "trouser" in bottom_type or "pant" in bottom_type:
                    return "wide_leg_trousers"
                elif "skirt" in bottom_type:
                    return "flowy_skirt"
            elif bottom["fit"] == "fitted_slim":
                if "jean" in bottom_type:
                    return "skinny_jeans"
                elif "trouser" in bottom_type or "pant" in bottom_type:
                    return "slim_trousers"
                elif "skirt" in bottom_type:
                    return "fitted_skirt"
        
        # Return the type as is if no special handling
        return bottom_type
    
    def determine_fit_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the overall fit type based on detected items
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Fit type description
        """
        if not clothing_items:
            return "balanced_silhouette"  # Default when no items detected
        
        # Count fit types
        fit_counts = {}
        for item in clothing_items:
            fit = item.get("fit", "balanced_silhouette")
            fit_counts[fit] = fit_counts.get(fit, 0) + 1
        
        # Find most common fit
        most_common_fit = max(fit_counts, key=fit_counts.get) if fit_counts else "balanced_silhouette"
        
        return most_common_fit
    
    def determine_footwear_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the footwear type
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Footwear type description
        """
        # This can be enhanced with SAM segmentation to detect shoes
        # For now, returning placeholder
        return "footwear_not_visible"
    
    def determine_accessories(self, clothing_items: List[Dict]) -> List[str]:
        """
        Determine accessories
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            list: List of accessory descriptions
        """
        # This can be enhanced with SAM segmentation to detect accessories
        # For now, returning empty list
        return []