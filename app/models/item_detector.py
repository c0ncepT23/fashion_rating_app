# app/models/item_detector.py
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import asyncio
import torch
import os
from ultralytics import YOLO
import cv2

from app.config import settings
from app.utils.constants import CLOTHING_CATEGORIES
from app.utils.image_processing import pil_to_cv2, cv2_to_pil

class ItemDetector:
    """
    Detector for clothing items in images using YOLOv8m
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the clothing item detector with YOLOv8m model
        
        Args:
            model_path: Path to the trained YOLOv8m model weights
        """
        # Load model weights
        if model_path is None:
            model_path = os.path.join(settings.MODEL_DIR, 'yolov8m_deepfashion2.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv8m model weights not found at {model_path}")
        
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Set confidence threshold
        self.conf_threshold = 0.25
        
        # Map DeepFashion2 class indices to names (adjust based on your training)
        # This mapping should match your YOLOv8m training on DeepFashion2
        self.class_mapping = {
            0: "short sleeve top",
            1: "long sleeve top",
            2: "short sleeve outwear",
            3: "long sleeve outwear",
            4: "vest",
            5: "sling",
            6: "shorts",
            7: "trousers",
            8: "skirt",
            9: "short sleeve dress",
            10: "long sleeve dress",
            11: "vest dress",
            12: "sling dress"
        }
        
        # Category mapping - maps DeepFashion2 classes to our app categories
        self.category_mapping = {
            "short sleeve top": "top",
            "long sleeve top": "top",
            "short sleeve outwear": "outerwear",
            "long sleeve outwear": "outerwear",
            "vest": "outerwear",
            "sling": "top",
            "shorts": "bottom",
            "trousers": "bottom",
            "skirt": "bottom",
            "short sleeve dress": "dress",
            "long sleeve dress": "dress",
            "vest dress": "dress",
            "sling dress": "dress"
        }
    
    async def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect clothing items in an image using YOLOv8m
        
        Args:
            image: PIL Image object
            
        Returns:
            list: Detected clothing items with metadata
        """
        # Convert PIL image to OpenCV format
        cv_image = pil_to_cv2(image)
        
        # In a real async setting, we would run inference in a thread pool
        # For demonstration, we use a simple await with a dummy coroutine
        await asyncio.sleep(0)
        
        # Run inference with YOLOv8
        results = self.model(cv_image, conf=self.conf_threshold)
        
        # Process results
        clothing_items = []
        
        # Get image dimensions
        img_height, img_width = cv_image.shape[:2]
        
        # Check if any detections exist
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            for i, detection in enumerate(results[0].boxes):
                # Extract bounding box
                box = detection.xyxy[0].cpu().numpy()  # x1, y1, x2, y2 format
                x1, y1, x2, y2 = map(int, box)
                
                # Get class id and confidence
                cls_id = int(detection.cls[0].item())
                confidence = detection.conf[0].item()
                
                # Map class id to name
                class_name = self.class_mapping.get(cls_id, f"unknown_{cls_id}")
                
                # Determine category
                category = self.category_mapping.get(class_name, "unknown")
                
                # Calculate item position in image (for spatial relationships)
                center_x = (x1 + x2) / 2 / img_width
                center_y = (y1 + y2) / 2 / img_height
                width = (x2 - x1)
                height = (y2 - y1)
                area = width * height
                
                # Create item dictionary
                item = {
                    "type": class_name,
                    "category": category,
                    "confidence": float(confidence),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "position": {
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "width": float(width / img_width),
                        "height": float(height / img_height),
                        "area": float(area / (img_width * img_height))
                    }
                }
                
                # Extract crop for further analysis
                if 0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height:
                    item_crop = cv_image[y1:y2, x1:x2]
                    # Convert back to PIL for consistent API
                    item["crop"] = cv2_to_pil(item_crop)
                
                # Add the item to our list
                clothing_items.append(item)
        
        # Add post-processing for better fashion analysis
        enhanced_items = self._enhance_detections(clothing_items, image)
        
        return enhanced_items
    
    def _enhance_detections(self, clothing_items: List[Dict], image: Image.Image) -> List[Dict]:
        """
        Enhance detections with additional fashion-specific attributes
        
        Args:
            clothing_items: List of detected clothing items
            image: Original PIL image
            
        Returns:
            list: Enhanced clothing items
        """
        # Add additional attributes for fashion analysis
        for item in clothing_items:
            # Determine fit type based on category
            if item["category"] == "top":
                if "sleeve" in item["type"] and "long" in item["type"]:
                    item["fit"] = "structured" if item["confidence"] > 0.8 else "relaxed"
                else:
                    item["fit"] = "casual"
            
            elif item["category"] == "bottom":
                if "trouser" in item["type"]:
                    # Analyze width to determine if wide-leg, skinny, etc.
                    width_ratio = item["position"]["width"] / item["position"]["height"]
                    if width_ratio > 0.6:
                        item["fit"] = "wide_leg"
                    elif width_ratio < 0.4:
                        item["fit"] = "skinny"
                    else:
                        item["fit"] = "regular"
                elif "shorts" in item["type"]:
                    item["fit"] = "relaxed"
                elif "skirt" in item["type"]:
                    item["fit"] = "flowy" if item["position"]["width"] > 0.3 else "fitted"
            
            elif item["category"] == "outerwear":
                item["fit"] = "structured" if "long" in item["type"] else "casual"
            
            elif item["category"] == "dress":
                item["fit"] = "elegant" if "long" in item["type"] else "casual"
                
            # Extract dominant color (simplified - will be replaced by color analyzer)
            item["dominant_color"] = "unknown"  # Placeholder
            
            # Mark as accessory if in certain categories
            if item["category"] == "accessory":
                item["impact_score"] = 0.8  # Placeholder impact score
        
        return clothing_items
    
    def determine_fit_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the overall fit type based on detected items
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Fit type description
        """
        if not clothing_items:
            return "unknown"
        
        # Count fit types
        fit_counts = {}
        for item in clothing_items:
            fit = item.get("fit", "unknown")
            fit_counts[fit] = fit_counts.get(fit, 0) + 1
        
        # Find most common fit
        most_common_fit = max(fit_counts, key=fit_counts.get) if fit_counts else "unknown"
        
        # Map to more descriptive fit types
        fit_mapping = {
            "structured": "structured_contemporary",
            "casual": "relaxed_casual",
            "elegant": "elegant_formal",
            "wide_leg": "loose_flowy",
            "skinny": "fitted_slim",
            "regular": "balanced_silhouette",
            "relaxed": "relaxed_casual",
            "flowy": "loose_flowy",
            "fitted": "fitted_slim"
        }
        
        return fit_mapping.get(most_common_fit, "balanced_silhouette")
    
    def determine_pants_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the pants/bottoms type
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Pants/bottoms type description
        """
        bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
        
        if not bottom_items:
            return "unknown_bottom"
        
        # Sort by confidence and take the highest
        bottom = max(bottom_items, key=lambda x: x["confidence"])
        
        # Determine pants type based on detected class
        if "trouser" in bottom["type"]:
            # Check fit attribute to determine style
            if bottom.get("fit") == "wide_leg":
                return "wide_leg_trousers"
            elif bottom.get("fit") == "skinny":
                return "skinny_trousers"
            else:
                return "straight_leg_trousers"
        elif "shorts" in bottom["type"]:
            return "casual_shorts"
        elif "skirt" in bottom["type"]:
            if bottom.get("fit") == "flowy":
                return "flowing_skirt"
            else:
                return "fitted_skirt"
        else:
            return f"{bottom['type']}"
    
    def determine_top_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the top type
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Top type description
        """
        top_items = [item for item in clothing_items if item["category"] in ["top", "outerwear"]]
        
        if not top_items:
            return "unknown_top"
        
        # If there are multiple tops (layering), describe the combination
        if len(top_items) > 1:
            # Sort by position (y-coordinate) to determine layering
            top_items.sort(key=lambda x: x["position"]["center_y"])
            
            # Get the innermost and outermost layers
            inner_layer = top_items[0]["type"].replace("_", " ")
            outer_layer = top_items[-1]["type"].replace("_", " ")
            
            return f"{inner_layer}_with_{outer_layer}"
        else:
            # Single top item
            top_type = top_items[0]["type"].replace("_", " ")
            return top_type
    
    def determine_footwear_type(self, clothing_items: List[Dict]) -> str:
        """
        Determine the footwear type
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            str: Footwear type description
        """
        # Note: DeepFashion2 doesn't have detailed footwear classes
        # We may need a specialized footwear detector or use a placeholder
        footwear_items = [item for item in clothing_items if item["category"] == "footwear"]
        
        if not footwear_items:
            return "footwear_not_visible"
        
        # Get the most confident footwear detection
        footwear = max(footwear_items, key=lambda x: x["confidence"])
        
        # Combine type with placeholder attributes (these would come from specialized models)
        footwear_styles = ["casual", "formal", "athletic", "boots", "sandals", "loafers", "heels"]
        footwear_colors = ["black", "brown", "white", "tan", "multi_colored"]
        
        # Use hash of detection box for deterministic but varying description
        box_hash = hash(str(footwear["box"]))
        style_idx = box_hash % len(footwear_styles)
        color_idx = (box_hash // 10) % len(footwear_colors)
        
        style = footwear_styles[style_idx]
        color = footwear_colors[color_idx]
        
        return f"{color}_{style}"
    
    def determine_accessories(self, clothing_items: List[Dict]) -> List[str]:
        """
        Determine the accessories
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            list: List of accessory descriptions
        """
        # DeepFashion2 doesn't have comprehensive accessory classes
        # This would be enhanced with a specialized accessory detector
        accessory_items = [item for item in clothing_items if item["category"] == "accessory"]
        
        if not accessory_items:
            return []
        
        accessories = []
        for acc in accessory_items:
            # Get basic type
            acc_type = acc["type"].replace("_", " ")
            accessories.append(acc_type)
        
        # Add placeholder accessories for demonstration
        if len(accessories) < 2 and len(clothing_items) > 3:
            possible_accessories = ["sunglasses", "watch", "necklace", "earrings", "bracelet", "belt"]
            # Use hash of all items for deterministic but varying accessories
            item_hash = hash(str([item["box"] for item in clothing_items]))
            for i in range(min(2, len(possible_accessories))):
                acc = possible_accessories[(item_hash + i) % len(possible_accessories)]
                if acc not in accessories:
                    accessories.append(acc)
        
        return accessories[:3]  # Limit to top 3