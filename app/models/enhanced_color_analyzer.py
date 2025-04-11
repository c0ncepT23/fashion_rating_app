# app/models/enhanced_color_analyzer.py
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
from typing import List, Dict, Tuple, Any
import colorsys
import math
import asyncio

class EnhancedColorAnalyzer:
    """
    Enhanced color analyzer using DeepLabV3+ for segmentation and 
    improved color analysis
    """
    
    def __init__(self):
        # Load segmentation model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seg_model = deeplabv3_resnet101(pretrained=True).to(self.device)
        self.seg_model.eval()
        
        # Create transform for segmentation input
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Clothing class IDs in COCO dataset
        # 15: person, 18: dog, etc. - we'll filter to keep only relevant classes
        self.clothing_class_ids = [15]  # Person class, we'll refine this later
        
        # Load color name database
        self.color_names = self._load_color_names()
    
    def _load_color_names(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Load color name dictionary with improved color recognition
        """
        color_dict = {
            # Basic colors
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'pink': (255, 192, 203),
            'crimson': (220, 20, 60),
            'maroon': (128, 0, 0),
            'brown': (165, 42, 42),
            'orange': (255, 165, 0),
            'coral': (255, 127, 80),
            'gold': (255, 215, 0),
            'yellow': (255, 255, 0),
            'olive': (128, 128, 0),
            'lime': (0, 255, 0),
            'green': (0, 128, 0),
            'aqua': (0, 255, 255),
            'teal': (0, 128, 128),
            'blue': (0, 0, 255),
            'navy': (0, 0, 128),
            'purple': (128, 0, 128),
            'magenta': (255, 0, 255),
            'gray': (128, 128, 128),
            'silver': (192, 192, 192),
            
            # Extended fashion colors
            'beige': (245, 245, 220),
            'burgundy': (128, 0, 32),
            'lavender': (230, 230, 250),
            'mint': (189, 252, 201),
            'salmon': (250, 128, 114),
            'turquoise': (64, 224, 208),
            'indigo': (75, 0, 130),
            'forest_green': (34, 139, 34),
            'khaki': (240, 230, 140),
            'plum': (221, 160, 221),
            'sky_blue': (135, 206, 235),
            'pale_yellow': (255, 255, 224),
            'cream': (255, 253, 208),
            'ruby_red': (224, 17, 95),
            'tomato_red': (255, 99, 71),
            'emerald': (80, 200, 120),
            'sapphire': (15, 82, 186),
            'slate': (112, 128, 144),
            'charcoal': (54, 69, 79)
        }
        return color_dict

    async def analyze_outfit_colors(self, image: Image.Image, clothing_items: List[Dict]) -> Dict[str, Any]:
        """
        Analyze colors using improved segmentation and color detection
        
        Args:
            image: PIL Image
            clothing_items: Detected clothing items
            
        Returns:
            dict: Color analysis results
        """
        # Convert to tensor for segmentation
        img_tensor = self._preprocess_image(image)
        
        # Perform segmentation to isolate clothing
        segmentation_mask = self._segment_image(img_tensor)
        
        # Convert PIL image to cv2
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
        
        # Extract colors using segmentation mask
        colors = self._extract_colors_from_mask(cv_image, segmentation_mask)
        
        # Get named colors
        named_colors = []
        for color in colors:
            color_name = self.closest_color_name(color)
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])  # BGR to RGB
            named_colors.append({
                'name': color_name,
                'rgb': (color[2], color[1], color[0]),  # BGR to RGB
                'hex': hex_color
            })
        
        # Analyze color harmony
        harmony_result = self._analyze_color_harmony(
            [(c['rgb'][0], c['rgb'][1], c['rgb'][2]) for c in named_colors]
        )
        
        # Determine seasonal palette
        seasonal_palette = self._detect_seasonal_palette([c['name'] for c in named_colors])
        
        # Apply the harmony and seasonal info to the first color
        if named_colors:
            named_colors[0]['harmony_type'] = harmony_result['type']
            named_colors[0]['harmony_score'] = harmony_result['score']
            named_colors[0]['seasonal_palette'] = seasonal_palette
        
        # Create result
        result = {
            'color_palette': named_colors,
            'harmony': harmony_result,
            'seasonal_palette': seasonal_palette,
            'item_colors': {}
        }
        
        # Add individual item colors
        for i, item in enumerate(clothing_items):
            # Create a mask for this item
            item_mask = np.zeros(segmentation_mask.shape, dtype=np.uint8)
            x1, y1, x2, y2 = item['box']
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(segmentation_mask.shape[1], x2), min(segmentation_mask.shape[0], y2)
            item_mask[y1:y2, x1:x2] = 1
            
            # Combine with segmentation mask
            combined_mask = item_mask & (segmentation_mask > 0)
            
            # Extract colors for this item
            item_colors = self._extract_colors_from_mask(cv_image, combined_mask, n_colors=3)
            
            # Convert to named colors
            item_named_colors = []
            for color in item_colors:
                color_name = self.closest_color_name(color)
                hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
                item_named_colors.append({
                    'name': color_name,
                    'rgb': (color[2], color[1], color[0]),
                    'hex': hex_color
                })
            
            # Store in result and update the item
            result['item_colors'][i] = item_named_colors
            
            # Update the item's dominant color
            if item_named_colors:
                item['dominant_color'] = item_named_colors[0]['name']
                item['colors'] = item_named_colors
        
        return result
    
    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Preprocess image for segmentation model"""
        # Resize for model
        img = pil_image.resize((513, 513))
        
        # Transform
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor
    
    def _segment_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Perform segmentation to isolate clothing"""
        with torch.no_grad():
            output = self.seg_model(img_tensor)['out'][0]
        
        # Get segmentation mask
        output_predictions = output.argmax(0).cpu().numpy()
        
        # Resize to original size if needed
        if output_predictions.shape[0] != img_tensor.shape[2] or output_predictions.shape[1] != img_tensor.shape[3]:
            output_predictions = cv2.resize(
                output_predictions, 
                (img_tensor.shape[3], img_tensor.shape[2]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Process to keep only clothing classes (person = 15 in COCO, we'll refine this)
        clothing_mask = np.zeros_like(output_predictions)
        for class_id in self.clothing_class_ids:
            clothing_mask = clothing_mask | (output_predictions == class_id)
        
        return clothing_mask
    
    def _extract_colors_from_mask(self, image: np.ndarray, mask: np.ndarray, n_colors=8) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from masked region"""
        # Apply mask
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        
        # Get non-zero pixels
        non_zero = masked_image[mask > 0]
        
        if len(non_zero) == 0:
            return []
        
        # Use K-means to find dominant colors
        non_zero = non_zero.reshape(-1, 3).astype(np.float32)
        
        # Limit number of colors based on available pixels
        k = min(n_colors, len(non_zero))
        if k == 0:
            return []
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(non_zero, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count occurrences of each label
        counts = np.bincount(labels.flatten())
        
        # Sort centers by count
        sorted_indices = np.argsort(counts)[::-1]
        centers = centers[sorted_indices]
        
        # Convert to integer tuples
        colors = [(int(center[0]), int(center[1]), int(center[2])) for center in centers]
        
        return colors
    
    def closest_color_name(self, bgr_color: Tuple[int, int, int]) -> str:
        """Find closest named color"""
        # Convert BGR to RGB
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        
        min_distance = float('inf')
        closest_name = "unknown"
        
        for name, color in self.color_names.items():
            distance = self._color_distance(rgb_color, color)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        return closest_name
    
    def _color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate weighted color distance in RGB space"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        # Weighted Euclidean distance
        r_mean = (r1 + r2) / 2
        r_weight = 2 + r_mean / 256
        g_weight = 4  # Higher weight for green (human eyes are more sensitive)
        b_weight = 2
        
        return math.sqrt(
            r_weight * (r1 - r2)**2 + 
            g_weight * (g1 - g2)**2 + 
            b_weight * (b1 - b2)**2
        )
    
    def _analyze_color_harmony(self, colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Analyze color harmony"""
        if not colors:
            return {"type": "unknown", "score": 0}
        
        # Convert RGB to HSV for better color analysis
        hsv_colors = []
        for color in colors:
            r, g, b = color
            r, g, b = r/255.0, g/255.0, b/255.0
            hsv = colorsys.rgb_to_hsv(r, g, b)
            hsv_colors.append(hsv)
        
        # Extract hues
        hues = [color[0] for color in hsv_colors]
        
        # Determine harmony type and score
        harmony_type = "custom"
        harmony_score = 0
        
        # Check for monochromatic
        hue_range = max(hues) - min(hues) if hues else 0
        if hue_range < 0.05:
            harmony_type = "monochromatic"
            harmony_score = 90
        
        # Check for complementary
        elif len(hues) >= 2:
            for i in range(len(hues)):
                for j in range(i+1, len(hues)):
                    hue_diff = abs(hues[i] - hues[j])
                    hue_diff = min(hue_diff, 1 - hue_diff)  # Account for circular nature
                    if 0.45 <= hue_diff <= 0.55:
                        harmony_type = "complementary"
                        harmony_score = 85
        
        # Check for analogous
        elif hue_range < 0.25:
            harmony_type = "analogous"
            harmony_score = 80
        
        # Calculate saturation and value statistics
        if harmony_score == 0:
            saturations = [color[1] for color in hsv_colors]
            values = [color[2] for color in hsv_colors]
            
            sat_variance = np.var(saturations)
            val_variance = np.var(values)
            
            # Score based on consistency
            consistency_score = 100 - (sat_variance + val_variance) * 100
            harmony_score = max(40, min(70, consistency_score))
            
            # Determine type based on characteristics
            if np.mean(saturations) < 0.3:
                harmony_type = "neutral"
            elif np.mean(values) > 0.7:
                harmony_type = "bright"
            elif np.mean(values) < 0.3:
                harmony_type = "dark"
            else:
                harmony_type = "balanced"
        
        return {
            "type": harmony_type,
            "score": int(harmony_score)
        }
    
    def _detect_seasonal_palette(self, color_names: List[str]) -> str:
        """Determine seasonal color palette"""
        if not color_names:
            return "unknown"
        
        # Define seasonal characteristics
        seasonal_colors = {
            "spring": ["coral", "mint", "yellow", "light_blue", "peach", "pale_yellow", "bright"],
            "summer": ["pastel_blue", "lavender", "pink", "mint", "light_gray", "pale", "soft"],
            "autumn": ["olive", "rust", "brown", "burgundy", "forest_green", "mustard", "earthy"],
            "winter": ["black", "white", "navy", "red", "blue", "emerald", "crimson", "clear"]
        }
        
        # Count matches for each season
        matches = {season: 0 for season in seasonal_colors}
        
        for color in color_names:
            for season, palette in seasonal_colors.items():
                if any(keyword in color.lower() for keyword in palette):
                    matches[season] += 1
        
        # Find season with most matches
        best_season = max(matches, key=matches.get)
        
        # If no clear winner, check color properties
        if matches[best_season] <= 1:
            # Red is commonly associated with winter and autumn
            if any("red" in color.lower() for color in color_names[:3]):
                return "winter" if "black" in color_names or "white" in color_names else "autumn"
            
            # Blues and greens are common in summer and spring
            if any("blue" in color.lower() or "green" in color.lower() for color in color_names[:3]):
                return "summer" if any("pastel" in color.lower() for color in color_names) else "spring"
        
        return best_season
    
    # Legacy method for backward compatibility
    async def extract_palette(self, image: Image.Image, clothing_items: List[Dict] = None) -> List[Dict]:
        """
        Extract the dominant color palette (legacy method)
        """
        result = await self.analyze_outfit_colors(image, clothing_items or [])
        return result["color_palette"]