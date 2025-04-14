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
    Enhanced color analyzer using DeepLabV3+ for segmentation, 
    ResNet18 for feature extraction, and improved color analysis
    (Modified to work with CLIP-only approach without bounding boxes)
    """
    
    def __init__(self):
        # Load segmentation model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seg_model = deeplabv3_resnet101(pretrained=True).to(self.device)
        self.seg_model.eval()
        
        # Load ResNet18 for feature extraction
        self.resnet = torchvision.models.resnet18(pretrained=True).to(self.device)
        # Remove classification layer to get feature maps
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.resnet.eval()
        
        # Create transforms
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
           # Reds & Pinks
            'red': (255, 0, 0),
            'coral': (255, 127, 80),  # Key fashion color
            'tomato_red': (255, 99, 71),
            'salmon': (250, 128, 114),
            'ruby_red': (224, 17, 95),
            'burgundy': (128, 0, 32),
            'maroon': (128, 0, 0),
            'crimson': (220, 20, 60),
            'pink': (255, 192, 203),
            'hot_pink': (255, 105, 180),
            'magenta': (255, 0, 255),
            'blush_pink': (255, 228, 225),
            
            # Oranges
            'orange': (255, 165, 0),
            'peach': (255, 218, 185),
            'terracotta': (226, 114, 91),
            
            # Yellows
            'yellow': (255, 255, 0),
            'gold': (255, 215, 0),
            'mustard': (255, 219, 88),
            'cream': (255, 253, 208),
            
            # Blues
            'blue': (0, 0, 255),
            'navy': (0, 0, 128),
            'light_blue': (173, 216, 230),
            'sky_blue': (135, 206, 235),
            'royal_blue': (65, 105, 225),
            'cobalt': (0, 71, 171),
            'denim_blue': (92, 136, 218),
            
            # Greens
            'green': (0, 128, 0),
            'mint': (189, 252, 201),
            'olive': (128, 128, 0),
            'forest_green': (34, 139, 34),
            'sage_green': (186, 202, 186),
            'emerald': (0, 201, 87),
            
            # Purples
            'purple': (128, 0, 128),
            'lavender': (230, 230, 250),
            'plum': (221, 160, 221),
            'indigo': (75, 0, 130),
            
            # Neutral & Earth Tones
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'silver': (192, 192, 192),
            'beige': (245, 245, 220),
            'tan': (210, 180, 140),
            'brown': (165, 42, 42),
            'khaki': (240, 230, 140),
            'charcoal': (54, 69, 79),
            'off_white': (250, 249, 246),
            'ivory': (255, 255, 240),
            'taupe': (72, 60, 50),
            'camel': (193, 154, 107),
        }
        return color_dict

    async def analyze_outfit_colors(
            self, 
            image: Image.Image, 
            clothing_items: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
        """
        Analyze colors using improved segmentation, ResNet features, and color detection
        (Modified for CLIP-only approach)
        
        Args:
            image: PIL Image
            clothing_items: Detected clothing items
            
        Returns:
            dict: Color analysis results
        """
        # Convert to tensor for segmentation and feature extraction
        img_tensor = self._preprocess_image(image)
        
        # Convert PIL image to numpy for OpenCV operations
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
        
        # Perform segmentation to isolate clothing
        segmentation_mask = self._segment_image(img_tensor)
        
        # Get ResNet features for attention map
        attention_map = self._get_attention_map(image)
        # Resize attention map to match segmentation mask dimensions
        attention_map_resized = cv2.resize(attention_map, 
                                      (segmentation_mask.shape[1], segmentation_mask.shape[0]), 
                                      interpolation=cv2.INTER_LINEAR)
        
        # Combine segmentation mask with attention map
        combined_weight_map = segmentation_mask.astype(float) * attention_map_resized
        
        # Normalize combined weights
        if combined_weight_map.max() > 0:
            combined_weight_map = combined_weight_map / combined_weight_map.max()
        
        # Extract colors using weighted clustering
        global_colors = self._extract_weighted_colors(cv_image, combined_weight_map)
        
        # Get named colors
        global_named_colors = []
        for color in global_colors:
            color_name = self.closest_color_name(color)
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])  # BGR to RGB
            global_named_colors.append({
                'name': color_name,
                'rgb': (color[2], color[1], color[0]),  # BGR to RGB
                'hex': hex_color
            })
        
        # Analyze color harmony
        harmony_result = self._analyze_color_harmony(
            [(c['rgb'][0], c['rgb'][1], c['rgb'][2]) for c in global_named_colors]
        )
        
        # Determine seasonal palette
        seasonal_palette = self._detect_seasonal_palette([c['name'] for c in global_named_colors])
        
        # Apply the harmony and seasonal info to the first color
        if global_named_colors:
            global_named_colors[0]['harmony_type'] = harmony_result['type']
            global_named_colors[0]['harmony_score'] = harmony_result['score']
            global_named_colors[0]['seasonal_palette'] = seasonal_palette
        
        # For CLIP-only approach where we don't have precise bounding boxes,
        # we'll analyze color for each item based on category positions
        item_colors = {}
        
        for i, item in enumerate(clothing_items):
            if "position" in item:
                # Create a mask based on the item's position
                # This is an approximation since we don't have exact bounding boxes
                pos = item["position"]
                center_x = int(pos["center_x"] * cv_image.shape[1])
                center_y = int(pos["center_y"] * cv_image.shape[0])
                width = int(pos["width"] * cv_image.shape[1])
                height = int(pos["height"] * cv_image.shape[0])
                
                # Create coordinates for a box around this position
                x1 = max(0, center_x - width // 2)
                y1 = max(0, center_y - height // 2)
                x2 = min(cv_image.shape[1], center_x + width // 2)
                y2 = min(cv_image.shape[0], center_y + height // 2)
                
                # Create a mask for this region
                item_mask = np.zeros(segmentation_mask.shape, dtype=np.uint8)
                item_mask[y1:y2, x1:x2] = 1
                
                # Combine with segmentation mask
                item_mask = item_mask & (segmentation_mask > 0)
                
                # Extract colors for this item
                item_weight_map = item_mask.astype(float) * attention_map_resized
                if np.max(item_weight_map) > 0:
                    item_weight_map = item_weight_map / np.max(item_weight_map)
                
                # Extract colors for this item
                item_colors_bgr = self._extract_weighted_colors(cv_image, item_weight_map, n_colors=3)
                
                # Convert to named colors
                item_named_colors = []
                for color in item_colors_bgr:
                    color_name = self.closest_color_name(color)
                    hex_color = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
                    item_named_colors.append({
                        'name': color_name,
                        'rgb': (color[2], color[1], color[0]),
                        'hex': hex_color
                    })
                
                # Store in result and update the item
                item_colors[i] = item_named_colors
                
                # Update the item's dominant color
                if item_named_colors:
                    item["dominant_color"] = item_named_colors[0]['name']
                    item["colors"] = item_named_colors
            else:
                # For items without position information, use global colors
                item_colors[i] = global_named_colors[:3]
                if global_named_colors:
                    item["dominant_color"] = global_named_colors[0]['name']
                    item["colors"] = global_named_colors[:3]
        
        # Create the color analysis result
        result = {
            "color_palette": global_named_colors,
            "harmony": harmony_result,
            "seasonal_palette": seasonal_palette,
            "item_colors": item_colors
        }
    
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
    
    def _get_attention_map(self, image: Image.Image) -> np.ndarray:
        """
        Generate attention map using ResNet features
        
        Args:
            image: PIL Image
            
        Returns:
            numpy.ndarray: Attention map highlighting important regions
        """
        # Resize image for ResNet
        resized_img = image.resize((224, 224))
        
        # Transform for ResNet
        img_tensor = self.transform(resized_img).unsqueeze(0).to(self.device)
        
        # Get feature maps from ResNet
        with torch.no_grad():
            feature_maps = self.resnet(img_tensor)
        
        # Convert feature maps to spatial attention
        # Take mean across channels to get attention map
        attention = feature_maps.mean(dim=1).squeeze().cpu().numpy()
        
        return attention
    
    def _extract_weighted_colors(self, image: np.ndarray, weights: np.ndarray, n_colors=8) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors using weighted k-means clustering
        
        Args:
            image: Input image (OpenCV format)
            weights: Weight map for pixel importance
            n_colors: Number of colors to extract
            
        Returns:
            list: List of (B, G, R) color tuples
        """
        # Ensure weights has the same dimensions as image
        if weights.shape[:2] != image.shape[:2]:
            weights = cv2.resize(weights, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Reshape image and weights
        pixels = image.reshape(-1, 3).astype(np.float32)
        flat_weights = weights.flatten()
        
        # Filter out pixels with very low weights
        valid_indices = flat_weights > 0.05
        valid_pixels = pixels[valid_indices]
        valid_weights = flat_weights[valid_indices]
        
        if len(valid_pixels) == 0:
            return []
        
        # Adjust number of clusters if needed
        k = min(n_colors, len(valid_pixels))
        if k <= 1:
            return [tuple(map(int, valid_pixels[0]))]
        
        # Use KMeans with weighted initialization
        # Select initial centroids weighted by importance
        init_indices = np.random.choice(
            len(valid_pixels), 
            size=k, 
            replace=False, 
            p=valid_weights/np.sum(valid_weights)
        )
        init_centroids = valid_pixels[init_indices]
        
        # Perform K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.1)
        _, labels, centers = cv2.kmeans(
            valid_pixels, k, 
            init_centroids, 
            criteria, 
            30, 
            cv2.KMEANS_PP_CENTERS
        )
        
        # Calculate weighted cluster sizes
        cluster_weights = np.zeros(k)
        for i, label in enumerate(labels.flatten()):
            cluster_weights[label] += valid_weights[i]
        
        # Sort clusters by weighted size
        sorted_indices = np.argsort(cluster_weights)[::-1]
        centers = centers[sorted_indices]
        
        # Convert to integer tuples
        colors = [(int(center[0]), int(center[1]), int(center[2])) for center in centers]
        
        return colors
    
    def closest_color_name(self, bgr_color: Tuple[int, int, int]) -> str:
        """
        Find closest named color with improved accuracy for fashion contexts
        
        Args:
            bgr_color: BGR color tuple
        
        Returns:
            str: Named color
        """
        # Convert BGR to RGB
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        
        min_distance = float('inf')
        closest_name = "unknown"
        
        # Special case for very dark colors (near black)
        r, g, b = rgb_color
        brightness = (r + g + b) / 3
        
        if brightness < 30:
            return "black"
        
        # Special case for very light colors (near white)
        if brightness > 240:
            return "white"
        elif brightness > 225:
            # More refined check for off-white vs. white
            saturation = max(r, g, b) - min(r, g, b)
            if saturation < 10:
                return "white"
            else:
                return "off_white"
        elif brightness > 210:
            # Check for cream or ivory
            if r > g + 5 and r > b + 5:  # Warmer tone
                return "cream"
            return "ivory"
        
        # Special handling for fashion-important colors like coral
        # Check coral first with special weighting
        coral_distance = self._color_distance(rgb_color, self.color_names["coral"])
        if coral_distance < 40:  # Lower threshold for coral detection
            return "coral"
        
        # For light blues (jeans)
        if 50 < brightness < 200 and b > r and b > g:
            light_blue_distance = self._color_distance(rgb_color, self.color_names["light_blue"])
            denim_distance = self._color_distance(rgb_color, self.color_names["denim_blue"])
            
            if light_blue_distance < 50 or denim_distance < 50:
                return "light_blue" if light_blue_distance < denim_distance else "denim_blue"
        
        # For lavender/purple detection (often misidentified)
        if 100 < brightness < 230 and b > r and r > g:
            lavender_distance = self._color_distance(rgb_color, self.color_names["lavender"])
            if lavender_distance < 45:
                return "lavender"
        
        # Standard color matching for everything else
        for name, color in self.color_names.items():
            distance = self._color_distance(rgb_color, color)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        return closest_name
    
    def _color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """
        Calculate color distance using a more perceptually accurate model (CIE76)
        
        Args:
            color1: First RGB color tuple
            color2: Second RGB color tuple
            
        Returns:
            float: Perceptual distance between colors
        """
        # Convert RGB to Lab color space for more perceptual accuracy
        lab1 = self._rgb_to_lab(color1)
        lab2 = self._rgb_to_lab(color2)
        
        # Calculate Euclidean distance in Lab space
        delta_l = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]
        
        return math.sqrt(delta_l**2 + delta_a**2 + delta_b**2)

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Convert RGB color to CIE-Lab color space
        
        Args:
            rgb: RGB color tuple
            
        Returns:
            tuple: (L, a, b) in CIE-Lab color space
        """
        # Convert RGB to XYZ
        r, g, b = rgb
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
        
        # Apply gamma correction
        r = self._gamma_correct(r)
        g = self._gamma_correct(g)
        b = self._gamma_correct(b)
        
        # Convert to XYZ
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        
        # Normalize for D65 white point
        x = x / 0.95047
        y = y / 1.0
        z = z / 1.08883
        
        # Convert XYZ to Lab
        x = self._lab_func(x)
        y = self._lab_func(y)
        z = self._lab_func(z)
        
        L = 116 * y - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        return (L, a, b)

    def _gamma_correct(self, c: float) -> float:
        """Apply gamma correction to a color channel"""
        return ((c > 0.04045) and (((c + 0.055) / 1.055) ** 2.4)) or (c / 12.92)

    def _lab_func(self, t: float) -> float:
        """Helper function for Lab conversion"""
        return ((t > 0.008856) and (t ** (1/3))) or ((7.787 * t) + (16/116))
    
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