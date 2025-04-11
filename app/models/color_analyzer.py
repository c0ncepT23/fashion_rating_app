# app/models/color_analyzer.py
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import asyncio
import colorsys
import math
from sklearn.cluster import KMeans

from app.utils.image_processing import (
    pil_to_cv2, 
    cv2_to_pil, 
    extract_masks_from_detections,
    extract_dominant_colors
)

class ColorAnalyzer:
    """
    Enhanced analyzer for extracting and analyzing color palettes from fashion images
    Works with YOLOv8m detections to analyze colors per garment
    """
    
    def __init__(self):
        """
        Initialize the color analyzer
        """
        # Load color name database
        self.color_names = self._load_color_names()
    
    def _load_color_names(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Load a comprehensive color name dictionary with improved black detection
        
        Returns:
            dict: Dictionary mapping color names to RGB values
        """
        color_dict = {
            # True black and variations - expanded with more variations
            'black': (0, 0, 0),
            'rich_black': (10, 10, 10),
            'off_black': (20, 20, 20),
            'dark_charcoal': (30, 30, 30),
            
            # Whites
            'white': (255, 255, 255),
            'off_white': (250, 249, 246),
            'cream': (255, 253, 208),
            'ivory': (255, 255, 240),
            
            # Grays - spaced further from black
            'charcoal': (54, 69, 79),  # Moved from original position
            'dark_gray': (80, 80, 80),  # Added to create more separation from black
            'gray': (128, 128, 128),
            'silver': (192, 192, 192),
            'light_gray': (220, 220, 220),
            
            # Basic colors
            'red': (255, 0, 0),
            'green': (0, 128, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            
            # Rest of your existing colors...
            'navy': (0, 0, 128),
            'teal': (0, 128, 128),
            'olive': (128, 128, 0),
            'maroon': (128, 0, 0),
            'beige': (245, 245, 220),
            'khaki': (240, 230, 140),
            'coral': (255, 127, 80),
            'turquoise': (64, 224, 208),
            'lavender': (230, 230, 250),
            'mint': (189, 252, 201),
            'burgundy': (128, 0, 32),
            'forest_green': (34, 139, 34),
            'mustard': (255, 219, 88),
            'tan': (210, 180, 140),
            'mauve': (204, 153, 204),
            'rust': (183, 65, 14),
            
            # Fashion colors
            'millennial_pink': (244, 194, 194),
            'sage_green': (186, 202, 186),
            'terracotta': (226, 114, 91),
            'neon_yellow': (255, 255, 0),
            'neon_green': (57, 255, 20),
            'neon_pink': (255, 20, 147),
            'pastel_blue': (174, 198, 207),
            'pastel_pink': (255, 209, 220),
            'pastel_purple': (215, 183, 219),
            'camel': (193, 154, 107),
            'blush': (222, 93, 131),
            'seafoam': (159, 226, 191),
            'slate': (112, 128, 144),
            'periwinkle': (204, 204, 255),
            'emerald': (0, 201, 87),
            'cobalt': (0, 71, 171),
            'plum': (142, 69, 133),
            'ochre': (204, 119, 34),
            'taupe': (72, 60, 50),
            'indigo': (75, 0, 130),
            'magenta': (255, 0, 255)
        }
        return color_dict
    
    def _color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """
        Calculate the color distance with special handling for black detection
        
        Args:
            color1: First RGB color tuple
            color2: Second RGB color tuple
            
        Returns:
            float: Distance between colors
        """
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        # Special case for blacks - if color is very dark, prioritize matching to black variants
        is_dark = (r1 + g1 + b1) < 60  # Threshold for "dark" colors
        
        if is_dark:
            # For dark colors, put more weight on darkness level
            darkness_diff = abs((r1 + g1 + b1) - (r2 + g2 + b2))
            
            # If comparing to a black variant, reduce the distance
            is_black_variant = (r2 + g2 + b2) < 60
            if is_black_variant:
                return darkness_diff * 0.7  # Reduced distance for black variants
        
        # Standard weighted Euclidean distance for non-dark colors
        red_mean = (r1 + r2) / 2
        r_weight = 2 + red_mean / 256
        g_weight = 4  # Green has the highest weight
        b_weight = 2  # Blue has the lowest weight
        
        return math.sqrt(
            r_weight * (r1 - r2)**2 + 
            g_weight * (g1 - g2)**2 + 
            b_weight * (b1 - b2)**2
        )
    
    def closest_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Find the closest named color for an RGB value
        
        Args:
            rgb_color: RGB color tuple
            
        Returns:
            str: Name of the closest color
        """
        # Convert from BGR (OpenCV) to RGB if needed
        if isinstance(rgb_color, np.ndarray) and rgb_color.size == 3:
            rgb_color = tuple(map(int, rgb_color))
            
        # Find the closest color name
        min_distance = float('inf')
        closest_name = "unknown"
        
        for name, color in self.color_names.items():
            distance = self._color_distance(rgb_color, color)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        return closest_name
    
    
    async def analyze_outfit_colors(
            self, 
            image: Image.Image, 
            clothing_items: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
        """
        Analyze colors for the entire outfit and individual items
        
        Args:
            image: Original PIL image
            clothing_items: Detected clothing items from YOLOv8m
            
        Returns:
            dict: Color analysis results
        """
        # Convert to OpenCV format for processing
        cv_image = pil_to_cv2(image)
        
        # Extract masks for each clothing item
        masks = extract_masks_from_detections(cv_image, clothing_items)
        
        # Analyze global colors (from the entire image)
        global_colors = extract_dominant_colors(cv_image, n_colors=8)
        
        # Convert to named colors
        global_named_colors = [self.closest_color_name(color) for color in global_colors]
        
        # Analyze colors per item
        for i, (item, mask) in enumerate(zip(clothing_items, masks)):
            # Extract dominant color for this item
            item_colors = extract_dominant_colors(cv_image, mask, n_colors=3)
            
            if item_colors:
                # Get primary color
                primary_color = item_colors[0]
                primary_color_name = self.closest_color_name(primary_color)
                
                # Add color information to the item
                item["colors"] = [{
                    "name": self.closest_color_name(color),
                    "rgb": color,
                    "hex": '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])  # RGB from BGR
                } for color in item_colors]
                
                item["dominant_color"] = primary_color_name
        
        # Analyze color harmony
        harmony_result = self._analyze_color_harmony(global_colors)
        seasonal_palette = self._detect_seasonal_palette(global_named_colors)
        
        # Check if we have any black/near-black colors in individual items
        black_variants = ['black', 'rich_black', 'off_black']
        has_black_item = any(item.get("dominant_color") in black_variants for item in clothing_items)
        
        # Check if we have any white/off-white colors in individual items
        white_variants = ['white', 'off_white', 'cream', 'ivory']
        has_white_item = any(item.get("dominant_color") in white_variants for item in clothing_items)
        
        # Create color palette with priority ordering
        if has_black_item or has_white_item:
            # If we have black or white items, prioritize them in the palette
            # First collect all colors from global and items
            all_colors = []
            
            # Add item dominant colors first (more reliable than global extraction)
            for item in clothing_items:
                if "dominant_color" in item:
                    color_name = item["dominant_color"]
                    # Find the corresponding color data
                    for color_data in item.get("colors", []):
                        if color_data["name"] == color_name:
                            all_colors.append(color_data)
                            break
            
            # Add global colors
            for i, color in enumerate(global_colors):
                color_name = self.closest_color_name(color)
                # Check if this color is already in our list
                if not any(c["name"] == color_name for c in all_colors):
                    all_colors.append({
                        "name": color_name,
                        "rgb": (color[2], color[1], color[0]),  # BGR to RGB
                        "hex": '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
                    })
            
            # Sort colors by priority categories
            color_palette = []
            
            # First add black variants if present
            black_colors = [c for c in all_colors if c["name"] in black_variants]
            color_palette.extend(black_colors)
            
            # Then add white variants if present
            white_colors = [c for c in all_colors if c["name"] in white_variants]
            color_palette.extend(white_colors)
            
            # Then add remaining colors
            other_colors = [c for c in all_colors if c["name"] not in black_variants and c["name"] not in white_variants]
            color_palette.extend(other_colors)
            
            # Deduplicate while preserving order
            seen = set()
            color_palette = [c for c in color_palette if not (c["name"] in seen or seen.add(c["name"]))]
        else:
            # Default color palette creation if no black/white detected
            color_palette = [{
                "name": self.closest_color_name(color),
                "rgb": (color[2], color[1], color[0]),  # Convert BGR to RGB
                "hex": '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
            } for color in global_colors]
        
        # Add harmony information to the result
        if color_palette:
            color_palette[0]["harmony_type"] = harmony_result["type"]
            color_palette[0]["harmony_score"] = harmony_result["score"]
            color_palette[0]["seasonal_palette"] = seasonal_palette
        
        # Compile the color analysis result
        result = {
            "color_palette": color_palette,
            "harmony": harmony_result,
            "seasonal_palette": seasonal_palette,
            "item_colors": {i: item.get("colors", []) for i, item in enumerate(clothing_items) if "colors" in item}
        }
    
        return result
    
    async def extract_palette(self, image: Image.Image, clothing_items: List[Dict] = None) -> List[Dict]:
        """
        Extract the dominant color palette from an image
        Legacy method to maintain compatibility with existing code
        
        Args:
            image: PIL Image object
            clothing_items: Optional list of detected clothing items
            
        Returns:
            list: Dominant colors with metadata
        """
        # Use the more comprehensive method and extract just the palette
        result = await self.analyze_outfit_colors(image, clothing_items or [])
        return result["color_palette"]
    
    def _analyze_color_harmony(self, colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """
        Analyze the color harmony of a palette
        
        Args:
            colors: List of RGB/BGR color tuples
            
        Returns:
            dict: Harmony type and score
        """
        if not colors:
            return {"type": "unknown", "score": 0}
        
        # Convert BGR to HSV for better color theory analysis
        hsv_colors = []
        for color in colors:
            # Normalize to 0-1 range and convert to HSV
            b, g, r = color  # BGR format from OpenCV
            r, g, b = r/255.0, g/255.0, b/255.0  # Normalize to 0-1
            hsv = colorsys.rgb_to_hsv(r, g, b)
            hsv_colors.append(hsv)
        
        # Extract hues for harmony analysis
        hues = [color[0] for color in hsv_colors]
        
        # Calculate harmony score and type
        harmony_score = 0
        harmony_type = "custom"
        
        # Check for monochromatic (similar hues)
        hue_range = max(hues) - min(hues) if hues else 0
        if hue_range < 0.05:
            harmony_type = "monochromatic"
            harmony_score = 90
        
        # Check for complementary (opposite hues)
        elif len(hues) >= 2:
            for i in range(len(hues)):
                for j in range(i+1, len(hues)):
                    hue_diff = abs(hues[i] - hues[j])
                    hue_diff = min(hue_diff, 1 - hue_diff)  # Account for circular nature of hue
                    if 0.45 <= hue_diff <= 0.55:  # About 180 degrees in hue
                        harmony_type = "complementary"
                        harmony_score = 85
                        break
        
        # Check for analogous (adjacent hues)
        elif hue_range < 0.25:
            harmony_type = "analogous"
            harmony_score = 80
        
        # Check for triadic (three evenly spaced hues)
        elif len(hues) >= 3:
            # Sort hues
            sorted_hues = sorted(hues)
            
            # Check if hues are approximately evenly spaced
            spacing1 = (sorted_hues[1] - sorted_hues[0]) % 1
            spacing2 = (sorted_hues[2] - sorted_hues[1]) % 1
            
            if abs(spacing1 - spacing2) < 0.05 and abs(spacing1 - 0.33) < 0.1:
                harmony_type = "triadic"
                harmony_score = 75
        
        # If we haven't identified a specific harmony type
        if harmony_score == 0:
            # Calculate saturation and value variance
            saturations = [color[1] for color in hsv_colors]
            values = [color[2] for color in hsv_colors]
            
            sat_variance = np.var(saturations)
            val_variance = np.var(values)
            
            # More consistent saturation and value often looks more harmonious
            consistency_score = 100 - (sat_variance + val_variance) * 100
            harmony_score = max(40, min(70, consistency_score))
            
            # Determine harmony type based on characteristics
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
        """
        Detect seasonal color palette based on color names
        
        Args:
            color_names: List of color names
            
        Returns:
            str: Seasonal palette name
        """
        if not color_names:
            return "unknown"
        
        # Define seasonal color palettes
        seasonal_palettes = {
            "spring": ["coral", "mint", "yellow", "light_blue", "peach", "bright", "warm"],
            "summer": ["pastel_blue", "lavender", "soft_pink", "mint", "light_gray", "cool", "soft"],
            "autumn": ["olive", "rust", "mustard", "burgundy", "terracotta", "warm", "muted"],
            "winter": ["black", "white", "navy", "red", "emerald", "cool", "clear"]
        }
        
        # Count matches for each season
        season_matches = {season: 0 for season in seasonal_palettes}
        
        for color in color_names:
            for season, palette in seasonal_palettes.items():
                if any(keyword in color for keyword in palette):
                    season_matches[season] += 1
        
        # Find the season with most matches
        max_matches = 0
        best_season = "unknown"
        
        for season, matches in season_matches.items():
            if matches > max_matches:
                max_matches = matches
                best_season = season
        
        # If no clear match, use a different approach
        if max_matches <= 1:
            # Extract color properties from the names
            warm_colors = ["red", "orange", "yellow", "gold", "rust", "brown", "terracotta", "mustard", "coral"]
            cool_colors = ["blue", "green", "purple", "teal", "mint", "turquoise", "lavender", "seafoam"]
            bright_colors = ["neon", "bright", "vibrant", "pure"]
            muted_colors = ["muted", "dusty", "sage", "mauve", "pastel"]
            
            # Count color characteristics
            warm_count = sum(1 for color in color_names if any(w in color for w in warm_colors))
            cool_count = sum(1 for color in color_names if any(c in color for c in cool_colors))
            bright_count = sum(1 for color in color_names if any(b in color for b in bright_colors))
            muted_count = sum(1 for color in color_names if any(m in color for m in muted_colors))
            
            # Determine season based on characteristics
            if warm_count > cool_count:
                if bright_count > muted_count:
                    return "spring"  # Warm and bright
                else:
                    return "autumn"  # Warm and muted
            else:
                if bright_count > muted_count:
                    return "winter"  # Cool and clear
                else:
                    return "summer"  # Cool and soft
        
        return best_season