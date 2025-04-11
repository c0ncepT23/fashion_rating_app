# app/models/style_classifier.py
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import asyncio
import random
import os

from app.config import settings
from app.utils.constants import STYLE_CATEGORIES
from app.utils.image_processing import pil_to_cv2

class StyleClassifier:
    """
    Rule-based classifier for fashion styles
    Works with YOLOv8m detections and color analysis
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the style classifier
        
        Args:
            model_path: Optional path to a pretrained model (not used in rule-based approach)
        """
        # Define style categories
        self.style_categories = sorted(list(STYLE_CATEGORIES.keys()))
        
        # Load style rules for rule-based classification
        self.style_rules = self._load_style_rules()
    
    def _load_style_rules(self) -> Dict[str, Any]:
        """
        Load rules for style classification
        
        Returns:
            dict: Style classification rules
        """
        # In production, these would be loaded from a JSON file
        # For now, we'll hardcode some basic rules
        rules = {
            "casual": {
                "items": ["short sleeve top", "long sleeve top", "shorts", "trousers", "jeans"],
                "colors": ["blue", "gray", "white", "black"],
                "not_items": ["formal_dress", "suit"],
                "weight": 1.0
            },
            "formal": {
                "items": ["suit", "blazer", "long sleeve outwear", "trousers", "long sleeve dress"],
                "colors": ["black", "navy", "gray", "white"],
                "not_items": ["shorts", "short sleeve top"],
                "weight": 1.0
            },
            "streetwear": {
                "items": ["short sleeve top", "hoodie", "shorts", "trousers"],
                "colors": ["black", "white", "red", "neon"],
                "not_items": ["long sleeve dress", "long sleeve outwear"],
                "weight": 1.0
            },
            "bohemian": {
                "items": ["vest", "skirt", "sling dress", "long sleeve dress"],
                "colors": ["terracotta", "mustard", "sage_green", "beige"],
                "not_items": ["suit", "blazer"],
                "weight": 1.0
            },
            "minimalist": {
                "items": ["short sleeve top", "long sleeve top", "trousers", "skirt"],
                "colors": ["black", "white", "navy", "gray", "beige"],
                "not_colors": ["neon_yellow", "neon_pink", "bright"],
                "weight": 1.0
            },
            "edgy_chic": {
                "items": ["short sleeve outwear", "long sleeve outwear", "vest", "trousers"],
                "colors": ["black", "gray", "white"],
                "weight": 1.2
            },
            "preppy": {
                "items": ["long sleeve top", "long sleeve outwear", "trousers", "skirt"],
                "colors": ["navy", "red", "white", "green", "beige"],
                "weight": 1.0
            },
            "sporty": {
                "items": ["short sleeve top", "shorts", "trousers"],
                "colors": ["blue", "red", "white", "gray"],
                "weight": 1.0
            },
            "vintage": {
                "items": ["long sleeve top", "skirt", "long sleeve dress", "vest dress"],
                "colors": ["beige", "pastel_pink", "pastel_blue", "burgundy"],
                "weight": 1.0
            },
            "fashion_week_off_duty": {
                "items": ["long sleeve outwear", "short sleeve outwear", "vest", "trousers", "skirt"],
                "colors": ["black", "white", "beige", "gray"],
                "weight": 1.5  # Higher weight for this trendy style
            }
        }
        return rules
    
    async def classify(
        self, 
        image: Image.Image, 
        clothing_items: List[Dict[str, Any]] = None,
        color_analysis: Dict[str, Any] = None,
        features: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Classify the fashion style using rules-based approach
        
        Args:
            image: PIL Image object
            clothing_items: Optional detected clothing items
            color_analysis: Optional color analysis results
            features: Optional pre-extracted features
            
        Returns:
            dict: Style classification results
        """
        # Use rule-based approach if we have clothing items and colors
        if clothing_items is not None and color_analysis is not None:
            rule_result = await self._classify_rules(clothing_items, color_analysis)
            return rule_result
        else:
            # Fallback to a basic classification
            return self._fallback_classification(image)
    
    async def _classify_rules(
        self, 
        clothing_items: List[Dict[str, Any]], 
        color_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify style using rule-based approach
        
        Args:
            clothing_items: Detected clothing items
            color_analysis: Color analysis results
            
        Returns:
            dict: Classification results
        """
        # Extract item types and colors
        item_types = [item["type"] for item in clothing_items]
        
        # Extract color names
        color_names = []
        if "color_palette" in color_analysis:
            color_names = [color["name"] for color in color_analysis["color_palette"]]
        
        # Calculate scores for each style based on rules
        style_scores = {}
        
        for style, rules in self.style_rules.items():
            score = 0.0
            weight = rules.get("weight", 1.0)
            
            # Check for matching items
            for item in rules.get("items", []):
                if any(item in it for it in item_types):
                    score += 1.0
            
            # Check for matching colors
            for color in rules.get("colors", []):
                if any(color in c for c in color_names):
                    score += 0.5
            
            # Check for excluding items
            for item in rules.get("not_items", []):
                if any(item in it for it in item_types):
                    score -= 1.5
            
            # Check for excluding colors
            for color in rules.get("not_colors", []):
                if any(color in c for c in color_names):
                    score -= 0.5
            
            # Apply weight and normalize
            score = max(0, score) * weight
            style_scores[style] = score
        
        # Normalize scores to probabilities
        total_score = sum(style_scores.values())
        if total_score > 0:
            style_probs = {style: score / total_score for style, score in style_scores.items()}
        else:
            # If all scores are zero, use uniform distribution
            style_probs = {style: 1.0 / len(self.style_rules) for style in self.style_rules}
        
        # Get top style
        top_style = max(style_probs, key=style_probs.get)
        top_prob = style_probs[top_style]
        
        # Get top 3 styles
        top_styles = sorted(style_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Get style description
        style_description = STYLE_CATEGORIES.get(top_style, "A distinctive fashion style.")
        
        # Calculate coherence - how well the items match the identified style
        coherence = min(0.9, top_prob)  # Cap at 0.9 for rule-based approach
        
        # Prepare results
        results = {
            "style": top_style,
            "confidence": top_prob,
            "coherence": coherence,
            "description": style_description,
            "top_styles": top_styles,
            "all_probabilities": style_probs,
            "method": "rule_based"
        }
        
        return results
    
    def _fallback_classification(self, image: Image.Image) -> Dict[str, Any]:
        """
        Fallback classification when other methods fail
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Basic classification result
        """
        # This is a very basic fallback using simple image features
        
        # Convert to OpenCV for analysis
        cv_image = pil_to_cv2(image)
        
        # Extract some basic features
        avg_color = np.mean(cv_image, axis=(0, 1))
        brightness = np.mean(avg_color)
        saturation = np.std(avg_color)
        
        # Determine a style based on simple heuristics
        if brightness < 80:  # Dark image
            primary_style = "edgy_chic"
            secondary_style = "formal"
        elif saturation < 30:  # Low saturation (monochrome)
            primary_style = "minimalist"
            secondary_style = "formal"
        elif brightness > 200:  # Very bright
            primary_style = "casual"
            secondary_style = "streetwear"
        else:  # Medium brightness and saturation
            primary_style = "casual"
            secondary_style = "minimalist"
        
        # Create probabilities based on these heuristics
        style_probs = {style: 0.05 for style in self.style_categories}
        style_probs[primary_style] = 0.4
        style_probs[secondary_style] = 0.2
        
        # Get top 3 styles
        top_styles = sorted(style_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Get style description
        style_description = STYLE_CATEGORIES.get(primary_style, "A distinctive fashion style.")
        
        # Prepare result
        result = {
            "style": primary_style,
            "confidence": 0.4,  # Low confidence for fallback
            "coherence": 0.3,   # Low coherence for fallback
            "description": style_description,
            "top_styles": top_styles,
            "all_probabilities": style_probs,
            "method": "fallback"
        }
        
        return result
    
    def get_style_description(self, style: str) -> str:
        """
        Get the description for a style category
        
        Args:
            style: Style category name
            
        Returns:
            str: Style description
        """
        return STYLE_CATEGORIES.get(style, "A distinctive fashion style.")