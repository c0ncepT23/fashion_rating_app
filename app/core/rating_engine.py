# app/core/rating_engine.py
from typing import Dict, List, Tuple, Any
import numpy as np
from app.config import settings

def calculate_fit_score(
    clothing_items: List[Dict], 
    style_result: Dict[str, Any]
) -> int:
    """
    Calculate the score for fit/silhouette based on detected items and style
    
    Args:
        clothing_items: List of detected clothing items
        style_result: Style classification result
        
    Returns:
        int: Score from 0-25
    """
    if not clothing_items:
        return 12  # Default middle score when no items detected
    
    # Start with a base score
    base_score = 15
    
    # Check for appropriate items for the style
    style = style_result.get("style", "casual")
    
    # Extract item types and categories
    item_types = [item["type"] for item in clothing_items]
    categories = [item["category"] for item in clothing_items]
    
    # Check balance of outfit (are all essential categories present?)
    essential_categories = ["top", "bottom", "footwear"]
    missing_categories = [cat for cat in essential_categories if cat not in categories]
    
    # Deduct points for missing essential categories
    base_score -= len(missing_categories) * 3
    
    # Look for fit attributes assigned by the detector
    fit_types = [item.get("fit", "unknown") for item in clothing_items]
    
    # Check if the fit types are consistent with the style
    style_fit_mappings = {
        "casual": ["relaxed", "casual", "loose"],
        "formal": ["structured", "tailored", "fitted"],
        "streetwear": ["oversized", "relaxed", "loose"],
        "minimalist": ["clean", "structured", "fitted"],
        "edgy_chic": ["structured", "fitted", "tailored"],
        "bohemian": ["flowy", "loose", "relaxed"],
        "vintage": ["tailored", "fitted", "structured"],
        "preppy": ["clean", "tailored", "structured"],
        "fashion_week_off_duty": ["oversized", "structured", "layered"]
    }
    
    # Get preferred fit types for the detected style
    preferred_fits = style_fit_mappings.get(style, ["balanced", "proportioned"])
    
    # Add points for matching fit types
    matching_fits = sum(1 for fit in fit_types if any(pref in fit for pref in preferred_fits))
    base_score += matching_fits * 2
    
    # Check proportions (e.g., tops vs bottoms)
    top_items = [item for item in clothing_items if item["category"] == "top"]
    bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
    
    if top_items and bottom_items:
        # Check for balanced proportions
        top_area = sum(item["position"]["area"] for item in top_items)
        bottom_area = sum(item["position"]["area"] for item in bottom_items)
        
        if bottom_area > 0:
            ratio = top_area / bottom_area
            # Ideal ratio depends on style
            if style in ["casual", "streetwear", "sporty"]:
                # More relaxed styles can have more varied proportions
                if 0.7 <= ratio <= 1.7:
                    base_score += 3
            else:
                # More formal/structured styles benefit from balanced proportions
                if 0.8 <= ratio <= 1.3:
                    base_score += 3
    
    # Cap score between 0-25
    return max(0, min(25, base_score))

def calculate_color_score(
    color_analysis: Dict[str, Any],
    style_result: Dict[str, Any]
) -> int:
    """
    Calculate the score for color coordination
    
    Args:
        color_analysis: Color analysis results
        style_result: Style classification result
        
    Returns:
        int: Score from 0-25
    """
    # Start with base score
    base_score = 15
    
    # Add points based on harmony score if available
    if "harmony" in color_analysis and "score" in color_analysis["harmony"]:
        harmony_score = color_analysis["harmony"]["score"]
        # Scale harmony score (0-100) to points (0-10)
        harmony_points = min(10, harmony_score / 10)
        base_score += harmony_points
    
    # Get color palette and style
    style = style_result.get("style", "casual")
    
    if "color_palette" in color_analysis and color_analysis["color_palette"]:
        color_palette = color_analysis["color_palette"]
        color_names = [color["name"] for color in color_palette]
        
        # Style-specific color preferences
        style_color_mappings = {
            "minimalist": ["black", "white", "gray", "navy", "beige"],
            "casual": ["blue", "gray", "white", "black", "red"],
            "formal": ["black", "navy", "gray", "white", "burgundy"],
            "edgy_chic": ["black", "gray", "white", "red"],
            "bohemian": ["terracotta", "mustard", "olive", "rust", "beige"],
            "streetwear": ["black", "white", "red", "neon", "blue"],
            "fashion_week_off_duty": ["black", "white", "camel", "gray"]
        }
        
        # Check if color palette matches style preferences
        preferred_colors = style_color_mappings.get(style, [])
        
        if preferred_colors:
            matching_colors = sum(1 for color in color_names if any(pref in color for pref in preferred_colors))
            # Add points based on matches (max 5)
            style_color_points = min(5, matching_colors)
            base_score += style_color_points
    
    # Ensure score is between 0-25
    return max(0, min(25, base_score))

def calculate_footwear_score(
    clothing_items: List[Dict],
    color_analysis: Dict[str, Any],
    style_result: Dict[str, Any]
) -> int:
    """
    Calculate the score for footwear
    
    Args:
        clothing_items: List of detected clothing items
        color_analysis: Color analysis results
        style_result: Style classification result
        
    Returns:
        int: Score from 0-20
    """
    # Start with base score
    base_score = 10
    
    # Find footwear items
    footwear_items = [item for item in clothing_items if item["category"] == "footwear"]
    
    if not footwear_items:
        # No footwear detected
        return max(0, base_score - 5)
    
    # Get style
    style = style_result.get("style", "casual")
    
    # Style-specific footwear preferences
    style_footwear_mappings = {
        "casual": ["sneakers", "loafers", "sandals", "boots"],
        "formal": ["heels", "oxfords", "loafers", "formal"],
        "streetwear": ["sneakers", "boots", "chunky"],
        "minimalist": ["clean", "simple", "minimal"],
        "edgy_chic": ["boots", "chunky", "platform"],
        "bohemian": ["sandals", "boots", "flats", "earthy"],
        "fashion_week_off_duty": ["boots", "loafers", "statement", "designer"]
    }
    
    # Check if footwear matches style preferences
    footwear_type = footwear_items[0]["type"]
    preferred_footwear = style_footwear_mappings.get(style, [])
    
    if preferred_footwear and any(pref in footwear_type.lower() for pref in preferred_footwear):
        base_score += 5
    
    # Check color coordination with outfit
    if "color_palette" in color_analysis and color_analysis["color_palette"]:
        outfit_colors = [color["name"] for color in color_analysis["color_palette"]]
        
        # Get footwear color
        footwear_color = footwear_items[0].get("dominant_color", "unknown")
        
        # Check if footwear color matches/complements outfit colors
        if footwear_color in outfit_colors:
            base_score += 5
        elif footwear_color in ["black", "white", "navy", "brown", "tan", "beige"]:
            # Neutral footwear generally works well
            base_score += 3
    
    # Ensure score is between 0-20
    return max(0, min(20, base_score))

def calculate_accessories_score(
    clothing_items: List[Dict],
    color_analysis: Dict[str, Any],
    style_result: Dict[str, Any]
) -> int:
    """
    Calculate the score for accessories
    
    Args:
        clothing_items: List of detected clothing items
        color_analysis: Color analysis results
        style_result: Style classification result
        
    Returns:
        int: Score from 0-15
    """
    # Start with base score
    base_score = 8
    
    # Find accessory items
    accessory_items = [item for item in clothing_items if item["category"] == "accessory"]
    
    # If no accessories, give a middling score
    if not accessory_items:
        return base_score - 3
    
    # Get style
    style = style_result.get("style", "casual")
    
    # Evaluate number of accessories (not too few, not too many)
    num_accessories = len(accessory_items)
    
    if 1 <= num_accessories <= 3:
        # Good number of accessories for most styles
        base_score += 3
    elif num_accessories > 3:
        # Too many accessories for most styles
        if style in ["bohemian", "fashion_week_off_duty"]:
            # These styles can handle more accessories
            base_score += 2
        else:
            base_score -= 1
    
    # Check if accessories match the style
    style_accessory_mappings = {
        "minimalist": ["simple", "clean", "minimal", "watch"],
        "casual": ["cap", "backpack", "simple", "watch"],
        "formal": ["subtle", "refined", "elegant", "watch"],
        "streetwear": ["cap", "beanie", "chunky", "statement"],
        "edgy_chic": ["leather", "metal", "statement", "chunky"],
        "bohemian": ["layered", "natural", "earthy", "colorful"],
        "fashion_week_off_duty": ["statement", "sunglasses", "designer", "scarf"]
    }
    
    # Check if accessories match style preferences
    accessory_types = [item["type"] for item in accessory_items]
    preferred_accessories = style_accessory_mappings.get(style, [])
    
    matching_accessories = sum(1 for acc in accessory_types 
                              if any(pref in acc.lower() for pref in preferred_accessories))
    
    # Add points based on matches (max 4)
    style_accessory_points = min(4, matching_accessories)
    base_score += style_accessory_points
    
    # Ensure score is between 0-15
    return max(0, min(15, base_score))

def calculate_style_score(style_result: Dict[str, Any]) -> int:
    """
    Calculate the score for overall style coherence
    
    Args:
        style_result: Style classification result
        
    Returns:
        int: Score from 0-15
    """
    # Start with base score
    base_score = 7
    
    # Add points based on style confidence
    confidence = style_result.get("confidence", 0.5)
    coherence = style_result.get("coherence", 0.5)
    
    # Convert confidence (0-1) to points (0-5)
    confidence_points = min(5, int(confidence * 10))
    
    # Convert coherence (0-1) to points (0-3)
    coherence_points = min(3, int(coherence * 6))
    
    base_score += confidence_points + coherence_points
    
    # Ensure score is between 0-15
    return max(0, min(15, base_score))

def generate_scores(
    clothing_items: List[Dict], 
    color_analysis: Dict[str, Any], 
    style_result: Dict[str, Any]
) -> Tuple[int, Dict[str, int]]:
    # Calculate individual component scores
    fit_score = calculate_fit_score(clothing_items, style_result)
    color_score = calculate_color_score(color_analysis, style_result)
    footwear_score = calculate_footwear_score(clothing_items, color_analysis, style_result)
    accessories_score = calculate_accessories_score(clothing_items, color_analysis, style_result)
    style_score = calculate_style_score(style_result)
    
    # Create score breakdown
    score_breakdown = {
        "fit": fit_score,
        "color": color_score,
        "footwear": footwear_score,
        "accessories": accessories_score,
        "style": style_score
    }
    
    # Get maximum possible scores
    max_scores = {
        "fit": 25,
        "color": 25,
        "footwear": 20,
        "accessories": 15,
        "style": 15
    }
    
    # Calculate weighted overall score
    overall_score = 0
    weights = settings.SCORE_WEIGHTS
    
    # Properly normalize each component by its maximum score and apply weight
    for component, score in score_breakdown.items():
        weight = weights.get(component, 0.2)
        max_component_score = max_scores.get(component, 100)
        # Convert to percentage then apply weight
        normalized_score = (score / max_component_score) * 100 * weight
        overall_score += normalized_score
    
    # Round to nearest integer
    overall_score = round(overall_score)
    
    return overall_score, score_breakdown