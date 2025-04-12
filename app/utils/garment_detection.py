# app/utils/garment_detection.py
from typing import List, Dict, Any

def detect_potential_dress(clothing_items: List[Dict], color_analysis: Dict[str, Any]) -> bool:
    """
    Detect when upper and lower garments are likely the same piece (i.e., a dress)
    
    Args:
        clothing_items: List of detected clothing items
        color_analysis: Color analysis results
        
    Returns:
        bool: True if a dress is likely present
    """
    # Find top and bottom items
    top_items = [item for item in clothing_items if item["category"] in ["top", "outerwear"]]
    bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
    
    # If we have both top and bottom items
    if top_items and bottom_items:
        # Check if they have similar colors
        if "dominant_color" in top_items[0] and "dominant_color" in bottom_items[0]:
            top_color = top_items[0]["dominant_color"]
            bottom_color = bottom_items[0]["dominant_color"]
            
            # Check for color similarity (exact match or similar color family)
            color_families = {
                "red": ["red", "ruby_red", "crimson", "burgundy", "maroon", "hot_pink"],
                "blue": ["blue", "navy", "light_blue", "cobalt", "royal_blue", "denim_blue"],
                "green": ["green", "mint", "sage_green", "emerald", "forest_green"],
                "pink": ["pink", "hot_pink", "blush_pink", "magenta"],
                "purple": ["purple", "lavender", "plum", "indigo", "mauve"],
                "yellow": ["yellow", "gold", "mustard"],
                "orange": ["orange", "coral", "peach", "terracotta"],
                "brown": ["brown", "tan", "camel", "khaki", "beige"],
                "black": ["black", "charcoal", "off_black"],
                "white": ["white", "cream", "off_white", "ivory"],
                "gray": ["gray", "silver", "light_gray", "dark_gray"]
            }
            
            # Check if colors are in the same family
            same_family = False
            for family, colors in color_families.items():
                if top_color in colors and bottom_color in colors:
                    same_family = True
                    break
            
            if top_color == bottom_color or same_family:
                # Check spatial relationship - are they adjacent?
                if "position" in top_items[0] and "position" in bottom_items[0]:
                    top_bottom = top_items[0]["position"]["center_y"] + (top_items[0]["position"]["height"] / 2)
                    bottom_top = bottom_items[0]["position"]["center_y"] - (bottom_items[0]["position"]["height"] / 2)
                    
                    # If they're close to each other
                    if abs(top_bottom - bottom_top) < 0.1:  # Threshold for adjacency
                        return True
                        
                # If we don't have position data, check box overlap
                elif "box" in top_items[0] and "box" in bottom_items[0]:
                    top_box = top_items[0]["box"]
                    bottom_box = bottom_items[0]["box"]
                    
                    # If the top item's bottom is close to the bottom item's top
                    if abs(top_box[3] - bottom_box[1]) < 50:  # Pixel threshold
                        return True
    
    # Also check CLIP predictions for dress types
    for item in clothing_items:
        if "clip_analysis" in item and "garment" in item["clip_analysis"]:
            for match in item["clip_analysis"]["garment"]["top_matches"]:
                if "dress" in match["type"].lower() and match["confidence"] > 0.3:
                    return True
    
    # No dress detected
    return False

def get_dress_type_from_items(clothing_items: List[Dict], color_analysis: Dict[str, Any]) -> str:
    """
    Determine the dress type from detected items and color analysis
    
    Args:
        clothing_items: List of detected clothing items
        color_analysis: Color analysis results
        
    Returns:
        str: Descriptive dress type
    """
    # Default dress type
    dress_type = "dress"
    
    # Try to get color from color analysis
    if color_analysis and "color_palette" in color_analysis and color_analysis["color_palette"]:
        main_color = color_analysis["color_palette"][0]["name"]
        dress_type = f"{main_color}_dress"
    
    # Check for specific dress features
    has_sleeve = False
    is_short = False
    
    for item in clothing_items:
        if "type" in item:
            item_type = item["type"].lower()
            if "sleeve" in item_type:
                has_sleeve = True
            if "short" in item_type or "mini" in item_type:
                is_short = True
    
    # Add style details to dress description
    if has_sleeve:
        if "short" in dress_type or is_short:
            dress_type = "short_sleeve_" + dress_type
        else:
            dress_type = "long_sleeve_" + dress_type
    elif is_short:
        dress_type = "mini_" + dress_type
    
    return dress_type