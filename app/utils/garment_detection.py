# app/utils/garment_detection.py
from typing import List, Dict, Any

def detect_potential_dress(clothing_items: List[Dict], color_analysis: Dict[str, Any]) -> bool:
    """
    Detect when upper and lower garments are likely the same piece (i.e., a dress)
    (Modified for CLIP-only approach)
    
    Args:
        clothing_items: List of detected clothing items
        color_analysis: Color analysis results
        
    Returns:
        bool: True if a dress is likely present
    """
    # Direct check - see if any item is categorized as a dress
    dress_items = [item for item in clothing_items if item.get("category") == "dress"]
    if dress_items:
        return True
    
    # Check if "dress" appears in any item type
    for item in clothing_items:
        item_type = item.get("type", "").lower()
        if "dress" in item_type or "gown" in item_type:
            return True
    
    # Find top and bottom items
    top_items = [item for item in clothing_items if item.get("category") in ["top", "outerwear"]]
    bottom_items = [item for item in clothing_items if item.get("category") == "bottom"]
    
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
    
    # Check CLIP analysis results if available
    for item in clothing_items:
        if "clip_analysis" in item and "garment" in item["clip_analysis"]:
            # Check best_match first
            if "best_match" in item["clip_analysis"]["garment"]:
                best_match = item["clip_analysis"]["garment"]["best_match"]
                if "type" in best_match and "dress" in best_match["type"].lower():
                    if best_match.get("confidence", 0) > 0.3:
                        return True
            
            # Check top_matches if available
            if "top_matches" in item["clip_analysis"]["garment"]:
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
    
    # Check for dress items directly
    dress_items = [item for item in clothing_items if item.get("category") == "dress"]
    if dress_items:
        dress_item = dress_items[0]
        if "type" in dress_item:
            dress_type = dress_item["type"]
    
    # Try to get color from color analysis
    if color_analysis and "color_palette" in color_analysis and color_analysis["color_palette"]:
        main_color = color_analysis["color_palette"][0]["name"]
        dress_type = f"{main_color}_{dress_type}"
    
    # Check for specific dress features
    has_sleeve = False
    is_short = False
    is_floral = False
    
    for item in clothing_items:
        if "type" in item:
            item_type = item["type"].lower()
            if "sleeve" in item_type:
                has_sleeve = True
            if "short" in item_type or "mini" in item_type:
                is_short = True
            if "floral" in item_type or ("has_pattern" in item and item["has_pattern"]):
                is_floral = True
    
    # Add style details to dress description
    if is_floral:
        dress_type = "floral_" + dress_type
    elif has_sleeve:
        if "short" in dress_type or is_short:
            dress_type = "short_sleeve_" + dress_type
        else:
            dress_type = "long_sleeve_" + dress_type
    elif is_short:
        dress_type = "mini_" + dress_type
    
    return dress_type