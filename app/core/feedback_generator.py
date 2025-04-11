# app/core/feedback_generator.py
from typing import Dict, List, Any
import random

from app.utils.constants import (
    FIT_DETAILS,
    COLOR_DETAILS,
    FOOTWEAR_DETAILS,
    ACCESSORIES_DETAILS,
    STYLE_DETAILS
)

# Templates for generating feedback
FIT_TEMPLATES = {
    "excellent": [
        "The {top_type} creates sharp contrast with the {bottom_type}. The proportions balance perfectly - {fit_detail}.",
        "Perfect balance between {top_type} and {bottom_type}. The {fit_detail} creates a flattering silhouette.",
        "The structured {top_type} pairs elegantly with the {bottom_type}. The overall silhouette {fit_detail}."
    ],
    "good": [
        "The {top_type} works well with the {bottom_type}. The proportions are good, with {fit_detail}.",
        "Nice pairing of {top_type} with {bottom_type}. The silhouette is flattering with {fit_detail}.",
        "The {top_type} and {bottom_type} create a harmonious proportion. {fit_detail}."
    ],
    "average": [
        "The {top_type} and {bottom_type} combination works, but {fit_detail} could be improved.",
        "Decent fit overall. The {top_type} and {bottom_type} work together, though {fit_detail}.",
        "The proportions between {top_type} and {bottom_type} are acceptable. Consider {fit_detail}."
    ],
    "poor": [
        "The {top_type} doesn't complement the {bottom_type} well. Consider {fit_detail}.",
        "The proportions between {top_type} and {bottom_type} could be better. Try {fit_detail}.",
        "The fit needs improvement. The {top_type} and {bottom_type} combination {fit_detail}."
    ]
}

COLOR_TEMPLATES = {
    "excellent": [
        "The {main_color} {item_type} creates a striking focal point against the {secondary_color} elements. {color_detail}.",
        "Masterful use of {main_color} and {secondary_color}. The color combination {color_detail}.",
        "The {main_color} and {secondary_color} palette is sophisticated and balanced. {color_detail}."
    ],
    "good": [
        "Good use of {main_color} with {secondary_color} accents. {color_detail}.",
        "The {main_color} {item_type} pairs nicely with the {secondary_color} elements. {color_detail}.",
        "Nice color coordination with {main_color} and {secondary_color}. {color_detail}."
    ],
    "average": [
        "The {main_color} and {secondary_color} combination works, but {color_detail}.",
        "Acceptable color pairing of {main_color} and {secondary_color}. Consider {color_detail}.",
        "The colors work, though {color_detail} would enhance the outfit."
    ],
    "poor": [
        "The {main_color} and {secondary_color} combination clashes. Try {color_detail}.",
        "Consider revisiting the color scheme. The {main_color} {item_type} {color_detail}.",
        "The color palette needs refinement. {color_detail}."
    ]
}

FOOTWEAR_TEMPLATES = {
    "excellent": [
        "The {footwear_type} perfectly complements this outfit. {footwear_detail}.",
        "Excellent choice with the {footwear_type}. {footwear_detail}.",
        "The {footwear_type} elevates the entire look. {footwear_detail}."
    ],
    "good": [
        "The {footwear_type} works well with this outfit. {footwear_detail}.",
        "Good selection with the {footwear_type}. {footwear_detail}.",
        "The {footwear_type} coordinates nicely with the outfit. {footwear_detail}."
    ],
    "average": [
        "The {footwear_type} is adequate for this outfit. Consider {footwear_detail}.",
        "The {footwear_type} works but {footwear_detail} would enhance the look.",
        "Acceptable choice of {footwear_type}, though {footwear_detail}."
    ],
    "poor": [
        "The {footwear_type} doesn't complement this outfit well. Try {footwear_detail}.",
        "Consider replacing the {footwear_type} with something that {footwear_detail}.",
        "The {footwear_type} detracts from the outfit. Opt for {footwear_detail}."
    ]
}

ACCESSORIES_TEMPLATES = {
    "excellent": [
        "The {accessories_list} add the perfect finishing touch. {accessories_detail}.",
        "Excellent accessorizing with {accessories_list}. {accessories_detail}.",
        "The {accessories_list} elevate the outfit beautifully. {accessories_detail}."
    ],
    "good": [
        "Good choice of accessories with {accessories_list}. {accessories_detail}.",
        "The {accessories_list} complement the outfit well. {accessories_detail}.",
        "Nice accessorizing with {accessories_list}. {accessories_detail}."
    ],
    "average": [
        "The {accessories_list} work with this outfit. Consider {accessories_detail}.",
        "Acceptable accessories, though {accessories_detail} would enhance the look.",
        "The {accessories_list} are adequate. Try {accessories_detail} for more impact."
    ],
    "poor": [
        "The accessories don't enhance this outfit. Consider {accessories_detail}.",
        "Reconsider the accessory choices. Try {accessories_detail}.",
        "The {accessories_list} distract from the outfit. Opt for {accessories_detail}."
    ],
    "none": [
        "This outfit could benefit from some accessories. Consider adding {accessories_detail}.",
        "Try adding some accessories like {accessories_detail} to complete the look.",
        "The outfit is missing accessories. {accessories_detail} would enhance it."
    ]
}

STYLE_TEMPLATES = {
    "excellent": [
        "Successfully combines elements for a cohesive {style} look. {style_detail}.",
        "Excellent execution of {style} style. {style_detail}.",
        "This outfit perfectly embodies {style} aesthetics. {style_detail}."
    ],
    "good": [
        "Good interpretation of {style} style. {style_detail}.",
        "This outfit effectively captures {style} elements. {style_detail}.",
        "Nice execution of {style} aesthetics. {style_detail}."
    ],
    "average": [
        "This outfit has {style} elements, but {style_detail} would make it more cohesive.",
        "Acceptable {style} look. Consider {style_detail} to enhance the style.",
        "The {style} influence is present, though {style_detail}."
    ],
    "poor": [
        "The {style} elements aren't cohesive. Try {style_detail}.",
        "This outfit doesn't fully capture {style} aesthetics. Consider {style_detail}.",
        "The style direction needs refinement. {style_detail}."
    ]
}

def get_score_category(score: int, max_score: int) -> str:
    """
    Convert numerical score to category
    
    Args:
        score: Numerical score
        max_score: Maximum possible score for this category
        
    Returns:
        str: Score category (excellent, good, average, poor)
    """
    percentage = score / max_score
    
    if percentage >= 0.85:
        return "excellent"
    elif percentage >= 0.7:
        return "good"
    elif percentage >= 0.5:
        return "average"
    else:
        return "poor"

def format_feedback_template(template: str, replacements: Dict[str, str]) -> str:
    """
    Format a feedback template with the provided replacements
    
    Args:
        template: Template string with placeholders
        replacements: Dictionary of replacement values
        
    Returns:
        str: Formatted feedback string
    """
    try:
        return template.format(**replacements)
    except KeyError as e:
        # Fallback for missing keys
        return f"This outfit shows {replacements.get('style', 'distinctive')} elements."

def generate_fit_feedback(clothing_items: List[Dict], score: int) -> str:
    """
    Generate feedback on fit/silhouette
    
    Args:
        clothing_items: Detected clothing items
        score: Fit score
        
    Returns:
        str: Formatted feedback
    """
    # Extract necessary information from clothing items
    top_items = [item for item in clothing_items if item["category"] == "top"]
    bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
    
    top_type = top_items[0]["type"] if top_items else "top"
    bottom_type = bottom_items[0]["type"] if bottom_items else "bottom"
    
    # Get appropriate template based on score
    category = get_score_category(score, 25)
    templates = FIT_TEMPLATES[category]
    template = random.choice(templates)
    
    # Select appropriate detail based on category
    fit_detail = random.choice(FIT_DETAILS[category])
    
    # Format the template
    replacements = {
        "top_type": top_type,
        "bottom_type": bottom_type,
        "fit_detail": fit_detail
    }
    
    return format_feedback_template(template, replacements)

def generate_color_feedback(color_analysis: Dict[str, Any], score: int) -> str:
    """
    Generate feedback on color palette
    
    Args:
        color_analysis: Extracted color palette
        score: Color score
        
    Returns:
        str: Formatted feedback
    """
    if "color_palette" not in color_analysis or not color_analysis["color_palette"]:
        return "Consider adding more color variation to create visual interest."
    
    color_palette = color_analysis["color_palette"]
    
    # Extract main and secondary colors
    main_color = color_palette[0]["name"]
    secondary_color = color_palette[1]["name"] if len(color_palette) > 1 else main_color
    
    # Get appropriate template based on score
    category = get_score_category(score, 25)
    templates = COLOR_TEMPLATES[category]
    template = random.choice(templates)
    
    # Select appropriate detail based on category
    color_detail = random.choice(COLOR_DETAILS[category])
    
    # Format the template
    replacements = {
        "main_color": main_color,
        "secondary_color": secondary_color,
        "item_type": "elements",  # This would ideally come from the actual items
        "color_detail": color_detail
    }
    
    return format_feedback_template(template, replacements)

def generate_footwear_feedback(clothing_items: List[Dict], score: int) -> str:
    """
    Generate feedback on footwear
    
    Args:
        clothing_items: Detected clothing items
        score: Footwear score
        
    Returns:
        str: Formatted feedback
    """
    footwear_items = [item for item in clothing_items if item["category"] == "footwear"]
    
    if not footwear_items:
        return "Consider adding appropriate footwear to complete the look."
    
    # Extract footwear type
    footwear_type = footwear_items[0]["type"]
    
    # Get appropriate template based on score
    category = get_score_category(score, 20)
    templates = FOOTWEAR_TEMPLATES[category]
    template = random.choice(templates)
    
    # Select appropriate detail based on category
    footwear_detail = random.choice(FOOTWEAR_DETAILS[category])
    
    # Format the template
    replacements = {
        "footwear_type": footwear_type,
        "footwear_detail": footwear_detail
    }
    
    return format_feedback_template(template, replacements)

def generate_accessories_feedback(clothing_items: List[Dict], score: int) -> str:
    """
    Generate feedback on accessories
    
    Args:
        clothing_items: Detected clothing items
        score: Accessories score
        
    Returns:
        str: Formatted feedback
    """
    accessories = [item for item in clothing_items if item["category"] == "accessory"]
    
    if not accessories:
        # No accessories detected
        template = random.choice(ACCESSORIES_TEMPLATES["none"])
        accessories_detail = random.choice(ACCESSORIES_DETAILS["none"])
        
        replacements = {
            "accessories_detail": accessories_detail
        }
        
        return format_feedback_template(template, replacements)
    
    # Format accessories list for readable text
    accessories_types = [item["type"] for item in accessories]
    if len(accessories_types) == 1:
        accessories_list = accessories_types[0]
    elif len(accessories_types) == 2:
        accessories_list = f"{accessories_types[0]} and {accessories_types[1]}"
    else:
        accessories_list = ", ".join(accessories_types[:-1]) + f", and {accessories_types[-1]}"
    
    # Get appropriate template based on score
    category = get_score_category(score, 15)
    templates = ACCESSORIES_TEMPLATES[category]
    template = random.choice(templates)
    
    # Select appropriate detail based on category
    accessories_detail = random.choice(ACCESSORIES_DETAILS[category])
    
    # Format the template
    replacements = {
        "accessories_list": accessories_list,
        "accessories_detail": accessories_detail
    }
    
    return format_feedback_template(template, replacements)

def generate_style_feedback(style: str, score: int) -> str:
    """
    Generate feedback on overall style
    
    Args:
        style: Detected style
        score: Style score
        
    Returns:
        str: Formatted feedback
    """
    # Get appropriate template based on score
    category = get_score_category(score, 15)
    templates = STYLE_TEMPLATES[category]
    template = random.choice(templates)
    
    # Format style name for readability
    formatted_style = style.replace("_", " ")
    
    # Select appropriate detail based on category and style
    style_detail_options = STYLE_DETAILS.get(style, STYLE_DETAILS["default"])
    style_detail = random.choice(style_detail_options[category])
    
    # Format the template
    replacements = {
        "style": formatted_style,
        "style_detail": style_detail
    }
    
    return format_feedback_template(template, replacements)

def generate_feedback(
    clothing_items: List[Dict],
    color_analysis: Dict[str, Any],
    style_result: Dict[str, Any],
    score_breakdown: Dict[str, int]
) -> Dict[str, str]:
    """
    Generate detailed feedback for all components of the outfit
    
    Args:
        clothing_items: Detected clothing items
        color_analysis: Color analysis results
        style_result: Style classification result
        score_breakdown: Score breakdown by component
        
    Returns:
        dict: Feedback for each component
    """
    # Generate feedback for each component
    fit_feedback = generate_fit_feedback(
        clothing_items=clothing_items,
        score=score_breakdown["fit"]
    )
    
    color_feedback = generate_color_feedback(
        color_analysis=color_analysis,
        score=score_breakdown["color"]
    )
    
    footwear_feedback = generate_footwear_feedback(
        clothing_items=clothing_items,
        score=score_breakdown["footwear"]
    )
    
    accessories_feedback = generate_accessories_feedback(
        clothing_items=clothing_items,
        score=score_breakdown["accessories"]
    )
    
    style_feedback = generate_style_feedback(
        style=style_result["style"],
        score=score_breakdown["style"]
    )
    
    # Compile all feedback
    feedback = {
        "fit": fit_feedback,
        "color": color_feedback,
        "footwear": footwear_feedback,
        "accessories": accessories_feedback,
        "style": style_feedback
    }
    
    return feedback