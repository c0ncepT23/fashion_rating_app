# test_fashion_rating.py
import asyncio
import json
import os
import sys
import numpy as np
from pprint import pprint

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.pipeline import process_fashion_image

def clean_for_serialization(obj):
    """Remove non-serializable objects from dictionaries and lists"""
    if isinstance(obj, dict):
        return {k: clean_for_serialization(v) for k, v in obj.items() 
                if not hasattr(v, 'mode') and k != 'crop' and k != 'sam_mask'}
    elif isinstance(obj, list):
        return [clean_for_serialization(item) for item in obj]
    elif hasattr(obj, 'mode') and hasattr(obj, 'getpixel'):  # PIL Image
        return "[Image object]"
    elif isinstance(obj, np.ndarray):  # Handle NumPy arrays
        return "[NumPy array]"
    else:
        return obj

async def test_fashion_rating(image_path):
    """
    Test the fashion rating pipeline with a sample image
    
    Args:
        image_path: Path to the test image
    """
    print(f"Testing fashion rating with image: {image_path}")
    
    # Process the image
    result = await process_fashion_image(image_path)
    
    # Pretty print the result
    print("\nRating Result:")
    print(f"Overall Score: {result['score']}/100")
    
    print("\nStyle Analysis:")
    print(f"Style: {result['labels']['style']}")
    print(f"Fit: {result['labels']['fit']}")
    print(f"Top: {result['labels']['top_type']}")
    print(f"Bottom: {result['labels']['pants_type']}")
    print(f"Footwear: {result['labels']['footwear_type']}")
    print(f"Accessories: {', '.join(result['labels']['accessories']) if result['labels']['accessories'] else 'None'}")
    print(f"Color Palette: {', '.join(result['labels']['color_palette'])}")
    
    print("\nScore Breakdown:")
    for component, score in result['score_breakdown'].items():
        max_score = 25 if component in ["fit", "color"] else 20 if component == "footwear" else 15
        print(f"{component.capitalize()}: {score}/{max_score}")
    
    print("\nFeedback:")
    for component, feedback in result['feedback'].items():
        print(f"{component.capitalize()}: {feedback}")
    
    # Save the result to a JSON file
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, os.path.basename(image_path) + ".json")
    
    # Clean the result for serialization
    serializable_result = clean_for_serialization(result)
    
    with open(output_path, "w") as f:
        json.dump(serializable_result, f, indent=2)
    
    print(f"\nFull result saved to: {output_path}")

if __name__ == "__main__":
    # Get image path from command line argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default test image if none provided
        image_path = "test_images/outfit1.jpg"
    
    # Run the test
    asyncio.run(test_fashion_rating(image_path))