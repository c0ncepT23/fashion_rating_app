# scripts/batch_evaluate.py
import os
import sys
import json
import asyncio
import argparse
from typing import Dict, Any, List

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.pipeline import process_fashion_image

async def process_images(image_dir: str, output_file: str):
    """
    Process all images in a directory and save results
    
    Args:
        image_dir: Directory containing images to process
        output_file: Path to save results
    """
    results = {}
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
        
        img_path = os.path.join(image_dir, img_file)
        
        try:
            # Process the image
            result = await process_fashion_image(img_path)
            
            # Add image path
            result["image_path"] = img_path
            
            # Store result with image path as key
            results[img_path] = result
            
            print(f"  Score: {result['score']}")
            print(f"  Style: {result['labels']['style']}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Save all results
    with open(output_file, 'w') as f:
        # Need to convert results to a list for JSON serialization
        results_list = [
            {
                "image_path": key, 
                **{k: v for k, v in value.items() if k != 'clothing_items'}
            } 
            for key, value in results.items()
        ]
        json.dump(results_list, f, indent=2)
    
    print(f"Saved results for {len(results)} images to {output_file}")
    return results

async def main():
    parser = argparse.ArgumentParser(description='Batch process fashion images')
    parser.add_argument('--image_dir', required=True, help='Directory containing images to process')
    parser.add_argument('--output', default='test_results.json', help='Path to save results')
    
    args = parser.parse_args()
    
    await process_images(args.image_dir, args.output)

if __name__ == "__main__":
    asyncio.run(main())