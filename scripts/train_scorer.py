# scripts/train_scorer.py
import os
import sys
import json
import argparse
import asyncio
from typing import List, Dict, Tuple, Any

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.learning_scorer import LearningBasedScorer
from app.core.pipeline import process_fashion_image

def load_expert_ratings(ratings_file: str) -> List[Dict]:
    """
    Load expert ratings from file
    
    Args:
        ratings_file: Path to expert ratings JSON file
        
    Returns:
        list: Expert ratings
    """
    with open(ratings_file, 'r') as f:
        return json.load(f)

def match_images_to_ratings(image_dir: str, expert_ratings: List[Dict]) -> List[Tuple[str, Dict]]:
    """
    Match images in directory to expert ratings
    
    Args:
        image_dir: Directory containing training images
        expert_ratings: List of expert ratings
        
    Returns:
        list: Matching (image_path, rating) pairs
    """
    matches = []
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images in {image_dir}")
    print(f"Looking for matches with {len(expert_ratings)} expert ratings")
    
    for img_file in image_files:
        # Find matching expert rating
        matching_ratings = [r for r in expert_ratings if r.get("image") == img_file]
        
        if matching_ratings:
            img_path = os.path.join(image_dir, img_file)
            matches.append((img_path, matching_ratings[0]))
    
    print(f"Found {len(matches)} matches between images and expert ratings")
    return matches

async def process_matched_images(matches: List[Tuple[str, Dict]]) -> List[Dict]:
    """
    Process matched images and combine with expert ratings
    
    Args:
        matches: List of (image_path, expert_rating) pairs
        
    Returns:
        list: Combined training data
    """
    training_data = []
    
    for i, (img_path, expert_rating) in enumerate(matches):
        print(f"Processing image {i+1}/{len(matches)}: {os.path.basename(img_path)}")
        
        try:
            # Process image with current pipeline
            result = await process_fashion_image(img_path)
            
            # Remove any non-serializable objects (like PIL Images)
            # Clean clothing items - they might contain image objects
            cleaned_items = []
            for item in result.get("clothing_items", []):
                cleaned_item = {k: v for k, v in item.items() if k != "crop" and not hasattr(v, 'mode')}
                cleaned_items.append(cleaned_item)
            
            # Create combined training record
            training_record = {
                "image_path": img_path,
                "clothing_items": cleaned_items,
                "color_analysis": result.get("color_analysis", {}),
                "style_result": {
                    "style": expert_rating["labels"]["style"],
                    "confidence": 0.9,  # Use expert style with high confidence
                    "coherence": 0.9
                },
                "score": expert_rating["score"],
                "score_breakdown": expert_rating["score_breakdown"]
            }
            
            training_data.append(training_record)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return training_data

async def process_single_image(img_path: str) -> Dict:
    """
    Process a single image for testing
    
    Args:
        img_path: Path to image
        
    Returns:
        dict: Processing result
    """
    print(f"Processing test image: {img_path}")
    
    try:
        # Process the image
        result = await process_fashion_image(img_path)
        print(f"Processed {img_path} successfully")
        return result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return {}

async def main():
    parser = argparse.ArgumentParser(description='Train fashion scoring model')
    parser.add_argument('--image_dir', default='training_images', help='Directory containing training images')
    parser.add_argument('--ratings', required=True, help='Path to expert ratings JSON file')
    parser.add_argument('--output', default='models/scoring', help='Directory to save trained models')
    parser.add_argument('--test', help='Test the model on a single image after training')
    
    args = parser.parse_args()
    
    # Load expert ratings
    expert_ratings = load_expert_ratings(args.ratings)
    print(f"Loaded {len(expert_ratings)} expert ratings")
    
    # Match images to ratings
    matches = match_images_to_ratings(args.image_dir, expert_ratings)
    
    if not matches:
        print("No matches found between images and ratings. Check filenames!")
        return
    
    # Process matched images
    training_data = await process_matched_images(matches)
    
    if not training_data:
        print("No training data was generated. Check image processing pipeline!")
        return
    
    print(f"Generated {len(training_data)} training examples")
    
    # Save training data for reference
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "training_data.json"), 'w') as f:
        json.dump(training_data, f)
    
    # Train the scorer
    scorer = LearningBasedScorer(model_dir=args.output)
    scorer.train(training_data)
    
    # Test on a single image if requested
    if args.test:
        result = await process_single_image(args.test)
        if result:
            print("\nTest Result:")
            print(f"Score: {result['score']}")
            print("Score Breakdown:")
            for component, score in result.get('score_breakdown', {}).items():
                print(f"  {component.capitalize()}: {score}")
            print("\nFeedback:")
            for component, feedback in result.get('feedback', {}).items():
                print(f"  {component.capitalize()}: {feedback}")

if __name__ == "__main__":
    asyncio.run(main())