# scripts/format_expert_ratings.py
import json
import os
import argparse

def format_expert_ratings(input_file, output_file):
    """
    Format expert ratings into consistent JSON structure
    
    Args:
        input_file: Path to input expert ratings file
        output_file: Path to save formatted expert ratings
    """
    with open(input_file, 'r') as f:
        expert_data = json.load(f)
    
    formatted_data = []
    
    for item in expert_data:
        # Ensure consistency in structure
        formatted_item = {
            "image": item.get("image_path", "unknown.jpg"),
            "score": item.get("score", 0),
            "labels": item.get("labels", {}),
            "score_breakdown": item.get("score_breakdown", {
                "fit": 0,
                "color": 0,
                "footwear": 0,
                "accessories": 0,
                "style": 0
            }),
            "feedback": item.get("feedback", {})
        }
        
        formatted_data.append(formatted_item)
    
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Formatted {len(formatted_data)} expert ratings to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format expert ratings')
    parser.add_argument('--input', required=True, help='Path to input expert ratings file')
    parser.add_argument('--output', default='expert_ratings_formatted.json', help='Path to save formatted expert ratings')
    
    args = parser.parse_args()
    
    format_expert_ratings(args.input, args.output)