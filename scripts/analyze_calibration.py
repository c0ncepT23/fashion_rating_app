# scripts/analyze_calibration.py
import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def load_results(results_file: str) -> List[Dict]:
    """Load system results from file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def load_expert_ratings(ratings_file: str) -> List[Dict]:
    """Load expert ratings from file"""
    with open(ratings_file, 'r') as f:
        return json.load(f)

def calculate_calibration_factors(system_results: List[Dict], expert_ratings: List[Dict]) -> Dict:
    """
    Calculate calibration factors by comparing system results to expert ratings
    
    Args:
        system_results: List of system results
        expert_ratings: List of expert ratings
        
    Returns:
        dict: Suggested calibration factors by style and component
    """
    # Create DataFrame for analysis
    rows = []
    
    for system_result in system_results:
        # Get image filename (not full path)
        img_filename = os.path.basename(system_result["image_path"])
        
        # Find matching expert rating
        expert_rating = next((r for r in expert_ratings if r.get("image") == img_filename), None)
        if not expert_rating:
            print(f"No expert rating found for {img_filename}")
            continue
        
        # Get style
        style = system_result["labels"]["style"]
        
        # Extract scores
        for component in ["fit", "color", "footwear", "accessories", "style"]:
            system_score = system_result["score_breakdown"].get(component, 0)
            expert_score = expert_rating["score_breakdown"].get(component, 0)
            
            # Only consider non-zero scores
            if system_score > 0 and expert_score > 0:
                rows.append({
                    "image": img_filename,
                    "style": style,
                    "component": component,
                    "system_score": system_score,
                    "expert_score": expert_score,
                    "ratio": expert_score / system_score if system_score > 0 else 1.0
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Calculate median ratio by style and component
    style_component_ratios = df.groupby(['style', 'component'])['ratio'].median().reset_index()
    
    # Convert to dictionary format
    calibration_factors = {}
    for _, row in style_component_ratios.iterrows():
        style = row['style']
        component = row['component']
        ratio = row['ratio']
        
        # Limit to reasonable range
        ratio = max(0.7, min(1.3, ratio))
        
        if style not in calibration_factors:
            calibration_factors[style] = {}
        
        calibration_factors[style][component] = round(ratio, 2)
    
    # Fill in missing components with 1.0
    all_components = ["fit", "color", "footwear", "accessories", "style"]
    for style in calibration_factors:
        for component in all_components:
            if component not in calibration_factors[style]:
                calibration_factors[style][component] = 1.0
    
    return calibration_factors

def print_calibration_code(calibration_factors: Dict):
    """Print Python code for calibration factors"""
    print("# Updated calibration factors")
    print("calibration_factors = {")
    for style, components in calibration_factors.items():
        print(f'    "{style}": {{')
        for component, value in components.items():
            print(f'        "{component}": {value},')
        print("    },")
    print("}")

def main():
    parser = argparse.ArgumentParser(description='Analyze calibration factors')
    parser.add_argument('--results', required=True, help='Path to system results JSON file')
    parser.add_argument('--expert', required=True, help='Path to expert ratings JSON file')
    
    args = parser.parse_args()
    
    # Load data
    system_results = load_results(args.results)
    expert_ratings = load_expert_ratings(args.expert)
    
    print(f"Loaded {len(system_results)} system results and {len(expert_ratings)} expert ratings")
    
    # Calculate calibration factors
    calibration_factors = calculate_calibration_factors(system_results, expert_ratings)
    
    # Print Python code for updated calibration factors
    print_calibration_code(calibration_factors)

if __name__ == "__main__":
    main()