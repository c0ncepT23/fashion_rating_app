# app/models/feature_extractor.py
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import asyncio
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import random

class FeatureExtractor:
    """
    Extracts features from fashion images for detailed analysis
    """
    
    def __init__(self):
        """
        Initialize the feature extractor
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        self.model = models.resnet50(pretrained=True)
        
        # Remove classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Set up image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    async def extract_features(self, image: Image.Image, clothing_items: List[Dict] = None) -> Dict[str, Any]:
        """
        Extract features from a fashion image
        
        Args:
            image: PIL Image object
            clothing_items: Optional list of detected clothing items
            
        Returns:
            dict: Extracted features
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # In a real async setting, we would run inference in a thread pool
        # For demonstration, we use a simple await with a dummy coroutine
        await asyncio.sleep(0)
        
        # Extract global image features
        with torch.no_grad():
            global_features = self.model(img_tensor).squeeze().cpu().numpy()
        
        # Calculate various outfit properties
        # In a real application, these would be determined by sophisticated models
        # Here we use simpler approaches and some randomized values for demonstration
        
        # Analyze proportions based on clothing items
        good_proportions = True
        if clothing_items:
            # Check for balanced top/bottom ratio
            top_items = [item for item in clothing_items if item["category"] in ["top", "outerwear"]]
            bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
            
            if top_items and bottom_items:
                # Calculate area ratio between top and bottom
                top_area = sum(item["position"]["area"] for item in top_items)
                bottom_area = sum(item["position"]["area"] for item in bottom_items)
                
                # Ideal ratio is around 1:1 to 3:2
                ratio = top_area / bottom_area if bottom_area > 0 else 1.0
                good_proportions = 0.8 <= ratio <= 1.5
        
        # Generate simulated feature values
        style_confidence = random.uniform(0.7, 0.95)  # Confidence of style classifier
        style_coherence = random.uniform(0.65, 0.95)  # How well items work together
        
        # Analyze fit appropriateness
        appropriate_fit_for_style = True
        
        # Identify any fit issues (simulated)
        fit_issues = []
        if random.random() < 0.3:  # 30% chance of having an issue
            possible_issues = ["proportion_imbalance", "oversized", "too_tight", "length_issue"]
            fit_issues = [random.choice(possible_issues)]
        
        # Compile all features
        features = {
            "global_features": global_features.tolist(),  # Convert numpy array to list for JSON serialization
            "good_proportions": good_proportions,
            "appropriate_fit_for_style": appropriate_fit_for_style,
            "fit_issues": fit_issues,
            "style_confidence": style_confidence,
            "style_coherence": style_coherence
        }
        
        return features
    
    def analyze_outfit_balance(self, clothing_items: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the balance and proportion of an outfit
        
        Args:
            clothing_items: List of detected clothing items
            
        Returns:
            dict: Balance analysis results
        """
        # This is a simplified implementation
        # In a real application, this would use more sophisticated analysis
        
        # Extract item positions
        items_by_category = {}
        for item in clothing_items:
            category = item["category"]
            if category not in items_by_category:
                items_by_category[category] = []
            items_by_category[category].append(item)
        
        # Check vertical balance (top vs bottom)
        top_items = items_by_category.get("top", []) + items_by_category.get("outerwear", [])
        bottom_items = items_by_category.get("bottom", [])
        
        vertical_balance = 0.5  # Default balanced
        if top_items and bottom_items:
            top_area = sum(item["position"]["area"] for item in top_items)
            bottom_area = sum(item["position"]["area"] for item in bottom_items)
            total_area = top_area + bottom_area
            
            if total_area > 0:
                top_ratio = top_area / total_area
                vertical_balance = top_ratio
        
        # Check horizontal balance (left vs right)
        left_items = [item for item in clothing_items if item["position"]["center_x"] < 0.5]
        right_items = [item for item in clothing_items if item["position"]["center_x"] >= 0.5]
        
        horizontal_balance = 0.5  # Default balanced
        if left_items and right_items:
            left_area = sum(item["position"]["area"] for item in left_items)
            right_area = sum(item["position"]["area"] for item in right_items)
            total_area = left_area + right_area
            
            if total_area > 0:
                left_ratio = left_area / total_area
                horizontal_balance = left_ratio
        
        # Calculate "golden ratio" adherence
        # The golden ratio is approximately 1.618 (or roughly 0.62 / 0.38 split)
        golden_ratio = 0.618
        vertical_golden_ratio_adherence = 1.0 - abs(vertical_balance - golden_ratio)
        
        return {
            "vertical_balance": vertical_balance,
            "horizontal_balance": horizontal_balance,
            "golden_ratio_adherence": vertical_golden_ratio_adherence
        }