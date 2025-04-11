# app/core/learning_scorer.py
import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU")

class TorchRegressor(nn.Module):
    """
    Enhanced PyTorch regression model for fashion scoring
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        # Build a more sophisticated network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class LearningBasedScorer:
    """
    Learning-based scoring system for fashion ratings
    Uses neural networks trained on expert ratings, with GPU acceleration
    """
    
    def __init__(self, model_dir=None, training_data_path=None):
        """
        Initialize the learning-based scorer
        
        Args:
            model_dir: Directory to load/save trained models
            training_data_path: Path to expert ratings JSON file
        """
        self.model_dir = model_dir or "models/scoring"
        self.training_data_path = training_data_path
        self.models = {}
        self.feature_names = []
        self.trained = False
        
        # Try to load pre-trained models
        if os.path.exists(os.path.join(self.model_dir, "feature_names.json")):
            self._load_models()
    
    def extract_features(self, clothing_items, color_analysis, style_result):
        """
        Extract advanced features for scoring model
        
        Args:
            clothing_items: Detected clothing items
            color_analysis: Color analysis results
            style_result: Style classification result
            
        Returns:
            dict: Features for scoring model
        """
        features = {}
        
        # Basic counts
        features["num_items"] = len(clothing_items)
        features["num_colors"] = len(color_analysis.get("color_palette", []))
        
        # Category presence and proportion features
        categories = [item["category"] for item in clothing_items]
        unique_categories = set(categories)
        for category in ["top", "bottom", "outerwear", "dress", "footwear", "accessory"]:
            features[f"has_{category}"] = int(category in unique_categories)
            features[f"num_{category}"] = sum(1 for cat in categories if cat == category)
        
        # Color sophistication features
        if "color_palette" in color_analysis and color_analysis["color_palette"]:
            # High contrast detection (e.g., black and white)
            has_black = any(color["name"].lower().endswith("black") for color in color_analysis["color_palette"][:3])
            has_white = any(color["name"].lower().endswith("white") for color in color_analysis["color_palette"][:3])
            features["high_contrast"] = int(has_black and has_white)
            
            # Color harmony 
            if "harmony" in color_analysis:
                features["color_harmony_score"] = color_analysis["harmony"].get("score", 50) / 100
                harmony_type = color_analysis["harmony"].get("type", "unknown")
                for ht in ["monochromatic", "complementary", "analogous", "triadic", "neutral"]:
                    features[f"harmony_{ht}"] = int(harmony_type == ht)
        
        # Style features - more detailed
        style = style_result.get("style", "casual")
        features["style_confidence"] = style_result.get("confidence", 0.5)
        features["style_coherence"] = style_result.get("coherence", 0.5)
        
        # Style-specific features
        styles = ["casual", "formal", "streetwear", "minimalist", "bohemian", 
                 "edgy_chic", "preppy", "vintage", "fashion_week_off_duty"]
        for s in styles:
            features[f"style_{s}"] = int(style == s)
            
            # Style match strength (how well the detected items match this style)
            # Higher for the dominant style
            if style == s:
                features[f"style_match_{s}"] = 0.9
            else:
                # Get probability if available
                features[f"style_match_{s}"] = style_result.get("all_probabilities", {}).get(s, 0.1)
        
        # Fit features
        features["wide_leg_trousers"] = any(
            item.get("fit") == "wide_leg" and item["category"] == "bottom" 
            for item in clothing_items
        )
        
        features["structured_top"] = any(
            item.get("fit") == "structured" and item["category"] in ["top", "outerwear"] 
            for item in clothing_items
        )
        
        features["relaxed_fit"] = any(
            item.get("fit") in ["relaxed", "casual", "loose"] 
            for item in clothing_items
        )
        
        # Proportion features
        top_items = [item for item in clothing_items if item["category"] in ["top", "outerwear"]]
        bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
        
        if top_items and bottom_items:
            try:
                # Calculate silhouette balance - ratio of top width to bottom width
                top_width = max(item["position"]["width"] for item in top_items)
                bottom_width = max(item["position"]["width"] for item in bottom_items)
                
                if bottom_width > 0:
                    features["width_ratio"] = top_width / bottom_width
                    
                    # Fashion-specific proportion features
                    features["oversized_silhouette"] = int(features["width_ratio"] > 1.2)
                    features["balanced_silhouette"] = int(0.8 <= features["width_ratio"] <= 1.2)
                    features["bottom_heavy_silhouette"] = int(features["width_ratio"] < 0.8)
            except (KeyError, ZeroDivisionError):
                features["width_ratio"] = 1.0
                features["balanced_silhouette"] = 1
        
        # Accessory sophistication
        accessory_items = [item for item in clothing_items if item["category"] == "accessory"]
        features["accessory_count"] = len(accessory_items)
        features["has_sunglasses"] = int(any("sunglasses" in str(item.get("type", "")).lower() for item in accessory_items))
        features["has_jewelry"] = int(any(jewel in str(item.get("type", "")).lower() for item in accessory_items 
                                       for jewel in ["necklace", "ring", "bracelet", "chain"]))
        
        # Color combination features
        if "color_palette" in color_analysis and len(color_analysis["color_palette"]) >= 2:
            main_color = color_analysis["color_palette"][0]["name"]
            secondary_color = color_analysis["color_palette"][1]["name"]
            
            # Classic combinations
            classic_combos = [
                ("black", "white"), ("navy", "white"), ("black", "beige"),
                ("gray", "white"), ("navy", "red"), ("black", "red")
            ]
            
            # Check if colors make classic combo (in any order)
            features["classic_color_combo"] = int(
                any((main_color in combo and secondary_color in combo) for combo in classic_combos)
            )
        
        # Before returning, ensure all features are numeric
        # Convert any non-numeric values to 0
        for key in list(features.keys()):
            if not isinstance(features[key], (int, float, bool, np.number)):
                # Try to convert to float if possible
                try:
                    features[key] = float(features[key])
                except (ValueError, TypeError):
                    features[key] = 0.0
        
        # Fill in missing features from our known feature list if needed
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in features:
                    features[feature] = 0
        
        return features
    
    def train(self, training_data=None):
        """
        Train scoring models on expert ratings using GPU-accelerated PyTorch
        
        Args:
            training_data: Optional list of expert-rated outfits
        """
        # Load training data if not provided
        if training_data is None:
            if self.training_data_path and os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'r') as f:
                    training_data = json.load(f)
            else:
                raise ValueError("No training data provided or found")
        
        print(f"Training on {len(training_data)} expert-rated outfits...")
        
        # Extract features and targets from training data
        features_list = []
        targets = {
            "fit": [],
            "color": [],
            "footwear": [],
            "accessories": [],
            "style": [],
            "overall": []
        }
        
        for outfit in training_data:
            # Extract features
            try:
                features = self.extract_features(
                    outfit.get("clothing_items", []),
                    outfit.get("color_analysis", {}),
                    outfit.get("style_result", {})
                )
                
                # Ensure all feature values are numeric
                for key in list(features.keys()):
                    if not isinstance(features[key], (int, float, bool, np.number)):
                        try:
                            features[key] = float(features[key])
                        except (ValueError, TypeError):
                            features[key] = 0.0
                
                features_list.append(features)
                
                # Extract targets
                score_breakdown = outfit.get("score_breakdown", {})
                targets["fit"].append(score_breakdown.get("fit", 0))
                targets["color"].append(score_breakdown.get("color", 0))
                targets["footwear"].append(score_breakdown.get("footwear", 0))
                targets["accessories"].append(score_breakdown.get("accessories", 0))
                targets["style"].append(score_breakdown.get("style", 0))
                targets["overall"].append(outfit.get("score", 0))
            except Exception as e:
                print(f"Error processing outfit for training: {e}")
                continue
        
        # Convert to DataFrame for preprocessing
        features_df = pd.DataFrame(features_list)
        
        # Handle any remaining non-numeric values
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Remember feature names for inference
        self.feature_names = features_df.columns.tolist()
        
        # Save feature names
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "feature_names.json"), 'w') as f:
            json.dump(self.feature_names, f)
        
        # Convert to PyTorch tensors
        X = torch.tensor(features_df.values.astype(np.float32), dtype=torch.float32).to(device)
        
        # Training hyperparameters
        epochs = 200
        batch_size = 16 if len(training_data) > 16 else max(1, len(training_data) // 2)
        learning_rate = 0.001
        
        # Train a model for each component and overall score
        for component in targets.keys():
            print(f"Training model for {component}...")
            
            # Convert target to tensor
            y = torch.tensor(targets[component], dtype=torch.float32).view(-1, 1).to(device)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Create model
            model = TorchRegressor(input_dim=X.shape[1], hidden_dims=[128, 64, 32]).to(device)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            model.train()
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Print progress every 20 epochs
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate model
            model.eval()
            with torch.no_grad():
                predictions = model(X)
                test_loss = criterion(predictions, y)
                print(f"Final loss: {test_loss:.4f}")
            
            # Save the model
            self.models[component] = model
            torch.save(model.state_dict(), os.path.join(self.model_dir, f"{component}_model.pt"))
        
        # Remove architecture changed flag if it exists
        flag_path = os.path.join(self.model_dir, "architecture_changed.flag")
        if os.path.exists(flag_path):
            os.remove(flag_path)
            
        self.trained = True
        print("Training complete!")
    
    def _load_models(self):
        """Load pre-trained PyTorch models from disk"""
        try:
            # Load feature names
            with open(os.path.join(self.model_dir, "feature_names.json"), 'r') as f:
                self.feature_names = json.load(f)
            
            # Check if models exist but architecture has changed
            if os.path.exists(os.path.join(self.model_dir, "architecture_changed.flag")):
                print("Model architecture has changed. Models need to be retrained.")
                self.trained = False
                return
            
            # Load component models
            components = ["fit", "color", "footwear", "accessories", "style", "overall"]
            for component in components:
                model_path = os.path.join(self.model_dir, f"{component}_model.pt")
                if os.path.exists(model_path):
                    try:
                        # Create model with the correct input dimension
                        model = TorchRegressor(input_dim=len(self.feature_names), hidden_dims=[128, 64, 32]).to(device)
                        # Load saved weights
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        model.eval()  # Set to evaluation mode
                        self.models[component] = model
                    except Exception as e:
                        print(f"Error loading model for {component}: {e}")
                        # Mark architecture as changed if there's a mismatch
                        with open(os.path.join(self.model_dir, "architecture_changed.flag"), 'w') as f:
                            f.write("Model architecture has changed. Please retrain models.")
                        self.trained = False
                        return
            
            if self.models:
                self.trained = True
                print(f"Loaded {len(self.models)} pre-trained PyTorch models")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.trained = False
    
    def calibrate_scores(self, component_scores, style):
        """
        Calibrate scores to match expert expectations for different styles
        
        Args:
            component_scores: Dictionary of raw predicted scores
            style: Detected style
            
        Returns:
            dict: Calibrated component scores
        """
        # Updated calibration factors
        calibration_factors = {
            "bohemian": {
                "accessories": 1.0,
                "color": 1.0,
                "fit": 1.0,
                "footwear": 1.0,
                "style": 1.27,
            },
            "casual": {
                "accessories": 1.09,
                "color": 0.95,
                "fit": 1.0,
                "footwear": 1.06,
                "style": 1.27,
            },
            "fashion_week_off_duty": {
                "accessories": 1.0,
                "color": 1.05,
                "fit": 0.91,
                "footwear": 1.0,
                "style": 1.3,
            },
            "formal": {
                "accessories": 1.0,
                "color": 1.3,
                "fit": 0.89,
                "footwear": 1.04,
                "style": 1.3,
            },
            "minimalist": {
                "accessories": 1.12,
                "color": 1.11,
                "fit": 0.95,
                "footwear": 1.06,
                "style": 1.27,
            },
            "vintage": {
                "accessories": 1.14,
                "color": 1.0,
                "fit": 1.05,
                "footwear": 1.3,
                "style": 1.3,
            },
            # Keep original values for any missing styles
            "streetwear": {"fit": 0.95, "color": 1.1, "footwear": 1.15, "accessories": 1.1, "style": 1.0},
            "edgy_chic": {"fit": 1.0, "color": 1.0, "footwear": 1.0, "accessories": 1.0, "style": 1.0},
            "preppy": {"fit": 1.0, "color": 1.0, "footwear": 1.0, "accessories": 1.0, "style": 1.0},
        }
        
        # Default calibration if style not found
        default_calibration = {"fit": 1.0, "color": 1.0, "footwear": 1.0, "accessories": 1.0, "style": 1.0}
        
        # Get calibration factors for this style
        factors = calibration_factors.get(style, default_calibration)
        
        # Apply calibration
        calibrated_scores = {}
        for component, score in component_scores.items():
            factor = factors.get(component, 1.0)
            max_scores = {"fit": 25, "color": 25, "footwear": 20, "accessories": 15, "style": 15}
            max_score = max_scores.get(component, 100)
            
            # Apply factor and ensure within valid range
            calibrated_score = round(score * factor)
            calibrated_score = max(0, min(max_score, calibrated_score))
            
            calibrated_scores[component] = calibrated_score
        
        return calibrated_scores
    
    def predict(self, clothing_items, color_analysis, style_result):
        """
        Predict scores using trained PyTorch models with calibration
        
        Args:
            clothing_items: Detected clothing items
            color_analysis: Color analysis results
            style_result: Style classification result
            
        Returns:
            tuple: (overall_score, score_breakdown)
        """
        if not self.trained:
            # Fall back to rule-based scoring if not trained
            from app.core.rating_engine import generate_scores
            return generate_scores(clothing_items, color_analysis, style_result)
        
        # Extract features
        features = self.extract_features(clothing_items, color_analysis, style_result)
        
        # Create feature tensor
        feature_values = [features.get(feature, 0) for feature in self.feature_names]
        
        # Ensure all values are numeric
        feature_values = [float(val) if isinstance(val, (int, float, bool, np.number)) else 0.0 
                         for val in feature_values]
        
        feature_tensor = torch.tensor([feature_values], dtype=torch.float32).to(device)
        
        # Predict scores
        score_breakdown = {}
        
        with torch.no_grad():  # No need to track gradients for inference
            for component, model in self.models.items():
                if component != "overall":  # Skip overall for now
                    # Get model prediction
                    prediction = model(feature_tensor).item()
                    
                    # Clamp scores to valid ranges
                    max_scores = {"fit": 25, "color": 25, "footwear": 20, "accessories": 15, "style": 15}
                    max_score = max_scores.get(component, 100)
                    score = max(0, min(max_score, round(prediction)))
                    
                    score_breakdown[component] = score
        
        # Get style for calibration
        style = style_result.get("style", "casual")
        
        # Calibrate scores based on style
        score_breakdown = self.calibrate_scores(score_breakdown, style)
        
        # Predict or calculate overall score
        if "overall" in self.models:
            overall_prediction = self.models["overall"](feature_tensor).item()
            overall_score = max(0, min(100, round(overall_prediction)))
        else:
            # Calculate weighted sum if no overall model
            weights = {
                "fit": 0.25,
                "color": 0.25,
                "footwear": 0.20,
                "accessories": 0.15,
                "style": 0.15
            }
            
            max_scores = {"fit": 25, "color": 25, "footwear": 20, "accessories": 15, "style": 15}
            overall_score = sum(score * weights.get(component, 0.2) * 100 / max_scores.get(component, 100) 
                              for component, score in score_breakdown.items())
            overall_score = min(100, round(overall_score))
        
        return overall_score, score_breakdown