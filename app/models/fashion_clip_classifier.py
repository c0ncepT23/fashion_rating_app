# app/models/fashion_clip_classifier.py
from typing import List, Dict, Any
import torch
import open_clip
from PIL import Image

class FashionCLIPClassifier:
    """
    Classifier using Open-Fashion-CLIP for accurate fashion item classification
    """
    
    def __init__(self, weights_path='weights/openfashionclip.pt'):
        """
        Initialize the Fashion CLIP classifier
        
        Args:
            weights_path: Path to OpenFashionCLIP weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        print("Loading OpenFashionCLIP model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B/32')
        state_dict = torch.load(weights_path, map_location=self.device)
        
        # Handle different state dict formats
        if 'CLIP' in state_dict:
            self.model.load_state_dict(state_dict['CLIP'])
        else:
            self.model.load_state_dict(state_dict)
            
        self.model = self.model.eval().requires_grad_(False).to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        # Initialize fashion categories
        self._init_fashion_categories()
        
        print("OpenFashionCLIP model loaded successfully!")
        
    def _init_fashion_categories(self):
        """Initialize fashion categories for classification"""
        # Define garment types with more specific fashion terms
        self.garment_types = [
            "t-shirt", "tank top", "crop top", 
            "blouse", "shirt", "dress shirt",
            "sweater", "cardigan", "hoodie", 
            "jacket", "blazer", "coat",
            "jeans", "skinny jeans", "straight leg jeans", "wide leg jeans",
            "trousers", "pants", "slacks", "chinos",
            "shorts", "bermuda shorts", "skirt", "mini skirt", "midi skirt", "maxi skirt",
            "dress", "mini dress", "midi dress", "maxi dress", 
            "sleeveless dress", "sundress", "slip dress",
            "tunic", "asymmetric tunic", "peplum top",
            "jumpsuit", "romper", "overalls"
        ]
        
        # More specific styles
        self.styles = [
            "casual", "formal", "business casual", "streetwear", 
            "bohemian", "vintage", "preppy", "athleisure",
            "minimalist", "maximalist", "edgy", "romantic",
            "retro", "classic", "avant-garde", "contemporary"
        ]
        
        # Create templates for better classification
        self.templates = [
            "a photo of a {}",
            "a picture of a {}",
            "a {} garment",
            "a {} clothing item"
        ]
        
    def classify_garment(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify a fashion item using OpenFashionCLIP
        
        Args:
            image: PIL Image of the clothing item
            
        Returns:
            dict: Classification results with garment type and style
        """
        # Preprocess image
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Prepare text inputs for garment types
        text_inputs = []
        for template in self.templates[:1]:  # Use just first template for speed
            for garment in self.garment_types:
                text_inputs.append(template.format(garment))
                
        tokenized_text = self.tokenizer(text_inputs).to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(tokenized_text)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        # Get top matches
        values, indices = similarity[0].topk(5)
        
        # Extract results
        top_matches = []
        for i, idx in enumerate(indices):
            template_idx = idx.item() // len(self.garment_types)
            garment_idx = idx.item() % len(self.garment_types)
            garment_type = self.garment_types[garment_idx]
            
            top_matches.append({
                "type": garment_type,
                "confidence": values[i].item()
            })
        
        # Now classify style
        style_inputs = [f"a {top_matches[0]['type']} in {style} style" for style in self.styles]
        tokenized_styles = self.tokenizer(style_inputs).to(self.device)
        
        with torch.no_grad():
            style_features = self.model.encode_text(tokenized_styles)
            style_features /= style_features.norm(dim=-1, keepdim=True)
            style_similarity = (100.0 * image_features @ style_features.T).softmax(dim=-1)
        
        style_values, style_indices = style_similarity[0].topk(3)
        top_styles = [
            {"style": self.styles[idx.item()], "confidence": val.item()}
            for val, idx in zip(style_values, style_indices)
        ]
        
        return {
            "garment": {
                "best_match": top_matches[0],
                "top_matches": top_matches
            },
            "style": {
                "best_match": top_styles[0],
                "top_matches": top_styles
            }
        }