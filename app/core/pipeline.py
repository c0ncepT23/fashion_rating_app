# app/core/pipeline.py
import os
from typing import Dict, Any
from PIL import Image
import asyncio
import time
import logging

from app.config import settings
from app.models.item_detector import ItemDetector
#from app.models.color_analyzer import ColorAnalyzer
from app.models.style_classifier import StyleClassifier
from app.core.feedback_generator import generate_feedback
from app.core.learning_scorer import LearningBasedScorer
from app.models.enhanced_color_analyzer import EnhancedColorAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

# Initialize the learning-based scorer
learning_scorer = LearningBasedScorer(model_dir="models/scoring")

async def process_fashion_image(image_path: str) -> Dict[str, Any]:
    """
    Main pipeline for processing a fashion image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Complete fashion rating results
    """
    try:
        start_time = time.time()
        logger.info(f"Starting analysis of image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Initialize models with your trained YOLOv8 model
        item_detector = ItemDetector(model_path=r"C:\Users\Vamsi\fashion_rating_app\model_weights\best.pt")
        color_analyzer = EnhancedColorAnalyzer()
        style_classifier = StyleClassifier()
        
        # Step 1: Detect clothing items
        logger.info("Detecting clothing items...")
        clothing_items = await item_detector.detect(image)
        logger.info(f"Detected {len(clothing_items)} clothing items")
        
        # Step 2: Analyze colors
        logger.info("Analyzing color palette...")
        color_analysis = await color_analyzer.analyze_outfit_colors(image, clothing_items)
        
        # Step 3: Classify style
        logger.info("Classifying style...")
        style_result = await style_classifier.classify(
            image=image, 
            clothing_items=clothing_items,
            color_analysis=color_analysis
        )
        logger.info(f"Detected style: {style_result['style']} with confidence {style_result['confidence']:.2f}")
        
        # Step 4: Determine specific item types
        fit_type = item_detector.determine_fit_type(clothing_items)
        pants_type = item_detector.determine_pants_type(clothing_items) 
        top_type = item_detector.determine_top_type(clothing_items)
        footwear_type = item_detector.determine_footwear_type(clothing_items)
        accessories = item_detector.determine_accessories(clothing_items)
        
        # Step 5: Generate scores using learning-based scorer
        logger.info("Generating scores...")
        score, score_breakdown = learning_scorer.predict(
            clothing_items=clothing_items,
            color_analysis=color_analysis,
            style_result=style_result
        )
        logger.info(f"Overall score: {score}/100")
        
        # Step 6: Generate feedback
        logger.info("Generating feedback...")
        feedback = generate_feedback(
            clothing_items=clothing_items,
            color_analysis=color_analysis,
            style_result=style_result,
            score_breakdown=score_breakdown
        )
        
        # Step 7: Create and format results
        color_palette = [color["name"] for color in color_analysis.get("color_palette", [])]
        
        labels = {
            "fit": fit_type,
            "pants_type": pants_type,
            "top_type": top_type,
            "color_palette": color_palette[:5] if color_palette else ["unknown"],
            "footwear_type": footwear_type,
            "accessories": accessories,
            "style": style_result["style"]
        }
        
        result = {
            "score": score,
            "labels": labels,
            "score_breakdown": score_breakdown,
            "feedback": feedback,
            # Store these for potential training data
            "clothing_items": clothing_items,
            "color_analysis": color_analysis,
            "style_result": style_result
        }
        
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in processing pipeline: {str(e)}")
        raise