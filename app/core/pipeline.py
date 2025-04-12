# app/core/pipeline.py
import os
from typing import Dict, List, Any
from PIL import Image
import asyncio
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Set up file logging
import logging.handlers
log_file = "fashion_analysis.log"
file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

from app.config import settings
from app.models.item_detector import ItemDetector
from app.models.enhanced_color_analyzer import EnhancedColorAnalyzer
from app.models.style_classifier import StyleClassifier
from app.core.feedback_generator import generate_feedback
from app.core.learning_scorer import LearningBasedScorer
from app.models.fashion_clip_classifier import FashionCLIPClassifier

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
        
        # Initialize models
        item_detector = ItemDetector(model_path=r"C:\Users\Vamsi\fashion_rating_app\model_weights\best.pt")
        color_analyzer = EnhancedColorAnalyzer()
        style_classifier = StyleClassifier()
        fashion_clip = FashionCLIPClassifier(weights_path='weights/openfashionclip.pt')
        
        # Step 1: Detect clothing items
        logger.info("Detecting clothing items...")
        clothing_items = await item_detector.detect(image)
        logger.info(f"Detected {len(clothing_items)} clothing items")
        
        # Log initial YOLOv8 detections
        logger.info("Initial YOLOv8 detections:")
        for i, item in enumerate(clothing_items):
            logger.info(f"Item {i}: type={item['type']}, category={item['category']}, confidence={item['confidence']:.2f}")
            if 'box' in item:
                logger.info(f"  Box: {item['box']}")
        
        # Check specifically for top items
        top_items = [item for item in clothing_items if item["category"] in ["top", "outerwear"]]
        logger.info(f"Found {len(top_items)} top/outerwear items in YOLOv8 detection")
        
        # Check for top detection in the image even if not categorized as top
        possible_top_regions = []
        for item in clothing_items:
            # Items in the upper half of the image might be tops
            if "position" in item and item["position"]["center_y"] < 0.5:
                possible_top_regions.append(item)
                logger.info(f"Possible top region from position: {item['type']}")
        
        # Step 1.5: Enhance detections with CLIP
        logger.info("Enhancing detections with FashionCLIP...")
        enhanced_items = []
        for i, item in enumerate(clothing_items):
            if "crop" in item:
                logger.info(f"Item {i} has a crop with size: {item['crop'].size}")
                # Get CLIP analysis for this cropped item
                clip_result = fashion_clip.classify_garment(item["crop"])
                
                # Enhance the item with CLIP results
                enhanced_item = item.copy()
                
                # Update type with more specific description from CLIP
                if clip_result["garment"]["best_match"]["confidence"] > 0.3:  # Only use if confident
                    logger.info(f"Item {i} CLIP classification: {clip_result['garment']['best_match']['type']} with confidence {clip_result['garment']['best_match']['confidence']:.2f}")
                    enhanced_item["type"] = clip_result["garment"]["best_match"]["type"]
                    
                    # Update category based on type
                    if "dress" in enhanced_item["type"]:
                        enhanced_item["category"] = "dress"
                    elif any(term in enhanced_item["type"].lower() for term in ["shirt", "blouse", "top", "tee", "tank", "tunic", "jacket", "blazer", "coat"]):
                        enhanced_item["category"] = "top"
                        logger.info(f"Item {i} categorized as top based on CLIP type: {enhanced_item['type']}")
                    elif any(term in enhanced_item["type"].lower() for term in ["jean", "trouser", "pant", "skirt", "short"]):
                        enhanced_item["category"] = "bottom"
                
                # Store all CLIP results for later use
                enhanced_item["clip_analysis"] = clip_result
                enhanced_items.append(enhanced_item)
            else:
                logger.info(f"Item {i} has NO crop")
                enhanced_items.append(item)
        # Add our misclassification fix right here
        # Fix misclassified items by using YOLOv8 category information
        for item in clothing_items:
            if "clip_analysis" in item and "category" in item:
                item_type = item["clip_analysis"]["garment"]["best_match"]["type"]
                confidence = item["clip_analysis"]["garment"]["best_match"]["confidence"]
                
                # If YOLOv8 says it's a top/outerwear but CLIP thinks it's pants,
                # try to get the second-best match from CLIP
                if item["category"] in ["top", "outerwear"] and any(term in item_type.lower() for term in ["jean", "trouser", "pant"]):
                    logger.info(f"Potential misclassification: YOLOv8 says {item['category']} but CLIP says {item_type}")
                    
                    # Check for other top-like classifications in CLIP results
                    for match in item["clip_analysis"]["garment"]["top_matches"]:
                        alt_type = match["type"]
                        alt_conf = match["confidence"]
                        
                        if any(term in alt_type.lower() for term in ["shirt", "blouse", "jacket", "blazer"]):
                            logger.info(f"Using alternative CLIP classification: {alt_type} with confidence {alt_conf:.2f}")
                            item["type"] = alt_type
                            break
                    
                    # If no suitable alternative found, use the YOLOv8 type
                    if any(term in item["type"].lower() for term in ["jean", "trouser", "pant"]):
                        logger.info(f"Falling back to YOLOv8 type: {item['category']}")
                        item["type"] = item["category"]
        
        # Replace original items with enhanced ones
        clothing_items = enhanced_items
        
        # Step 2: Analyze colors
        logger.info("Analyzing color palette...")
        color_analysis = await color_analyzer.analyze_outfit_colors(image, clothing_items)
        
        # Step 3: Classify style (enhance with CLIP)
        logger.info("Classifying style...")
        style_result = await style_classifier.classify(
            image=image, 
            clothing_items=clothing_items,
            color_analysis=color_analysis
        )
        
        # Enhance style with CLIP if any items were analyzed
        clip_styles = []
        clip_confidences = []
        for item in clothing_items:
            if "clip_analysis" in item and "style" in item["clip_analysis"]:
                for style_match in item["clip_analysis"]["style"]["top_matches"]:
                    clip_styles.append(style_match["style"])
                    clip_confidences.append(style_match["confidence"])
        
        # If we have CLIP style information, consider it
        if clip_styles:
            # Find the most common style from CLIP
            from collections import Counter
            style_counter = Counter(clip_styles)
            most_common_style = style_counter.most_common(1)[0][0]
            
            # Map CLIP style to our style categories
            style_mapping = {
                "casual": "casual",
                "formal": "formal",
                "streetwear": "streetwear",
                "bohemian": "bohemian",
                "vintage": "vintage",
                "minimalist": "minimalist",
                "athleisure": "sporty",
                "edgy": "edgy_chic",
                "preppy": "preppy",
                "romantic": "romantic",
                "retro": "vintage",
                "contemporary": "fashion_week_off_duty"
            }
            
            # If the CLIP style maps to one of our categories and has high confidence, use it
            if most_common_style in style_mapping:
                mapped_style = style_mapping[most_common_style]
                avg_confidence = sum(clip_confidences) / len(clip_confidences)
                
                if avg_confidence > 0.2:  # Threshold for using CLIP style
                    style_result["style"] = mapped_style
                    style_result["confidence"] = avg_confidence
        
        logger.info(f"Detected style: {style_result['style']} with confidence {style_result['confidence']:.2f}")
        
        # Step 4: Use CLIP analysis for item types
        clip_top_types = []
        clip_bottom_types = []
        clip_dress_types = []
        clip_outerwear_types = []

        # Extract CLIP classifications for each item with confidence
        logger.info("Extracting CLIP classifications...")
        for item in clothing_items:
            if "clip_analysis" in item and "garment" in item["clip_analysis"]:
                item_type = item["clip_analysis"]["garment"]["best_match"]["type"]
                confidence = item["clip_analysis"]["garment"]["best_match"]["confidence"]
                
                logger.info(f"CLIP classified item as {item_type} with confidence {confidence:.2f}")
                
                # Only use classifications with reasonable confidence
                if confidence > 0.25:
                    # Explicitly categorize by recognizable terms
                    if any(term in item_type.lower() for term in ["jean", "trouser", "slack", "pant", "chino", "skirt", "short"]):
                        logger.info(f"Adding {item_type} to bottom types")
                        clip_bottom_types.append((item_type, confidence))
                    elif any(term in item_type.lower() for term in ["shirt", "blouse", "top", "tee", "tank", "tunic", "jacket", "blazer", "coat", "sweater", "hoodie", "cardigan"]):
                        logger.info(f"Adding {item_type} to top types")
                        clip_top_types.append((item_type, confidence))
                    elif any(term in item_type.lower() for term in ["dress", "gown"]):
                        logger.info(f"Adding {item_type} to dress types")
                        clip_dress_types.append((item_type, confidence))
                    elif any(term in item_type.lower() for term in ["jacket", "blazer", "coat", "cardigan"]):
                        logger.info(f"Adding {item_type} to outerwear types")
                        clip_outerwear_types.append((item_type, confidence))

        # If no tops were detected, try analyzing the whole upper portion of the image
        if not clip_top_types and not top_items:
            logger.info("No tops detected - analyzing upper image portion")
            # Create a crop of the upper third of the image
            upper_crop = image.crop((0, 0, image.width, image.height // 3))
            # Analyze with CLIP
            upper_analysis = fashion_clip.classify_garment(upper_crop)
            logger.info(f"Upper image CLIP analysis: {upper_analysis['garment']['best_match']}")
            
            # Check if CLIP found a top-like item
            upper_type = upper_analysis['garment']['best_match']['type']
            upper_conf = upper_analysis['garment']['best_match']['confidence']
            
            if upper_conf > 0.25 and any(term in upper_type.lower() for term in 
                                    ["shirt", "blouse", "top", "tee", "tank", "jacket", "coat"]):
                clip_top_types.append((upper_type, upper_conf))
                logger.info(f"Added whole-image top detection: {upper_type} ({upper_conf:.2f})")

        # Sort by confidence (highest first)
        clip_bottom_types.sort(key=lambda x: x[1], reverse=True)
        clip_top_types.sort(key=lambda x: x[1], reverse=True)
        clip_dress_types.sort(key=lambda x: x[1], reverse=True)
        clip_outerwear_types.sort(key=lambda x: x[1], reverse=True)

        # Log what we found
        logger.info(f"CLIP bottom types: {clip_bottom_types}")
        logger.info(f"CLIP top types: {clip_top_types}")
        logger.info(f"CLIP dress types: {clip_dress_types}")
        logger.info(f"CLIP outerwear types: {clip_outerwear_types}")

        # Determine primary types using CLIP data or fall back to YOLOv8
        top_type = "unknown_top"
        pants_type = "unknown_bottom"

        if clip_dress_types:
            # If dress is detected, it's the primary garment
            top_type = clip_dress_types[0][0]
            pants_type = "unknown_bottom"  # No separate bottom for dresses
            logger.info(f"Using dress type: {top_type}")
        else:
            # Handle top and bottom separately
            if clip_top_types:
                top_type = clip_top_types[0][0]
                logger.info(f"Using CLIP top type: {top_type}")
            else:
                top_type = item_detector.determine_top_type(clothing_items)
                logger.info(f"Using YOLO top type: {top_type}")
                
            if clip_bottom_types:
                pants_type = clip_bottom_types[0][0]
                logger.info(f"Using CLIP bottom type: {pants_type}")
            else:
                pants_type = item_detector.determine_pants_type(clothing_items)
                logger.info(f"Using YOLO bottom type: {pants_type}")

        # For fit type, use a combination of CLIP and YOLOv8
        if any(item.get("fit") == "wide_leg" for item in clothing_items) or any("wide" in b_type[0].lower() for b_type in clip_bottom_types):
            fit_type = "loose_flowy"
            logger.info("Using loose_flowy fit type based on wide leg detection")
        elif any(item.get("fit") == "skinny" for item in clothing_items) or any("skinny" in b_type[0].lower() for b_type in clip_bottom_types):
            fit_type = "fitted_slim"
            logger.info("Using fitted_slim fit type based on skinny detection")
        else:
            # Fall back to YOLOv8
            fit_type = item_detector.determine_fit_type(clothing_items)
            logger.info(f"Using YOLO fit type: {fit_type}")

        # For footwear and accessories, still rely on YOLOv8 as CLIP crops might not contain these
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