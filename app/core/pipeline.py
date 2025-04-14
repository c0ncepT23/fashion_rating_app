# app/core/pipeline.py
import os
from typing import Dict, List, Any
from PIL import Image
import asyncio
import time
import logging
import torch
import numpy as np
import cv2
import json
from collections import Counter

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
from app.core.feedback_reconciliation import reconcile_fashion_analysis
from app.utils.garment_detection import detect_potential_dress, get_dress_type_from_items

# Import SAM
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model once
sam_checkpoint_path = os.path.join("weights", "sam_vit_b_01ec64.pth")
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# Initialize the learning-based scorer
learning_scorer = LearningBasedScorer(model_dir="models/scoring")

async def process_fashion_image(image_path: str) -> Dict[str, Any]:
    """
    Main pipeline for processing a fashion image (SAM+CLIP version)
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Complete fashion rating results
    """
    try:
        # Initialize dictionary to store all model outputs
        model_outputs = {}
        
        start_time = time.time()
        logger.info(f"Starting analysis of image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Initialize models
        fashion_clip = FashionCLIPClassifier(weights_path='weights/openfashionclip.pt')
        # Use SAM predictor with CLIP-based detector
        item_detector = ItemDetector(fashion_clip=fashion_clip, sam_predictor=predictor)
        color_analyzer = EnhancedColorAnalyzer()
        style_classifier = StyleClassifier()
        
        # Step 1: Detect clothing items using SAM+CLIP
        logger.info("Detecting clothing items with SAM+CLIP...")
        clothing_items = await item_detector.detect(image)
        model_outputs["sam_clip_detections"] = clothing_items.copy()
        logger.info(f"Detected {len(clothing_items)} clothing items with SAM+CLIP")
        
        # Log what was detected
        for i, item in enumerate(clothing_items):
            logger.info(f"Item {i}: type={item.get('type', 'unknown')}, category={item.get('category', 'unknown')}, confidence={item.get('confidence', 0):.2f}")
            if 'position' in item:
                logger.info(f"  Position: center_x={item['position']['center_x']:.2f}, center_y={item['position']['center_y']:.2f}")
        
        # Convert PIL image to numpy for additional processing
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR for OpenCV
        
        # Special case for detecting patterns in garments
        pattern_detection = []
        for i, item in enumerate(clothing_items):
            # Use SAM mask if available, otherwise use general image analysis
            if "sam_mask" in item:
                mask = item["sam_mask"]
                masked_img = cv2.bitwise_and(cv_image, cv_image, mask=mask.astype(np.uint8)*255)
                if np.sum(mask) > 0:  # Check if mask has any pixels
                    # Convert to grayscale
                    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
                    # Calculate texture features using gradient magnitude
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                    # Only consider pixels within the mask
                    mask_pixels = np.sum(mask)
                    if mask_pixels > 0:
                        pattern_score = float(np.sum(gradient_magnitude * mask) / mask_pixels)
                    else:
                        pattern_score = 0
                else:
                    # Fallback for empty mask
                    pattern_score = 0
            else:
                # For items without a mask, analyze the entire image
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                pattern_score = float(np.mean(gradient_magnitude))
            
            # Determine if this item has a pattern
            has_pattern = pattern_score > 20  # Threshold determined empirically
            
            item["has_pattern"] = has_pattern
            pattern_detection.append({
                "item_index": i,
                "pattern_score": pattern_score,
                "has_pattern": has_pattern
            })
            
            if has_pattern:
                logger.info(f"Detected pattern in item {i} with score {pattern_score:.2f}")
        
        model_outputs["pattern_detection"] = pattern_detection
        
        # Step 2: Analyze colors using the masks from SAM
        logger.info("Analyzing color palette...")
        color_analysis = await color_analyzer.analyze_outfit_colors(image, clothing_items)
        model_outputs["color_analysis"] = color_analysis
        
        # Step 3: Classify style
        logger.info("Classifying style...")
        style_result = await style_classifier.classify(
            image=image, 
            clothing_items=clothing_items,
            color_analysis=color_analysis
        )
        model_outputs["style_classification"] = style_result
        
        # Special check for patterns in bohemian or romantic styles
        has_floral_pattern = any(
            item.get("has_pattern", False) and 
            any(floral_term in str(item.get("type", "")).lower() for floral_term in ["floral", "flower", "print"]) 
            for item in clothing_items
        )
        
        has_pattern = any(item.get("has_pattern", False) for item in clothing_items)
        
        # If we detect patterns, adjust style scores
        if has_pattern:
            if has_floral_pattern:
                # Floral patterns strongly suggest bohemian or romantic styles
                if style_result["style"] not in ["bohemian", "romantic"]:
                    # Check if any garment has "dress" in its type
                    has_dress = any("dress" in str(item.get("type", "")).lower() for item in clothing_items)
                    if has_dress:
                        style_result["style"] = "bohemian" if style_result.get("confidence", 0) < 0.4 else style_result["style"]
                        logger.info(f"Adjusted style to bohemian based on floral dress detection")
        
        logger.info(f"Detected style: {style_result['style']} with confidence {style_result['confidence']:.2f}")
        
        # Step 4: Extract key garment types for labeling
        logger.info("Extracting garment types...")
        top_type = item_detector.determine_top_type(clothing_items)
        pants_type = item_detector.determine_pants_type(clothing_items)
        fit_type = item_detector.determine_fit_type(clothing_items)
        footwear_type = item_detector.determine_footwear_type(clothing_items)
        accessories = item_detector.determine_accessories(clothing_items)
        
        # Add pattern information to names if detected
        if has_pattern:
            # Check if there's a dress with pattern
            dress_items = [item for item in clothing_items if item["category"] == "dress"]
            if dress_items and dress_items[0].get("has_pattern", False):
                if has_floral_pattern:
                    top_type = "floral_" + dress_items[0]["type"]
                else:
                    top_type = "patterned_" + dress_items[0]["type"]
                pants_type = "none"  # No separate bottom for dresses
            
            # Check for patterned top
            top_items = [item for item in clothing_items if item["category"] in ["top", "outerwear"]]
            if top_items and top_items[0].get("has_pattern", False):
                if has_floral_pattern:
                    top_type = "floral_" + top_items[0]["type"]
                else:
                    top_type = "patterned_" + top_items[0]["type"]
                    
            # Check for patterned bottom
            bottom_items = [item for item in clothing_items if item["category"] == "bottom"]
            if bottom_items and bottom_items[0].get("has_pattern", False):
                if has_floral_pattern:
                    pants_type = "floral_" + bottom_items[0]["type"]
                else:
                    pants_type = "patterned_" + bottom_items[0]["type"]
        
        # Step 5: Generate scores
        logger.info("Generating scores...")
        score, score_breakdown = learning_scorer.predict(
            clothing_items=clothing_items,
            color_analysis=color_analysis,
            style_result=style_result
        )
        model_outputs["score_prediction"] = {
            "overall_score": score,
            "score_breakdown": score_breakdown
        }
        logger.info(f"Overall score: {score}/100")
        
        # Step 6: Generate feedback
        logger.info("Generating feedback...")
        feedback = generate_feedback(
            clothing_items=clothing_items,
            color_analysis=color_analysis,
            style_result=style_result,
            score_breakdown=score_breakdown
        )
        model_outputs["feedback_generation"] = feedback
        
        # Step 7: Create and format results
        # Deduplicate colors and avoid repetition
        unique_colors = []
        unique_color_names = set()
        for color in color_analysis.get("color_palette", []):
            if color["name"] not in unique_color_names:
                unique_colors.append(color)
                unique_color_names.add(color["name"])
                
        color_palette = [color["name"] for color in unique_colors]
        
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
        
        # Step 8: Reconcile results to ensure consistency
        reconciled_result = reconcile_fashion_analysis(
            result=result, 
            clothing_items=clothing_items, 
            color_analysis=color_analysis
        )
        
        # Log all model outputs
        log_dir = log_model_outputs(image_path, model_outputs)
        reconciled_result["log_dir"] = log_dir
        
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        
        return reconciled_result
    
    except Exception as e:
        logger.error(f"Error in processing pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def log_model_outputs(image_path, model_outputs, output_dir="model_logs"):
    """
    Log all model outputs to structured files for analysis
    
    Args:
        image_path: Path to the input image
        model_outputs: Dictionary of outputs from different models
        output_dir: Directory to save logs
    """
    import os
    import json
    import time
    import numpy as np
    from PIL import Image
    
    # Create a timestamp for the log
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Get base filename without extension
    base_name = os.path.basename(image_path).split('.')[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory for this image analysis
    log_dir = os.path.join(output_dir, f"{base_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Helper function to convert non-serializable objects
    def sanitize_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size < 1000 else f"<ndarray: shape={obj.shape}>"
        elif hasattr(obj, 'mode') and hasattr(obj, 'getpixel'):  # PIL Image
            return f"<PIL.Image: mode={obj.mode}, size={obj.size}>"
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(item) for item in obj]
        else:
            return str(obj) if not isinstance(obj, (int, float, bool, str, type(None))) else obj
    
    # Process and save each model's output
    for model_name, output in model_outputs.items():
        # Skip None or empty outputs
        if output is None:
            continue
            
        # Convert to JSON-serializable format
        if isinstance(output, dict):
            serializable_output = {k: sanitize_for_json(v) for k, v in output.items()}
        elif isinstance(output, list):
            serializable_output = [sanitize_for_json(item) for item in output]
        else:
            serializable_output = sanitize_for_json(output)
        
        # Save to JSON file
        output_file = os.path.join(log_dir, f"{model_name}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_output, f, indent=2)
        except TypeError as e:
            print(f"Error serializing {model_name}: {e}")
            # Fallback to string representation
            with open(output_file, 'w') as f:
                f.write(str(serializable_output))
    
    # Create a summary file with all outputs
    summary_file = os.path.join(log_dir, "analysis_summary.json")
    with open(summary_file, 'w') as f:
        summary = {
            "image_path": image_path,
            "timestamp": timestamp,
            "models_used": list(model_outputs.keys())
        }
        json.dump(summary, f, indent=2)
    
    return log_dir

def visualize_model_outputs(image_path, log_dir):
    """
    Create visualizations for all model outputs
    
    Args:
        image_path: Path to the original image
        log_dir: Directory containing model output logs
    """
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    cv_image = np.array(image)
    cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
    
    # Create output directory
    viz_dir = os.path.join(log_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load all available model outputs
    model_files = [f for f in os.listdir(log_dir) if f.endswith('.json') and f != "analysis_summary.json"]
    
    # Create a consolidated visualization
    consolidated = cv_image.copy()
    
    # Add title and basic information
    cv2.putText(
        consolidated,
        "Fashion Analysis Summary",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    
    # Visualize SAM+CLIP detections
    if "sam_clip_detections.json" in model_files:
        with open(os.path.join(log_dir, "sam_clip_detections.json"), 'r') as f:
            detection_data = json.load(f)
        
        # Create a copy of the image for drawing
        detection_viz = image.copy()
        draw = ImageDraw.Draw(detection_viz)
        
        for i, item in enumerate(detection_data):
            if "position" in item:
                pos = item["position"]
                h, w = image.height, image.width
                
                # Calculate box from position
                center_x = int(float(pos["center_x"]) * w)
                center_y = int(float(pos["center_y"]) * h)
                width = int(float(pos["width"]) * w)
                height = int(float(pos["height"]) * h)
                
                # Create box coordinates
                x1 = center_x - width // 2
                y1 = center_y - height // 2
                x2 = center_x + width // 2
                y2 = center_y + height // 2
                
                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
                
                # Add label
                label = f"{item.get('type', 'unknown')} ({item.get('confidence', 0):.2f})"
                draw.text((x1, y1 - 15), label, fill="blue")
                
                # Add to consolidated visualization
                cv2.rectangle(consolidated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    consolidated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
        
        # Save detection visualization
        detection_viz.save(os.path.join(viz_dir, "sam_clip_detections.jpg"))
    
    # Visualize Color Analysis
    if "color_analysis.json" in model_files:
        with open(os.path.join(log_dir, "color_analysis.json"), 'r') as f:
            color_data = json.load(f)
        
        # Create a color palette visualization
        if "color_palette" in color_data and color_data["color_palette"]:
            palette = color_data["color_palette"]
            
            # Create a simple color palette image
            palette_height = 100
            palette_width = 500
            palette_img = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            
            # Draw color swatches
            num_colors = min(len(palette), 5)  # Limit to 5 colors
            swatch_width = palette_width // num_colors
            
            for i, color in enumerate(palette[:num_colors]):
                # Extract RGB values
                if "hex" in color:
                    hex_color = color["hex"].lstrip('#')
                    try:
                        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    except:
                        r, g, b = 128, 128, 128  # Default gray if hex parsing fails
                elif "rgb" in color:
                    rgb_str = color["rgb"].strip("()").split(",")
                    try:
                        r, g, b = map(int, rgb_str)
                    except:
                        r, g, b = 128, 128, 128  # Default gray if RGB parsing fails
                else:
                    continue
                
                # Fill the swatch
                x_start = i * swatch_width
                x_end = (i + 1) * swatch_width
                palette_img[:, x_start:x_end] = [b, g, r]  # BGR for OpenCV
                
                # Add color name
                cv2.putText(
                    palette_img, 
                    color.get("name", ""), 
                    (x_start + 5, palette_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255) if (r + g + b) / 3 < 128 else (0, 0, 0),
                    1
                )
            
            # Save the palette visualization
            cv2.imwrite(os.path.join(viz_dir, "color_palette.jpg"), palette_img)
    
    # Add style information to consolidated visualization
    if "style_classification.json" in model_files:
        with open(os.path.join(log_dir, "style_classification.json"), 'r') as f:
            style_data = json.load(f)
        
        style = style_data.get("style", "unknown")
        confidence = style_data.get("confidence", 0)
        
        cv2.putText(
            consolidated,
            f"Style: {style} ({confidence:.2f})",
            (20, consolidated.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    
    # Add score to consolidated visualization
    if "score_prediction.json" in model_files:
        with open(os.path.join(log_dir, "score_prediction.json"), 'r') as f:
            score_data = json.load(f)
        
        overall_score = score_data.get("overall_score", 0)
        
        cv2.putText(
            consolidated,
            f"Score: {overall_score}/100",
            (20, consolidated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    
    # Save consolidated visualization
    cv2.imwrite(os.path.join(viz_dir, "consolidated_analysis.jpg"), consolidated)
    
    return viz_dir

def generate_html_report(image_path, result, log_dir, output_file=None):
    """
    Generate an HTML report with all model outputs and visualizations
    
    Args:
        image_path: Path to original image
        result: Final analysis result
        log_dir: Directory with model outputs
        output_file: Path to save HTML report
    """
    import base64
    from PIL import Image
    
    # If no output file specified, create one in the log directory
    if output_file is None:
        output_file = os.path.join(log_dir, "fashion_analysis_report.html")
    
    # Function to convert an image to base64 for embedding in HTML
    def img_to_base64(img_path):
        try:
            with open(img_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            logger.error(f"Error encoding image {img_path}: {e}")
            return ""
    
    # Read model outputs
    model_files = [f for f in os.listdir(log_dir) if f.endswith('.json') and f != "analysis_summary.json"]
    model_data = {}
    
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        try:
            with open(os.path.join(log_dir, model_file), 'r') as f:
                model_data[model_name] = json.load(f)
        except Exception as e:
            logger.error(f"Error reading model file {model_file}: {e}")
            model_data[model_name] = {"error": str(e)}
    
    # Get visualization images
    viz_dir = os.path.join(log_dir, "visualizations")
    viz_images = {}
    
    if os.path.exists(viz_dir):
        for img_file in os.listdir(viz_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name = os.path.splitext(img_file)[0]
                viz_images[img_name] = img_to_base64(os.path.join(viz_dir, img_file))
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fashion Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .model-output {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
            .image-container {{ margin: 10px 0; }}
            pre {{ overflow-x: auto; }}
            .score {{ font-size: 24px; font-weight: bold; }}
            .color-swatch {{ display: inline-block; width: 20px; height: 20px; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <h1>Fashion Analysis Report</h1>
        
        <div class="section">
            <h2>Original Image</h2>
            <img src="{img_to_base64(image_path)}" style="max-width: 500px;">
        </div>
        
        <div class="section">
            <h2>Analysis Result</h2>
            <div class="score">Overall Score: {result['score']}/100</div>
            
            <h3>Style Analysis</h3>
            <ul>
                <li><strong>Style:</strong> {result['labels']['style']}</li>
                <li><strong>Fit:</strong> {result['labels']['fit']}</li>
                <li><strong>Top:</strong> {result['labels']['top_type']}</li>
                <li><strong>Bottom:</strong> {result['labels']['pants_type']}</li>
                <li><strong>Footwear:</strong> {result['labels']['footwear_type']}</li>
                <li><strong>Accessories:</strong> {', '.join(result['labels']['accessories']) if result['labels']['accessories'] else 'None'}</li>
            </ul>
            
            <h3>Color Palette</h3>
            <div>
    """
    
    # Add color swatches from hex colors if available
    color_palette = result['labels']['color_palette']
    if color_palette:
        for color_name in color_palette:
            # Get hex color from model data if available
            hex_color = "#808080"  # Default gray
            if "color_analysis" in model_data and "color_palette" in model_data["color_analysis"]:
                for color in model_data["color_analysis"]["color_palette"]:
                    if color.get("name") == color_name and "hex" in color:
                        hex_color = color["hex"]
                        break
            html_content += f'<span class="color-swatch" style="background-color: {hex_color};"></span>'
        html_content += f'{", ".join(color_palette)}'
    
    html_content += """
            </div>
            
            <h3>Score Breakdown</h3>
            <ul>
    """
    
    # Add score breakdown
    for component, score in result['score_breakdown'].items():
        max_score = 25 if component in ["fit", "color"] else 20 if component == "footwear" else 15
        html_content += f'<li><strong>{component.capitalize()}:</strong> {score}/{max_score}</li>'
    
    html_content += """
            </ul>
            
            <h3>Feedback</h3>
            <ul>
    """
    
    # Add feedback
    for component, feedback in result['feedback'].items():
        html_content += f'<li><strong>{component.capitalize()}:</strong> {feedback}</li>'
    
    html_content += """
            </ul>
        </div>
    """
    
    # Add model visualizations
    if viz_images:
        html_content += """
        <div class="section">
            <h2>Model Visualizations</h2>
        """
        
        for name, img_data in viz_images.items():
            html_content += f"""
            <div class="image-container">
                <h3>{name.replace('_', ' ').title()}</h3>
                <img src="{img_data}" style="max-width: 100%;">
            </div>
            """
        
        html_content += "</div>"
    
    # Add raw model outputs
    html_content += """
        <div class="section">
            <h2>Raw Model Outputs</h2>
    """
    
    for model_name, data in model_data.items():
        html_content += f"""
            <h3>{model_name.replace('_', ' ').title()}</h3>
            <div class="model-output">
                <pre>{json.dumps(data, indent=2)}</pre>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    try:
        with open(output_file, 'w') as f:
            f.write(html_content)
        logger.info(f"Generated HTML report: {output_file}")
    except Exception as e:
        logger.error(f"Error writing HTML report: {e}")
    
    return output_file