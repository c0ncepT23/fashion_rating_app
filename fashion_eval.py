import anthropic
import base64
import json
import os
from glob import glob
import time

# Initialize Claude client
api_key = "ANTHROPIC_API_KEY"  # Replace with your actual API key
client = anthropic.Anthropic(api_key=api_key)

# Your image directory
image_dir = r"C:\Users\Vamsi\fashion_rating_app\training_images\working folder"
image_files = glob(os.path.join(image_dir, "*.jpg"))

print(f"Found {len(image_files)} images to process")

# Process in batches of 10
batch_size = 10
all_evaluations = []

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(image_dir), "results")
os.makedirs(output_dir, exist_ok=True)

# Number of images to process
start_image_number = 146
end_image_number = 629
total_images = end_image_number - start_image_number + 1

# Calculate number of batches
total_batches = (total_images + batch_size - 1) // batch_size

for batch_index in range(total_batches):
    batch_start = batch_index * batch_size
    batch_end = min(batch_start + batch_size, total_images)
    
    # Calculate image numbers for this batch
    start_num = start_image_number + batch_start
    end_num = start_image_number + batch_end - 1
    
    batch_num = batch_index + 1
    print(f"Processing batch {batch_num}/{total_batches} (images casual_{start_num}.jpg to casual_{end_num}.jpg)...")
    
    # Get the subset of images for this batch
    batch_images = image_files[batch_start:batch_end]
    
    # Create filenames for this batch
    filenames = [f"casual_{start_image_number + batch_start + i}.jpg" for i in range(len(batch_images))]
    
    # Format the prompt text
    prompt_text = f"""You are a top fashion consultant who works at Zara. Please analyze these outfits and provide detailed fashion feedback in the following JSON format for each image:
    {{
        "image": "filename.jpg",
        "score": 86,
        "labels": {{
            "fit": "style_description",
            "top_type": "description",
            "bottom_type": "description",
            "color_palette": ["color1", "color2", "color3"],
            "footwear_type": "description",
            "accessories": ["item1", "item2"],
            "style": "overall_style_category"
        }},
        "score_breakdown": {{
            "fit": 22,
            "color": 20,
            "footwear": 16,
            "accessories": 14,
            "style": 14
        }},
        "feedback": {{
            "fit": "Detailed feedback on fit and silhouette.",
            "color": "Feedback on color coordination.",
            "footwear": "Comments on shoe choice.",
            "accessories": "Thoughts on accessories.",
            "style": "Overall style assessment."
        }}
    }}

    Return a valid JSON object containing an array of evaluations under the key "outfits". 
    Use these filenames for the images: {', '.join(filenames)}
    """

    # Create message content with images
    content = [{"type": "text", "text": prompt_text}]

    # Add images to content
    for idx, img_path in enumerate(batch_images):
        try:
            with open(img_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded
                    }
                })
            print(f"  Added image {filenames[idx]}")
        except Exception as e:
            print(f"  Error reading image {img_path}: {str(e)}")

    # Maximum retries
    max_retries = 3
    retry_count = 0
    success = False

    while retry_count < max_retries and not success:
        try:
            # Use the beta API
            print("  Sending API request...")
            response = client.beta.messages.create(
                model="claude-3-opus-20240229",  # Or claude-3-sonnet-20240229 for faster, cheaper processing
                max_tokens=4000,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            
            # Extract response text
            response_text = ""
            for content_block in response.content:
                if content_block.type == "text":
                    response_text = content_block.text
            
            # Save raw response for debugging
            batch_response_file = os.path.join(output_dir, f"batch_{batch_num}_raw_response.txt")
            with open(batch_response_file, "w", encoding="utf-8") as f:
                f.write(response_text)
            
            print("  Response received and saved")
            
            # Try to parse the JSON
            batch_evaluations = []
            try:
                # Try to parse the entire response as JSON first
                json_data = json.loads(response_text)
                batch_evaluations = json_data.get("outfits", [])
            except json.JSONDecodeError:
                # If that fails, try to find JSON within code blocks or elsewhere
                import re
                json_matches = re.findall(r'```json\s*([\s\S]*?)\s*```|```\s*([\s\S]*?)\s*```|(\{[\s\S]*\})', response_text)
                for match in json_matches:
                    json_str = next((m for m in match if m), None)
                    if json_str:
                        try:
                            json_data = json.loads(json_str)
                            if isinstance(json_data, dict):
                                batch_evaluations = json_data.get("outfits", [])
                            elif isinstance(json_data, list):
                                batch_evaluations = json_data
                            if batch_evaluations:
                                break
                        except:
                            continue
            
            # If we got evaluations, we succeeded
            if batch_evaluations:
                success = True
                
                # Ensure filenames are assigned correctly
                for idx, eval_data in enumerate(batch_evaluations):
                    if idx < len(filenames):
                        if "image" not in eval_data or not eval_data["image"]:
                            eval_data["image"] = filenames[idx]
                
                all_evaluations.extend(batch_evaluations)
                
                # Save batch results
                batch_file = os.path.join(output_dir, f"batch_{batch_num}_results.json")
                with open(batch_file, "w") as f:
                    json.dump({"outfits": batch_evaluations}, f, indent=2)
                
                print(f"  ✓ Successfully processed batch {batch_num} with {len(batch_evaluations)} evaluations")
            else:
                print(f"  ✗ No evaluations found in response for batch {batch_num}")
                retry_count += 1
                
        except Exception as e:
            print(f"  ✗ Error processing batch {batch_num}: {str(e)}")
            retry_count += 1
            time.sleep(2)  # Wait a bit before retrying
    
    # Write cumulative results after each batch
    cumulative_file = os.path.join(output_dir, "fashion_evaluations_current.json")
    with open(cumulative_file, "w") as f:
        json.dump({"outfits": all_evaluations}, f, indent=2)
    
    # Sleep between batches to avoid rate limits
    if batch_num < total_batches:
        wait_time = 3  # seconds
        print(f"Waiting {wait_time} seconds before processing next batch...")
        time.sleep(wait_time)

# Write final results to file
final_file = os.path.join(output_dir, "fashion_evaluations_complete.json")
with open(final_file, "w") as f:
    json.dump({"outfits": all_evaluations}, f, indent=2)

print(f"Completed processing {len(all_evaluations)} outfits")
print(f"Results saved to {os.path.abspath(output_dir)}")