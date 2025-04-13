import json
import os
import glob

def combine_batch_files(input_dir, output_file):
    # Path to the results directory
    results_dir = input_dir
    
    # Find all batch result files
    batch_files = glob.glob(os.path.join(results_dir, "batch_*_results.json"))
    
    # Sort them numerically
    batch_files.sort(key=lambda x: int(x.split('batch_')[1].split('_')[0]))
    
    print(f"Found {len(batch_files)} batch files to combine")
    
    # Initialize an empty list to store all outfit evaluations
    all_outfits = []
    
    # Process each batch file
    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                
            # Extract outfits array from each file
            if 'outfits' in batch_data:
                outfits = batch_data['outfits']
                all_outfits.extend(outfits)
                print(f"Added {len(outfits)} outfits from {os.path.basename(batch_file)}")
            else:
                print(f"Warning: No 'outfits' key found in {os.path.basename(batch_file)}")
        except Exception as e:
            print(f"Error processing {batch_file}: {str(e)}")
    
    # Save all outfits to a single JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_outfits, f, indent=2)
        
        print(f"Successfully combined {len(all_outfits)} outfits into {output_file}")
    except Exception as e:
        print(f"Error saving combined file: {str(e)}")

if __name__ == "__main__":
    # Configure these paths for your environment
    input_directory = r"C:\Users\Vamsi\fashion_rating_app\training_images\results"
    output_file = r"C:\Users\Vamsi\fashion_rating_app\training_images\all_fashion_evaluations.json"
    
    combine_batch_files(input_directory, output_file)