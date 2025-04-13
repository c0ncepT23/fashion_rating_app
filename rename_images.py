import os
import shutil
from pathlib import Path

# Source directory containing the images
source_dir = r"C:\Users\Vamsi\fashion_rating_app\training_images\next set"

# Destination directory for renamed images
# You can use the same directory or specify a different one
dest_dir = r"C:\Users\Vamsi\fashion_rating_app\training_images\renamed"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Starting index number
start_number = 126

# Valid image extensions
valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']

# Get all image files in source directory
image_files = []
for file in os.listdir(source_dir):
    file_path = os.path.join(source_dir, file)
    if os.path.isfile(file_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_files.append(file)

# Sort the files to ensure a consistent order
image_files.sort()

# Rename and copy files
for i, filename in enumerate(image_files):
    # Get file extension
    _, extension = os.path.splitext(filename)
    
    # Create new filename
    new_filename = f"casual_{i + start_number}{extension.lower()}"
    
    # Source and destination paths
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, new_filename)
    
    # Copy file with new name
    shutil.copy2(source_path, dest_path)
    print(f"Renamed {filename} to {new_filename}")

print(f"Renamed {len(image_files)} images, starting from casual_{start_number}")
print(f"Renamed images are saved in: {dest_dir}")