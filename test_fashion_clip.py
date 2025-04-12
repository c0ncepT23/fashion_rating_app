# test_fashion_clip.py
import asyncio
from PIL import Image
from app.models.fashion_clip_classifier import FashionCLIPClassifier

async def test_fashion_clip():
    # Load an image (update this path to one of your test images)
    image_path = "test_images/casual_2.jpg"
    image = Image.open(image_path)
    
    # Initialize classifier
    classifier = FashionCLIPClassifier(weights_path='weights/openfashionclip.pt')
    
    # Classify the image
    result = classifier.classify_garment(image)
    
    # Print results
    print(f"\nClassification results for {image_path}:")
    print(f"Best garment match: {result['garment']['best_match']['type']} with confidence {result['garment']['best_match']['confidence']:.2f}")
    print("\nTop 5 garment matches:")
    for match in result['garment']['top_matches']:
        print(f"- {match['type']}: {match['confidence']:.2f}")
    
    print(f"\nBest style match: {result['style']['best_match']['style']} with confidence {result['style']['best_match']['confidence']:.2f}")
    print("\nTop 3 style matches:")
    for match in result['style']['top_matches']:
        print(f"- {match['style']}: {match['confidence']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_fashion_clip())