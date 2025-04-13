check format_expert_ratings.py
expert_ratings_formatted.json
train_scorer
learning_scorer.py

to train: python scripts/train_scorer.py --image_dir training_images --ratings expert_ratings_formatted.json --output models/scoring
to test: python test_fashion_rating.py test_images/casual_1.jpg