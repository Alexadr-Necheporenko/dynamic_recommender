import pandas as pd
import numpy as np
from data_processing import DataPreprocessor
from recommender import DynamicRecommender

def main():
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess MovieLens data
    # Note: You need to download the MovieLens dataset first
    # https://grouplens.org/datasets/movielens/
    ratings_path = "path_to_ratings.csv"  # Update this path
    df = preprocessor.load_movielens_data(ratings_path)
    
    # Normalize features
    df_normalized = preprocessor.normalize_features(df)
    
    # Fill missing values
    df_clean = preprocessor.fill_missing_values(df_normalized)
    
    # Create user sequences for dynamic modeling
    user_sequences, user_timestamps = preprocessor.create_user_sequences(
        df_clean, sequence_length=10
    )
    
    # Initialize recommender system
    num_users = len(df_clean['userId'].unique())
    num_items = len(df_clean['movieId'].unique())
    recommender = DynamicRecommender(num_users, num_items)
    
    # Train static matrix factorization model
    user_ids = df_clean['userId'].values
    item_ids = df_clean['movieId'].values
    ratings = df_clean['rating'].values
    
    print("Training static model...")
    recommender.train_static_model(user_ids, item_ids, ratings)
    
    # Train dynamic models for each user
    print("Training dynamic models...")
    recommender.train_dynamic_models(user_sequences)
    
    # Example: Get recommendations for a specific user
    test_user_id = 1
    if test_user_id in user_sequences:
        user_history = user_sequences[test_user_id][-1]  # Last sequence for the user
        recommendations = recommender.get_recommendations(
            test_user_id, user_history, n_recommendations=5
        )
        
        print(f"\nTop 5 recommendations for user {test_user_id}:")
        for item_id, pred_rating in recommendations:
            print(f"Item {item_id}: Predicted rating = {pred_rating:.2f}")
    
if __name__ == "__main__":
    main() 