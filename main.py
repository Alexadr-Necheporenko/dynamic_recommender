import pandas as pd
import numpy as np
from data_processing import DataPreprocessor
from recommender import DynamicRecommender
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dynamic_models import SystemIdentification

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    # Create directories for plots
    plots_dir = "plots"
    ensure_dir(plots_dir)
    ensure_dir(os.path.join(plots_dir, "static"))
    ensure_dir(os.path.join(plots_dir, "dynamic"))
    ensure_dir(os.path.join(plots_dir, "sindy"))
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess MovieLens data
    # Note: You need to download the MovieLens dataset first
    # https://grouplens.org/datasets/movielens/
    ratings_path = "ml-32m/ratings.csv"  # Updated path to the ratings file
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
    print("Training static model...")
    user_ids = df_clean['userId'].values
    item_ids = df_clean['movieId'].values
    ratings = df_clean['rating'].values
    
    static_losses = recommender.train_static_model(user_ids, item_ids, ratings)
    
    # Plot and save static model training loss
    plt.figure(figsize=(10, 6))
    plt.plot(static_losses)
    plt.title('Static Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "static", "training_loss.png"))
    plt.close()
    
    # Combine all user sequences for SINDy
    print("\nTraining global SINDy model...")
    all_sequences = np.vstack([seq for seq in user_sequences.values()])
    X_sindy = all_sequences[:, :-1]  # All but last rating in each sequence
    y_sindy = all_sequences[:, -1]   # Last rating in each sequence
    
    # Reshape X_sindy to 2D
    X_sindy = X_sindy.reshape(X_sindy.shape[0], -1)
    
    # Train SINDy model on combined data
    sindy_model = SystemIdentification(model_type='sindy')
    sindy_model.train_model(X_sindy, y_sindy)
    
    # Create and save SINDy visualization
    print("Creating global SINDy visualization...")
    sindy_model.model.plot_3d_dynamics(
        save_path=os.path.join(plots_dir, "sindy", "global_dynamics.png")
    )
    
    # Train other dynamic models for each user
    print("\nTraining NARX and LSTM models for individual users...")
    recommender.train_dynamic_models(user_sequences)
    
    # Compare models for a sample user
    test_users = list(user_sequences.keys())[:2]  # Take first 2 users for demonstration
    
    # Create results directory for tables
    results_dir = "results"
    ensure_dir(results_dir)
    
    # Initialize empty lists to store all recommendations
    all_static_recommendations = []
    all_dynamic_recommendations = []
    all_summary_statistics = []
    
    for idx, test_user_id in enumerate(test_users, 1):
        print(f"\nAnalyzing recommendations for user {idx}:")
        
        # Get user history
        user_history = user_sequences[test_user_id][-1]  # Last sequence for the user
        
        # Compare recommendations from different models
        comparison_df = recommender.compare_recommendations(
            test_user_id, user_history, n_recommendations=10
        )
        
        # Plot and save model metrics comparison
        print("\nModel Metrics Comparison:")
        user_plot_dir = os.path.join(plots_dir, "dynamic", f"user_{idx}")
        ensure_dir(user_plot_dir)
        recommender.plot_model_comparison(
            test_user_id,
            save_path=os.path.join(user_plot_dir, "metrics_comparison.png")
        )
        
        # Separate static and dynamic recommendations
        static_recs = comparison_df[comparison_df['Model'] == 'STATIC'].copy()
        dynamic_recs = comparison_df[comparison_df['Model'] != 'STATIC'].copy()
        
        # Add user ID column
        static_recs['User'] = f"User {idx}"
        dynamic_recs['User'] = f"User {idx}"
        
        # Store recommendations
        all_static_recommendations.append(static_recs)
        all_dynamic_recommendations.append(dynamic_recs)
        
        # Get summary statistics
        summary_stats = comparison_df.groupby('Model')['Predicted Rating'].agg(['mean', 'std', 'min', 'max']).reset_index()
        summary_stats['User'] = f"User {idx}"
        all_summary_statistics.append(summary_stats)
        
        print(f"\nProcessed recommendations for User {idx}")
    
    # Combine and save all recommendations
    all_static_df = pd.concat(all_static_recommendations, ignore_index=True)
    all_dynamic_df = pd.concat(all_dynamic_recommendations, ignore_index=True)
    all_stats_df = pd.concat(all_summary_statistics, ignore_index=True)
    
    # Save combined recommendations
    all_static_df.to_csv(os.path.join(results_dir, "static_recommendations.csv"), index=False)
    all_dynamic_df.to_csv(os.path.join(results_dir, "dynamic_recommendations.csv"), index=False)
    all_stats_df.to_csv(os.path.join(results_dir, "model_statistics.csv"), index=False)
    
    print("\nFinal Statistics:")
    print("\nStatic Recommendations:")
    print(all_static_df)
    print("\nDynamic Recommendations:")
    print(all_dynamic_df)
    print("\nModel Statistics:")
    print(all_stats_df)

if __name__ == "__main__":
    main() 