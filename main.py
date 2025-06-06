import pandas as pd
import numpy as np
from data_processing import DataPreprocessor
from recommender import DynamicRecommender
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dynamic_models import SystemIdentification
import torch
import warnings
import random

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._regression')

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_metrics(pred, true):
    """Calculate metrics with proper error handling"""
    metrics = {}
    
    # Ensure arrays are properly shaped
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    
    # Basic metrics
    metrics['mse'] = np.mean((pred - true) ** 2)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = np.mean(np.abs(pred - true))
    
    # Calculate R² only if we have enough samples and variance
    if len(pred) >= 2 and np.var(true) > 0:
        try:
            ss_res = np.sum((true - pred) ** 2)
            ss_tot = np.sum((true - np.mean(true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot)
        except:
            metrics['r2'] = np.nan
    else:
        metrics['r2'] = np.nan
    
    return metrics

def split_user_sequences(user_sequences, test_size=0.2, random_state=42):
    """Split user sequences into training and testing sets"""
    train_sequences = {}
    test_sequences = {}
    
    # Ensure reproducibility
    rng = np.random.RandomState(random_state)
    
    for user_id, sequences in user_sequences.items():
        if len(sequences) > 1:
            # Split each user's sequences
            if len(sequences) > 5: # Ensure enough data for splitting
                 # Take a fraction of sequences for testing
                 num_test = max(1, int(len(sequences) * test_size))
                 # Randomly select sequences for testing without replacement
                 test_indices = rng.choice(len(sequences), size=num_test, replace=False)
                 
                 train_indices = np.setdiff1d(range(len(sequences)), test_indices)

                 train_sequences[user_id] = [sequences[i] for i in train_indices]
                 test_sequences[user_id] = [sequences[i] for i in test_indices]
            else:
                # If not enough sequences, use the last one for testing and rest for training
                train_sequences[user_id] = sequences[:-1]
                test_sequences[user_id] = [sequences[-1]]
        elif len(sequences) == 1:
            # If only one sequence, use it for training and test is empty
             train_sequences[user_id] = sequences
             test_sequences[user_id] = []
        else:
            # Empty sequence, skip
            train_sequences[user_id] = []
            test_sequences[user_id] = []
            
    return train_sequences, test_sequences

def compare_dynamic_models(recommender, test_sequences, test_users, plots_dir):
    """Compare different dynamic models (LSTM, NARX, SINDY) and analyze their performance"""
    print("\n=== Порівняння динамічних моделей ===")
    
    # Create directory for model comparisons
    model_comparison_dir = os.path.join(plots_dir, "model_comparison")
    ensure_dir(model_comparison_dir)
    
    # Initialize metrics storage for all predictions
    all_predictions = {
        'LSTM': {'pred': [], 'true': [], 'adaptive': []},
        'NARX': {'pred': [], 'true': [], 'adaptive': []},
        'SINDY': {'pred': [], 'true': [], 'adaptive': []}
    }
    
    # Collect predictions for all test sequences
    total_sequences = 0
    successful_predictions = 0
    
    for user_id in test_users:
        if user_id not in test_sequences:
            continue
            
        user_test_sequences = test_sequences[user_id]
        if not user_test_sequences:
            continue
            
        total_sequences += len(user_test_sequences)
        
        for test_sequence in user_test_sequences:
            if len(test_sequence) < 2:
                continue
                
            for model_type in ['LSTM', 'NARX', 'SINDY']:
                if model_type not in recommender.dynamic_models.get(user_id, {}):
                    continue
                    
                try:
                    model = recommender.dynamic_models[user_id][model_type]['model']
                    
                    # Get predictions for each step in the sequence
                    for i in range(len(test_sequence) - 1):
                        # Prepare input based on model type
                        if model_type == 'SINDY':
                            # For SINDY, use the sequence up to current point
                            input_seq = test_sequence[:i+1].copy()
                            if len(input_seq) < model.input_dim:
                                input_seq = np.pad(input_seq, (model.input_dim - len(input_seq), 0), mode='constant')
                            elif len(input_seq) > model.input_dim:
                                input_seq = input_seq[-model.input_dim:]
                            input_seq = input_seq.reshape(1, -1)
                        else:
                            # For LSTM/NARX, reshape to (1, sequence_length, 1)
                            input_seq = test_sequence[:i+1].reshape(1, -1, 1)
                        
                        # Get dynamic prediction
                        pred = model.predict(input_seq)
                        if isinstance(pred, np.ndarray):
                            pred_value = float(pred.ravel()[0])
                        else:
                            pred_value = float(pred)
                        
                        # Get adaptive prediction
                        current_item_id = int(test_sequence[i])
                        adaptive_pred = recommender.get_adaptive_prediction(user_id, current_item_id, model_type)
                        
                        # Store predictions and true value
                        all_predictions[model_type]['pred'].append(pred_value)
                        all_predictions[model_type]['true'].append(float(test_sequence[i+1]))
                        all_predictions[model_type]['adaptive'].append(adaptive_pred)
                        
                        successful_predictions += 1
                        
                except Exception as e:
                    print(f"Помилка при оцінці моделі {model_type} для користувача {user_id}: {str(e)}")
                    continue
    
    print(f"\nЗагальна кількість тестових послідовностей: {total_sequences}")
    print(f"Успішних прогнозів: {successful_predictions}")
    
    # Calculate baseline metrics
    all_true_values = []
    for model_type in all_predictions:
        all_true_values.extend(all_predictions[model_type]['true'])
    
    if not all_true_values:
        print("Недостатньо даних для аналізу")
        return
    
    # Calculate metrics for each model
    model_metrics = {}
    for model_type in all_predictions:
        if not all_predictions[model_type]['pred']:
            continue
            
        pred = np.array(all_predictions[model_type]['pred'])
        true = np.array(all_predictions[model_type]['true'])
        adaptive = np.array(all_predictions[model_type]['adaptive'])
        
        # Calculate metrics for dynamic predictions
        metrics = calculate_metrics(pred, true)
        metrics['n_samples'] = len(pred)
        
        # Calculate metrics for adaptive predictions
        adaptive_metrics = calculate_metrics(adaptive, true)
        adaptive_metrics['n_samples'] = len(adaptive)
        
        # Combine metrics
        model_metrics[model_type] = {
            'dynamic': metrics,
            'adaptive': adaptive_metrics
        }
    
    # Calculate baseline metrics
    baseline_pred = np.mean(all_true_values)
    baseline_metrics = calculate_metrics(
        np.full_like(all_true_values, baseline_pred),
        np.array(all_true_values)
    )
    
    print("\nБазові метрики:")
    print(f"Baseline MSE: {baseline_metrics['mse']:.4f}")
    print(f"Baseline RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"Кількість зразків: {len(all_true_values)}")
    
    # Create comparison plots
    metrics = ['mse', 'rmse', 'mae', 'r2']
    metric_names = {'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²'}
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        data = []
        labels = []
        
        for model_type in model_metrics:
            if metric in model_metrics[model_type]['dynamic'] and not np.isnan(model_metrics[model_type]['dynamic'][metric]):
                data.append(model_metrics[model_type]['dynamic'][metric])
                labels.append(f"{model_type} (Динамічний)")
                data.append(model_metrics[model_type]['adaptive'][metric])
                labels.append(f"{model_type} (Адаптивний)")
        
        if data:
            plt.bar(labels, data)
            plt.title(f'Порівняння {metric_names[metric]} для різних моделей')
            plt.ylabel(metric_names[metric])
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(model_comparison_dir, f'{metric}_comparison.png'))
            plt.close()
    
    # Create detailed comparison table
    print("\nПідсумкова таблиця порівняння моделей:")
    print("-" * 120)
    print(f"{'Модель':<15} {'Підхід':<12} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Покращення':>12} {'Зразки':>8}")
    print("-" * 120)
    
    for model_type in model_metrics:
        # Dynamic approach
        metrics = model_metrics[model_type]['dynamic']
        r2_str = f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A"
        dynamic_improvement = ((baseline_metrics['rmse'] - metrics['rmse']) / baseline_metrics['rmse']) * 100
        print(f"{model_type:<15} {'Динамічний':<12} {metrics['mse']:>10.4f} {metrics['rmse']:>10.4f} "
              f"{metrics['mae']:>10.4f} {r2_str:>10} {dynamic_improvement:>11.2f}% {metrics['n_samples']:>8}")
        
        # Adaptive approach
        metrics = model_metrics[model_type]['adaptive']
        r2_str = f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A"
        adaptive_improvement = ((baseline_metrics['rmse'] - metrics['rmse']) / baseline_metrics['rmse']) * 100
        print(f"{model_type:<15} {'Адаптивний':<12} {metrics['mse']:>10.4f} {metrics['rmse']:>10.4f} "
              f"{metrics['mae']:>10.4f} {r2_str:>10} {adaptive_improvement:>11.2f}% {metrics['n_samples']:>8}")
    print("-" * 120)
    
    # Create prediction vs true value scatter plots for each model
    for model_type in model_metrics:
        if not all_predictions[model_type]['pred']:
            continue
            
        plt.figure(figsize=(10, 10))
        
        # Plot dynamic predictions
        plt.subplot(2, 1, 1)
        plt.scatter(all_predictions[model_type]['true'], 
                   all_predictions[model_type]['pred'],
                   alpha=0.5, label='Динамічний')
        plt.plot([0, 1], [0, 1], 'r--', label='Ідеальний прогноз')
        plt.title(f'Динамічні прогнози {model_type}')
        plt.xlabel('Справжні значення')
        plt.ylabel('Прогнозовані значення')
        plt.legend()
        plt.grid(True)
        
        # Plot adaptive predictions
        plt.subplot(2, 1, 2)
        plt.scatter(all_predictions[model_type]['true'],
                   all_predictions[model_type]['adaptive'],
                   alpha=0.5, label='Адаптивний')
        plt.plot([0, 1], [0, 1], 'r--', label='Ідеальний прогноз')
        plt.title(f'Адаптивні прогнози {model_type}')
        plt.xlabel('Справжні значення')
        plt.ylabel('Прогнозовані значення')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_comparison_dir, f'{model_type}_predictions_scatter.png'))
        plt.close()

def predict_next_participants(recommender, user_sequences, num_participants=5, plots_dir="plots"):
    """Create prediction plots for a few participants for each dynamic method"""
    # Create directory for prediction plots
    prediction_dir = os.path.join(plots_dir, "predictions")
    ensure_dir(prediction_dir)
    
    # Get all available users
    available_users = list(user_sequences.keys())
    if len(available_users) < num_participants:
        num_participants = len(available_users)
    
    # Select sample participants
    random.seed(42)  # For reproducibility
    selected_users = random.sample(available_users, num_participants)
    
    methods = {
        'LSTM': 'LSTM',
        'NARX': 'NARX',
        'SINDY': 'SINDY'
    }
    
    # For each selected user, create plots comparing dynamic methods
    for user_id in selected_users:
        if user_id not in user_sequences or len(user_sequences[user_id]) < 2:
            continue
        
        user_sequences_array = user_sequences[user_id]
        # Use the last sequence as the test sequence
        test_sequence = user_sequences_array[-1]
        # True values are the subsequent items in the test sequence
        true_ratings = [float(item) for item in test_sequence[1:]]
        
        # Collect predictions for all dynamic methods for this user's sequence
        user_method_predictions = {
            'LSTM': [],
            'NARX': [],
            'SINDY': []
        }
        
        for method, method_name in methods.items():
            # Ensure the model exists for this user and method
            if method not in recommender.dynamic_models.get(user_id, {}):
                continue
                
            try:
                model_data = recommender.dynamic_models[user_id][method]
                model = model_data['model']
                
                # Generate dynamic predictions for the sequence
                for i in range(len(test_sequence) - 1):
                    # Use the sequence up to the current item as input history
                    current_history = test_sequence[:i+1].copy() # Ensure it's a copy
                    
                    # Reshape and prepare input based on model type
                    if method == 'SINDY':
                        # SINDY expects a sequence of the trained input dimension
                        # We need to pad the current history if it's shorter
                        input_dim = model.input_dim # Get input dimension from the trained SINDY model
                        if len(current_history) < input_dim:
                             # Pad at the beginning with zeros
                            input_seq = np.pad(current_history, (input_dim - len(current_history), 0), mode='constant')
                        elif len(current_history) > input_dim:
                            # Use the last part of the history
                            input_seq = current_history[-input_dim:]
                        else:
                            input_seq = current_history
                            
                        input_seq = input_seq.reshape(1, -1) # Reshape to (1, input_dim)
                        
                    else: # LSTM or NARX
                         # NN models expect (batch_size, sequence_length, 1)
                         # We need to ensure sequence_length matches what the model was trained on
                         # This might still be an issue if sequence_length is fixed in model init.
                         # For now, reshape based on current history length.
                        input_seq = current_history.reshape(1, -1, 1)
                        
                    # Get prediction
                    pred = model.predict(input_seq)
                    
                    # Ensure prediction is a scalar
                    if isinstance(pred, np.ndarray):
                        pred_value = float(pred.ravel()[0])
                    else:
                        pred_value = float(pred)
                    user_method_predictions[method].append(pred_value)
                
            except Exception as e:
                print(f"Помилка при отриманні динамічних прогнозів для користувача {user_id} та методу {method}: {str(e)}")
                # Optional: print traceback for debugging
                # import traceback
                # traceback.print_exc()
                continue
        
        # Debugging: print SINDY predictions if method is SINDY
        if 'SINDY' in user_method_predictions and user_method_predictions['SINDY']:
            print(f"\nSINDY прогнози для користувача {user_id}: {user_method_predictions['SINDY'][:10]}...") # Print first 10 predictions
            if all(p == 0.0 or abs(p) < 1e-9 for p in user_method_predictions['SINDY']): # Check for near-zero values
                 print(f"Увага: Всі SINDY прогнози для користувача {user_id} дорівнюють або дуже близькі до 0.")

        # Create a single plot comparing dynamic methods for this user
        plt.figure(figsize=(12, 6))
        # Use the length of true_ratings for the x-axis, as predictions should align with these.
        x = range(len(true_ratings))
        
        plt.plot(x, true_ratings, marker='o', linestyle='-', label='Справжні значення')
        
        # Plot predictions for each dynamic method (LSTM, NARX, SINDY)
        for method, method_name in methods.items():
            if method in user_method_predictions and user_method_predictions[method]: # Check if method exists and has predictions
                 plt.plot(x, user_method_predictions[method], marker='x', linestyle='--', label=f'{method_name} прогноз')

        
        plt.title(f'Порівняння динамічних прогнозів для користувача {user_id}')
        plt.xlabel('Елемент послідовності')
        plt.ylabel('Рейтинг')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(prediction_dir, f'user_{user_id}_dynamic_methods_comparison.png'))
        plt.close()

def main():
    # Create directories for plots
    plots_dir = "plots"
    ensure_dir(plots_dir)
    ensure_dir(os.path.join(plots_dir, "static"))
    ensure_dir(os.path.join(plots_dir, "dynamic"))
    ensure_dir(os.path.join(plots_dir, "sindy"))
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    # Assuming 'ratings.csv' and 'movies.csv' are in the ml-32m directory
    ratings_path = "ml-32m/ratings_sampled_50k.csv" # Use the sampled data path
    movies_path = "ml-32m/movies_sampled_50k.csv" # Use the sampled data path

    # Check if ratings_path exists
    if not os.path.exists(ratings_path):
         print(f"Помилка: Файл рейтингів не знайдено за шляхом: {ratings_path}")
         # Optionally, fall back to default or exit
         ratings_path = "ml-32m/ratings.csv"
         print(f"Спроба завантажити файл за замовчуванням: {ratings_path}")
         if not os.path.exists(ratings_path):
             print("Помилка: Файл рейтингів за замовчуванням також не знайдено. Вихід.")
             exit()

    # Load and preprocess data using existing methods
    print(f"\nLoading data from {ratings_path}...")
    # Use load_movielens_data to load ratings
    df = preprocessor.load_movielens_data(ratings_path)
    print("Data loaded successfully.")

    # Normalize features
    print("Normalizing features...")
    df_normalized = preprocessor.normalize_features(df)
    print("Features normalized.")

    # Fill missing values
    print("Filling missing values...")
    # Use fill_missing_values
    processed_df = preprocessor.fill_missing_values(df_normalized)
    print("Missing values filled.")

    # Create user sequences for dynamic modeling
    user_sequences, user_timestamps = preprocessor.create_user_sequences(
        processed_df, sequence_length=10
    )
    
    # Split user sequences into training and testing sets
    train_sequences, test_sequences = split_user_sequences(user_sequences, test_size=0.2)
    
    # Initialize recommender system
    num_users = len(processed_df['userId'].unique())
    num_items = len(processed_df['movieId'].unique())
    recommender = DynamicRecommender(num_users, num_items)
    
    # Train static matrix factorization model
    print("Навчання статичної моделі...")
    user_ids = processed_df['userId'].values
    item_ids = processed_df['movieId'].values
    ratings = processed_df['rating'].values
    
    static_losses = recommender.train_static_model(user_ids, item_ids, ratings)
    
    # Plot and save static model training loss
    plt.figure(figsize=(10, 6))
    plt.plot(static_losses)
    plt.title('Втрати при навчанні статичної моделі')
    plt.xlabel('Епоха')
    plt.ylabel('Втрати')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "static", "training_loss.png"))
    plt.close()
    
    # Train dynamic models for each user using training sequences
    print("\nНавчання динамічних моделей...")
    recommender.train_dynamic_models(train_sequences)
    
    # Initialize storage for all predictions
    all_predictions = {
        'LSTM': {'pred': [], 'true': [], 'adaptive': []},
        'NARX': {'pred': [], 'true': [], 'adaptive': []},
        'SINDY': {'pred': [], 'true': [], 'adaptive': []}
    }
    
    # Collect predictions for all users using test sequences for evaluation
    print("\nЗбір прогнозів та оцінка ефективності на тестових даних...")
    
    # For now, let's pass test_sequences to compare_dynamic_models
    compare_dynamic_models(recommender, test_sequences, list(test_sequences.keys()), plots_dir)

    # --- Додаткове порівняння та візуалізація підходів --- #
    print("\n=== Додаткове порівняння та візуалізація підходів ===")
    
    # Обираємо випадкового користувача для демонстрації порівняння рекомендацій
    test_users_with_seq = [user_id for user_id, sequences in test_sequences.items() if sequences]
    if test_users_with_seq:
        # Вибираємо користувача з достатньою кількістю послідовностей в тренувальному наборі, щоб були треновані динамічні моделі
        # Або просто беремо першого користувача з тестовими послідовностями для демонстрації
        sample_user_id = test_users_with_seq[0]
        sample_test_sequence = test_sequences[sample_user_id][0] # Використовуємо першу тестову послідовність як історію для порівняння
        
        print(f"Демонстрація порівняння рекомендацій для користувача: {sample_user_id}")
        
        # Отримуємо порівняльну таблицю рекомендацій для випадкового користувача
        # Використовуємо частину тестової послідовності як історію для цієї демонстрації
        # Моделі для порівняння: static, lstm, narx, sindy
        comparison_df = recommender.compare_recommendation_approaches(
            sample_user_id,
            np.array(sample_test_sequence),
            n_recommendations=10, # Кількість рекомендацій для порівняння
            model_types=['LSTM', 'NARX', 'SINDY'] # Зазначаємо, які динамічні моделі включити
        )
        
        print("\nТаблиця порівняння рекомендацій для випадкового користувача:")
        # Merge with movie titles for display
        # Load full movie titles data to ensure all titles are available
        full_movies_path = "ml-32m/movies.csv" # Assuming full movies file is here
        try:
            full_movies_df = pd.read_csv(full_movies_path)
            movie_titles_map = full_movies_df[['movieId', 'title']].drop_duplicates().set_index('movieId')
        except FileNotFoundError:
            print(f"Warning: Full movies file not found at {full_movies_path}. Movie titles may not be displayed.")
            movie_titles_map = pd.DataFrame(columns=['title'], index=pd.Index([], name='movieId')) # Create empty map if file not found

        comparison_df_with_titles = comparison_df.copy()
        # Ensure 'ID об\'єкта' column is of compatible type with movie_titles_map index (e.g., int)
        comparison_df_with_titles['ID об\'єкта'] = comparison_df_with_titles['ID об\'єкта'].astype(int)
        # Perform the join using the full movie titles map
        comparison_df_with_titles = comparison_df_with_titles.join(movie_titles_map, on='ID об\'єкта', how='left')

        # If a title is still missing after join (shouldn't happen if ID exists in full movies file),
        # you could fill it with a placeholder like 'Unknown Title'
        # comparison_df_with_titles['title'] = comparison_df_with_titles['title'].fillna('Unknown Title')

        # Rearrange columns for better display
        cols = ['Підхід', 'Ранг', 'ID об\'єкта', 'title', 'Прогнозований рейтинг']
        # Ensure all columns exist before reordering
        cols_to_keep = [col for col in cols if col in comparison_df_with_titles.columns]
        comparison_df_with_titles = comparison_df_with_titles[cols_to_keep]

        print(comparison_df_with_titles)
        
        # Побудова графіків порівняння рекомендацій
        recommendation_comparison_dir = os.path.join(plots_dir, "recommendation_comparison")
        ensure_dir(recommendation_comparison_dir)
        
        print("\nПобудова графіків порівняння рекомендацій...")
        recommender.plot_recommendation_comparison(
            comparison_df,
            save_path=os.path.join(recommendation_comparison_dir, "recommendation_comparison_plots.png"),
            n_recommendations=10 # Pass the n_recommendations value
        )
        print(f"Графіки порівняння рекомендацій збережено у {recommendation_comparison_dir}")
        
    else:
        print("\nНемає тестових користувачів з послідовностями для демонстрації порівняння рекомендацій.")

if __name__ == "__main__":
    main() 