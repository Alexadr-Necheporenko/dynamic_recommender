import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dynamic_models import SystemIdentification
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DynamicMatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_factors: int = 50):
        super(DynamicMatrixFactorization, self).__init__()
        
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        
        # Initialize weights
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embeds = self.user_factors(user_ids)
        item_embeds = self.item_factors(item_ids)
        # Sum along the factors dimension and ensure output is 1D
        return (user_embeds * item_embeds).sum(dim=1).squeeze(-1)

class DynamicRecommender:
    def __init__(self, num_users: int, num_items: int, 
                 num_factors: int = 50, learning_rate: float = 0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        
        # Initialize matrix factorization model
        self.mf_model = DynamicMatrixFactorization(num_users, num_items, num_factors)
        
        # Dictionary to store dynamic models for each user
        self.dynamic_models: Dict[int, Dict[str, SystemIdentification]] = {}
        
    def train_static_model(self, user_ids: np.ndarray, item_ids: np.ndarray, 
                          ratings: np.ndarray, epochs: int = 100, batch_size: int = 64):
        """
        Train the static matrix factorization model
        """
        optimizer = torch.optim.Adam(self.mf_model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to PyTorch tensors
        user_ids = torch.LongTensor(user_ids)
        item_ids = torch.LongTensor(item_ids)
        ratings = torch.FloatTensor(ratings)
        
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            # Mini-batch training
            for i in range(0, len(ratings), batch_size):
                batch_users = user_ids[i:i + batch_size]
                batch_items = item_ids[i:i + batch_size]
                batch_ratings = ratings[i:i + batch_size]
                
                # Forward pass
                predictions = self.mf_model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(ratings) // batch_size)
            losses.append(avg_loss)
            
        return losses
    
    def train_dynamic_models(self, user_sequences: Dict[int, np.ndarray], 
                           sequence_length: int = 10, test_size: float = 0.2,
                           save_dir: str = None):
        """
        Train dynamic models for each user using different methods
        """
        model_types = ['narx', 'lstm', 'sindy']
        
        for user_id, sequences in user_sequences.items():
            # Split sequences into features and targets
            X = sequences[:, :-1]  # All but last rating in each sequence
            y = sequences[:, -1]   # Last rating in each sequence
            
            # Reshape X to have the correct dimensions [batch_size, sequence_length, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            self.dynamic_models[user_id] = {}
            
            # Train each type of model
            for model_type in model_types:
                print(f"Training {model_type.upper()} model for user {user_id}...")
                
                # Initialize and train dynamic model for user
                dynamic_model = SystemIdentification(model_type=model_type)
                dynamic_model.train_model(X_train, y_train)
                
                # Evaluate model
                metrics = dynamic_model.evaluate_model(X_test, y_test)
                
                # Store model and metrics
                self.dynamic_models[user_id][model_type] = {
                    'model': dynamic_model,
                    'metrics': metrics
                }
    
    def predict(self, user_id: int, item_id: int, user_history: np.ndarray = None,
                model_type: str = 'narx') -> float:
        """
        Make predictions combining static and dynamic models
        """
        # Static prediction
        with torch.no_grad():
            static_pred = self.mf_model(
                torch.LongTensor([user_id]),
                torch.LongTensor([item_id])
            ).item()
        
        # Dynamic prediction if available
        dynamic_pred = None
        if user_id in self.dynamic_models and user_history is not None:
            if model_type in self.dynamic_models[user_id]:
                dynamic_model = self.dynamic_models[user_id][model_type]['model']
                
                if model_type == 'sindy':
                    # For SINDy, we need to ensure the input is properly shaped
                    if len(user_history.shape) == 1:
                        history_reshaped = user_history.reshape(1, -1)
                    else:
                        history_reshaped = user_history.reshape(1, -1)
                    dynamic_pred = dynamic_model.model.predict(history_reshaped)[0]
                else:
                    # For neural network models
                    history_reshaped = user_history.reshape(1, -1, 1)
                    with torch.no_grad():
                        if model_type == 'narx':
                            dynamic_pred = dynamic_model.model(torch.FloatTensor(history_reshaped))[0].item()
                        else:
                            dynamic_pred = dynamic_model.model(torch.FloatTensor(history_reshaped)).item()
        
        # Combine predictions
        if dynamic_pred is not None:
            final_pred = (static_pred + dynamic_pred) / 2
        else:
            final_pred = static_pred
        
        return final_pred
    
    def get_recommendations(self, user_id: int, user_history: np.ndarray = None,
                          n_recommendations: int = 10, model_type: str = 'narx') -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user using specified model type
        """
        predictions = []
        
        # Generate predictions for all items
        for item_id in range(self.num_items):
            pred = self.predict(user_id, item_id, user_history, model_type)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def compare_recommendations(self, user_id: int, user_history: np.ndarray = None,
                              n_recommendations: int = 10) -> pd.DataFrame:
        """
        Compare recommendations from different models
        """
        model_types = ['static'] + list(self.dynamic_models.get(user_id, {}).keys())
        all_recommendations = {}
        
        for model_type in model_types:
            if model_type == 'static':
                # Get static recommendations
                static_preds = []
                with torch.no_grad():
                    for item_id in range(self.num_items):
                        pred = self.mf_model(
                            torch.LongTensor([user_id]),
                            torch.LongTensor([item_id])
                        ).item()
                        static_preds.append((item_id, pred))
                static_preds.sort(key=lambda x: x[1], reverse=True)
                all_recommendations['static'] = static_preds[:n_recommendations]
            else:
                # Get dynamic model recommendations
                all_recommendations[model_type] = self.get_recommendations(
                    user_id, user_history, n_recommendations, model_type
                )
        
        # Create comparison DataFrame
        comparison_data = []
        for model_type, recommendations in all_recommendations.items():
            for rank, (item_id, pred_rating) in enumerate(recommendations, 1):
                comparison_data.append({
                    'Model': model_type.upper(),
                    'Rank': rank,
                    'Item ID': item_id,
                    'Predicted Rating': pred_rating
                })
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self, user_id: int, save_path: str = None):
        """
        Plot comparison of model metrics for a user and optionally save to file
        """
        if user_id not in self.dynamic_models:
            print(f"No dynamic models available for user {user_id}")
            return
        
        metrics = ['mse', 'rmse', 'r2']
        model_metrics = {}
        
        for model_type, model_data in self.dynamic_models[user_id].items():
            model_metrics[model_type] = {
                metric: model_data['metrics'][metric]
                for metric in metrics
            }
        
        # Create DataFrame for plotting
        df_metrics = pd.DataFrame(model_metrics).T
        
        # Plot metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, metric in enumerate(metrics):
            sns.barplot(data=df_metrics, y=df_metrics.index, x=metric, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 