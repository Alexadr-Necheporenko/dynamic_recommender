import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dynamic_models import SystemIdentification
from sklearn.model_selection import train_test_split

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
        self.dynamic_models: Dict[int, SystemIdentification] = {}
        
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
        
        for epoch in range(epochs):
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
    
    def train_dynamic_models(self, user_sequences: Dict[int, np.ndarray], 
                           sequence_length: int = 10, test_size: float = 0.2):
        """
        Train dynamic models for each user
        """
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
            
            # Initialize and train dynamic model for user
            dynamic_model = SystemIdentification(model_type='narx')
            dynamic_model.train_model(X_train, y_train)
            
            # Store model and evaluation metrics
            metrics = dynamic_model.evaluate_model(X_test, y_test)
            self.dynamic_models[user_id] = {
                'model': dynamic_model,
                'metrics': metrics
            }
    
    def predict(self, user_id: int, item_id: int, user_history: np.ndarray = None) -> float:
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
            dynamic_model = self.dynamic_models[user_id]['model']
            # Reshape user_history to match training data format [batch_size, sequence_length, features]
            history_reshaped = user_history.reshape(1, -1, 1)
            with torch.no_grad():
                dynamic_pred = dynamic_model.model(
                    torch.FloatTensor(history_reshaped)
                )[0].item()
        
        # Combine predictions
        if dynamic_pred is not None:
            # Simple average of both predictions
            final_pred = (static_pred + dynamic_pred) / 2
        else:
            final_pred = static_pred
        
        return final_pred
    
    def get_recommendations(self, user_id: int, user_history: np.ndarray = None,
                          n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user
        """
        predictions = []
        
        # Generate predictions for all items
        for item_id in range(self.num_items):
            pred = self.predict(user_id, item_id, user_history)
            predictions.append((item_id, pred))
        
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations] 