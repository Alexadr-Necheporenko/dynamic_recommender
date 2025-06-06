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
from sklearn.metrics import mean_squared_error

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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_size)
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
        
    def fit(self, X, y, epochs=50, batch_size=32, verbose=0):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Convert to tensors and ensure correct shapes
        X = torch.FloatTensor(X)  # Shape: (batch_size, sequence_length, 1)
        y = torch.FloatTensor(y)  # Shape: (batch_size, 1)
        
        # Store normalization parameters
        self.X_mean = X.mean()
        self.X_std = X.std()
        self.y_mean = y.mean()
        self.y_std = y.std()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        X = (X - self.X_mean) / (self.X_std + epsilon)
        y = (y - self.y_mean) / (self.y_std + epsilon)
        
        # Ensure y has the same batch size as X
        if y.shape[0] != X.shape[0]:
            y = y[:X.shape[0]]
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x)
            # Use stored normalization parameters
            epsilon = 1e-8
            x = (x - self.X_mean) / (self.X_std + epsilon)
            pred = self(x)
            # Denormalize prediction
            pred = pred * (self.y_std + epsilon) + self.y_mean
            return pred.numpy()

class NARX(nn.Module):
    def __init__(self, input_size, output_size):
        super(NARX, self).__init__()
        self.hidden_size = 32
        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
        
    def fit(self, X, y, epochs=50, verbose=0):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Convert to tensors and ensure correct shapes
        X = torch.FloatTensor(X)  # Shape: (batch_size, sequence_length, 1)
        y = torch.FloatTensor(y)  # Shape: (batch_size, 1)
        
        # Store normalization parameters
        self.X_mean = X.mean()
        self.X_std = X.std()
        self.y_mean = y.mean()
        self.y_std = y.std()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        X = (X - self.X_mean) / (self.X_std + epsilon)
        y = (y - self.y_mean) / (self.y_std + epsilon)
        
        # Ensure y has the same batch size as X
        if y.shape[0] != X.shape[0]:
            y = y[:X.shape[0]]
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x)
            # Use stored normalization parameters
            epsilon = 1e-8
            x = (x - self.X_mean) / (self.X_std + epsilon)
            pred = self(x)
            # Denormalize prediction
            pred = pred * (self.y_std + epsilon) + self.y_mean
            return pred.numpy()

class SystemIdentification:
    def __init__(self):
        self.model = None
        self.input_dim = None
        # Use separate scalers for input (X) and output (y)
        self.X_scaler = None
        self.y_scaler = None
        
    def fit(self, X, y):
        # Store input dimension
        self.input_dim = X.shape[1]
        
        # Normalize the data using separate scalers
        from sklearn.preprocessing import StandardScaler
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Fit scalers on the provided data
        X_scaled = self.X_scaler.fit_transform(X.reshape(X.shape[0], -1))
        y_scaled = self.y_scaler.fit_transform(y.reshape(y.shape[0], -1))
        
        # Проста лінійна регресія для прикладу
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_scaled)
        
        # Debugging: print learned coefficients and intercept
        print(f"SINDY (Linear Regression) Coeffs: {self.model.coef_}")
        print(f"SINDY (Linear Regression) Intercept: {self.model.intercept_}")
        
    def predict(self, x):
        # Ensure input has correct dimensions (batch_size, n_features)
        # It should typically be (1, input_dim) for a single prediction
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        elif len(x.shape) == 3:
            # This case should ideally not happen if input is a sequence of ratings for SINDY
            # If it happens, reshape to (batch_size, sequence_length)
            x = x.reshape(x.shape[0], -1)
            
        # Pad or truncate if necessary to match the training input dimension
        if x.shape[1] < self.input_dim:
            # Pad at the beginning with zeros, matching training logic
            x = np.pad(x, ((0, 0), (self.input_dim - x.shape[1], 0)), mode='constant')
        elif x.shape[1] > self.input_dim:
            # Use the last part of the sequence, matching training logic
            x = x[:, -self.input_dim:]
            
        # Scale input using the X_scaler fitted during training
        # Ensure x_scaled has shape (batch_size, input_dim)
        x_scaled = self.X_scaler.transform(x)
        
        # Get prediction in the scaled space
        pred_scaled = self.model.predict(x_scaled)
        
        # Inverse transform the prediction using the y_scaler
        # Assuming pred_scaled has shape (batch_size, 1)
        pred = self.y_scaler.inverse_transform(pred_scaled)
        
        # Convert to scalar value
        if isinstance(pred, np.ndarray):
            pred = pred.ravel()[0]
        return float(pred)

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
    
    def train_dynamic_models(self, user_sequences):
        """Train dynamic models for each user"""
        print("Training dynamic models for each user...")
        total_users = len(user_sequences)
        processed_users = 0
        
        for user_id, sequences in user_sequences.items():
            processed_users += 1
            if processed_users % 100 == 0:
                print(f"Processed {processed_users}/{total_users} users...")
            
            if len(sequences) < 2:
                continue
            
            # Store sequences for this user
            self.dynamic_models[user_id] = {}
            
            # Convert sequences to numpy arrays
            sequences_array = np.array(sequences)
            
            # Train each model type
            for model_type in ['LSTM', 'NARX', 'SINDY']:
                try:
                    # Prepare data
                    X = sequences_array[:-1].copy()  # Make a copy to avoid modifying original
                    y = sequences_array[1:].copy()   # Make a copy to avoid modifying original
                    
                    # Initialize and train model
                    if model_type == 'LSTM':
                        model = LSTM(1, 32, 1)  # input_size=1 for single feature
                        # Reshape data for LSTM
                        # Each sequence becomes a batch, and we predict the next value
                        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)  # Shape: (batch_size, sequence_length, 1)
                        y_reshaped = y[:, -1].reshape(-1, 1)  # Take last value of each sequence as target
                        model.fit(X_reshaped, y_reshaped, epochs=100, batch_size=32, verbose=0)
                    elif model_type == 'NARX':
                        model = NARX(1, 1)  # input_size=1 for single feature
                        # Reshape data for NARX
                        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)  # Shape: (batch_size, sequence_length, 1)
                        y_reshaped = y[:, -1].reshape(-1, 1)  # Take last value of each sequence as target
                        model.fit(X_reshaped, y_reshaped, epochs=100, verbose=0)
                    else:  # SINDY
                        model = SystemIdentification()
                        # Prepare data for SINDY: predict the next value based on the sequence
                        X_sindy = []
                        y_sindy = []
                        # For each sequence, create (history, next_value) pairs
                        for seq in sequences:
                            if len(seq) > 1:
                                for i in range(len(seq) - 1):
                                    # Use history up to current item to predict the next
                                    X_sindy.append(seq[:i+1].tolist()) # History as features
                                    y_sindy.append(seq[i+1]) # Next value as target
                        
                        if X_sindy and y_sindy:
                            # Pad sequences to the fixed sequence_length from DataPreprocessor for consistent input dimension
                            # Assumes sequence_length is accessible or passed
                            # For now, let's assume a fixed length, e.g., the one used in DataPreprocessor (10)
                            # Need to ensure this matches how sequences were generated initially.
                            # Assuming sequence_length = 10 based on main.py
                            fixed_sequence_length = 10 # This should ideally come from DataPreprocessor

                            X_padded = np.array([np.pad(x, (fixed_sequence_length - len(x), 0), 'constant') for x in X_sindy])

                            # Ensure padded sequences have the correct fixed length
                            # Filter out any sequences that couldn't be padded correctly (shouldn't happen with constant mode)
                            valid_indices = [i for i, x in enumerate(X_padded) if len(x) == fixed_sequence_length]
                            X_padded = X_padded[valid_indices]
                            y_sindy = np.array(y_sindy)[valid_indices].reshape(-1, 1)

                            # Ensure data is properly shaped for SINDY
                            model.fit(X_padded, y_sindy)
                        else:
                            # If no valid data for SINDY, skip training for this user
                            print(f"Недостатньо даних для тренування SINDY моделі для користувача {user_id}")
                            continue # Skip storing model if not trained
                    
                    # Store model and last sequence
                    # Use the entire last sequence as the base for future dynamic predictions
                    if model_type in ['LSTM', 'NARX']:
                         # For NN models, store the actual last sequence
                         last_sequence_to_store = sequences[-1].copy()
                    elif model_type == 'SINDY':
                        # For SINDY, store the entire last sequence, padding will be handled in predict
                        last_sequence_to_store = sequences[-1].copy()

                    # Ensure we only store the model if training was successful
                    if model is not None:
                        # Додайте відступ тут
                        self.dynamic_models[user_id][model_type] = {
                            'model': model,
                            'last_sequence': last_sequence_to_store
                        }
                    
                except Exception as e:
                    print(f"Error training {model_type} model for user {user_id}: {str(e)}")
                    continue
    
    def get_static_prediction(self, user_id, item_id):
        """Get prediction from static matrix factorization model"""
        with torch.no_grad():
            prediction = self.mf_model(
                torch.LongTensor([user_id]),
                torch.LongTensor([item_id])
            ).item()
        return prediction
        
    def get_dynamic_prediction(self, user_id, item_id, model_type):
        """Get prediction from dynamic model"""
        if user_id not in self.dynamic_models or model_type not in self.dynamic_models[user_id]:
            return None
            
        try:
            model_data = self.dynamic_models[user_id][model_type]
            model = model_data['model']
            # Make a copy to avoid modifying original
            # Use the stored last sequence for prediction input
            sequence_to_predict = model_data['last_sequence'].copy()
            
            try:
                if model_type == 'SINDY':
                    # For SINDy, we need to ensure the input is properly shaped
                    # Use the input dimension stored in the trained SINDY model
                    input_dim = model.input_dim
                    if len(sequence_to_predict.shape) == 1:
                         # If 1D array, reshape to (1, -1)
                         sequence_to_predict = sequence_to_predict.reshape(1, -1)
                         
                    # Pad or truncate if necessary to match the training input dimension
                    if sequence_to_predict.shape[1] < input_dim:
                         # Pad at the beginning with zeros
                        sequence_to_predict = np.pad(sequence_to_predict, ((0, 0), (input_dim - sequence_to_predict.shape[1], 0)), mode='constant')
                    elif sequence_to_predict.shape[1] > input_dim:
                        # Use the last part of the sequence
                        sequence_to_predict = sequence_to_predict[:, -input_dim:]
                    
                    # Ensure input is 2D (batch_size, n_features)
                    if len(sequence_to_predict.shape) == 1:
                        sequence_to_predict = sequence_to_predict.reshape(1, -1)

                    # SINDY model (SystemIdentification) has the LinearRegression model stored in self.model
                    # but we should use the predict method of the SystemIdentification wrapper.
                    dynamic_pred = model.predict(sequence_to_predict)
                    
                else: # LSTM or NARX (NN models)
                    # For neural network models, reshape to (batch_size, sequence_length, 1)
                    if len(sequence_to_predict.shape) == 1:
                        history_reshaped = sequence_to_predict.reshape(1, -1, 1)
                    elif len(sequence_to_predict.shape) == 2:
                         # If 2D (batch_size, sequence_length), reshape to (batch_size, sequence_length, 1)
                        history_reshaped = sequence_to_predict.reshape(sequence_to_predict.shape[0], sequence_to_predict.shape[1], 1)
                    else:
                         # Assume it's already in the correct 3D shape or raise an error
                         history_reshaped = sequence_to_predict

                    # Call the predict method of the LSTM/NARX object
                    dynamic_pred = model.predict(history_reshaped)
                    
                # Ensure dynamic_pred is a scalar
                if isinstance(dynamic_pred, np.ndarray):
                     dynamic_pred = dynamic_pred.ravel()[0]
                dynamic_pred = float(dynamic_pred)
            
            except Exception as e:
                print(f"Error calculating dynamic prediction for user {user_id} with {model_type}: {str(e)}")
                # Optional: print traceback for debugging
                # import traceback
                # traceback.print_exc()
                dynamic_pred = None # Set to None if prediction fails

            return dynamic_pred
            
        except Exception as e:
            print(f"Error getting dynamic prediction for user {user_id}: {str(e)}")
            return None
    
    def get_adaptive_prediction(self, user_id, item_id, model_type):
        """Get prediction from adaptive model"""
        static_pred = self.get_static_prediction(user_id, item_id)
        dynamic_pred = self.get_dynamic_prediction(user_id, item_id, model_type)
        
        if dynamic_pred is None:
            return static_pred
            
        # Combine predictions with weights
        static_weight = 0.7
        dynamic_weight = 0.3
        
        return float(static_weight * static_pred + dynamic_weight * dynamic_pred)
    
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
                model_data = self.dynamic_models[user_id][model_type]
                dynamic_model = model_data['model']
                # Make a copy to avoid modifying original
                # Use the stored last sequence for prediction input
                # The compare_dynamic_models function uses a segment of the test sequence as history, not the stored last sequence.
                # So user_history is the correct input here.
                sequence_to_predict = user_history.copy()
                
                try:
                    if model_type == 'SINDY':
                    # For SINDy, we need to ensure the input is properly shaped
                        # Use the input dimension stored in the trained SINDY model
                        input_dim = dynamic_model.input_dim
                        if len(sequence_to_predict.shape) == 1:
                             # If 1D array, reshape to (1, -1)
                             sequence_to_predict = sequence_to_predict.reshape(1, -1)
                             
                        # Pad or truncate if necessary to match the training input dimension
                        if sequence_to_predict.shape[1] < input_dim:
                             # Pad at the beginning with zeros
                            sequence_to_predict = np.pad(sequence_to_predict, ((0, 0), (input_dim - sequence_to_predict.shape[1], 0)), mode='constant')
                        elif sequence_to_predict.shape[1] > input_dim:
                            # Use the last part of the sequence
                            sequence_to_predict = sequence_to_predict[:, -input_dim:]
                        
                        # Ensure input is 2D (batch_size, n_features)
                        if len(sequence_to_predict.shape) == 1:
                            sequence_to_predict = sequence_to_predict.reshape(1, -1)

                        # SINDY model (SystemIdentification) has the LinearRegression model stored in self.model
                        # but we should use the predict method of the SystemIdentification wrapper.
                        dynamic_pred = dynamic_model.predict(sequence_to_predict)
                        
                    else: # LSTM or NARX (NN models)
                        # For neural network models, reshape to (batch_size, sequence_length, 1)
                        if len(sequence_to_predict.shape) == 1:
                            history_reshaped = sequence_to_predict.reshape(1, -1, 1)
                        elif len(sequence_to_predict.shape) == 2:
                             # If 2D (batch_size, sequence_length), reshape to (batch_size, sequence_length, 1)
                            history_reshaped = sequence_to_predict.reshape(sequence_to_predict.shape[0], sequence_to_predict.shape[1], 1)
                        else:
                             # Assume it's already in the correct 3D shape or raise an error
                             history_reshaped = sequence_to_predict

                        # Call the predict method of the LSTM/NARX object
                        dynamic_pred = dynamic_model.predict(history_reshaped)
                        
                    # Ensure dynamic_pred is a scalar
                    if isinstance(dynamic_pred, np.ndarray):
                         dynamic_pred = dynamic_pred.ravel()[0]
                    dynamic_pred = float(dynamic_pred)
                
                except Exception as e:
                    print(f"Error calculating dynamic prediction for user {user_id} with {model_type}: {str(e)}")
                    # Optional: print traceback for debugging
                    # import traceback
                    # traceback.print_exc()
                    dynamic_pred = None # Set to None if prediction fails
        
        # Combine predictions
        if dynamic_pred is not None:
            # You might want a more sophisticated combination, but for now average.
            final_pred = (static_pred + dynamic_pred) / 2
        else:
            final_pred = static_pred
        
        # Ensure final_pred is a float
        return float(final_pred)
    
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
    
    def compare_recommendation_approaches(self, user_id: int, user_history: np.ndarray,
                                        n_recommendations: int = 10, 
                                        model_types: List[str] = None) -> pd.DataFrame:
        """
        Порівняння різних підходів до рекомендацій
        """
        if model_types is None:
            model_types = list(self.dynamic_models.get(user_id, {}).keys())
        
        # Отримуємо статичні рекомендації
        static_recs = []
        with torch.no_grad():
            for item_id in range(self.num_items):
                pred = self.mf_model(
                    torch.LongTensor([user_id]),
                    torch.LongTensor([item_id])
                ).item()
                static_recs.append((item_id, pred))
        static_recs.sort(key=lambda x: x[1], reverse=True)
        static_recs = static_recs[:n_recommendations]
        
        # Дані для порівняння
        comparison_data = []
        
        # Додаємо статичні рекомендації
        for rank, (item_id, pred) in enumerate(static_recs, 1):
            comparison_data.append({
                'Підхід': 'Статичний',
                'Ранг': rank,
                'ID об\'єкта': item_id,
                'Прогнозований рейтинг': pred
            })
        
        # Додаємо рекомендації з динамічними моделями
        for model_type in model_types:
            try:
                # Стандартні динамічні рекомендації
                dyn_recs = self.get_recommendations(
                    user_id, user_history, n_recommendations, model_type
                )

                print(f"\n--- DEBUGGING RECOMMENDATIONS for user {user_id} ({model_type}) ---")
                print(f"Dynamic Recommendations ({model_type.upper()}):")
                print(dyn_recs)
                print("---------------------------------------")
                
                for rank, (item_id, pred) in enumerate(dyn_recs, 1):
                    comparison_data.append({
                        'Підхід': f'Динамічний ({model_type.upper()})',
                        'Ранг': rank,
                        'ID об\'єкта': item_id,
                        'Прогнозований рейтинг': pred
                    })
                    
                # Адаптивні рекомендації з поведінковою моделлю
                adapt_recs = self.get_adaptive_recommendations(
                    user_id, user_history, n_recommendations, model_type
                )

                print(f"Adaptive Recommendations ({model_type.upper()}):")
                print(adapt_recs)
                print("---------------------------------------")
                
                for rank, (item_id, pred) in enumerate(adapt_recs, 1):
                    comparison_data.append({
                        'Підхід': f'Адаптивний ({model_type.upper()})',
                        'Ранг': rank,
                        'ID об\'єкта': item_id,
                        'Прогнозований рейтинг': pred
                    })
            except ValueError:
                continue  # Пропускаємо моделі, які не доступні
        
        # Створюємо DataFrame з даними
        return pd.DataFrame(comparison_data)
    
    def evaluate_recommendation_approaches(self, user_id: int, user_history: np.ndarray,
                                         test_items: List[Tuple[int, float]],
                                         model_type: str = 'narx') -> dict:
        """
        Оцінка ефективності різних підходів до рекомендацій
        
        Args:
            user_id: Ідентифікатор користувача
            user_history: Історія взаємодій користувача
            test_items: Список тестових об'єктів з реальними рейтингами
            model_type: Тип моделі для використання
            
        Returns:
            dict: Метрики ефективності рекомендацій
        """
        # Розділяємо тестові дані
        test_item_ids = [item[0] for item in test_items]
        test_ratings = [item[1] for item in test_items]
        
        # Отримуємо прогнози різних підходів
        static_preds = []
        dynamic_preds = []
        adaptive_preds = []
        
        with torch.no_grad():
            for item_id in test_item_ids:
                # Статичні прогнози
                static_pred = self.mf_model(
                    torch.LongTensor([user_id]),
                    torch.LongTensor([item_id])
                ).item()
                static_preds.append(static_pred)
                
                # Динамічні прогнози
                try:
                    dynamic_pred = self.predict(user_id, item_id, user_history, model_type)
                    dynamic_preds.append(dynamic_pred)
                    
                    # Адаптивні прогнози
                    # Отримуємо прогноз поведінки
                    future_behavior = self.predict_future_behavior(user_id, user_history, 1, model_type)[0]
                    
                    # Адаптуємо рейтинг
                    behavior_weight = 0.3
                    if future_behavior > 0.5:
                        adaptive_weight = behavior_weight * future_behavior
                    else:
                        adaptive_weight = behavior_weight * (1 - future_behavior)
                        
                    adaptive_pred = static_pred * (1 + adaptive_weight * (static_pred - 0.5))
                    adaptive_pred = max(0, min(1, adaptive_pred))
                    adaptive_preds.append(adaptive_pred)
                except ValueError:
                    # Якщо немає моделі, використовуємо статичні прогнози
                    dynamic_preds.append(static_pred)
                    adaptive_preds.append(static_pred)
        
        # Обчислюємо метрики для кожного підходу
        metrics = {}
        
        # MSE
        metrics['static_mse'] = mean_squared_error(test_ratings, static_preds)
        metrics['dynamic_mse'] = mean_squared_error(test_ratings, dynamic_preds)
        metrics['adaptive_mse'] = mean_squared_error(test_ratings, adaptive_preds)
        
        # RMSE
        metrics['static_rmse'] = np.sqrt(metrics['static_mse'])
        metrics['dynamic_rmse'] = np.sqrt(metrics['dynamic_mse'])
        metrics['adaptive_rmse'] = np.sqrt(metrics['adaptive_mse'])
        
        # MAE
        metrics['static_mae'] = np.mean(np.abs(np.array(test_ratings) - np.array(static_preds)))
        metrics['dynamic_mae'] = np.mean(np.abs(np.array(test_ratings) - np.array(dynamic_preds)))
        metrics['adaptive_mae'] = np.mean(np.abs(np.array(test_ratings) - np.array(adaptive_preds)))
        
        # Відсоткове покращення
        metrics['dynamic_vs_static_improvement'] = (metrics['static_rmse'] - metrics['dynamic_rmse']) / metrics['static_rmse'] * 100
        metrics['adaptive_vs_static_improvement'] = (metrics['static_rmse'] - metrics['adaptive_rmse']) / metrics['static_rmse'] * 100
        metrics['adaptive_vs_dynamic_improvement'] = (metrics['dynamic_rmse'] - metrics['adaptive_rmse']) / metrics['dynamic_rmse'] * 100
        
        return metrics
        
    def plot_recommendation_comparison(self, comparison_df: pd.DataFrame, 
                                     save_path: str = None,
                                     n_recommendations: int = 10):
        """
        Візуалізація порівняння різних підходів до рекомендацій
        
        Args:
            comparison_df: DataFrame з даними порівняння (з compare_recommendation_approaches)
            save_path: Шлях для збереження графіка
            n_recommendations: Кількість рекомендацій, що використовується для визначення перекриття на тепловій карті.
        """
        if comparison_df.empty:
            print("Немає даних для візуалізації порівняння рекомендацій.")
            return

        # Обчислюємо середні рейтинги по кожному підходу та рангу
        # Використовуємо pivot_table для зручності обчислення середнього рейтингу для кожного підходу на кожному рангу
        avg_ratings_pivot = comparison_df.pivot_table(
            values='Прогнозований рейтинг',
            index='Ранг',
            columns='Підхід',
            aggfunc='mean'
        )
        
        # Візуалізуємо розподіл рейтингів по рангу
        plt.figure(figsize=(14, 8))
        
        # Графік порівняння рейтингів
        sns.lineplot(data=avg_ratings_pivot, markers=True, dashes=False)
        
        plt.title('Порівняння середніх прогнозованих рейтингів за рангом рекомендації')
        plt.xlabel('Ранг рекомендації')
        plt.ylabel('Середній прогнозований рейтинг')
        plt.grid(True)
        plt.legend(title='Підхід')
        
        # Зберігаємо або показуємо графік середніх рейтингів
        if save_path:
            base_name = os.path.splitext(os.path.basename(save_path))[0]
            avg_ratings_save_path = os.path.join(os.path.dirname(save_path), f'{base_name}_avg_ratings.png')
            plt.savefig(avg_ratings_save_path)
            plt.close()
        else:
            plt.show()

        # --- Графіки порівняння прогнозованих рейтингів для топ-N об'єктів (розділені) ---

        # Отримуємо унікальний список топ-N об'єктів з comparison_df
        # Можемо взяти топ N з будь-якого підходу, оскільки вони виявилися однаковими
        top_n_item_ids = comparison_df[comparison_df['Підхід'] == comparison_df['Підхід'].iloc[0]] \
                           .sort_values(by='Ранг').head(n_recommendations)['ID об\'єкта'].tolist()

        # Фільтруємо DataFrame, щоб залишити лише дані для цих топ-N об'єктів
        top_items_comparison_df = comparison_df[comparison_df['ID об\'єкта'].isin(top_n_item_ids)].copy()

        # Перетворюємо ID об'єктів на строковий тип або категорійний
        top_items_comparison_df['ID об\'єкта'] = top_items_comparison_df['ID об\'єкта'].astype(str)

        # Визначаємо групи підходів для окремих графіків
        approach_groups = {
            'LSTM': ['Статичний', 'Динамічний (LSTM)', 'Адаптивний (LSTM)'],
            'NARX': ['Статичний', 'Динамічний (NARX)', 'Адаптивний (NARX)'],
            'SINDY': ['Статичний', 'Динамічний (SINDY)', 'Адаптивний (SINDY)']
        }

        # Генеруємо окремі графіки для кожної групи
        for model_type, approaches in approach_groups.items():
            # Фільтруємо дані для поточної групи підходів
            group_df = top_items_comparison_df[top_items_comparison_df['Підхід'].isin(approaches)].copy()

            if group_df.empty:
                print(f"Немає даних для групи {model_type}.")
                continue

            plt.figure(figsize=(12, 8))
            sns.barplot(data=group_df, x='ID об\'єкта', y='Прогнозований рейтинг', hue='Підхід')

            plt.title(f'Порівняння прогнозованих рейтингів для топ-{n_recommendations} об\'єктів ({model_type})')
            plt.xlabel('ID об\'єкта')
            plt.ylabel('Прогнозований рейтинг')
            plt.grid(axis='y')
            plt.legend(title='Підхід')
            plt.tight_layout()

            # Зберігаємо або показуємо графік
            if save_path:
                base_name = os.path.splitext(os.path.basename(save_path))[0]
                group_save_path = os.path.join(os.path.dirname(save_path), f'{base_name}_predicted_ratings_{model_type.lower()}.png')
                plt.savefig(group_save_path)
                plt.close()
            else:
                plt.show()

    def get_adaptive_recommendations(self, user_id: int, user_history: np.ndarray = None,
                                   n_recommendations: int = 10, model_type: str = 'narx') -> List[Tuple[int, float]]:
        """
        Отримання рекомендацій за допомогою адаптивного підходу.

        Args:
            user_id: Ідентифікатор користувача.
            user_history: Історія взаємодій користувача.
            n_recommendations: Кількість рекомендацій для отримання.
            model_type: Тип динамічної моделі для адаптивного підходу.

        Returns:
            Список кортежів (item_id, predicted_rating) для N рекомендованих об'єктів.
        """
        all_item_ids = np.arange(self.num_items)
        adaptive_predictions = []

        with torch.no_grad():
            for item_id in all_item_ids:
                adaptive_pred = self.get_adaptive_prediction(user_id, item_id, model_type)
                adaptive_predictions.append((item_id, adaptive_pred))

        # Сортуємо за прогнозованим рейтингом у спадаючому порядку та беремо топ N
        adaptive_predictions.sort(key=lambda x: x[1], reverse=True)

        return adaptive_predictions[:n_recommendations] 