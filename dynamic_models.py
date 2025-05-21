import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import entropy
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.integrate import odeint
import os

class NARX(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super(NARX, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # NARX layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get the output from the last time step
        output = self.fc(lstm_out[:, -1, :])
        # Squeeze the last dimension if it's 1
        output = output.squeeze(-1)
        return output, hidden

class DynamicLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super(DynamicLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        # Squeeze the last dimension to match target shape
        predictions = predictions.squeeze(-1)
        return predictions

class SINDy:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.coefficients = None
        self.features = None
        self.n_features = None
        self.training_data = None
        
    def _build_library(self, X: np.ndarray) -> np.ndarray:
        """Build feature library with polynomial terms in 3D space"""
        # Reshape X to 2D if it's not already
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            # Ensure consistent feature dimension
            if X.shape[1] != self.n_features:
                if len(X.shape) > 2:
                    X = X.reshape(X.shape[0], -1)
                if X.shape[1] > self.n_features:
                    X = X[:, :self.n_features]
                elif X.shape[1] < self.n_features:
                    pad_width = ((0, 0), (0, self.n_features - X.shape[1]))
                    X = np.pad(X, pad_width, mode='constant')
        
        n_samples = X.shape[0]
        
        # Enhanced feature library for 3D space
        # [1, x, y, z, x^2, y^2, z^2, xy, xz, yz, sin(x), sin(y), sin(z)]
        library = np.ones((n_samples, 13))
        
        # Linear terms
        for i in range(min(3, self.n_features)):
            library[:, i+1] = X[:, i]
        
        # Quadratic terms
        for i in range(min(3, self.n_features)):
            library[:, i+4] = X[:, i]**2
        
        # Cross terms
        if self.n_features >= 2:
            library[:, 7] = X[:, 0] * X[:, 1]  # xy
        if self.n_features >= 3:
            library[:, 8] = X[:, 0] * X[:, 2]  # xz
            library[:, 9] = X[:, 1] * X[:, 2]  # yz
        
        # Trigonometric terms
        for i in range(min(3, self.n_features)):
            library[:, i+10] = np.sin(X[:, i])
        
        return library
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SINDy model"""
        self.training_data = (X.copy(), y.copy())
        # Build feature library
        self.features = self._build_library(X)
        
        # Use Lasso regression for sparse identification
        lasso = Lasso(alpha=self.threshold, fit_intercept=False)
        lasso.fit(self.features, y)
        
        # Store coefficients
        self.coefficients = lasso.coef_
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using identified dynamics"""
        features = self._build_library(X)
        return features @ self.coefficients
    
    def plot_3d_dynamics(self, save_path: str = None):
        """Plot 3D visualization of the system dynamics"""
        if self.training_data is None:
            print("No training data available. Please fit the model first.")
            return
        
        X, y = self.training_data
        y_pred = self.predict(X)
        
        fig = plt.figure(figsize=(15, 10))
        
        # First subplot: 3D scatter of actual vs predicted
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Create color maps for actual and predicted values
        norm = plt.Normalize(min(y.min(), y_pred.min()), max(y.max(), y_pred.max()))
        
        # Plot actual values
        scatter1 = ax1.scatter(X[:, 0], X[:, 1], y, 
                             c='blue', alpha=0.6, label='Actual')
        # Plot predicted values
        scatter2 = ax1.scatter(X[:, 0], X[:, 1], y_pred, 
                             c='red', alpha=0.4, label='Predicted')
        
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_zlabel('Target')
        ax1.set_title('Actual vs Predicted Values in 3D')
        ax1.legend()
        
        # Second subplot: Feature importance
        ax2 = fig.add_subplot(122)
        feature_names = ['Const', 'x', 'y', 'z', 'x²', 'y²', 'z²', 
                        'xy', 'xz', 'yz', 'sin(x)', 'sin(y)', 'sin(z)']
        coeffs = np.abs(self.coefficients)
        sorted_idx = np.argsort(coeffs)
        pos = np.arange(len(feature_names))
        
        ax2.barh(pos, coeffs[sorted_idx])
        ax2.set_yticks(pos)
        ax2.set_yticklabels(np.array(feature_names)[sorted_idx])
        ax2.set_xlabel('|Coefficient|')
        ax2.set_title('Feature Importance in SINDy Model')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class SystemIdentification:
    def __init__(self, model_type: str = 'narx'):
        self.model_type = model_type
        self.model = None
        self.training_history = {'loss': [], 'val_loss': []}
        
    def save_training_plot(self, save_path: str):
        """Save training history plot to file"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['loss'], label='Training Loss')
        if self.training_history['val_loss']:
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_type.upper()} Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
    def save_prediction_plot(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str):
        """Save prediction comparison plot to file"""
        if self.model_type == 'sindy':
            y_pred = self.model.predict(X_test)
            # For SINDy, create 3D visualization
            self.model.plot_3d_dynamics(save_path)
            return
        
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            if self.model_type == 'narx':
                y_pred, _ = self.model(X_test_tensor)
            else:
                y_pred = self.model(X_test_tensor)
            y_pred = y_pred.numpy()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.model_type.upper()} Predictions vs Actual')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
                   validation_split: float = 0.2,
                   save_dir: str = None) -> List[float]:
        """
        Train the dynamic system model
        """
        # Split data for validation
        val_size = int(len(X_train) * validation_split)
        if val_size > 0:
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
        
        if self.model_type == 'sindy':
            self.model = SINDy()
            self.model.fit(X_train, y_train)
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val) if val_size > 0 else None
            
            # Calculate losses
            train_loss = mean_squared_error(y_train, train_pred)
            val_loss = mean_squared_error(y_val, val_pred) if val_size > 0 else None
            
            self.training_history['loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            
            return [train_loss]
        
        # Convert data to PyTorch tensors for neural network models
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        if val_size > 0:
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
        
        # Adjust batch size if dataset is too small
        batch_size = min(batch_size, len(X_train))
        if batch_size == 0:
            batch_size = 1
        
        # Initialize model based on type
        if self.model_type == 'narx':
            self.model = NARX(
                input_size=X_train.shape[2],
                hidden_size=64,
                output_size=1
            )
        else:
            self.model = DynamicLSTM(
                input_size=X_train.shape[2],
                hidden_size=64
            )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                
                if self.model_type == 'narx':
                    outputs, _ = self.model(batch_X)
                else:
                    outputs = self.model(batch_X)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            self.training_history['loss'].append(avg_loss)
            
            # Validation
            if val_size > 0:
                self.model.eval()
                with torch.no_grad():
                    if self.model_type == 'narx':
                        val_outputs, _ = self.model(X_val)
                    else:
                        val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                    self.training_history['val_loss'].append(val_loss)
        
        if save_dir:
            self.save_training_plot(os.path.join(save_dir, f"{self.model_type}_training.png"))
            if val_size > 0:
                self.save_prediction_plot(X_val, y_val, 
                    os.path.join(save_dir, f"{self.model_type}_predictions.png"))
        
        return self.training_history['loss']
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model using various metrics
        """
        if self.model_type == 'sindy':
            y_pred = self.model.predict(X_test)
        else:
            self.model.eval()
            with torch.no_grad():
                X_test = torch.FloatTensor(X_test)
                y_test = torch.FloatTensor(y_test)
                
                if self.model_type == 'narx':
                    y_pred, _ = self.model(X_test)
                else:
                    y_pred = self.model(X_test)
                
                y_pred = y_pred.numpy()
                y_test = y_test.numpy()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate AIC and BIC
        n = len(y_test)
        if self.model_type == 'sindy':
            k = np.sum(np.abs(self.model.coefficients) > 1e-10)  # Number of non-zero terms
        else:
            k = sum(p.numel() for p in self.model.parameters())
        
        aic = n * np.log(mse) + 2 * k
        bic = n * np.log(mse) + k * np.log(n)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'aic': aic,
            'bic': bic
        }
    
    def plot_training_history(self, title: str = None):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['loss'], label='Training Loss')
        if self.training_history['val_loss']:
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title or f'{self.model_type.upper()} Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_prediction_comparison(self, X_test: np.ndarray, y_test: np.ndarray, title: str = None):
        """Plot actual vs predicted values"""
        if self.model_type == 'sindy':
            y_pred = self.model.predict(X_test)
        else:
            self.model.eval()
            with torch.no_grad():
                X_test = torch.FloatTensor(X_test)
                if self.model_type == 'narx':
                    y_pred, _ = self.model(X_test)
                else:
                    y_pred = self.model(X_test)
                y_pred = y_pred.numpy()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title or f'{self.model_type.upper()} Predictions vs Actual')
        plt.grid(True)
        plt.show() 