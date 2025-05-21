import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import entropy

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
        return predictions

class SystemIdentification:
    def __init__(self, model_type: str = 'narx'):
        self.model_type = model_type
        self.model = None
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> List[float]:
        """
        Train the dynamic system model
        """
        # Convert data to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
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
        
        losses = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
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
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            losses.append(avg_loss)
            
        return losses
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model using various metrics
        """
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
            k = sum(p.numel() for p in self.model.parameters())  # number of parameters
            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'aic': aic,
                'bic': bic
            } 