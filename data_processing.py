import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List

class DataPreprocessor:
    def __init__(self):
        self.user_scaler = MinMaxScaler()
        self.item_scaler = MinMaxScaler()
        self.rating_scaler = MinMaxScaler()
        
    def load_movielens_data(self, ratings_path: str) -> pd.DataFrame:
        """
        Load MovieLens dataset
        """
        df = pd.read_csv(ratings_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features
        """
        df_normalized = df.copy()
        df_normalized['userId'] = self.user_scaler.fit_transform(df[['userId']])
        df_normalized['movieId'] = self.item_scaler.fit_transform(df[['movieId']])
        df_normalized['rating'] = self.rating_scaler.fit_transform(df[['rating']])
        return df_normalized
    
    def create_user_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Create sequences of user interactions for time series analysis
        """
        user_sequences = {}
        user_timestamps = {}
        
        for user_id in df['userId'].unique():
            user_data = df[df['userId'] == user_id].sort_values('timestamp')
            
            # Create sequences of ratings and timestamps
            ratings = user_data['rating'].values
            times = user_data['timestamp'].values
            
            # Create overlapping sequences
            sequences = []
            time_sequences = []
            
            for i in range(len(ratings) - sequence_length + 1):
                sequences.append(ratings[i:i + sequence_length])
                time_sequences.append(times[i:i + sequence_length])
                
            if sequences:
                user_sequences[user_id] = np.array(sequences)
                user_timestamps[user_id] = np.array(time_sequences)
        
        return user_sequences, user_timestamps
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the dataset
        """
        # Create a copy to avoid modifying the original dataframe
        df_filled = df.copy()
        
        # Calculate user means
        user_means = df_filled.groupby('userId')['rating'].transform('mean')
        
        # Fill missing ratings with mean rating per user
        df_filled['rating'] = df_filled['rating'].fillna(user_means)
        
        # If still any missing values, fill with global mean
        global_mean = df_filled['rating'].mean()
        df_filled['rating'] = df_filled['rating'].fillna(global_mean)
        
        return df_filled 