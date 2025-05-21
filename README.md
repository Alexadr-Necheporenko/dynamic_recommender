# Dynamic Recommender System

This project implements a dynamic recommendation system that combines collaborative filtering with dynamic systems modeling. The system uses both Matrix Factorization and NARX (Nonlinear AutoRegressive with eXogenous inputs) models to capture both static and temporal patterns in user preferences.

## Features

- Data preprocessing and normalization
- Dynamic time series modeling using NARX and LSTM
- Matrix Factorization for collaborative filtering
- System identification and evaluation metrics
- Combined static and dynamic predictions

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `data_processing.py`: Contains data preprocessing and preparation utilities
- `dynamic_models.py`: Implements NARX and LSTM models for dynamic modeling
- `recommender.py`: Main recommendation system combining static and dynamic approaches
- `main.py`: Example script demonstrating system usage

## Usage

1. Download the MovieLens dataset from https://grouplens.org/datasets/movielens/
2. Update the `ratings_path` in `main.py` to point to your downloaded dataset
3. Run the example:

```bash
python main.py
```

## Implementation Details

### Data Processing
- Normalizes user, item, and rating features
- Creates time series sequences for each user
- Handles missing values

### Dynamic Modeling
- NARX implementation for temporal pattern recognition
- LSTM alternative for complex temporal dependencies
- System identification with various evaluation metrics

### Recommendation Generation
- Combines static (Matrix Factorization) and dynamic (NARX/LSTM) predictions
- Provides personalized recommendations based on user history
- Supports top-N recommendation generation

## Evaluation Metrics

The system provides various evaluation metrics:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)

## Example Output

```python
Top 5 recommendations for user 1:
Item 123: Predicted rating = 4.85
Item 456: Predicted rating = 4.72
Item 789: Predicted rating = 4.65
Item 234: Predicted rating = 4.58
Item 567: Predicted rating = 4.52
``` 