import pandas as pd  

data = pd.read_csv('ml-32m/ratings_with_movies.csv')

data.drop('timestamp', inplace=True, axis=1)

data.to_csv('ratings_with_movies_without_timestamp.csv')
