'''
This program reads two csv files and merges them based on a common key column.
'''
# import the pandas library
# you can install using the following command: pip install pandas

import pandas as pd

# Read the files into two dataframes.
df1 = pd.read_csv('ml-32m/ratings.csv')
df2 = pd.read_csv('ml-32m/movies_sampled.csv')

# Merge the two dataframes, using _ID column as key
df3 = pd.merge(df1, df2, on = 'movieId')
df3.set_index('movieId', inplace = True)

# Write it to a new CSV file
df3.to_csv('ratings_with_movies.csv')
