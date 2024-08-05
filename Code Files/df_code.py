'''Splits data into training, and testing set
Ensures each model uses the same sets
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main(dataset):
    seed = 466

    # Uncomment which desired path for each dataset
    # all the features
    # less features
    if dataset == 'all':
        #file = r'C:\Users\gwend\UofA\CMPUT 466\Final Mini Project 466\CSV Files\music_genre_466.csv' # all feats
        file = r"C:\Users\gwend\UofA\CMPUT 466\Final Mini Project 466\CSV Files\music_shortened_features_466.csv"

    else:
        #file = r'C:\Users\gwend\UofA\CMPUT 466\Final Mini Project 466\CSV Files\music_shortened_genre_466.csv'
        file = r"C:\Users\gwend\UofA\CMPUT 466\Final Mini Project 466\CSV Files\music_shortened_features+genre_466.csv"

    df = pd.read_csv(file)

    df = df.sample(frac=1, random_state=seed)
    y = df['genre']
    X = df.drop(columns=['genre'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

    class_names = y.unique().tolist()
    print('There are',len(class_names), 'classes')

    return X_train, X_test, y_train, y_test, class_names
