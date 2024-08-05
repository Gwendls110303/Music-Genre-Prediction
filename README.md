# Predicting Music Genres From Song Attributes

This project aims to classify music genres using various machine learning algorithms. The dataset consists of features extracted from music tracks, and the task is to predict the genre of each track.

## File Descriptions

**Code Files**
- **`project_main.py`**: Trains and evaluates each machine learning model.
- **`graphing.py`**: Generates bar graphs to visualize the accuracies of the models.
- **`decisiontrees_466.py`**: Contains the implementation of the decision tree classifier.
- **`knn_466.py`**: Contains the implementation of the K-nearest neighbors classifier.
- **`fnn_466.py`**: Contains the implementation of the feedforward neural network classifier.
- **`df_code.py`**: Splits the dataset into training and testing sets.
- **[`Dataset Preprocessing Code`](https://colab.research.google.com/drive/1IIpfNlytwfmCnXefq9ArI1xv-4OFxOuD#scrollTo=QGnYhvSVT6Gn)**: Combines the two music genre datasets

**CSV Files**
- **`music_genre.csv`**: from Prediction of Music Genre Dataset
- **`music_genre_spot.csv`**: from Spotify Tracks Grenre Dataset
- **`music_genre_466.csv`**: Full combined dataset
- **`music_shortened_features_466.csv`**: Full combined dataset with less features
- **`music_shortened_genre_466.csv`**: Less classes dataset
- **`music_shorted_features+genre_466.csv`**: Less classes dataset with less features


## Dependencies

To run the project, these dependencies are needed:

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- numpy
- torch

## Usage

1. (All the cleaned csv files are already included too) Clean the datasets using the Google Colab Code. Download `music_genre.csv` and `music_genre_spot.csv` into your Google Drive. Ensure that the path names in the code are changed to your directory.
2. Run `project_main.py` to train and evaluate the models. Copy the printed append statements into `graphing.py`.
3. Run `graphing.py` to generate visualizations of the model accuracies. Make sure to change the filepath names as needed.




