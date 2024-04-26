import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_data(path_to_file):
    """
    Load preprocessed tweet data from a CSV file into a pandas DataFrame.

    Parameters:
    path_to_file (str): Path to the CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing only preprocessed tweet text data.
    """
    # read in the data with proper column names
    df = pd.read_csv(path_to_file, delimiter="|", header=None, names=["id", "time", "text"], dtype={"id": str})

    # remove tweet id and timestamp columns
    df.drop(columns=["id", "time"], inplace=True)

    # remove any words that start with "@"
    df["text"] = df["text"].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith("@")]))

    # remove any hashtag symbols (#) from words
    df["text"] = df["text"].apply(lambda x: ' '.join([word.strip("#") for word in x.split()]))

    # remove any URLs
    df["text"] = df["text"].apply(lambda x: re.sub(r"http\S+", "", x))

    # convert every word to lowercase
    df["text"] = df["text"].apply(lambda x: x.lower())

    return df


def convert_to_zero_one_matrix(df):
    """
    Convert the DataFrame containing tweet text into 0/1 matrix.

    Parameters:
    df (DataFrame): DataFrame containing tweet text.

    Returns:
    sparse matrix: 0/1 matrix.
    """
    # Initialize TF-IDF vectorizer
    vectorizer = CountVectorizer(binary=True)

    # Convert tweet text into TF-IDF weighted feature vectors
    X = vectorizer.fit_transform(df["text"])

    return X


if __name__ == "__main__":
    filename = "../../data/Health-Tweets/bbchealth.txt"
    df = load_data(filename)
    print(df.head())

    X = convert_to_zero_one_matrix(df)
    print(type(X))
    print(X)