import numpy as np
import pandas as pd

import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from prefect import task

#task

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)


@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    
    return X, y


@task
def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@task
def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Pre-processing the data.
    """
    def clean_text(data):
        lemmatizer = WordNetLemmatizer()
    
        # Removing special characters and digits
        sentence = re.sub(r"[^a-zA-Z0-9\s]", " ", data)
    
        # change sentence to lower case
        sentence = sentence.lower()

        # tokenize into words
        tokens = sentence.split()
    
        # remove stop words                
        clean_tokens = [t for t in tokens if not t in stopwords.words("english")]
    
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
    
        sentence =  " ".join(clean_tokens)
    
        return sentence
    
    for col in X_train.columns:
        X_train[col] = X_train[col].apply(lambda doc: clean_text(doc))
    for col in X_test.columns:
        X_test[col] = X_test[col].apply(lambda doc: clean_text(doc))

    return X_train, X_test, y_train, y_test


@task
def vectorizing_data(X_train, X_test, y_train, y_test):
    """
    vectorizing the data.
    """
    vector = CountVectorizer()
    X_train_vec = vector.fit_transform(X_train)
    X_test_vec = vector.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test


@task
def train_model(X_train_vec, y_train, hyperparameters):
    """
    Training the machine learning model.
    """
    clf = DecisionTreeClassifier(**hyperparameters)
    clf.fit(X_train_vec, y_train)
    return clf


@task
def evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test):
    """
    Evaluating the model.
    """
    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)

    train_score = metrics.accuracy_score(y_train, y_train_pred)
    test_score = metrics.accuracy_score(y_test, y_test_pred)
    
    return train_score, test_score


from prefect import flow

# Workflow

@flow(name="Decision Tree Training Flow")
def workflow():
    DATA_PATH = "data/data.csv"
    INPUTS = ['Review text']
    OUTPUT = 'Review Sentiment'
    HYPERPARAMETERS = {'max_features' : 1000, 'max_depth': 10}
    
    # Load data
    reviews = load_data(DATA_PATH)

    # Identify Inputs and Output
    X, y = split_inputs_output(reviews, INPUTS, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    #preprocessing the data
    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)
    
    # vectorizing to data
    X_train_vec, X_test_vec, y_train, y_test = vectorizing_data(X_train['Review text'], X_test['Review text'], y_train, y_test)
    
    # Build a model
    model = train_model(X_train_vec, y_train, HYPERPARAMETERS)
    
    # Evaluation
    train_score, test_score = evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test)
    
    print("Train Score:", train_score)
    print("Test Score:", test_score)


if __name__ == "__main__":
    workflow.serve(
        name="Sentiment-Analysis-Review-Prediction", 
        cron="*/10 * * * *"
    )