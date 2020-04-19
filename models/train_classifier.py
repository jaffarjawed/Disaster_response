import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
stopwords = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    engine = create_engine('sqlite:////'+ database_filepath)
    df = pd.read_sql_table('Disaster', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    return X, Y

def tokenize(text):
    pattern = re.compile(r"[^A-Za-z]")
    text = re.sub(pattern, ' ', str(text))
    token = word_tokenize(text)
    token = [word for word in token if word not in stopwords]
    porter = PorterStemmer()
    lemma = WordNetLemmatizer()
    lemmatized = [lemma.lemmatize(word).lower().strip() for word in token]
    clean_token = [porter.stem(word) for word in lemmatized]
    return clean_token


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()