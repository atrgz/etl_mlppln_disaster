# import libraries
import sys
import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """Load message database and extract name of category columns

    INPUT:
    database_filepath: database with the data to generate a ML model

    OUTPUT:
    X: numpy array with the features
    Y: numpy array with the targets
    category_names: list with the name of the categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)
    X = df.message.values
    Y = df.drop(columns=['id', 'message', 'original', 'genre']).values
    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns.tolist()
    
    return (X, Y, category_names)


def tokenize(text):
    """Normalize, tokenize and lemmatize text

    INPUT:
    text: a text string

    OUTPUT:
    clean_tokens: an array containing the tokenization of the text input
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # keep only letters and numbers, normalize to lower case
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    tokens = [t for t in tokens if t not in stopwords.words("english")] # get rid of stop words

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build a pipeline for a ML model and optimize parameters with grid search

    INPUT:
    None

    OUTPUT:
    cv: ML model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    
    parameters = {
        'clf__estimator__n_estimators': [10, 50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    # This is a long process, verbose will print the current step
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Estimate and evaluate ML model

    INPUT:
    model: a machine learning model
    X_test: a features array to make predictions
    Y_test: a target array to evaluate predictions
    category_names: a list of category names to present results

    OUTPUT:
    None
    """
    Y_pred = model.predict(X_test) # make predictions using the model
    # interate each category to evaluate model
    for i, category in enumerate(category_names):
        report = classification_report(Y_test[:, i], Y_pred[:, i], output_dict=True, zero_division=0)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        
        print(f"{category} \n \tPrecision: {precision:.4f}, \tRecall: {recall:.4f}, \tF1-Score: {f1_score:.4f}\n")

def save_model(model, model_filepath):
    """Save the model to a pickle file

    INPUT:
    model: a machine learning model
    model_filepath: desired name for the pickle file

    OUTPUT:
    None
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    # check the numer of arguments
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