import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    INPUT:
    database_filepath: filepath database  
    
    OUTPUT:
    X: messages (input variable) 
    y: categories of the messages (output variable)
    category_names: category name for y
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    INPUT:
    text: raw text
    
    OUTPUT:
    clean_tokens: tokenized messages
    """    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model(clf = AdaBoostClassifier()):
    """
    INPUT:
    clf: classifier model 
    
    OUTPUT:
    cv = ML model pipeline after performing grid search
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    parameters = {
        'clf__estimator__learning_rate':[0.5, 1.0],
        'clf__estimator__n_estimators':[10,20]
    
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
    
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model: Machine Learning model
    X_test: test messages
    y_test: categories for test messages
    category_names: category name for y
    
    OUTPUT:
    none - print scores (precision, recall, f1-score) for each output category of the dataset.
    """
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    INPUT:
    model: Machine Learning model
    model_filepath: location to save the model
    
    OUTPUT:
    none
    """    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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