import sys

from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    """Load dataset from SQLite database"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('dataset', engine)
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X.values, Y.values, list(Y.columns)


def tokenize(text):
    """Use Twitter aware casual tokenizer followed by WordNetLemmatizer on extracted tokens"""

    # Implementation of casual_tokenize can be at: www.nltk.org/_modules/nltk/tokenize/casual.html
    tokens = casual_tokenize(text.lower())

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Build model pipeline"""
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', OneVsRestClassifier(RandomForestClassifier(), n_jobs=-1)),
    ])



def evaluate_model(model, X_test, Y_test, category_names):
    """Print a report with model accuracy, precision, recall and f1-score"""
    Y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Classification report per category:")
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Pickle model and save it at the given filepath"""
    with open(model_filepath, 'wb') as p_file:
        p_file.write(pickle.dumps(model))


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
