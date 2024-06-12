import sys
import pickle
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split



def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('CleanedData', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # Normalize text
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        # Lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('Category:', category_names[i])
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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