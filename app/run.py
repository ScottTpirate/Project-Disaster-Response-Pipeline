import json
import plotly
import joblib
import nltk
import pandas as pd


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    """
    Tokenize text using NLTK's word_tokenize and lemmatize each token.

    Args:
        text (str): Text to be tokenized.

    Returns:
        list: List of lemmatized lowercase tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('CleanedData', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Render the main page of the web app with data visualizations.

    Returns:
        str: Rendered HTML page including visualizations of message data.
    """
    # Extract data for the visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Example for additional visual: Category Distribution
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    
    # Example for additional visual: Message Length by Genre
    df['message_length'] = df['message'].apply(len)
    message_lengths = df.groupby('genre')['message_length'].apply(list)
    genre_list = list(message_lengths.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(x=genre_names, y=genre_counts)
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(x=category_names, y=category_counts)
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },
        {
            'data': [
                {'x': genre_list, 'y': message_lengths[genre], 'type': 'box', 'name': genre}
                for genre in genre_list
            ],
            'layout': {
                'title': 'Message Length Distribution by Genre',
                'yaxis': {'title': "Length of Messages"},
                'xaxis': {'title': "Genre"}
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handle user query and display model results.

    Returns:
        str: Rendered HTML page showing the classification results for user query.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    Run the Flask app.
    """
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()