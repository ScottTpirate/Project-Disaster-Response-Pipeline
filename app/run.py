import json
import plotly
import joblib
import nltk
import pandas as pd
from collections import Counter
import itertools
import numpy as np
from textblob import TextBlob

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Box, Histogram, Scatter, Pie
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
    # Prepare data for visualizations
    request_types = df[['request', 'offer']].sum()
    request_type_names = ['Request', 'Offer']

    aid_related = df[['food', 'water', 'shelter']]
    aid_related_totals = aid_related.sum().tolist()
    aid_related_names = ['Food', 'Water', 'Shelter']
    
    # Ensure only numeric columns are included in the sum
    numeric_cols = df.select_dtypes(include=[np.number]).drop(['id', 'related', 'aid_related', 'request', 'offer', 'direct_report'], axis=1)  # Select only numeric columns and drop unintesting ones
    feature_sums = numeric_cols.sum().sort_values(ascending=False)  # Sum and sort numeric columns


    # Graphs list
    graphs = [
        {
            'data': [
                Bar(x=aid_related_names, y=aid_related_totals,
                       text=aid_related_totals, textposition='auto',
                       marker=dict(color=['#FFD700', '#9EA0A1', '#CD7F32']))
            ],
            'layout': {
                'title': 'Messages by Aid Related',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Aid Type"}
            }
        },
        {
            'data': [
                Pie(labels=feature_sums.index, values=feature_sums.values, hole=.3, 
                       hoverinfo='label+percent', textinfo='value')
            ],
            'layout': {
                'title': 'Overview of All Features'
            }
        },
        {
            'data': [
                Pie(labels=request_type_names, values=request_types.tolist(), hole=.3)
            ],
            'layout': {
                'title': 'Distribution of Requests and Offers'
            }
        }
    ]

    # Encode the graphs for use in the HTML template
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render the HTML template, passing data and IDs for graphs
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