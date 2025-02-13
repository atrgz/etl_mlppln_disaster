# import libraries
import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Tokenize and lemmatize text

    INPUT:
    text: a text string

    OUTPUT:
    clean_tokens: an array containing the tokenization of the text input
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_sums = df.drop(columns=['id', 'message', 'original', 'genre']).sum()
    
    top_5_categories = category_sums.sort_values(ascending=False).head(5)
    top_5_category_names = top_5_categories.index.tolist()
    top_5_category_values = top_5_categories.values.tolist()
    
    bottom_5_categories = category_sums.sort_values(ascending=True).head(5)
    bottom_5_category_names = bottom_5_categories.index.tolist()
    bottom_5_category_values = bottom_5_categories.values.tolist()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_5_category_names,
                    y=top_5_category_values
                )
            ],

            'layout': {
                'title': 'Top 5 Categories',
                'yaxis': {
                    'title': "Number"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=bottom_5_category_names,
                    y=bottom_5_category_values
                )
            ],

            'layout': {
                'title': 'Bottom 5 Categories',
                'yaxis': {
                    'title': "Number"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()