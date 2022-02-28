import asyncio
import itertools
import json
import re
import time
from collections import Counter
from urllib.request import urlopen

import dash_bootstrap_components as dbc
import mysql.connector
import nltk
import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import stylecloud
from dash import Dash, dcc, html  # pip install dash (version 2.0.0 or higher)
from nltk.tokenize import word_tokenize
import dash_extensions as de
from datetime import datetime
import datetime as dt

nltk.download('punkt')

# Lotties: Emil at https://github.com/thedirtyfew/dash-extensions

url = "https://assets5.lottiefiles.com/packages/lf20_COGJwc/22 - Thumbs Up.json"
url2 = "https://assets5.lottiefiles.com/packages/lf20_tQN9eF/23 - Thumbs Down.json"
url3 =  "./assets/twitter.json"
url4 = "./assets/keyboard.json"

options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))


# def create_server_connection(host_name, user_name, user_password):
#     connection = None
#     try:
#         connection = mysql.connector.connect(
#             host=host_name,
#             user=user_name,
#             passwd=user_password,
#         )
#         print("MySQL Database connection successful")
#     except Error as err:
#         print(f"Error: '{err}'")
#     return connection


# connection = create_server_connection("mysql-db.cyg6qupmqicu.ap-southeast-2.rds.amazonaws.com", "admin", "Sentiment2022analysis")

def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection


pw = "Sentiment"

connection = create_db_connection("database.cyg6qupmqicu.ap-southeast-2.rds.amazonaws.com", "admin", pw, "HTL_SENTIMENT")
# # Stopword removal
# stopword = nltk.corpus.stopwords.words('english')

# words = set(nltk.corpus.words.words( ))


# def clean_text(text):
#     # Lower case
#     text = text.lower( )
#
#     text = re.split('\W+', text)  # tokenization
#
#     # Stopword removal
#     text = [word for word in text if word not in stopword]  # remove stopwords
#
#     # Remove non-english words / sentances
#     text = " ".join(w for w in text if w.lower( ) in words or not w.isalpha( ))
#
#     return text


with urlopen('https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson') as response:
    counties = json.load(response)

state_id_map = {}
for i in counties['features']:
    state_id_map[i['properties']['STATE_NAME']] = i['id']


app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
# server = app.server
df = pd.read_sql("SELECT * FROM tweet;", con=connection)
flat_list = list(itertools.chain.from_iterable(df.tweet.apply(lambda x: word_tokenize(x))))

# Create bigrams
bgs = nltk.bigrams(flat_list)

freq_count = nltk.FreqDist(bgs)

freq_count_df = pd.DataFrame(dict(freq_count).items( ))

freq_count_df = freq_count_df.sort_values(1, ascending=False).iloc[:20, :]

viz = df.groupby(by=['place', 'sentiment']).count( ).reset_index( )
pos_sentiment = viz[viz['sentiment'] == 1]
neg_sentiment = viz[viz['sentiment'] == 0]
pos_sentiment['place_id'] = pos_sentiment['place'].map(state_id_map)
neg_sentiment['place_id'] = neg_sentiment['place'].map(state_id_map)

# df['cleaned_tweet'] = df['tweet'].apply(lambda x: clean_text(x))
# pos_sentiment['log_tweet'] = np.log(pos_sentiment['tweet'])
# neg_sentiment['log_tweet'] = np.log(neg_sentiment['tweet'])

df = df[df.cleaned_tweet != '']

# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter( )
negative_counts = Counter( )
total_counts = Counter( )

positive_tweets = df[df['sentiment'] == 1].cleaned_tweet.values.tolist( )
negative_tweets = df[df['sentiment'] == 0].cleaned_tweet.values.tolist( )

for i in range(len(positive_tweets)):
    for word in positive_tweets[i].split(" "):
        positive_counts[word] += 1
        total_counts[word] += 1

for i in range(len(negative_tweets)):
    for word in negative_tweets[i].split(" "):
        negative_counts[word] += 1
        total_counts[word] += 1

pos_neg_ratios = Counter( )

# Calculate the ratios of positive and negative uses of the most common words
# Consider words to be "common" if they've been used at least 100 times
for term, cnt in list(total_counts.most_common( )):
    if (cnt > 50):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
        pos_neg_ratios[term] = pos_neg_ratio

# print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
# print("Pos-to-neg ratio for 'health' = {}".format(pos_neg_ratios["health"]))
# print("Pos-to-neg ratio for 'mental' = {}".format(pos_neg_ratios["mental"]))
# print("Pos-to-neg ratio for 'covid' = {}".format(pos_neg_ratios["covid"]))

# Convert ratios to logs
for word, ratio in pos_neg_ratios.most_common( ):
    if ratio <= 0:
        pos_neg_ratios[word] = -np.log(1 / (ratio + 0.1))
    else:
        pos_neg_ratios[word] = np.log(ratio)

# print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
# print("Pos-to-neg ratio for 'brilliant' = {}".format(pos_neg_ratios["health"]))
# print("Pos-to-neg ratio for 'disgusting' = {}".format(pos_neg_ratios["mental"]))
# print("Pos-to-neg ratio for 'suck' = {}".format(pos_neg_ratios["covid"]))

# words most frequently seen in a review with a "NEGATIVE" label
neg_df = pd.DataFrame(list(reversed(pos_neg_ratios.most_common( )))[0:500])

neg_df['freq'] = neg_df[0].apply(lambda x: dict(negative_counts.most_common( ))[x])

neg_df = neg_df.iloc[:30, :].set_index(0)

neg_df = neg_df.reset_index( )

pos_df = pd.DataFrame(pos_neg_ratios.most_common( )[:30])
pos_df['freq'] = pos_df[0].apply(
    lambda x: dict(positive_counts.most_common( ))[x] if x in dict(positive_counts.most_common( )) else 0)

stylecloud.gen_stylecloud(' '.join(positive_tweets), colors=['#ecf0f1', '#3498db', '#e74c3c'],
                          background_color='#0000', icon_name='fas fa-hashtag', output_name='./assets/pos_cloud.png')

# stylecloud.gen_stylecloud(' '.join(negative_tweets), colors=['#e74c3c', '#3498db', '#ecf0f1'],
#                           background_color='#0000', icon_name='fas fa-thumbs-down',
#                           output_name='./assets/neg_cloud.png')



def update_graph(df, color_name, tweet):
    fig_Heterogeneity = px.choropleth_mapbox(df,
                                             locations="place_id",
                                             geojson=counties,
                                             color="tweet",
                                             mapbox_style="carto-darkmatter",  # carto-darkmatter",
                                             hover_name='place',
                                             hover_data={'place_id': False},
                                             opacity=0.8, color_continuous_midpoint=0,
                                             color_continuous_scale=eval('px.colors.sequential.' + color_name),
                                             labels={'EC': 'Environmental Heterogeneity'},
                                             center={"lat": -25.2744, "lon": 133.7751},
                                             range_color=(0, df.tweet.max( )),
                                             zoom=2.5)
    fig_Heterogeneity.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, paper_bgcolor="black", plot_bgcolor='#fff',
                                    font={
                                        "color": "white"})
    fig_Heterogeneity.update_geos(fitbounds="locations", visible=False)

    return fig_Heterogeneity


def uni_gram_graph(df, color_name):
    bar_fig = px.bar(df[[0, 'freq']].sort_values(by='freq', ascending=True),
                     x='freq', y=0, orientation='h', text='freq', color='freq',
                     color_continuous_scale=color_name,
                     labels={
                         "0": "Unigrams",
                         "freq": "Frequency",
                         "freq": "Frequency"
                     }
                     )

    bar_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    bar_fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide',
                          plot_bgcolor='rgb(0,0,0)', paper_bgcolor='rgb(0,0,0)',
                          font={
                              "color": "white"
                          })
    return bar_fig


def bigram_graph(df, color_name, sentiment='pos'):
    if sentiment == 'pos':
        df1 = df[df['sentiment'] == 1]
    else:
        df1 = df[df['sentiment'] == 0]

    flat_list = list(itertools.chain.from_iterable(df1.tweet.apply(lambda x: word_tokenize(x))))

    # Create bigrams
    bgs = nltk.bigrams(flat_list)

    freq_count = nltk.FreqDist(bgs)

    freq_count_df = pd.DataFrame(dict(freq_count).items( ))

    freq_count_df = freq_count_df.sort_values(by=1, ascending=False).iloc[:20, :]

    freq_count_df[0] = freq_count_df[0].apply(lambda x: x[0] + '-' + x[1])

    bigram_bar_fig = px.bar(freq_count_df,
                            x=1, y=0, orientation='h', text=1, color=1, color_continuous_scale=color_name,
                            labels={
                                "0": "Bi-grams",
                                "1": "Frequency",
                                "1": "Frequency"
                            }
                            )

    bigram_bar_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    bigram_bar_fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide',
                                 plot_bgcolor='rgb(0,0,0)', paper_bgcolor='rgb(0,0,0)',
                                 font={
                                     "color": "white"
                                 })

    return bigram_bar_fig


def pos_area(df):
    df['date_only'] = df.date.apply(lambda x: x.split(' ')[0])
    df_date = df.groupby(['date_only', 'sentiment']).count( )

    df_date = df_date.reset_index( )

    df_date['sentiment'] = df_date['sentiment'].map({0: 'Negative', 1: 'Positive'})
    area_fig = px.area(df_date, x="date_only", y="cleaned_tweet", color="sentiment", line_group="sentiment",
                       labels={
                           "cleaned_tweet": "No. of tweets",
                           "date_only": "Dates",
                           "sentiment": "Sentiment"
                       }
                       )

    area_fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide',
                           plot_bgcolor='rgb(0,0,0)', paper_bgcolor='rgb(0,0,0)',
                           font={
                               "color": "white"
                           })

    return area_fig


asd = datetime.fromisoformat(df.date.max()) + dt.timedelta(days=1)
today = datetime.today()


connection.close()

# # ------------------------------------------------------------------------------
# # App layout
app.layout = dbc.Container([
    # First Row
    dbc.Row([
        # First row first col
        dbc.Col([
            dbc.Card([
                dbc.CardImg(src='./assets/HTL_Logo.PNG')
            ], className='mb-2'),

        ], width=2),

        # First Row Second Col
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H1("Twitter Health Sentiment Analysis", style={'fontFamily': "serif", 'margin-bottom': "0px", 'color': "black"}, className='animate-character')
                    ])
                ])
            ], style={'borderRadius': '0px 25px 25px 0px',
                                      'overflow': 'hidden', 'background': '#41B3A3'}),
        ], width=6),

        # First Row Second Col
        dbc.Col([
            dbc.Card([

                dbc.CardBody([
                    html.Div([
                        html.H6(children='Last updated at: {} '.format(df.date.max().split(' ')[0]), style={'margin-bottom': "0px", 'color': "white", "textAlign": "right"})
                    ]),

                    html.Div([
                        html.H6(children="Next update on: {} ".format(asd.date()), style={'margin-bottom': "0px", 'color': "white", "textAlign": "right"})
                    ])
                ])
            ], style={'background-color': 'black'}),
        ], width=4)
    ], className='mb-4 mt-4'),

    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(de.Lottie(options=options, width="50%", height="50%", url=url),
                               style={'borderRadius': '25px 0px 0px 0px',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Positive Tweets", style={'textAlign': 'center',
                                                      'color': "black",
                                                      'background-color': '#41B3A3'}),
                    html.H2(id='content-positive', children='{}%'.format(round(100*(len(positive_tweets)/(len(positive_tweets)+len(negative_tweets))))),
                            style={'textAlign': 'center',
                            'color': "black",
                            'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=3, style={'background-color': 'black'}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(de.Lottie(options=options, width="50%", height="50%", url=url2),
                               style={'borderRadius': '25px 0px 0px 0px', "height": "",
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Negative Tweets", style={'textAlign': 'center',
                                                      'color': "black",
                                                      'background-color': '#41B3A3'}),
                    html.H2(id='content-negative', children='{}%'.format(round(100*(len(negative_tweets)/(len(positive_tweets)+len(negative_tweets))))),
                            style={'textAlign': 'center',
                            'color': "black",
                            'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=3, style={'background-color': 'black'}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(de.Lottie(options=options, width="50%", height="50%", url=url3),
                               style={'borderRadius': '25px 0px 0px 0px',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Total Tweets Collected", style={'textAlign': 'center',
                                                      'color': "black",
                                                      'background-color': '#41B3A3'}),
                    html.H2(id='content-tweets', children='{}K'.format(round((len(positive_tweets)+len(negative_tweets))/1000,1)),
                            style={'textAlign': 'center',
                            'color': "black",
                            'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=3, style={'background-color': 'black'}),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(de.Lottie(options=options, width="50%", height="50%", url=url4),
                               style={'borderRadius': '25px 0px 0px 0px',
                                      'overflow': 'hidden', 'background': '#E8A87C'}),

                dbc.CardBody([
                    html.H6("Total Words Analyzed", style={'textAlign': 'center',
                                                      'color': "black",
                                                      'background-color': '#41B3A3'}),
                    html.H2(id='content-words', children='{}K'.format(round((sum(list(dict(positive_counts).values())) + sum(list(dict(negative_counts).values())))/1000,1)),
                            style={'textAlign': 'center',
                            'color': "black",
                            'background-color': '#41B3A3'})
                ], style={'borderRadius': '0px 0px 25px 0px', 'overflow': 'hidden', 'background-color': '#41B3A3'})

            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=3, style={'background-color': 'black'}),





    ], className="mb-20 mt-6"),

    # Second Row
    dbc.Row([
        # Second Row First Col
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(style={'margin': '0', 'display': 'flex'},
                              config={"displayModeBar": False, "showTips": False},
                              figure=update_graph(pos_sentiment, 'Tealgrn', 'Positive tweets'))
                ])
            ], style={'background-color': 'black'}),
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(style={'margin': '0'},
                              config={"displayModeBar": False, "showTips": False},
                              figure=update_graph(neg_sentiment, 'Redor', 'Negative tweets'))
                ])
            ], style={'background-color': 'black'}),
        ], width=6),

    ], className='mb-4 mt-4'),

    # Graph headlines (unigrams)
    dbc.Row([
        dbc.Col([
            dbc.CardHeader([
                    html.H5(" Top-20 Most common positive unigrams within positive tweets",
                            style={'padding': '8px', 'borderRadius': '0px 25px 0px 0px',
                                   'overflow': 'hidden', 'background': '#E8A87C', 'color': "black",
                                                      'background-color': '#E8A87C', 'height': '72px'})
            ])

        ], width=4),
        dbc.Col([
            dbc.CardHeader([
                    html.H5(" Tweets collected so far...",
                            style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                                   'overflow': 'hidden', 'background': '#E8A87C', 'color': "black",
                                   'background-color': '#E8A87C', 'height': '72px', 'position': 'relative'})
            ])

        ], width=4),
        dbc.Col([
            dbc.CardHeader([
                    html.H5(" Top-20 Most common negative unigrams within negative tweets",
                            style={'padding': '8px', 'borderRadius': '0px 25px 0px 0px',
                                   'overflow': 'hidden', 'background': '#E8A87C', 'color': "black",
                                   'background-color': '#E8A87C', 'height': '72px'})
            ])

        ], width=4)
    ], className='mt-8'),

    # Third Row (uni-grams graph)
    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                              config={"displayModeBar": False, "showTips": False},
                              figure=uni_gram_graph(pos_df, 'Tealgrn'))
                ], style={'background-color': 'rgb(0,0,0)', "outline": "solid #E8A87C"})
            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=4, style={'background-color': 'rgb(0,0,0)'}),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=pos_area(df))
                ], style={'background-color': 'rgb(0,0,0)', "outline": "solid #E8A87C"})
            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=4, style={'background-color': 'rgb(0,0,0)'}),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                              config={"displayModeBar": False, "showTips": False},
                              figure=uni_gram_graph(neg_df, 'Redor'))
                ], style={'background-color': 'rgb(0,0,0)', "outline": "solid #E8A87C"})
            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=4, style={'background-color': 'rgb(0,0,0)'})
    ], className='mb-4 mt-0', style={'background-color': 'rgb(0,0,0)'}),

    # Graph headlines (Bi-grams)
    dbc.Row([
        dbc.Col([
            dbc.CardHeader([
                    html.H4(" Top-20 Most common positive Bi-grams within positive tweets",
                                    style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                                           'overflow': 'hidden', 'background': '#E8A87C', 'color': "black",
                                           'background-color': '#E8A87C', 'height': '72px'})
            ])

        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.CardHeader([
                        html.H4("What are people talking about?",
                                style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                                       'overflow': 'hidden', 'background': '#E8A87C', 'color': "black",
                                       'background-color': '#E8A87C', 'height': '72px'})
                    ])

                ], style={'background-color': 'rgb(0,0,0)'})

            ], style={'background-color': 'rgb(0,0,0)'})

        ], width=4),

        dbc.Col([
            dbc.CardHeader([
                    html.H4(" Top-20 Most common negative Bi-grams within negative tweets",
                            style={'padding': '15px', 'borderRadius': '0px 25px 0px 0px',
                                   'overflow': 'hidden', 'background': '#E8A87C', 'color': "black",
                                   'background-color': '#E8A87C', 'height': '72px'})
            ])

        ], width=4)
    ], className="mb-0"),

    # Fourth row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                              config={"displayModeBar": False, "showTips": False},
                              figure=bigram_graph(df, 'Tealgrn', 'pos'))
                ], style={'background-color': 'rgb(0,0,0)', "outline": "solid #E8A87C"})
            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=4, style={'background-color': 'rgb(0,0,0)'}),

        # Wordcloud
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.CardImg(src='./assets/pos_cloud.png')
                ], style={'background-color': 'rgb(0,0,0)', "outline": "solid #E8A87C"})
            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=4, style={'background-color': 'rgb(0,0,0)'}),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(style={'display': 'flex', 'width': '100%', 'height': '200%', 'font_color': 'white'},
                              config={"displayModeBar": False, "showTips": False},
                              figure=bigram_graph(df, 'Redor', 'neg'))
                ], style={'background-color': 'rgb(0,0,0)', "outline": "solid #E8A87C"})
            ], style={'background-color': 'rgb(0,0,0)'})
        ], width=4, style={'background-color': 'rgb(0,0,0)'}),
    ]),

], style={'background-color': 'black'}, fluid=True)


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
    # app.run_server(host='0.0.0.0', port=8050, debug=True)