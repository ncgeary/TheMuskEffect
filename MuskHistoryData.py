#%%
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import twitterscraper as ts
import os
import re
import json
import datetime as dt


#%%
# ------------------
# Got this method of pulling tweets form here:
# ------------------
# https: // medium.com/@kevin.a.crystal/scraping-twitter-with-tweetscraper-and-python-ea783b40443b
# https: // github.com/jenrhill/Power_Outage_Identification/blob/master/code/1_Data_Collection_and_EDA.ipynb
# https: // www.youtube.com/watch?v = zF_Q2v_9zKY

user = 'elonmusk'
limit = 1000

tweets = ts.query_tweets_from_user(user=user, limit=limit)


# %%
df = pd.DataFrame(tweet.__dict__ for tweet in tweets)

# %%
#Sentimental Analysis Score

def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text).split())

def analyze_sentiment_score(self, tweet):
    analysis_score = TextBlob(self.clean_tweet(tweet.text))
    analysis_score = analysis_score.sentiment.polarity
    return analysis_score

def analyze_sentiment_result(self, tweet):

    analysis_result = TextBlob(self.clean_tweet(tweet.text))

    if analysis_result.sentiment.polarity > 0:
        return 'Positive'
    elif analysis_result.sentiment.polarity == 0:
        return '0'
    else:
        return 'Negative'


df['Sentiment Score'] = np.array(
    [tweet_analyzer.analyze_sentiment_score(tweet) for tweet in df['text']])
df['Sentiment Result'] = np.array(
    [tweet_analyzer.analyze_sentiment_result(tweet) for tweet in df['text']])

df.head()


# %%
mainData = df.copy()
mainData = mainData.drop(['has_media', 'img_urls', 'is_replied', 'is_reply_to',
                'links', 'parent_tweet_id',  'reply_to_users',
               'screen_name',  'text_html',
               'timestamp_epochs', 'tweet_id', 'tweet_url', 'user_id', 
               'video_url'],axis=1)

mainData['timestamp'] = mainData['timestamp'].dt.date

mainData = mainData.groupby('timestamp').agg(
    {'likes': 'sum', 'retweets': 'sum', 'replies':'sum','Sentiment Score': 'mean'})

mainData.reset_index().head()

# %%
#normalizing data
scaler = MinMaxScaler()

mainData['SIZE_likes'] = scaler.fit_transform(
    mainData['likes'].values.reshape(-1, 1))
mainData['SIZE_retweets'] = scaler.fit_transform(
    mainData['retweets'].values.reshape(-1, 1))
mainData['SIZE_replies'] = scaler.fit_transform(
    mainData['replies'].values.reshape(-1, 1))

mainData.describe()

# %%
# Get the data of the stock Tesla Stock (TSLA)
stockData = yf.download("TSLA", start="2019-1-1", end="2020-01-27")
stockData.reset_index().head()
stockData.head()

#%%

#Joining Dataframes
AllData = mainData.join(stockData, lsuffix='timestamp',
                        rsuffix='Date').reset_index()
                        

AllData

#need to figure out what to do on days with no tweets... you got this!!

 # %%
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=mainData.timestamp,
    y=mainData['Sentiment Score'],
    name="Sentiment Score",
    line_color='red',
    opacity=0.8))

# fig.add_trace(go.Scatter(
#     x=mainData.Date,
#     y=mainData['SIZE_stock'],
#     name="Stock Price",
#     line_color='green',
#     opacity=0.8))

fig.add_trace(go.Scatter(
    x=mainData.timestamp,
    y=mainData['SIZE_likes'],
    name="Likes",
    line_color='blue',
    opacity=0.8))

fig.add_trace(go.Scatter(
    x=mainData.timestamp	,
    y=mainData['SIZE_retweets'],
    name="Retweets",
    line_color='grey',
    opacity=0.8))

fig.add_trace(go.Scatter(
    x=mainData.timestamp	,
    y=mainData['SIZE_replies'],
    name="Replys",
    line_color='green',
    opacity=0.8))

# Use date string to set xaxis range
fig.update_layout(xaxis_range=['2019-1-1', '2020-1-25'],
                  title_text="Replys, Likes, & Retweets")
fig.show()


# %%
