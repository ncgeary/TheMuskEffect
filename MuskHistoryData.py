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
import yfinance as yf
import plotly
import plotly.express as px
import plotly.graph_objs as go


#%%
# ------------------
# Got this method of pulling tweets form here:
# ------------------
# https: // medium.com/@kevin.a.crystal/scraping-twitter-with-tweetscraper-and-python-ea783b40443b
# https: // github.com/jenrhill/Power_Outage_Identification/blob/master/code/1_Data_Collection_and_EDA.ipynb
# https: // www.youtube.com/watch?v = zF_Q2v_9zKY

user = 'elonmusk'
limit = 10000

tweets = ts.query_tweets_from_user(user=user, limit=limit)


#%%
class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """
    #clean tweets
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
        
    #creating sentimental score using TextBlob 
    def analyze_sentiment_score(self, tweet):
        analysis_score = TextBlob(self.clean_tweet(tweet))
        analysis_score = analysis_score.sentiment.polarity
        return analysis_score

    #Determining positive vs negative tweets
    def analyze_sentiment_result(self, tweet):

        analysis_result = TextBlob(self.clean_tweet(tweet))

        if analysis_result.sentiment.polarity >= 0.3:
            return 'Positive'
        elif analysis_result.sentiment.polarity <= -0.3:
            return 'Negative'
        else:
            return '0'

    def tweets_to_data_frame(self, tweets):
        df2 = pd.DataFrame(
            data=[tweet.timestamp for tweet in tweets], columns=['Date'])

        df2['Tweet'] = np.array([tweet.text for tweet in tweets])
        df2['Replied_Tweet'] = np.array([tweet.is_replied for tweet in tweets])
        df2['Likes'] = np.array([tweet.likes for tweet in tweets])
        df2['Reply_Count'] = np.array([tweet.replies for tweet in tweets])
        df2['Retweets'] = np.array([tweet.retweets for tweet in tweets])

        return df2

#%%
if __name__ == '__main__':
    tweet_analyzer = TweetAnalyzer()

    df2 = tweet_analyzer.tweets_to_data_frame(tweets)

    df2['Sentiment Score'] = np.array(
        [tweet_analyzer.analyze_sentiment_score(tweet) for tweet in df['Tweet']])
    df2['Sentiment Result'] = np.array(
        [tweet_analyzer.analyze_sentiment_result(tweet) for tweet in df['Tweet']])


mainData = df2.copy()
mainData.head()


# %%
# mainData['Tweet_Date'] = [d.date() for d in df['Date']]
# mainData['Tweet_Time'] = [d.time() for d in df['Date']]


# %%
# Truly determining what day the tweet will affect
# Later than 4pm est then the tweet will affect the next day
# Tweets during the day will affect the current day
def checkDates(d):
    if d.time().hour >= 16:
        return d + pd.Timedelta(days=1)
    else:
        return d


mainData['Tweet_Date'] = mainData['Date'].apply(
    lambda d: checkDates(pd.to_datetime(d))).dt.date

mainData = mainData.set_index('Tweet_Date')



# mainData = mainData.groupby('timestamp').agg(
#     {'likes': 'sum', 'retweets': 'sum', 'replies':'sum','Sentiment Score': 'mean'})

# mainData.reset_index().head()

# %%
#normalizing data
scaler = MinMaxScaler()

mainData['SIZE_likes'] = scaler.fit_transform(
    mainData['Likes'].values.reshape(-1, 1))
mainData['SIZE_retweets'] = scaler.fit_transform(
    mainData['Retweets'].values.reshape(-1, 1))
mainData['SIZE_replies'] = scaler.fit_transform(
    mainData['Reply_Count'].values.reshape(-1, 1))

mainData.info()



# %%
# Get the data of the stock Tesla Stock (TSLA)
stockData = yf.download("TSLA", start="2019-4-1", end="2020-02-7")
stockData.reset_index().head()

stockData['SIZE_END_DAY_STOCK'] = scaler.fit_transform(
    stockData['Close'].values.reshape(-1, 1))

stockData.info()

#%%

#Joining Dataframes
AllData = mainData.join(stockData, lsuffix='Tweet_Date',
                        rsuffix='Date')
                        

AllData = AllData.drop(columns = ['Date','Tweet'])

AllData = AllData.reset_index()

AllData.head()



# %%
fig = go.Figure()


fig.add_trace(go.Scatter(
    x=AllData.index,
    y=AllData['Sentiment Score'],
    name="Sentiment Score",
    line_color='red',
    opacity=0.8))

fig.add_trace(go.Scatter(
    x=AllData.index,
    y=AllData['SIZE_END_DAY_STOCK'],
    name="Stock Price",
    line_color='green',
    opacity=0.8))


# fig.add_trace(go.Scatter(
#     x=AllData.index,
#     y=AllData['SIZE_likes'],
#     name="Likes",
#     line_color='blue',
#     opacity=0.8))

# fig.add_trace(go.Scatter(
#     x=AllData.index,
#     y=AllData['SIZE_retweets'],
#     name="Retweets",
#     line_color='grey',
#     opacity=0.8))

# Use date string to set xaxis range
fig.update_layout(xaxis_range=['400', '550'],
                  title_text="Elon's Stock Price, Likes, Retweets")
fig.show()


# %%
