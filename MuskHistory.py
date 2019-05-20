from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from pymongo import MongoClient
import re
import json

import config


# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    ### Don't need, but could be fun later ###

    # def get_friend_list(self, num_friends):
    #     friend_list = []
    #     for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
    #         friend_list.append(friend)
    #     return friend_list

    # def get_home_timeline_tweets(self, num_tweets):
    #     home_timeline_tweets = []
    #     for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
    #         home_timeline_tweets.append(tweet)
    #     return home_timeline_tweets


# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(config.consumer_key,
                            config.consumer_secret)
        auth.set_access_token(config.access_token_key,
                              config.access_token_secret)
        return auth

# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """

    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, musk_id):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        stream.filter(follow = musk_id)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment_score(self, tweet):
        analysis_score = TextBlob(self.clean_tweet(tweet))
        analysis_score = analysis_score.sentiment.polarity
        return analysis_score

    def analyze_sentiment_result(self, tweet):

        analysis_result = TextBlob(self.clean_tweet(tweet))

        if analysis_result.sentiment.polarity > 0:
            return 'Positive'
        elif analysis_result.sentiment.polarity == 0:
            return '0'
        else:
            return 'Negative'

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(
            data=[tweet.created_at for tweet in tweets], columns=['Date'])

        df['Tweet'] = np.array([tweet.text for tweet in tweets])
        df['Tweet_id'] = np.array([tweet.id for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df


if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()

    # Elon Musk Twitter ID = 44196397
    musk_id = '44196397' 

    tweets = api.user_timeline(id=musk_id, count=1000)
    
    fetched_tweets_filename = "tweets.json"

    df = tweet_analyzer.tweets_to_data_frame(tweets)

    df['Sentiment Score'] = np.array([tweet_analyzer.analyze_sentiment_score(tweet) for tweet in df['Tweet']])
    df['Sentiment Result'] = np.array([tweet_analyzer.analyze_sentiment_result(tweet) for tweet in df['Tweet']])

    df.to_json('Muskhistory.json')

    client = MongoClient(config.mongoaccess)
    db = client.twitterhistory
    history = db['data']

    with open('Muskhistory.json') as Muskhistory:
        file_data = json.load(Muskhistory)

    # history.insert_one(file_data)
    # history.update_one({
    #     {"id": "5cd37f3370921d31bc017823"},
    #     {$set:{
    #         "Date":...,
    #         "Tweet":...,
    #         "Tweet_id":...,
    #         "likes":...,
    #         "retweets":...,
    #         "Sentiment Score": ...,
    #         "Sentiment Result": ...,
    #     }}
    # })

    client.close()

    # time_likes = pd.Series(data=df['likes'].values, index=df['Date'])
    # time_likes.plot(figsize=(16, 4), label="likes", legend=True)

    # time_retweets = pd.Series(data=df['retweets'].values, index=df['Date'])
    # time_retweets.plot(figsize=(16, 4), label="retweets", legend=True)
    # plt.show()

    # print()

    #twitter_streamer = TwitterStreamer()
    #twitter_streamer.stream_tweets(fetched_tweets_filename, musk_id)


