from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import config

class TwitterStreamer():
    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, musk_id):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = StdOutListener(fetched_tweets_filename)
        auth = OAuthHandler(config.consumer_key,
                            config.consumer_secret)
        auth.set_access_token(config.access_token_key,
                              config.access_token_secret)
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        # stream.filter(track=hash_tag_list)
        stream.filter(follow=musk_id)


# # # # TWITTER STREAM LISTENER # # # #
class StdOutListener(StreamListener):
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
        print(status)


if __name__ == '__main__':

    # Authenticate using config.py and connect to Twitter Streaming API.
    # hash_tag_list = ["Elon Musk", "Tesla", "SpaceX"]

    musk_id = ['44196397', '1124133518676774914']
    fetched_tweets_filename = "tweets.json"

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, musk_id)
