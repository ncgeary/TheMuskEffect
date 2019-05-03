import urllib
import urllib.request
from bs4 import BeautifulSoup

Muskurl = "https://twitter.com/elonmusk"

pagedig = urllib.request.urlopen(Muskurl)
soup = BeautifulSoup(pagedig,"html.parser")

for stamps in soup.findAll('a', ["data-time-ms"]):
    # stamp = stamps.text
    print(stamps)

#WORKS BUT NEED TO MATCH THE TWEETS TO THE DATE/TIME

# for tweets in soup.findAll('p', {'class': 'TweetTextSize'}):
#     tweet = tweets.text
#     print(tweet)



