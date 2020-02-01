import nltk
import warnings
warnings.filterwarnings('ignore')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import pprint
import pandas as pd

date_sentiments = {}

SP500=pd.read_csv('/Users/zhoujiawang/Desktop/constituents_csv.csv')

for i in SP500['Name']:
    page = urlopen('https://www.businesstimes.com.sg/search/'+i+'?page=1').read()
    soup = BeautifulSoup(page, features="html.parser")
    posts = soup.findAll("div", {"class": "media-body"})
    try:
        post=posts[0]
        url = post.a['href']
        date = post.time.text
        print(date, url)
        try:
            link_page = urlopen(url).read()
        except:
            url = url[:-2]
            link_page = urlopen(url).read()
        link_soup = BeautifulSoup(link_page)
        sentences = link_soup.findAll("p")
        passage = ""
        for sentence in sentences:
            passage += sentence.text
        sentiment = sia.polarity_scores(passage)['compound']
        date_sentiments.setdefault(i, []).append(sentiment)
    except Exception:pass

print(date_sentiments)
sorted_sentiment = sorted(date_sentiments.items(), key=lambda kv: kv[1])
print(sorted_sentiment[-15:])
