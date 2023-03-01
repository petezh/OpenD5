"""
Constructs the twitter_rumors distributions.
"""

import pandas as pd
import tweepy

from utils import encode_ascii

CONSUMER_KEY = None
CONSUMER_SECRET = None
OAUTH_TOKEN = None
OAUTH_TOKEN_SECRET = None

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)


def get_tweet_info(row):
    """
    Calls Twitter API to get tweet text.
    """

    id_of_tweet = int(row['Tweet ID'])
    try:
        tweet = api.get_status(id_of_tweet)
        row['text'] = tweet.text
        row['dt'] = tweet.created_at
    except:
        row['text'] = None
        row['dt'] = None
    return row


STAGES = ['early', 'mid', 'late']


def scrape():
    """
    Downloads Tweet IDs from Zenodo and scrapes Tweets from Twitter API.
    """

    URLS = {
        'redhawks': 'https://zenodo.org/record/2563864/files/DATASET_R1.xlsx',
        'muslim_waitress': 'https://zenodo.org/record/2563864/files/DATASET_R2.xlsx',
        'zuckerberg_yatch': 'https://zenodo.org/record/2563864/files/DATASET_R3.xlsx',
        'denzel_washington': 'https://zenodo.org/record/2563864/files/DATASET_R4.xlsx',
        'veggietales': 'https://zenodo.org/record/2563864/files/DATASET_R7.xlsx',
        'michael_jordan': 'https://zenodo.org/record/2563864/files/DATASET_R8.xlsx',
    }

    data = {}

    for rumor, url in URLS.items():

        print(rumor)

        df = pd.read_excel(url)
        df = df.sample(300)
        df = df.apply(get_tweet_info, axis=1)
        df = df.dropna(axis=0)
        df['stage'] = pd.qcut(df['dt'], 3, labels=STAGES)

        for stage in STAGES:
            data[f'{rumor}_{stage}'] = df[df.stage ==
                                          stage]['text'].apply(encode_ascii).tolist()

    return data
