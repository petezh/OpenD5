"""
Constructs the blm_countermovements distributions.
"""

import pandas as pd
import tweepy
from parameters import *
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

    id_of_tweet = int(row["ID"])
    try:
        tweet = api.get_status(id_of_tweet)
        row["text"] = tweet.text
    except:
        row["text"] = None
    return row


def scrape():
    """
    Scapes BLM countermovements Tweets.
    """

    paths = {
        "all_lives_matter": f"{MANUAL_FOLDER}/blm_countermovements/AllLivesMatter_IDs.csv",
        "blue_lives_matter": f"{MANUAL_FOLDER}/blm_countermovements/BlueLivesMatter_IDs.csv",
        "white_lives_matter": f"{MANUAL_FOLDER}/blm_countermovements/WhiteLivesMatter_IDs.csv",
    }

    df = pd.DataFrame()
    for movement, filepath in paths.items():
        with open(filepath, "r") as f:
            IDs = f.readlines()[1:]
            movement_df = pd.DataFrame({"movement": movement, "ID": IDs}).sample(
                n=1000, replace=False, random_state=0
            )
            df = df.append(movement_df)

    df = df.apply(get_tweet_info, axis=1)
    df = df.dropna(axis=0)

    data = {}
    for movement in paths:
        data[movement] = df[df.movement == movement].text.apply(encode_ascii).tolist()

    return data
