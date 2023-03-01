"""
Downloads and/or scrapes distributions.
"""

import argparse
import codecs
from collections import defaultdict, Counter
from datetime import timedelta, datetime
from functools import partial
import itertools
import json
import os
from os.path import join
import re
import yaml
from yaml.loader import SafeLoader

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from parameters import *
from utils import *

"""
**********
Processors
**********
"""

def process_abc_headlines():
    """
    ABC headlines are directly downloaded from Harvard Dataverse. The year
    is extracted from the publication date field. Samples are constructed from the
    headline text.
    """

    NAME = 'abc_headlines'
    URL = 'https://dataverse.harvard.edu/api/access/datafile/4460084'
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)
    
    df = pd.read_csv(join(directory, filename), sep='\t')
    df['year'] = df['publish_date'].astype(str).str[:4].astype(int)
    
    data = {}
    for year in df['year'].unique():
        data[str(year)] = df[df['year']==year]['headline_text'].tolist()
    
    save_output_json(data, NAME)

def process_ad_transcripts():
    """
    Ad transcripts are directly downloaded from Kaggle. The top eight industries by
    frequency are selected. Newlines are replaced with spaces.
    """
    
    NAME = 'ad_transcripts'

    df = pd.read_excel(join(MANUAL_FOLDER, NAME, 'Advertisement_Transcripts_deduped_edited.xlsx'))
    top_n = 8
    industries = df.Category.value_counts().index[:top_n].tolist()

    def clean(text):
        return text.replace('\n', ' ')

    data = {}
    for industry in industries:
        data[industry] = df[df.Category == industry].Ad_copy.apply(clean).to_list()
    
    save_output_json(data, NAME)

def process_admin_statements():
    """
    Administration statements are downloaded directly as PDFs from the official
    GitHub repository and preprocessed using pdfplumber. Extraneous
    symbols are removed and samples are split by paragraph.
    """

    NAME = 'admin_statements'
    URL = 'https://github.com/unitedstates/statements-of-administration-policy/archive/master.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)
    
    import scrape_admin_statements

    data = scrape_admin_statements.scrape()
    
    save_output_json(data, NAME)

def process_airline_reviews():
    """
    Airline reviews for airlines, airports, and seats are downloaded from a
    public Github repository. Names of aircrafts, airlines, countries, and traveller
    types are standardized. Ratings of 1, 4, or 5 on a scale of 5, and
    1, 5, 8, or 10 on a scale of 10 are kept.
    """

    NAME = 'airline_reviews'
    URLS = {
        'airlines':'https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv',
        'airports':'https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airport.csv',
        'seats':'https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/seat.csv'
    }

    df = pd.read_csv(URLS['seats'])

    B747 = ('BOEING 747-400', 'B747-400', 'Boeing 747-400')
    B777 = ('BOEING 777','BOEING 777-300','BOEING 777-300ER','B777-300','B777','Boeing 777-300','B777-300ER','Boeing 777-300ER')
    A380 = ('AIRBUS A380', 'A380')
    A340 = ('A340','AIRBUS A340','A340-600','AIRBUS A340-300','A340-300')
    A330 = ('A330','A330-300','AIRBUS A330','AIRBUS A330-300')

    data = {
        'seats_b747':df[df.aircraft.isin(B747)].content.tolist(),
        'seats_b777':df[df.aircraft.isin(B777)].content.tolist(),
        'seats_a380':df[df.aircraft.isin(A380)].content.tolist(),
        'seats_a340':df[df.aircraft.isin(A340)].content.tolist(),
        'seats_a330':df[df.aircraft.isin(A330)].content.tolist(),
        'seats_2x4x2':df[df.seat_layout=='2x4x2'].content.tolist(),
        'seats_3x4x3':df[df.seat_layout=='3x4x3'].content.tolist(),
        'seats_3x3x3':df[df.seat_layout=='3x3x3'].content.tolist(),
        'seats_3x3':df[df.seat_layout=='3x3'].content.tolist(),
        'seats_econ':df[df.cabin_flown=='Economy'].content.tolist(),
        'seats_prem':df[df.cabin_flown=='Premium Economy'].content.tolist(),
    }

    df = pd.read_csv(URLS['airlines'])
    airline_map = {
        'airline_spirit':'spirit-airlines',
        'airline_frontier':'frontier-airlines',
        'airline_british':'british-airways',
        'airline_ryan':'ryanair',
        'airline_jet':'jet-airways',
        'airline_emirates':'emirates',
        'airline_canada':'air-canada',
        'airline_canada_rogue':'air-canada-rouge',
        'airline_united':'united-airlines',
        'airline_american':'american-airlines',
        'airline_delta':'delta-air-lines',
    }
    for pair_name, airline in airline_map.items():
        data[pair_name] = df[df.airline_name==airline].content.tolist()
    df['author_country'].value_counts()[:20]
    country_map = {
        'author_uk':'United Kingdom',
        'author_us':'United States',
        'author_aus':'Australia',
        'author_cad':'Canada',
        'author_ger':'Germany',
        'author_fr':'France',
        'author_sg':'Singapore',
        'author_in':'India',
    }
    for pair_name, country in country_map.items():
        data[pair_name] = df[df.author_country==country].content.tolist()

    ratings_map = [
        ('overall', 'overall_rating',[1, 5, 8, 10]),
        ('comfort','seat_comfort_rating',[1, 5]),
        ('staff','cabin_staff_rating',[1, 5]),
        ('food','food_beverages_rating',[1, 5]),
        ('entertainment','inflight_entertainment_rating',[1, 5]),
        ('service','ground_service_rating',[1, 5]),
        ('value','value_money_rating',[1, 5]),
    ]
    for rating_type, col, vals in ratings_map:
        for val in vals:
            data[f'airline_{rating_type}_rating_{val}'] = df[df[col]==val].content.tolist()

    df = pd.read_csv(URLS['airports'])

    traveller_map = {
        'solo':'Solo Leisure',
        'couple':'Couple Leisure',
        'family':'FamilyLeisure',
        'business':'Business',
    }
    for pair_name, traveller in traveller_map.items():
        data[f'traveller_{pair_name}'] = df[df.type_traveller==traveller].content.tolist()

    ratings_map = [
        ('overall', 'overall_rating',[1, 5, 8, 10]),
        ('queue','queuing_rating',[1, 4, 5]),
        ('cleanliness','terminal_cleanliness_rating',[1, 4, 5]),
        ('shopping','airport_shopping_rating',[1, 4, 5]),
    ]
    
    for rating_type, col, vals in ratings_map:
        for val in vals:
            data[f'airport_{rating_type}_rating_{val}'] = df[df[col]==val].content.tolist()

    save_output_json(data, NAME)

def process_aita():
    """
    Posts from r/AmITheAsshole are downloaded from a praw scrape of Reddit.
    Topic areas are chosen based on common themes in posts and coarsely
    defined based on manual keywords. Each post can belong to multiple
    topic areas.
    """

    NAME = 'aita'
    FILE = f'{MANUAL_FOLDER}/{NAME}/aita_clean.csv'

    df = pd.read_csv(FILE)
    df = df[df.score > 10]
    df['text'] = df['title'] + '\n' + df['body']
    df.text = df.text.fillna(df.title).apply(unmark) # remove markdown formatting

    df = split_df(df, 'text')

    data = {}
    verdicts = {
        'a':'asshole',
        'nta':'not the asshole',
        'es':'everyone sucks',
        'nah':'no assholes here',
    }
    for v, verdict in verdicts.items():
        data[f'verdict_{v}'] = df[df.verdict==verdict].text.tolist()

    topic2keywords = {
        'work':['boss','coworker','customer'],
        'sex':['sex','blowjob','intercourse','orgasm','hooked up'],
        'ex':[' ex '],
        'husband':['my husband'],
        'wife':['my wife'],
        'race':['racism','racist','bigot'],
        'gender':['feminism','feminist','sexist','sexism'],
        'children':['baby','child','son','daughter'],
        'social_media':['instagram','facebook','snapchat','fb','social media'],
        'sexuality':['gay','lesbian','lgbt','queer','homosexual','fag'],
        'alcohol':['drunk','drinking','sober','drunken'],
        }
        
    for topic, kws in topic2keywords.items():
        index = df.text.str[0].str.lower().str.contains(kws[0])
        for kw in kws[1:]:
            index = index|df.text.str[0].str.lower().str.contains(kw)
        topic_df = df[index]
        data[f'topic_{topic}_is_asshole'] = topic_df[topic_df.is_asshole==1].text.tolist()
        data[f'topic_{topic}_not_asshole'] = topic_df[topic_df.is_asshole==0].text.tolist()

    save_output_json(data, NAME)

def process_all_the_news():
    """
    News articles are downloaded directly from Components website. The 
    titles are used as text samples.
    """

    NAME = 'all_the_news'
    FILES = ['articles1.csv', 'articles2.csv', 'articles3.csv']

    df = pd.DataFrame()
    
    for file in FILES:
        filename = join(MANUAL_FOLDER, NAME, file)
        df = df.append(pd.read_csv(filename))

    col_types = {
        'title':str,
        'content':str,
        'publication':'category',
        'author':'category',
        'date':'datetime64',
    }
    df = df[col_types.keys()].astype(col_types)
    
    snippets = df['title'].tolist()
    snippets = [snippet.split(' - ')[0] for snippet in snippets]
    sentences = sentence_tokenize(snippets)

    save_dataset(df, NAME)
    save_unlabeled_json(sentences, NAME)

def process_amazon_reviews():
    """
    Amazon reviews are downloaded from a 2018 crawl of the website. The
    first 100,000 review texts are treated as the text sample.
    """

    NAME = 'amazon_reviews'
    URLS = {
        'amazon_fashion':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz',
        'beauty':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz',
        'appliances':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Appliances.json.gz',
        'arts_crafts':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz',
        'automotive':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive_5.json.gz',
        'cds':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/CDs_and_Vinyl_5.json.gz',
        'cell_phones':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Cell_Phones_and_Accessories_5.json.gz',
        'digital_music':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Digital_Music.json.gz',
        'gift_cards':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Gift_Cards.json.gz',
        'grocery':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz',
        'industrial_scientific':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Industrial_and_Scientific_5.json.gz',
        'luxury_beauty':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz',
        'magazines':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Magazine_Subscriptions.json.gz',
        'music_instruments':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Musical_Instruments.json.gz',
        'office':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Office_Products_5.json.gz',
        'patio':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Patio_Lawn_and_Garden_5.json.gz',
        'pantry':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Prime_Pantry.json.gz',
        'software':'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Software.json.gz',
        'video_games':'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz',
    }

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'

    df = pd.DataFrame()
    for product, url in tqdm(URLS.items()):
        print(product)
        filename = product + '.json'
        # download_gz(url, directory, filename)
        product_df = pd.read_json(join(directory, filename), lines=True, nrows=100000)
        product_df['product_category'] = product
        df = df.append(product_df)
    
    df.vote = df.vote.astype(str).str.replace(',', '').astype(float).fillna(1)
    df['year'] = df['reviewTime'].str.split(', ').str[1]

    rename_map = {
        'reviewText':'text',
        'summary':'summary',
        'Abstract':'abstract',
        'overall':'stars',
        'vote':'votes',
    }
    df = df.rename(rename_map, axis=1)

    col_types = {
        'text':str,
        'summary':str,
        'year':int,
        'stars':int,
        'votes':int,
        'product_category':'category',
    }
    
    df = df[col_types.keys()].astype(col_types)

    save_dataset(df, NAME)

def process_armenian_jobs():
    """
    Armenian job postings dataset is downloaded from a snapshot on GitHub.
    Different IT jobs are manually coded and time intervals are defined in
    order to balance sample availlability.
    """


    NAME = 'armenian_jobs'
    URL = 'https://raw.githubusercontent.com/GurpreetKaur28/Analysing-Online-Job-Postings/master/data%20job%20posts.csv'
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.tsv'
    # download_file(URL, directory, filename)

    df = pd.read_csv(join(directory, filename))

    data = {}

    jobs = {
        'sw_dev':'Software Developer',
        'senior_sw_dev':'Senior Software Developer',
        'qa_eng':'QA Engineer',
        'senior_qa_eng':'Senior QA Engineer',
        'sw_eng':'Software Engineer',
        'senior_sw_eng':'Senior Software Engineer',
        'java_dev':'Java Developer',
        'senior_java_dev':'Senior Java Developer',
        'prgmr':'programmer',
    }

    df['job_desc'] = df['JobDescription'].str.replace('\n',' ')
    df['job_req'] = df['JobRequirment'].str.replace('\n',' ')
    df['app_proc'] = df['ApplicationP'].str.replace('\n','')

    desc_df = df[df['job_desc'].str.split().str.len() > 0]
    desc_df = split_df(desc_df, 'job_desc')
    req_df = df[df['job_req'].str.split().str.len() > 0]
    req_df = split_df(req_df, 'job_req')
    app_df = df[df['app_proc'].str.split().str.len() > 0]
    app_df = split_df(req_df, 'app_proc')

    for name, title in jobs.items():
        descriptions = list(set(desc_df[desc_df.Title == title]['job_desc']))
        requirements = list(set(req_df[req_df.Title == title]['job_req']))
        data[f'job_desc_{name}'] = descriptions
        data[f'job_req_{name}'] = requirements

    year_bins = [(2004, 2007), (2007, 2010), (2010, 2013), (2013, 2015)]
    for start_year, end_year in year_bins:
        requirements = list(set(req_df[(start_year <= req_df.Year) & (req_df.Year < end_year)]['job_req'].dropna()))
        app_process = list(set(app_df[(start_year <= app_df.Year) & (app_df.Year < end_year)]['app_proc'].dropna()))
        data[f'job_req_years_{start_year}_{end_year}'] = requirements
        data[f'app_process_years_{start_year}_{end_year}'] = app_process

    save_output_json(data, NAME)

def process_blm_countermovements():
    """
    Tweet IDs are downloaded from the original paper and, where available, collected
    from the current API. Due to API rate limits, only 1,000 Tweets are sampled from 
    each movement.
    """

    NAME = 'blm_countermovements'

    import scrape_blm_countermovements

    data = scrape_blm_countermovements.scrape()
    
    save_output_json(data, NAME)

def process_blogs():
    """
    Blogs are downloaded directly from Kaggle and the first
    1 million blog posts are kept.
    """

    NAME = 'blogs'
    FILE = 'blogtext.csv'

    ROWS = 1000000

    filename = join(MANUAL_FOLDER, NAME, FILE)

    df = pd.read_csv(filename,nrows=ROWS)

    # fix date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[~df['date'].isna()]

    # set col types
    col_types = {
        'gender':'category',
        'topic':'category',
        'sign':'category',
        'date':'datetime64[ns]',
        'text':str
    }
    df = df[col_types.keys()].astype(col_types)

    snippets = df['text'].tolist()[:100000]
    sentences = sentence_tokenize(snippets)

    save_dataset(df, NAME)
    save_unlabeled_json(sentences, NAME)

def process_cah():
    """
    Uses a private dataset of CAH rounds. Establishes
    Bayesian estimates of how funny each card and/or
    joke, which are used to group the cards.
    """

    NAME = 'cah'

    df = pd.read_csv(join(MANUAL_FOLDER, NAME, 'cah_lab_data_for_research.csv'))

    MIN_SECONDS = 5
    df = df[df['round_completion_seconds'] >= MIN_SECONDS]
    df = df.sample(frac=1, random_state=0)
    data = {}

    # for white cards
    base_rate = df['won'].mean()
    PRIOR_STRENGTH = 20
    ALPHA, BETA = PRIOR_STRENGTH * base_rate, PRIOR_STRENGTH * (1-base_rate)
    whitecard2freq = Counter(df['white_card_text'])
    whitecard2wins = Counter(df[df.won]['white_card_text'])
    whitecard2winrate = {}
    for card, freq in whitecard2freq.items():
        wins = whitecard2wins[card]
        whitecard2winrate[card] = (ALPHA+wins) / (PRIOR_STRENGTH+freq)

    sorted_whitecards = sorted(df['white_card_text'].unique(), key=whitecard2winrate.get, reverse=True)
    data['cards_funny'] = sorted_whitecards[:500]
    data['cards_not_funny'] = sorted_whitecards[-500:]

    # for pick 1 jokes
    df_pick1 = df[df['black_card_pick_num'] == 1]
    base_rate = df_pick1['won'].mean()
    PRIOR_STRENGTH = 3
    ALPHA, BETA = PRIOR_STRENGTH * base_rate, PRIOR_STRENGTH * (1-base_rate)
    df_pick1['joke'] = 'Black card: ' + df_pick1['black_card_text'] + '\nWhite card: ' + df_pick1['white_card_text']
    df_pick1 = df_pick1.groupby('joke')['white_card_text', 'won', 'round_skipped'].agg({'white_card_text':'count','won':'sum', 'round_skipped':'sum'}).reset_index()
    df_pick1.rename({'white_card_text':'freq'}, axis=1, inplace=True)
    df_pick1['winrate'] = df_pick1['won']/df_pick1['freq']
    df_pick1['winrate_bayesian'] = (df_pick1['won'] + ALPHA)/(df_pick1['freq'] + PRIOR_STRENGTH)


    sorted_jokes = df_pick1.sort_values('winrate_bayesian', ascending=False)['joke'].tolist()
    data['jokes_very_funny'] = sorted_jokes[:300]
    data['jokes_funny'] = sorted_jokes[300:1000]
    data['jokes_somewhat_funny'] = sorted_jokes[1000:5000]
    data['jokes_not_funny'] = sorted_jokes[-10000:]

    save_output_json(data, NAME)

def process_clickbait_headlines():
    """
    The Examiner headlines are directly downloaded from Kaggle. The
    year is extracted from the publication date field. Samples are
    constructed from the headline text.
    """

    NAME = 'clickbait_headlines'
    URL = 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/BFAZHR/WYSGGQ'
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)
    
    df = pd.read_csv(join(directory, filename))
    df['year'] = df['publish_date'].astype(str).str[:4].astype(int)
    
    data = {}
    for year in df['year'].unique():
        data[str(year)] = df[df['year']==year]['headline_text'].tolist()
    
    save_output_json(data, NAME)

def process_convincing_arguments():
    """
    Annotated arguments are downloaded from the Github repostiory. Arguments
    are sorted by rank. The bottom 400 are treated as "unconvincing", the
    top 200 are treated as "convincing", and the next 200 are treated as
    "somewhat convincing."
    """

    NAME = 'convincing_arguments'
    URL = 'https://github.com/UKPLab/acl2016-convincing-arguments/archive/master.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)

    data_path = f'{directory}/acl2016-convincing-arguments-master/data/UKPConvArg1-Ranking-CSV/*'
    files = glob.glob(data_path)
    
    df = pd.DataFrame()
    for file in files:
        topic = re.findall('/([\w-]+).csv', file)[0]
        topic_df = pd.read_csv(files[0], sep='\t')
        topic_df['topic'] = topic
        df = df.append(topic_df)

    def clean(text):
        return strip_tags(text).replace('\n', '')
    
    df['argument'] = df['argument'].apply(clean)

    sorted_df = df.sort_values('rank')
    unconvincing, convincing = sorted_df.iloc[-400:], sorted_df.iloc[:200]
    somewhat_convincing =  sorted_df.iloc[200:400]

    data = {}
    data['unconvincing'] = unconvincing['argument'].tolist()
    data['somewhat_convincing'] = somewhat_convincing['argument'].tolist()
    data['convincing'] = convincing['argument'].tolist()

    save_output_json(data, NAME)

def process_craigslist_negotiations():
    """
    Craigslist negotiations are downloaded from Huggingface. Sequences
    which contained a "quit" intention or "reject" intention are categorized
    as failures; those which contained an "accept" intention are categorized
    as successes. The mid-price is defined as the mean price of the items
    sold. Within each category, the items are sorted by mid-price. The top
    half is treated as high-price and the bottom half is treated as low-price.
    """

    NAME = 'craigslist_negotiations'

    from datasets import load_dataset
    df = load_dataset('craigslist_bargains', split='train').to_pandas()
    df['failure'] = df['dialogue_acts'].apply(lambda x: x['intent'].__contains__('quit') or x['intent'].__contains__('reject'))
    df['success'] = df['dialogue_acts'].apply(lambda x: x['intent'].__contains__('accept'))
    df['all_text'] = df['utterance'].str.join('\n')

    df['mid_price'] = df['items'].apply(lambda x: x['Price'].mean())
    df['category'] = df['items'].apply(lambda x: x['Category'][0])


    def split_high_low(category):
        cat_df = df[df.category==category]
        cat_df['half'] = pd.qcut(cat_df.mid_price, 2, labels=(0, 1))
        bottom_text = cat_df[cat_df.half == 0].all_text
        top_text = cat_df[cat_df.half == 1].all_text
        return bottom_text, top_text

    car_low, car_high = split_high_low('car')
    bike_low, bike_high = split_high_low('bike')
    housing_low, housing_high = split_high_low('housing')

    data = {
        'failure':df[df.failure]['all_text'].tolist(),
        'success':df[df.success]['all_text'].tolist(),
        'car_low':car_low.tolist(),
        'car_high':car_high.tolist(),
        'bike_low':bike_low.tolist(),
        'bike_high':bike_high.tolist(),
        'housing_low':housing_low.tolist(),
        'housing_high':housing_high.tolist(),
    }
    
    save_output_json(data, NAME)

def process_debate():
    """
    The train split is downloaded from Hugginface. For each sample, we
    use the abstract as the text. Arguments are categorized by type,
    debate camp of origin, and topic/specific argument. For topics,
    we use domain knowlege to list relevant keywords for each topic
    and include any sample with a file name that includes any keyword. A
    single sample can belong to multiple topics.
    """

    NAME = 'debate'
    HUGGINGFACE_NAME = "Hellisotherpeople/DebateSum"

    from datasets import load_dataset
    dataset = load_dataset(HUGGINGFACE_NAME)

    df = dataset['train'].to_pandas()
    rename_map = {
        'Full-Document':'body',
        'Extract':'summary',
        'Abstract':'abstract',
        'Citation':'citation',
        'OriginalDebateFileName':'file',
        'Tag':'arg_type',
        'DebateCamp':'debate_camp',
        'Year':'year',
    }
    df = df.rename(rename_map, axis=1)
    df['year'] = df['year'].replace('Unknown',np.nan).astype(float)

    argtype2cat = {
        'Kritiks': 'k',
        'Affirmatives': 'aff',
        'Case Negatives': 'case_neg',
        'Counterplans': 'cp',
        'Disadvantages': 'da',
        'Kritik Answers': 'a2_k',
        'Topicality': 't',
        'Theory': 'th',
        'Lincoln Douglas': 'ld',
        'Politics': 'politics',
        'Counterplan Answers': 'a2_cp',
        'Impact Files': 'imp',
        'Disadvantage Answers': 'a2_da',
        'Framework': 'fw',
        'misc':np.nan,
    }
    df['arg_type'] = df['arg_type'].map(argtype2cat)

    camp2cat = {
        'Gonzaga (GDI)':'gdi',
        'Dartmouth DDI':'ddi',
        'Northwestern (NHSI)':'nhsi',
        'Berkeley (CNDI)':'cdni',
        'Wyoming':'wyoming',
        'Georgetown (GDS)':'gds',
        'Texas (UTNIF)':'utnif',
        'Missouri State (MSDI)':'msdi',
        'Unknown':np.nan,
        'Kansas (JDI)':'jdi',
        'Michigan (7-week)':'mich_7week',
        'Sun Country (SCDI)':'scdi',
        'North Texas (UNT)':'unt',
        'Samford':'samford',
        'Emory (ENDI)':'endi',
        'Hoya-Spartan Scholars':'hss',
        'Michigan State (SDI)':'sdi',
        'Michigan (Classic)':'mich_classic',
        'Michigan (MNDI)':'mndi',
        'Wake Forest (RKS)':'rks',
        'Dartmouth DDIx':'ddi',
        'Georgia':'georgia',
        'Harvard':'harvard',
        'Weber State (WSDI)':'wsdi',
        'UT Dallas (UTD)':'utd',
        'NAUDL':'naudl',
        'Baylor':'bawlor',
        'Mean Green Comet':'mgc',
        'The Debate Intensive':'tdi',
        'National Symposium for Debate':'nsd',
    }
    df['debate_camp'] = df['debate_camp'].map(camp2cat) 

    argument2kws = {
        # kritiks
        'ableism':('ableism','disability','crip',),
        'anthro':('anthro',),
        'afropess':('afropessmism','afro pessmism','afro-pessimism','black nihilism','ontological terror','warren',),
        'antiblackness':('blackness',),
        'baudrillard':('baudrillard','baudy',),
        'cap':('capitalism','cap k',),
        'fem':('feminism','gender',),
        'foucault':('foucault',),
        'heidegger':('heidegger',),
        'militarism':('militarism',),
        'neolib':('neolib',),
        'psycho':('psychoanalysis','lacan',),
        'queerness':('queer pessimism','queer nihilism','queerpess','queer theory', 'queer k'),
        'security':('security k',),
        'settcol':('coloniality','settler colonialism','decolonization','settlerism',),
        # politics
        'midterms':('midterms',),
        'elections':('elections',),
        'politics':('politics da',),
        # counterplans
        'consult':('consult',),
        'states':('states cp','states counterplan','cp - states',),
        'advantage_cp':('advantage counterplan','cp - advantage','advantage cp',),
        'courts':('courts cp','courts counterplan','cp - courts',),
    }
    kw2argument = {}
    for argument, kws in argument2kws.items():
        kw2argument.update({kw:argument for kw in kws})

    def get_argument(filename:str):
        args = {kw2argument[kw] for kw in kw2argument if kw in filename.lower()}
        if len(args) != 1: return np.nan
        return args.pop()

    filename2argument = {
        filename:get_argument(filename) for filename in df['file'].unique()
    }
    df['argument'] = df['file'].map(filename2argument)

    col_types = {
        'body':str,
        'summary':str,
        'abstract':str,
        'citation':str,
        'year':'Int64',
        'arg_type':'category',
        'debate_camp':'category',
        'argument':'category',
    }
    df = df[col_types.keys()].astype(col_types)

    snippets = df['body'].sample(n=100000, random_state=0).tolist()
    sentences = sentence_tokenize(snippets)

    save_dataset(df, NAME)

    save_unlabeled_json(sentences, NAME)

def process_dice_jobs():
    """
    Job postings are downloaded from Kaggle. Posts from the six most popular
    companies are categorized by company. We remove miscellaneous characters
    and blank descriptions. We additionally apply our splitting procedure
    to reduce description length.
    """

    NAME = 'dice_jobs'

    df = pd.read_csv(join(MANUAL_FOLDER, NAME, 'Dice_US_jobs.csv'), encoding='latin-1')
    orgs = {
        'northup_grumman':'NORTHROP GRUMMAN',
        'leidos':'Leidos',
        'dell':'Dell',
        'deloitte':'Deloitte',
        'amazon':'Amazon',
        'jpm':'JPMorgan Chase'
    }

    def clean(text):
        return text.replace('\u00e5\u00ca', '')
    
    df['job_description'] = df['job_description'].apply(clean)
    df = df.drop_duplicates()

    df = df[df['job_description'].str.split().str.len() > 0]
    df = split_df(df, 'job_description', splitter=split_delimiter_)
    df = split_df(df, 'job_description')

    data = {}
    for name, org in orgs.items():
        descriptions = df[df.organization == org].job_description.dropna().tolist()
        data[name] = list(descriptions)

    save_output_json(data, NAME)

def process_diplomacy_deception():
    """
    Diplomacy dialogues are downloaded from Github (all splits). The data
    are ASCII encoded and newlines are removed. Each message and label is treated
    as a sample.
    """

    NAME = 'diplomacy_deception'
    URLS = {'test':'https://raw.githubusercontent.com/DenisPeskov/2020_acl_diplomacy/master/data/test.jsonl',
            'train':'https://raw.githubusercontent.com/DenisPeskov/2020_acl_diplomacy/master/data/train.jsonl',
            'validation':'https://raw.githubusercontent.com/DenisPeskov/2020_acl_diplomacy/master/data/validation.jsonl'}

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    for dataset, url in URLS.items():
        filename = f'{dataset}.txt'
        download_file(url, directory, filename)
    
    files = glob.glob(f'{directory}/*.txt')
    data = defaultdict(list)

    def clean(text):
        return encode_ascii(text).replace('\n', '')

    for file in files:
        df = pd.read_json(file, lines=True)
        messages = list(itertools.chain.from_iterable(pd.read_json(files[0], lines=True)['messages']))
        labels = list(itertools.chain.from_iterable(pd.read_json(files[0], lines=True)['sender_labels']))
        df = pd.DataFrame({'message':messages, 'label':labels})
        data['truth'].extend(df[df['label']==True]['message'].apply(clean).tolist())
        data['lie'].extend(df[df['label']==False]['message'].apply(clean).tolist())
    
    save_output_json(data, NAME)

def process_drug_experiences():
    """
    Drug experiences are downloaded from Github repository. For each
    sample, we remove HTML formatting, split samples by paragraphs, and
    keep only paragraphs with over 50 characters.
    """

    NAME = 'drug_experiences'
    URL = 'https://github.com/technillogue/erowid-w2v/archive/master.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)

    DRUGS = ['cocaine', 'dxm', 'lsd', 'mdma', 'mushrooms', 'oxycodone', 'salvia', 'tobacco']

    data = {}

    for drug in DRUGS:
        
        files = glob.glob(join(directory, f'erowid-w2v-master/core-experiences/{drug}/*'))
        experiences = []

        for file in files:
            with open(file, 'r') as f:
                text = "".join(f.readlines())
                text = strip_tags(text).replace('\r', '')
                experiences.extend(split_delimiter_(text))
        
        data[drug] = experiences
    
    save_output_json(data, NAME)

def process_echr_decisions():
    """
    Decisions are downloaded from a public archive. A random sample of
    500 decisions are selected from the files. The samples with any
    violated articles are categorized as "violation," while the rest are
    categorized as "no violation." 
    """

    NAME = 'echr_decisions'
    URL = 'https://archive.org/download/ECHR-ACL2019/ECHR_Dataset.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)
    
    path = f'{directory}/*_Anon/*.json'

    np.random.seed(0)
    files = np.random.choice(glob.glob(path), 500, replace=False)
    dicts = [json.load(open(f, 'r')) for f in files]
    
    data = defaultdict(list)
    for d in dicts:
        text = list(map(encode_ascii, d['TEXT']))
        if d['VIOLATED_ARTICLES']:
            data['violation'].extend(text)
        else:
            data['no_violation'].extend(text)
    
    save_output_json(data, NAME)

def process_essay_scoring():
    """
    Essays are downloaded from a Github repository. Only essays from
    set 5 are considered. Essays with a score of at least 3 are categorized
    as good essays, while essays with a score less than 3 are bad essays.
    """

    NAME = 'essay_scoring'
    URL = 'https://raw.githubusercontent.com/Turanga1/Automated-Essay-Scoring/master/training_set_rel3.tsv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.tsv'
    download_file(URL, directory, filename)

    df = pd.read_csv(join(directory, filename), sep='\t', encoding='latin-1')
    df = df[df.essay_set == 5]
    good_essays = df[df.domain1_score >= 3].essay.tolist()
    bad_essays = df[df.domain1_score < 3].essay.tolist()

    data = {
        'good_essays':good_essays,
        'bad_essays':bad_essays
    }
    
    save_output_json(data, NAME)

def process_fake_news():
    """
    Fake news articles are downloaded from author's website. Full articles
    are treated as text snippets.
    """
    
    NAME = 'fake_news'
    URL = 'http://web.eecs.umich.edu/~mihalcea/downloads/fakeNewsDatasets.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)

    files = glob.glob(f'{directory}/fakeNewsDatasets/fakeNewsDataset/**/*.txt')
    legit, fake = [], []
    for file in files:
        with open(file, 'r') as f:
            contents = f.read()
            if 'legit' in file:
                legit.append(contents)
            else:
                fake.append(contents)
    
    data = {
        'legit':legit,
        'fake':fake
    }

    save_output_json(data, NAME)

def process_fomc_speeches():
    """
    Fed speeches are downloaded from Kaggle. The macro indicator data
    are merged in on the year and month. Full speech text is split by paragraph
    and categorized by speaker, year, and macroeconomic indicator.
    """
    
    NAME = 'fomc_speeches'

    df = pd.read_csv(f'{MANUAL_FOLDER}/{NAME}/fed_speeches_1996_2020.csv')
    df = df.dropna()
    df['year_month'] = df['date'].astype(int).astype(str).str[:6]
    indicators_df = pd.read_csv(f'{MANUAL_FOLDER}/{NAME}/macro_indicators.csv')

    indicators_df['year_month'] = indicators_df.Date.astype(str).str[:6]
    df = df.merge(indicators_df, on='year_month', how='left')
    bins = 5
    df['unemp_cuts'] = pd.qcut(df['unemployment'], q=bins, labels=range(bins))
    df['growth_cuts'] = pd.qcut(df['growth rate'], q=bins, labels=range(bins))
    df['ir_cuts'] = pd.qcut(df['fed interest rate'], q=bins, labels=range(bins))

    df = split_df(df, 'text', splitter=split_delimiter_)

    data = {
        'greenspan_speeches':df[df.speaker == 'Chairman Alan Greenspan'].text.tolist(),
        'bernanke_speeches':df[df.speaker == 'Chairman Ben S. Bernanke'].text.tolist(),
        'powell_chair_speeches':df[df.speaker == 'Chairman Jerome H. Powell'].text.tolist(),
        'yellen_speeches':df[df.speaker == 'Chair Janet L. Yellen'].text.tolist(),
        'ferguson_speeches':df[df.speaker == 'Vice Chairman Roger W. Ferguson'].text.tolist(),
        'meyer_speeches':df[df.speaker == 'Governor Laurence H. Meyer'].text.tolist(),
        'powell_gov_speeches':df[df.speaker == 'Governor Jerome H. Powell'].text.tolist(),
        'greenspan_years':df[df.year_month <= '200601'].text.tolist(),
        'bernanke_years':df[(df.year_month >= '200602') & (df.year_month <= '201401')].text.tolist(),
        'yellen_years':df[(df.year_month >= '201402') & (df.year_month <= '201801')].text.tolist(),
        'powell_years':df[df.year_month >= '201802'].text.tolist(),
        'low_unemp':df[df.unemp_cuts == 0].text.tolist(),
        'high_unemp':df[df.unemp_cuts == bins-1].text.tolist(),
        'low_growth':df[df.growth_cuts == 0].text.tolist(),
        'high_growth':df[df.growth_cuts == bins-1].text.tolist(),
        'low_ir':df[df.ir_cuts == 0].text.tolist(),
        'high_ir':df[df.ir_cuts == bins-1].text.tolist(),
    }

    save_output_json(data, NAME)

def process_genius_lyrics():
    """
    Genius lyrics are downloaded from a Google Drive. The lyrics are merged with
    song metadata and treated as samples. We categorize lyrics by hand-selecting
    popular artists, common genres, time periods, and view counts (over 1M views
    is high, 500k-1M is medium).
    """

    NAME = 'genius_lyrics'
    ID = '1SKwubShEWa7CuAuSqTxwRu6_J4ZVkOvD'
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    # download_drive_zip(ID, directory)

    lyrics_df = pd.read_json(join(directory, 'genius-expertise/lyrics.jl'), lines=True)
    songs_df = pd.read_json(join(directory, 'genius-expertise/song_info.json'), lines=True)
    df = lyrics_df.merge(songs_df, left_on='song',right_on='url_name')

    clean = lambda s: [st.strip('\n') for st in re.split('\[.*\]', s) if st.strip('\n')]
    df['release_year'] = df['release_date'].str.split(', ').str[1]
    df = df[~df.release_year.isna()]
    df.release_year = df.release_year.astype(int)

    artists = {
        'drake':'Drake',
        'kanye':'Kanye-west',
        'kendrick':'Kendrick-lamar',
        'j_cole':'J-cole',
        'eminem':'Eminem',
        'logic':'Logic',
        'ariana_grande':'Ariana-grande',
        'beyonce':'Beyonce',
        'weeknd':'The-weeknd',
        'post_malone':'Post-malone'
    }
    genres = {
        'east_coast':'East Coast',
        'west_coast':'West Coast',
        'rap':'Rap',
        'trap':'Trap',
        'rock':'Rock',
        'alt_rock':'Alternative Rock',
        'pop':'Pop',
        'alt':'Alternative',
        'uk':'UK',
        'france':'France',
        'r_b':'R&B',
        'soul':'Soul',
    }
    years = range(1970, 2020)

    data = {}
    for name, artist in artists.items():
        data[name] = list(itertools.chain.from_iterable(df[df.primary_artist == artist].lyrics.apply(clean)))
    for name, genre in genres.items():
        data[name] = list(itertools.chain.from_iterable(df[df.tags.apply(lambda x: genre in x)].lyrics.apply(clean)))
    for year in years:
        data[year] = list(itertools.chain.from_iterable(df[df.release_year == year].lyrics.apply(clean)))
    data['high_views'] = list(itertools.chain.from_iterable(df[df.views > 1e6].lyrics.apply(clean)))
    data['mid_views'] = list(itertools.chain.from_iterable(df[(df.views > 5e5) & (df.views < 1e6)].lyrics.apply(clean)))
    
    snippets = df['lyrics'].tolist()
    sentences = sentence_tokenize(snippets)
    
    save_output_json(data, NAME)
    save_unlabeled_json(sentences, NAME)

def process_happy_moments():
    """
    The HappyDB dataset is downloaded from the official GitHub repository. Demographic
    data is cleaned and merged into the happy moments. Happy moment descriptions are
    treated as samples are categorized by type of happy moment, country of origin, and
    other demographic features.
    """

    NAME = 'happy_moments'
    
    URL = 'https://raw.githubusercontent.com/megagonlabs/HappyDB/master/happydb/data/cleaned_hm.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    
    filename = f'happy_moments.csv'
    download_file(URL, directory, filename)

    URL_DEMO = 'https://raw.githubusercontent.com/megagonlabs/HappyDB/master/happydb/data/demographic.csv'
    demo_filename = f'demographics.csv'
    download_file(URL_DEMO, directory, demo_filename)

    df = pd.read_csv(join(directory, filename))
    demo_df = pd.read_csv(join(directory, demo_filename))

    df = df.merge(demo_df, on='wid', how='left')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    data = {
        'affection':df[df.ground_truth_category=='affection'].cleaned_hm.tolist(),
        'bonding':df[df.ground_truth_category=='bonding'].cleaned_hm.tolist(),
        'enjoy_the_moment':df[df.ground_truth_category=='enjoy_the_moment'].cleaned_hm.tolist(),
        'leisure':df[df.ground_truth_category=='leisure'].cleaned_hm.tolist(),
        'usa':df[df.country=='USA'].cleaned_hm.tolist(),
        'india':df[df.country=='IND'].cleaned_hm.tolist(),
        'venezuela':df[df.country=='VEN'].cleaned_hm.tolist(),
        'canada':df[df.country=='CAN'].cleaned_hm.tolist(),
        '18-21':df[df.age.between(18, 21)].cleaned_hm.tolist(),
        '22-25':df[df.age.between(22, 25)].cleaned_hm.tolist(),
        '26-35':df[df.age.between(26, 35)].cleaned_hm.tolist(),
        '36-45':df[df.age.between(36, 45)].cleaned_hm.tolist(),
        '46+':df[df.age.between(46, 100)].cleaned_hm.tolist(),
        'male':df[df.gender == 'm'].cleaned_hm.tolist(),
        'female':df[df.gender == 'f'].cleaned_hm.tolist(),
        'parent':df[df.parenthood == 'y'].cleaned_hm.tolist(),
        'not_parent':df[df.parenthood == 'n'].cleaned_hm.tolist(),
        'single':df[df.marital == 'single'].cleaned_hm.tolist(),
        'married':df[df.marital == 'married'].cleaned_hm.tolist(),
        'divorced':df[df.marital == 'divorced'].cleaned_hm.tolist(),
        'separated':df[df.marital == 'separated'].cleaned_hm.tolist(),
    }
    
    save_output_json(data, NAME)

def process_huff_post_headlines():
    """
    Huffington Post headlines are downloaded from Kaggle. The short description
    of each article is treated as a sample and tokenized at the sentence
    level.
    """

    NAME = 'huff_post_headlines'
    FILE = 'News_Category_Dataset_v3.json'
    
    directory = join(MANUAL_FOLDER, NAME)
    df = pd.read_json(join(directory, FILE), lines=True)

    col_types = {
        'headline':str,
        'short_description':str,
        'category':'category',
        'authors':'category',
        'date':'datetime64'
    }
    df = df[col_types.keys()].astype(col_types)

    snippets = df['short_description'].tolist()
    sentences = sentence_tokenize(snippets)

    save_dataset(df, NAME)
    save_unlabeled_json(sentences, NAME)

def process_immigration_speeches():
    """
    Immigration speeches are downloaded from the replication package.
    The speech text is preprocessed to remove extraneous spaces. We engineer
    features corresponding to time periods, well-known speakers, other
    significant time periods, racial group under discussion, and geographic
    area of the United States.
    """

    NAME = 'immigration_speeches'
    PRES_FILE = 'pres_imm_segments_with_tone.jsonlist'
    CON_FILE = 'imm_segments_with_tone_and_metadata.jsonlist'

    def clean_text(text):
        """Tries to replace accidental periods and removes the space before periods."""

        text = text.replace(' .', '.')
        phrases = text.split('. ')
        new_text = ""
        for curr_phrase, next_phrase in zip(phrases, phrases[1:]):
            if next_phrase[0].isupper():
                new_text += curr_phrase + '. '
            else:
                new_text += curr_phrase + ' '
        return new_text.strip()

    pres_df = pd.read_json(join(MANUAL_FOLDER, NAME, PRES_FILE), lines=True)
    con_df = pd.read_json(join(MANUAL_FOLDER, NAME, CON_FILE), lines=True)

    # preprocess dataframes
    con_df['year'] = con_df['date'].astype(str).str[:4].astype(int)
    pres_df.text = pres_df.text.apply(clean_text)
    pres_df = pres_df[pres_df.text.astype(bool)] # remove empty strings
    pres_df = pres_df.rename({
        'anti_prob':'anti',
        'neutral_prob':'neutral',
        'pro_prob':'pro',
    }, axis=1)
    con_df.text = con_df.text.apply(clean_text)  # remove empty strings
    con_df = con_df[con_df.text.astype(bool)]
    both_cols = list(set(pres_df.columns).intersection(con_df.columns)) # get overlapping columns
    combined_df = pres_df[both_cols].append(con_df[both_cols])

    data = {}
    pres_map = { # white house speakers
        'trump':'Donald J. Trump',
        'obama':'Barack Obama',
        'bush_jr':'George W. Bush',
        'clinton':'William J. Clinton',
        'bush_sr':'George Bush',
        'reagan':'Ronald Reagan',
        'pence':'Mike Pence',
    }
    for speaker, full_name in pres_map.items():
        data[f'speaker_{speaker}'] = pres_df[pres_df.speaker==full_name].text.tolist()

    SPOKEN = {'spoken_addresses','inaugurals','news_conferences'} # split by spoken/written
    WRITTEN = {'written_messages','written_statements'}
    pres_df['is_written'] = pres_df.categories.apply(lambda c: bool(WRITTEN.intersection(c)))
    pres_df['is_spoken'] = pres_df.categories.apply(lambda c: bool(SPOKEN.intersection(c)))
    for speaker, full_name in pres_map.items():
        data[f'speaker_{speaker}_spoken'] = pres_df[(pres_df.speaker==full_name)&(pres_df.is_spoken)].text.tolist()
        data[f'speaker_{speaker}_written'] = pres_df[(pres_df.speaker==full_name)&(pres_df.is_written)].text.tolist()
    
    con_map = {
        'kennedy':('Mr. KENNEDY','MA'),
        'durbin':('Mr. DURBIN','IL'),
        'sessions':('Mr. SESSIONS','AL'),
        'king':('Mr. KING of Iowa','IA'),
        'simpson':('Mr. SIMPSON','WY'),
        'tancredo':('Mr. TANCREDO','CO'),
        'dickstein':('Mr. DICKSTEIN','NY'),
        'leahy':('Mr. LEAHY','VT'),
        'cornyn':('Mr. CORNYN','TX'),
    }
    for speaker, (full_name, state) in con_map.items():
        data[f'con_speaker_{speaker}'] = con_df[(con_df.speaker==full_name)&(con_df.state==state)].text.tolist()
    
    pres_periods = [
        ('obama',datetime(2009,1,26), datetime(2013,1,25)),
        ('obama',datetime(2013,1,26), datetime(2017,5,7)),
        ('bush_jr',datetime(2001,1,20), datetime(2001,9,10)),
        ('bush_jr',datetime(2001,9,11), datetime(2009,1,15)),
    ]
    for speaker, dt1, dt2 in pres_periods:
        full_name = pres_map[speaker]
        dt1_str = dt1.strftime("%m-%d-%Y")
        dt2_str = dt2.strftime("%m-%d-%Y")
        data[f'speaker_{speaker}_{dt1_str}_{dt2_str}'] = pres_df[(pres_df.speaker==full_name)&(pres_df.date.between(dt1, dt2))].text.tolist()
    
    combined_df['period_early'] = combined_df['year'].between(1873, 1934, inclusive='both')
    combined_df['period_mid'] = combined_df['year'].between(1935, 1956, inclusive='both')
    combined_df['period_late'] = combined_df['year'].between(1957, 2020, inclusive='both')
    con_df['period_early'] = con_df['year'].between(1873, 1934, inclusive='both')
    con_df['period_mid'] = con_df['year'].between(1935, 1956, inclusive='both')
    con_df['period_late'] = con_df['year'].between(1957, 2020, inclusive='both')

    CHINESE = ['Chinese','chinese','Chinee','Chines','-Chinese','Chineso','Chinese-','hinese','Chinamen']
    MEXICAN = ['Mexican','Mexicans','mexican','exican','Mlexican','Miexican']
    ITALIAN = ['Italian','italian','Italy']
    def contains_one_of(s, lst):
        return any(el in s for el in lst)
    combined_df['mentions_chinese'] = combined_df.text.apply(lambda s: contains_one_of(s, CHINESE))
    combined_df['mentions_mexican'] = combined_df.text.apply(lambda s: contains_one_of(s, MEXICAN))
    combined_df['mentions_italian'] = combined_df.text.apply(lambda s: contains_one_of(s, ITALIAN))
    for period in ('early','mid','late'):
        right_period = {
            'early':combined_df.period_early,
            'mid':combined_df.period_mid,
            'late':combined_df.period_late,
        }[period]
        data[f'period_{period}_pro'] = combined_df[right_period&(combined_df.pro > .995)].text.tolist()
        data[f'period_{period}_anti'] = combined_df[right_period&(combined_df.anti > .995)].text.tolist()
        data[f'period_{period}_chinese'] = combined_df[right_period&(combined_df.mentions_chinese)].text.tolist()
        data[f'period_{period}_mexican'] = combined_df[right_period&(combined_df.mentions_mexican)].text.tolist()
        data[f'period_{period}_italian'] = combined_df[right_period&(combined_df.mentions_italian)].text.tolist()

    NORTHERN = 'ME, MA, RI, CT, NH, VT, NY, PA, NJ, DE, OH, IN, MI, IL, MO, WI, MN, IA, KS, NE, SD, ND'.split(', ')
    WESTERN = 'CO, WY, MT, ID, WA, OR, UT, NV, CA, AK, HI'.split(', ')
    SOUTHERN = 'WV, VI, VA, KY, TN, NC, SC, GA, AL, MS, AR, LA, FL, TX, OK, NM, AZ'.split(', ')
    for period in ('early','mid','late'):
        right_period = {
            'early':con_df.period_early,
            'mid':con_df.period_mid,
            'late':con_df.period_late,
        }[period]
        data[f'period_{period}_democrat'] = con_df[right_period&(con_df.party=='D')].text.tolist()
        data[f'period_{period}_republican'] = con_df[right_period&(con_df.party=='R')].text.tolist()
        data[f'period_{period}_northern'] = con_df[right_period&(con_df.state.isin(NORTHERN))].text.tolist()
        data[f'period_{period}_western'] = con_df[right_period&(con_df.state.isin(WESTERN))].text.tolist()
        data[f'period_{period}_southern'] = con_df[right_period&(con_df.state.isin(SOUTHERN))].text.tolist()
        data[f'period_{period}_senate'] = con_df[right_period&(con_df.chamber=='S')].text.tolist()
        data[f'period_{period}_house'] = con_df[right_period&(con_df.chamber=='H')].text.tolist()

    save_output_json(data, NAME)

def process_kickstarter():
    """
    We download a 2018 crawl from Kickstarter from Kaggle. The project name is
    treated as the text sample.
    """

    NAME = 'kickstarter'

    file = 'ks-projects-201801.csv'
    directory = join(MANUAL_FOLDER, NAME)
    df = pd.read_csv(join(directory, file), encoding='latin-1')
    
    df['deadline'] = pd.to_datetime(df['deadline'])
    df['launched'] = pd.to_datetime(df['launched'])
    mapper = {
        'usd pledged':'usd_pledged',
        'category':'sub_category',
    }
    df = df.rename(mapper, axis=1)
    col_types = {
        'name':str,
        'sub_category':'category',
        'main_category':'category',
        'currency':'category',
        'deadline':'datetime64',
        'goal':float,
        'launched':'datetime64',
        'pledged':int,
        'state':'category',
        'backers':int,
        'country':'category',
        'usd_pledged':float,
        'usd_pledged_real':float,
        'usd_goal_real':float,
    }
    df = df[col_types.keys()].astype(col_types)

    data = {}
    STATES = {'failed','successful'}
    for state in STATES:
        data[f'state_{state}'] = df[df['state'] == state]['name'].tolist()
    
    save_output_json(data, NAME)
    save_dataset(df, NAME)

def process_microedit_humor():
    """
    Microedit dataset is downloaded from the author's website.
    We make the relevant edit to each text sample and treat the
    edited text sample as the data point. We bin the mean annotator
    grade into 4 and denote each as unfunny, netural, funny, and
    very funny, respectively.
    """

    NAME = 'microedit_humor'
    URL = 'https://cs.rochester.edu/u/nhossain/semeval-2020-,0k,-7-dataset.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)

    files = glob.glob(f'{directory}/**/subtask-1/*.csv')

    df = pd.DataFrame()

    for file in files:
        df = df.append(pd.read_csv(file))
        
    def make_edit(sentence, replacement):
        return re.sub('<[^\>]+>', replacement, sentence)

    clean_text = encode_ascii

    df['edited'] = df.apply(lambda x: make_edit(x.original, x.edit), axis=1)
    df['edited'] = df['edited'].apply(clean_text)
    df['bin'] = pd.cut(df['meanGrade'], 4, labels=range(4))
    
    data = {}
    for i, rank in enumerate(['unfunny', 'neutral', 'funny', 'very_funny']):
        data[rank] = df[df['bin'] == i]['edited'].tolist()
    
    save_output_json(data, NAME)

def process_monster_jobs():
    """
    Jobs on Monster.com are downloaded from Kaggle. Job descriptions are treated
    as samples and split at the paragraph and sentence level. We keep and categorize
    jobs from seventeen large cities.  
    """

    NAME = 'monster_jobs'

    df = pd.read_csv(join(MANUAL_FOLDER, NAME, 'monster_com-job_sample.csv'))
    
    locations = {
        'dallas':'Dallas, TX',
        'houston':'Houston, TX',
        'austin':'Austin, TX',
        'denver':'Denver, CO',
        'atlanta':'Atlanta, GA',
        'cincinatti':'Cincinnati, OH',
        'tampa':'Tampa, FL',
        'boston':'Boston, MA',
        'milwaukee':'Milwaukee, WI',
        'la':'Los Angeles, CA',
        'sf':'San Francisco, CA',
        'nashville':'Nashville, TN',
        'nyc':'New York, NY',
        'colombus':'Columbus, OH',
        'seattle':'Seattle, WA',
        'las_vegas':'Las Vegas, NV',
        'berkeley':'Berkeley, CA'
    }

    df = split_df(df, 'job_description', splitter=split_delimiter_)
    df = split_df(df, 'job_description')

    data = {}
    for name, loc in locations.items():
        descriptions = df[df.location.str.contains(loc)].job_description.dropna().tolist()
        data[name] = descriptions

    save_output_json(data, NAME)

def process_movie_tmdb():

    NAME = 'movie_tmdb'

    df = pd.read_csv('https://raw.githubusercontent.com/ErolGelbul/movies_data/master/movies_complete.csv')
    df = df[df['original_language'] == 'en'] # keep only english movies
    df = df[~df['overview'].isna()] # remove missing overviews
    data = {}

    # split by bin
    FEATURES = {
        'runtime':'runtime',
        'pop':'popularity',
        'budget':'budget_musd',
        'revenue':'revenue_musd'
    }
    for name, column in FEATURES.items():
        df[f'{column}_bin'] = pd.qcut(df[column], 5, labels=range(5))
        data[f'{name}_high'] = df[df[f'{column}_bin'] == 0]['overview'].tolist()
        data[f'{name}_mid'] = df[df[f'{column}_bin'] == 2]['overview'].tolist()
        data[f'{name}_low'] = df[df[f'{column}_bin'] == 4]['overview'].tolist()

    # release date
    data['pre_2000'] = df[df['release_date'] < '2000-01-01']['overview'].tolist()
    data['post_2000'] = df[df['release_date'] >= '2000-01-01']['overview'].tolist()

    companies = {
        'mgm': 'MGM',
        'warner':'Warner Bros.',       
        'paramount':'Paramount Pictures',
        'fox':'Twentieth Century Fox Film Corporation',
        'universal':'Universal Pictures',
    }

    for name, production_company in companies.items():
        data[f'company_{name}'] = df[(~df['production_companies'].isna()) & df['production_companies'].str.contains(production_company)]['overview'].tolist()
    data['multiple_companies'] = df[(~df['production_companies'].isna()) & df['production_companies'].str.contains('|')]['overview'].tolist()
    data['single_company'] = df[~(~df['production_companies'].isna() & df['production_companies'].str.contains('|'))]['overview'].tolist()

    snippets = df['overview'].tolist()
    sentences = sentence_tokenize(snippets)

    save_output_json(data, NAME)
    save_unlabeled_json(sentences, NAME)

def process_movie_wiki():
    """
    Wikipedia movie summaries are downloaded from Kaggle.
    """

    NAME = 'movie_wiki'

    file = 'wiki_movie_plots_deduped.csv'

    df = pd.read_csv(join(MANUAL_FOLDER, NAME, file))

    mapper = {
        'Release Year':'year',
        'Title':'title',
        'Origin/Ethnicity':'origin',
        'Director':'director',
        'Genre':'genre',
        'Plot':'plot'
    }
    df = df.rename(mapper, axis=1)
    col_types = {
        'year':int,
        'title':str,
        'origin':'category',
        'director':'category',
        'genre':'category',
        'plot':str
    }

    df = df[col_types.keys()].astype(col_types)

    snippets = df['plot'].tolist()
    sentences = sentence_tokenize(snippets)

    save_dataset(df, NAME)
    save_unlabeled_json(sentences, NAME)

def process_news_popularity():
    """
    Headlines are downloaded from a reproduction package. Headline and title text
    is cleaned and the title is treated as the text sample. The 100 most positive
    and negative or popular and unpopular articles on each topic are used as
    distributions.
    """

    NAME = 'news_popularity'
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/News_Final.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)

    df = pd.read_csv(join(directory, filename))

    def clean_text(text):
        return str(text).replace('&quot;', '"').replace('"""', '"')

    df['Headline'] = df['Headline'].apply(clean_text)
    df['Title'] = df['Title'].apply(clean_text)

    def top_bottom(group, col, n = 100):
        sorted = group.sort_values(col)
        return sorted.iloc[-n:], sorted.iloc[:n]

    def rank_sentiment(group):
        return top_bottom(group, 'SentimentTitle')

    def rank_fb(group):
        return top_bottom(group, 'Facebook')

    sentiment_groups = df.groupby('Topic').apply(rank_sentiment)
    data = {}
    for topic in sentiment_groups.keys():
        pos, neg = sentiment_groups[topic]
        data[topic + '_pos'] = pos['Title'].tolist()
        data[topic + '_neg'] = neg['Title'].tolist()

    fb_df = df[(df.Source == 'Bloomberg') & (df.Facebook >= 0) & (df.Topic.isin(['obama', 'economy', 'microsoft']))]
    fb_groups = fb_df.groupby('Topic').apply(rank_fb)
    for topic in fb_groups.keys():
        pop, unpop = fb_groups[topic]
        data[topic + '_pop'] = pop['Title'].tolist()
        data[topic + '_unpop'] = unpop['Title'].tolist()

    save_output_json(data, NAME)

def process_nli_benchmarks():
    """
    NLI benchmarks are downloaded from a public collection on Google Drive. We
    examine the premise and hypothesis separately as samples.
    """

    NAME = 'nli_benchmarks'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'

    IDs = {
        'wanli':('1VGRF7Rp0CUU0bUP5Lu8PXehW2hZkHBPH','1P-_hixzcdvAopWuWcB9VjYIDGo0ncrVI'),
        'tailor':'1-xhVxkeSNj_ROAyp-bR78ClyVioXMuxB',
        'qnli':'1muZgdXe8CJbyLMbNovAbjUe8D8qL2uJM',
        'nq-nil':'1s51pxVzBMr23JneWXFOPGSvk5oiO9j_C',
        'mnli':'1G0HGU0Rovz0zf4fSDpoTf6V6unCQ3uDb',
        'hans':'1a6uZwo0G-7aXGBCGLkkSv_FkDrBeke0a',
        'fever-nil':'1oe2Q43uyfFMomenlX8vJZ1M8z7hO92nn',
        'anli_r1':( '1FgB5rcydZjjlHpt637PILXsv5wwqZ1Tq','1-5wGKFGIuuRM7xJ_cibSAmr_3zAM6mb0'),
        'anli_r2':('1CXvJtsFg9IFU8knyNLaPT_1UBvOYwOoc','1oNvNxoBMV1iKgDXCDRpkMgbghscvzgCm'),
        'anli_r3':('1aLkhFUA-0ZN0vwi-0L9n_5XQs1CCzkpf','1drcod8uPONyvYpguOvEvJN7C2MyMFvbF'),
        'mnli_mismatched':'1FXpfI3xTfVxXYxDjo33JVDl0zP7QvBOF',
        'mnli_matched':'1ZeCWToJfViJbyhDnECAmrkb_0pOz_BHT',
    }

    dfs = {}
    for name, ids in IDs.items():
        if not isinstance(ids, tuple):
            ids = tuple([ids])
        df = pd.DataFrame()
        for index, drive_id in enumerate(ids):
            download_drive_file(drive_id, directory, f'{name}_{index}.jsonl')
        json_paths = glob.glob(join(directory, f'{name}_*.jsonl'))
        for json_path in json_paths:
            df = df.append(pd.read_json(json_path, lines=True))
        dfs[name] = df

    data = {}
    for name in IDs.keys():
        df = dfs[name]
        data[f'{name}_premise'] = df['premise'].tolist()
        data[f'{name}_hypothesis'] = df['hypothesis'].tolist()

    save_output_json(data, NAME)

def process_npt_conferences():
    """
    NPT conference notes are extracted from the accompanying replication package.
    Text is split by paragraph and only paragraphs longer than 50 characters
    are preserved. Text is split into three time ranges: pre-2008, 2008-2012, and
    post-2012.
    """

    NAME = 'npt_conferences'

    files = glob.glob(f'{MANUAL_FOLDER}/{NAME}/BarnumLoNPTReplication/data/docs_by_committee/**/*.txt')
    docs = []
    for file in files:
        year = re.findall('\d\d\d\d', file)[0]
        with open(file, 'r', encoding='latin1') as f:
            text = " ".join(f.readlines())
        doc = {
            'year':int(year),
            'text':text
        }
        docs.append(doc)

    df = pd.DataFrame(docs)

    def clean_text(text):
        paras = text.replace('\t', ' ').split('\n')
        return '\n\n'.join([p for p in filter(None, paras) if len(p) > 50])

    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.strip().str.len() > 0]
    df = split_df(df, 'text', partial(split_delimiter_, delimiter='\n\n'))
    df = split_df(df, 'text')

    pre_2008 = df[df.year < 2008].text.tolist()
    btw_2008_2012 = df[df.year.between(2008, 2012)].text.tolist()
    post_2012 = df[df.year > 2012].text.tolist()

    data = {
        'pre_2008':pre_2008,
        'btw_2008_2012':btw_2008_2012,
        'post_2012':post_2012,
    }

    save_output_json(data, NAME)

def process_open_deception():
    """
    Open domain lies are downloaded from the public dataset and lie textsare
    split into lies and truths.
    """

    NAME = 'open_deception'
    URL = 'http://web.eecs.umich.edu/~mihalcea/downloads/openDeception.2015.tar.gz'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_tar(URL, directory)
    
    input_file = f'{directory}/OpenDeception/7Truth7LiesDataset.csv'
    df = pd.read_csv(input_file, quotechar="'", escapechar="\\")
    
    data = {
        'lie':df[df['class']=='lie']['text'].tolist(),
        'truth': df[df['class']=='truth']['text'].tolist()
    }
    
    save_output_json(data, NAME)

def process_open_review():
    """
    Open review abstracts are accessed via the openreview API. We query
    for abstracts from the 2018-2021 ICLR blind submissions. Abstracts are
    classified based on rating: >=7 ("great"), 5-6 ("good"), and <=4 ("bad").
    """

    NAME = 'open_review'

    import scrape_open_review
    great_papers, good_papers, bad_papers = scrape_open_review.scrape()

    data = {
        'great_papers':list(map(encode_ascii, great_papers)),
        'good_papers':list(map(encode_ascii, good_papers)),
        'bad_papers':list(map(encode_ascii, bad_papers)),
    }
    
    save_output_json(data, NAME)

def process_oral_histories():
    """
    Oral histories are downloaded from the paper's accompanying Github repository.
    Histories are classified according to birth year of the author (pre-1930, 1930-1949, post-1950),
    the race of the speaker (black, Asian, white), college education (graduate/bachelors or none),
    and place of birth (South or not South, as defined by the Census Bureau). We treat
    the full oral history as the text sample.
    """
    
    NAME = 'oral_histories'
    URL = 'https://raw.githubusercontent.com/ohtap/ohtap/master/Research%20Question%202/updated_0510.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)
    
    df = pd.read_csv(join(directory, filename), sep='\t')
    southern_states = """MD
    DE
    VA
    WV
    KY
    TN
    NC
    SC
    FL
    GA
    AL
    MS
    LA
    AK
    TX
    OK""".split('\n')

    df['corrected_text'] = df['corrected_text'].apply(encode_ascii)

    data = {
        'pre_1930':df[df.birth_year < 1930].corrected_text.to_list(),
        '1930-50':df[(df.birth_year >= 1930) & (df.birth_year < 1950)].corrected_text.to_list(),
        'post_1950':df[df.birth_year > 1950].corrected_text.to_list(),
        'black':df[df.race == 'Black or African American'].corrected_text.to_list(),
        'white':df[df.race == 'White'].corrected_text.to_list(),
        'asian':df[df.race == 'Asian'].corrected_text.to_list(),
        'college_educated':df[df.education.isin(['Graduate or professional degree', 'Bachelor\'s degree'])].corrected_text.to_list(),
        'not_college_educated':df[~df.education.isin(['Graduate or professional degree', 'Bachelor\'s degree'])].corrected_text.to_list(),
        'south':df[df.interviewee_birth_state.isin(southern_states)].corrected_text.to_list(),
        'not_south':df[~df.interviewee_birth_state.isin(southern_states)].corrected_text.to_list(),
    }
    
    save_output_json(data, NAME)

def process_parenting_reddit_users():
    """
    Individual posts are retrieved with permission from the author. We sample
    5000 posts per year. When use authorship histories to estamate how long
    each author has been posting on parenting related subreddits and split
    according to various account ages. We use posts on mom- and dad- related
    subreddits to guess user gender and split accordingly.
    """

    NAME = 'parenting_reddit_users'
    ID = None
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_drive_file(ID, directory, filename)
    
    df = pd.read_csv(join(directory, filename))

    df = df.sample(frac=1, random_state=0) # shuffle
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    data = {}

    df['year'] = df['datetime'].dt.year
    for year in df.year.unique():
        data[f'year_{year}'] = df[df.year==year].selftext.sample(n=5000).tolist()

    first_post = df[['author', 'datetime']].groupby('author').min()
    get_first_post = dict(zip(first_post.index, first_post['datetime']))
    df['first_post'] = df['author'].apply(lambda x: get_first_post[x])
    df['age'] = df['datetime'] - df['first_post']

    labels = ['new', '30_days', '90_days', 'half_year', '1_year', '2_years', '3_years', '4_years', '5_years']
    days = (0, 30, 90, 180, 365, 2*365, 3*365, 4*365, 5*365, 100*365)
    bins = [timedelta(days=d) for d in days]
    df['stage'] = pd.cut(df['age'], bins=bins, right=False, labels=labels)

    for stage in labels:
        stage_text = df[df['stage'] == stage]['selftext']
        data[f'{stage}'] = stage_text.sample(min(5000, len(stage_text))).tolist()

    mom_subreddits = ('Mommit', 'NewMomStuff')
    dad_subreddits = ('daddit', 'NewDads')
    is_mom = df[['author', 'subreddit']].groupby('author').agg(lambda x: x.isin(mom_subreddits).any())
    is_dad = df[['author', 'subreddit']].groupby('author').agg(lambda x: x.isin(dad_subreddits).any())

    get_is_mom = dict(zip(is_mom.index, is_mom['subreddit']))
    get_is_dad = dict(zip(is_dad.index, is_dad['subreddit']))
    df['is_mom'] = df['author'].apply(lambda x: get_is_mom[x])
    df['is_dad'] = df['author'].apply(lambda x: get_is_dad[x])

    data['moms'] = df[df['is_mom'] & ~df['is_dad'] & ~df['subreddit'].isin(mom_subreddits)]['selftext'].tolist()
    data['dads'] = df[df['is_dad'] & ~df['is_mom'] & ~df['subreddit'].isin(dad_subreddits)]['selftext'].tolist()

    save_output_json(data, NAME)

def process_parenting_subreddits():
    """
    Posts from various subreddits are downloaded from the paper's
    GitHub repository. We clean the text and split the posts according to
    the topic(s) each post is tagged with.
    """

    NAME = 'parenting_subreddits'
    URL = 'https://raw.githubusercontent.com/GT-SALT/Parenting_OnlineUsage/main/data/0527_reddit_1300_parenting_clean.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)
    
    df = pd.read_csv(join(directory, filename))
    df.head()

    df = df.dropna()
    topics = set(itertools.chain.from_iterable(df['topics'].str.split(',')))
    data = {}
    for topic in topics:
        text = df[df['topics'].str.contains(topic)]['text']
        clean_text = text.apply(lambda s: codecs.unicode_escape_decode(s)[0]).apply(encode_ascii)
        data[topic] = clean_text.tolist()

    save_output_json(data, NAME)

def process_poetry():
    """
    Poems are downloaded from a 2019 scrape of the PoetryFoundation website
    from Kaggle. The text is cleaned and split according subject tags and
    authorship.
    """

    NAME = 'poetry'
    FILE_NAME = 'PoetryFoundationData.csv'
    
    FILE = join(MANUAL_FOLDER, NAME, FILE_NAME)

    df = pd.read_csv(FILE, index_col=0)
    df = df.reset_index(drop=True)
    def clean_text(text):
        return encode_ascii(text.replace('\r\r\n','\n').strip())

    df['Poem'] = df['Poem'].apply(clean_text).astype(str)
    df = df[df['Poem'] != '']

    snippets = df['Poem'].tolist()
    sentences = sentence_tokenize(snippets)

    df = split_df(df, 'Poem')

    data = {}
    subjects = {
        'living':'Living',
        'death':'Death',
        'time':'Time & Brevity',
        'nature':'Nature',
        'relationships':'Relationships',
        'family':'Family & Ancestors',
        'commentary':'Social Commentaries',
        'history':'History & Politics',
    }
    tags_df = df[~df.Tags.isna()]
    for subject, tag in subjects.items():
        data[f'subject_{subject}'] = tags_df[tags_df.Tags.str.contains(tag)].Poem.tolist()
    poets = {
        'tennyson':'Alfred, Lord Tennyson',
        'shakespeare':'William Shakespeare',
        'dickinson':'Emily Dickinson',
        'wordsworth':'William Wordsworth',
    }
    for poet_name, poet in poets.items():
        data[f'poet_{poet_name}'] = df[df.Poet==poet].Poem.tolist()
    

    save_output_json(data, NAME)
    save_unlabeled_json(sentences, NAME)

def process_political_ads():
    """
    Ads are downloaded from the Ad Obvserver website, which maintains
    an aggregate of all collected ads. We extract targetting metadata from the
    targettings field and define splits according to age, gender, location,
    interests, time, and political lean.
    """
    
    NAME = 'political_ads'
    FILE = 'fb_monolith.csv'

    directory = join(MANUAL_FOLDER, NAME)
    df = pd.read_csv(join(directory, FILE))

    df.rename({'Unnamed: 8':'start_date','Unnamed: 9':'end_date'}, axis=1, inplace=True)
    df.columns = df.columns.str.replace(' ', '')

    df = df[~df['ad_text'].isna()]
    df['targetings'] = df['targetings'].apply(lambda x: eval(x))

    df['targetings'].apply(lambda x: x['AGE_GENDER'][0]['Gender'] if 'AGE_GENDER' in x else np.nan)
    df['age_gender'] = df['targetings'].apply(lambda x: x['AGE_GENDER'] if 'AGE_GENDER' in x else np.nan)
    df['locations'] = df['targetings'].apply(lambda x: x['LOCATION'] if 'LOCATION' in x else np.nan)
    df['interests'] = df['targetings'].apply(lambda x: x['INTERESTS'] if 'INTERESTS' in x else np.nan)
    df['observed_at'] = pd.to_datetime(df['observed_at'])
    data = {}
    is_dem = df['political_value'] > 0
    is_rep = df['political_value'] < 0
    ages = {
        'children':'6-53',
        'teenagers':'13-53',
        'adults':'18-53',
    }
    for age, val in ages.items():
        data[f'age_{age}'] = df[df['age_gender'].str[0].str['Age'] == val]['ad_text'].tolist()
    genders = {
        'male':'MALE',
        'female':'FEMALE',
        'any':'ANY',
    }
    for gender, val in genders.items():
        data[f'gender_{gender}'] = df[df['age_gender'].str[0].str['Gender'] == val]['ad_text'].tolist()
    locations = {
        'us':'the United States',
        'cad':'Canada',
        'uk':'the United Kingdom',
        'aus':'Australia',
        'ca':'California',
        'ga':'Georgia',
        'nyc':'New York, New York',
        'tx':'Texas',
    }
    for location, val in locations.items():
        data[f'location_{location}'] = df[df['locations'].str[0] == {'Location', val}]['ad_text'].tolist()
    interests = {
        'social_issues':'Politics and social issues',
        'politics':'Politics',
        'community_issues':'Community issues',
        'education':'Education',
        'family':'Family'
    }   
    is_dem = df['political_value'] > 0
    is_rep = df['political_value'] < 0
    for interest, val in interests.items():
        data[f'interest_{interest}'] = df[df['interests'].str[0].str['Interests'] == val]['ad_text'].tolist()
        data[f'interest_{interest}_dem'] = df[is_dem & (df['interests'].str[0].str['Interests'] == val)]['ad_text'].tolist()
        data[f'interest_{interest}_rep'] = df[is_rep & (df['interests'].str[0].str['Interests'] == val)]['ad_text'].tolist()
    us_df = df[df['locations'].str[0] == {'the United States', 'Location'}]
    is_dem = us_df['political_value'] > 0
    is_rep = us_df['political_value'] < 0
    time_ranges = {
        '21_jan_mar':('2021-1-1','2021-4-1'),
        '21_jul_sep':('2021-7-1','2021-10-1'),
    }
    for time, (start, end) in time_ranges.items():
        data[f'time_{time}'] = us_df[us_df['observed_at'].between(start, end)]['ad_text'].tolist()
        data[f'time_{time}_dem'] = us_df[is_dem & us_df['observed_at'].between(start, end)]['ad_text'].tolist()
        data[f'time_{time}_rep'] = us_df[is_rep & us_df['observed_at'].between(start, end)]['ad_text'].tolist()

    save_output_json(data, NAME)

def process_politifact():

    NAME = 'politifact'
    FILE = 'politifact_factcheck_data.json'

    directory = join(MANUAL_FOLDER, NAME)
    df = pd.read_json(join(directory, FILE), lines=True)

    col_types = {
        'statement':str,
        'verdict':'category',
        'statement_originator':'category',
        'statement_source':'category',
        'factchecker':'category',
        'statement_date':'datetime64',
        'factcheck_date':'datetime64',
    }
    df = df[col_types.keys()].astype(col_types)

    save_dataset(df, NAME)

def process_rate_my_prof():
    """
    Downloads sample of RateMyProfessor.com reviews from online repo. We clean the text
    and guess the gender of the reviewed lecturer from the first name using the
    gender_guesser package. Due to data availability, we consider only male and female
    names. To improve the quality of the classification, we remove
    any posts which use pronouns from the opposing sex (e.g. "him"). 
    """

    NAME = 'rate_my_prof'
    URL = 'https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/fvtfjyvw7d-2.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)
    
    import gender_guesser.detector as gender

    d = gender.Detector()
    get_gender = lambda name: d.get_gender(name)

    df = pd.read_csv(join(directory, 'RateMyProfessor_Sample data.csv'))
    df['comments'] = df['comments'].apply(lambda s: codecs.unicode_escape_decode(s)[0] if isinstance(s, str) else "")
    df['first_name'] = df['professor_name'].str.split().str[0]
    df['gender'] = df['first_name'].apply(get_gender)
    df['gender'].value_counts()

    df = df.sample(frac=1, random_state = 0)

    data = {
        'female':list(map(str, df[df['gender']=='female']['comments'])),
        'male':list(map(str, df[df['gender']=='male']['comments']))
    }

    data['female'] = list(filter(None, [t for t in data['female'] if ' him ' not in t and ' his ' not in t]))
    data['male'] = list(filter(None, [t for t in data['male'] if ' she ' not in t and ' her ' not in t]))

    save_output_json(data, NAME)

def process_radiology_diagnosis():
    """
    Radiology diagnoses are downloaded from a GitHub copy of the original
    task dataset. We parse the metadata to retrieve the dianostic code,
    decision type, impression, and pateint history. Referencing the associated
    ICD codes, we convert codes to colloquial diagnoses (e.g. 786.2 denotes cough).
    We treat the histories and impressions as samples and split them according to
    diagnosis and level of consensus.
    """

    NAME = 'radiology_diagnosis'
    URL = 'https://raw.githubusercontent.com/ngoduyvu/Convolution-text-classification/master/2007ChallengeTrainData.xml'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.xml'
    download_file(URL, directory, filename)

    import xmltodict
    with open(join(directory, filename), 'r') as f:
        data_dict = xmltodict.parse(f.read())

    def is_consensus(codes):
        for code in codes:
            if code['@origin'] == 'CMC_MAJORITY':
                majority = code['#text']
        agree = set()
        for code in codes:
            if code['#text'] == majority and not code['@origin'] == 'CMC_MAJORITY':
                agree.add(code['@origin'])
        return len(agree) == 3

    reports = data_dict['docs']['doc']
    samples = []
    for report in reports:
        codes = report['codes']['code']
        for code in codes:
            if code['@origin'] == 'CMC_MAJORITY':
                c = code['#text']
        for text in report['texts']['text']:
            if text['@type'] == 'CLINICAL_HISTORY':
                history = text['#text']
            if text['@type'] == 'IMPRESSION':
                impression = text['#text']
        consensus = is_consensus(codes)
        samples.append({
            'code':c,
            'history':history,
            'impression':impression,
            'consensus':consensus
        })

    df = pd.DataFrame(samples)

    disease_map = {
        'cough':"786.2",
        'fever':"780.6",
        'pneumonia':"486",
        'uti':"599.0",
        'vesicoureteral_reflux':"593.70",
    }
    data = {}
    for disease, code in disease_map.items():
        data[f'{disease}_history'] = df[df.code==code].history.tolist()
        data[f'{disease}_impression'] = df[df.code==code].impression.tolist()
    for group, sub_df in [('all',df), ('cough',df[df.code==disease_map['cough']])]:
        data[f'{group}_consensus_history'] = sub_df[sub_df.consensus].history.tolist()
        data[f'{group}_consensus_impression'] = sub_df[sub_df.consensus].impression.tolist()
        data[f'{group}_no_consensus_history'] = sub_df[~sub_df.consensus].history.tolist()
        data[f'{group}_no_consensus_impression'] = sub_df[~sub_df.consensus].impression.tolist()

    save_output_json(data, NAME)

def process_reddit_humor():
    """
    Jokes are downloaded from the dev and test splits of the dataset. We clean teh text
    and split the dataset according to whether they are labeled as funny.
    """

    NAME = 'reddit_humor'
    URLS = {'dev':'https://raw.githubusercontent.com/orionw/RedditHumorDetection/master/data/dev.tsv',
            'test':'https://raw.githubusercontent.com/orionw/RedditHumorDetection/master/data/dev.tsv'}
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    for dataset, url in URLS.items():
        filename = f'{dataset}.tsv'
        download_file(url, directory, filename)

    files = glob.glob(f'{directory}/*.tsv')
    data = defaultdict(list)

    df = pd.DataFrame()
    for file in files:
        df = df.append(pd.read_csv(file, names=['index', 'funny', 'type', 'text'], encoding='latin-1'))
    df = df.drop(['index', 'type'], axis=1)

    data = {}
    def process(text):
        return encode_ascii(text.replace('_____', ' '))
    data['funny'] = df[df['funny'] == 1]['text'].apply(process).tolist()
    data['unfunny'] = df[df['funny'] == 0]['text'].apply(process).tolist()

    save_output_json(data, NAME)

def process_reddit_stress():
    """
    Reddit posts are downloaded from a GitHub mirror. We split the
    post text based on which subreddit they are posted on (related to PTSD, anxiety,
    or stress generally).
    """
    
    NAME = 'reddit_stress'
    URL = 'http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)
    
    get_df = lambda split: pd.read_csv(join(directory, f'dreaddit-{split}.csv'), encoding='latin-1')
    dfs = [get_df(split) for split in ('train','test')]
    df = pd.concat(dfs)

    topic2subreddits = {
        'abuse':('domesticviolence','survivorsofabuse'),
        'anxiety':('anxiety',),
        'stress':('stress',),
        'financial':('almosthomeless','assistance','food_pantry','homeless'),
        'ptsd':('ptsd',),
        'social':('relationships',),
    }

    data = {}
    for topic, subreddits in topic2subreddits.items():
        posts = []
        for subreddit in subreddits:
            posts.extend(df[df['subreddit'] == subreddit].text.tolist())
        data[topic] = posts

    save_output_json(data, NAME)

def process_reuters_authorship():
    """
    Reuters articles are downloaded from the UCI repository. The articles are
    split according to author.
    """

    NAME = 'reuters_authorship'
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)
    
    files = glob.glob(f'{directory}/**/**/*.txt')
    data = defaultdict(list)
    for file in files:
        author = file.split('/')[3]
        with open(file, 'r') as f:
            text = f.read()
            data[author].extend(split_truncate_(text))
    
    save_output_json(data, NAME)

def process_riddles():
    """
    The 3000 most common English words are manually copied from a 
    website. Words with between 5 and 8 characters are kept. We create
    two popular riddles. First, we split words based on whether they have
    a duplicate character. We exclude any words with multiple "doubles" or
    more than 2 of any character. This is the "green glass door" riddle.
    Second, we split words based on whether they have the letter T. This
    is the "grandma doesn't like tea" riddle.
    """
    
    NAME = 'riddles'
    WORD_FILE = 'manual/riddles/words.txt'

    with open(WORD_FILE, 'r') as f:
        all_words = [s.strip() for s in f.readlines() if len(s) in range(5, 9)]

    def valid_task_word(w):
        char_counts = Counter(w)
        no_double_double = len([v for v in char_counts.values() if v >= 2]) <= 1
        no_triple = max(char_counts.values()) <= 2
        return no_double_double and no_triple
    ggd_words = [w for w in all_words if valid_task_word(w)]

    def is_doubled(w):
        return max(Counter(w).values()) == 2

    ggd_pos = [w for w in ggd_words if is_doubled(w)]
    ggd_neg = [w for w in ggd_words if not is_doubled(w)]
    data = {'ggd_pos':ggd_pos, 'ggd_neg':ggd_neg}
    
    def contains_t(w):
        return 't' in w

    t_pos = [w for w in all_words if not contains_t(w)]
    t_neg = [w for w in all_words if contains_t(w)]
    data['t_pos'] = t_pos
    data['t_neg'] = t_neg
    
    save_output_json(data, NAME)

def process_scotus_cases():
    """
    Supreme Court cases are downloaded from a GitHub repository. We identify
    state/federal parties by manually defining keywords. We split based on the winning
    party, the identity of each party, and the type of decision. We then define
    several time periods and relevant political eras and split decisions accordingly.
    Finally, we split according to the ruling's policy area and how it changes
    over time.
    """

    NAME = 'scotus_cases'
    URL = 'https://raw.githubusercontent.com/smitp415/CSCI_544_Final_Project/main/clean_data.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)
    
    STATE_FILE = join(MANUAL_FOLDER, NAME, 'states.txt')

    STATES = [s.strip() for s in open(STATE_FILE, 'r').readlines()]
    TERRITORIES = ['District of Columbia', 'Puerto Rico']
    STATE_NAMES = STATES + [f'State of {state}' for state in STATES] + [f'Commonwealth of {state}' for state in STATES]
    S_T_NAMES = STATE_NAMES + TERRITORIES
    S_T_NAMES = S_T_NAMES + [f'{n}, et al.' for n in S_T_NAMES]

    df = pd.read_csv(join(directory, filename), index_col=0)
    df['facts'] = df['facts'].apply(strip_tags)
    df['first_party_winner'] = df['first_party_winner'].apply(bool)

    US_NAMES = ('United States of America', 'United States', 'United States, et al.')
    data = {
        'first_party_win':df[df.first_party_winner].facts.tolist(),
        'first_party_lose':df[~df.first_party_winner].facts.tolist(),
        'dispo_reversed_remanded':df[df.disposition=='reversed/remanded'].facts.tolist(),
        'dispo_affirmed':df[df.disposition=='affirmed'].facts.tolist(),
        'dispo_reversed':df[df.disposition=='reversed'].facts.tolist(),
        'dispo_vacated_remanded':df[df.disposition=='vacated/remanded'].facts.tolist(),
        'decision_majority':df[df.decision_type=='majority opinion'].facts.tolist(),
        'decision_plurality':df[df.decision_type=='plurality opinion'].facts.tolist(),
        'decision_per_curiam':df[df.decision_type=='per curiam'].facts.tolist(), # unanimous
        'party_1st_us':df[df.first_party.isin(US_NAMES)].facts.tolist(),
        'party_2nd_us':df[df.second_party.isin(US_NAMES)].facts.tolist(),
        'party_neither_us':df[~df.first_party.isin(US_NAMES)&~df.second_party.isin(US_NAMES)].facts.tolist(),
        'party_1st_state':df[df.first_party.isin(S_T_NAMES)].facts.tolist(),
        'party_2nd_state':df[df.second_party.isin(S_T_NAMES)].facts.tolist(),
        'party_neither_state':df[~df.first_party.isin(S_T_NAMES)&~df.second_party.isin(S_T_NAMES)].facts.tolist(),
    }

    TERMS = {'1789-1850', '1850-1900', '1900-1940', '1940-1955'}

    for term in TERMS:
        data[f'era_{term}'] = df[df.term==term].facts.tolist()

    # issues over time
    eras = {
        'chief_warren':(1953, 1969),
        'chief_burger':(1969, 1986),
        'chief_rehnquist':(1986, 2005),
        'chief_roberts':(2005, 2021),
        'pres_obama':(2009, 2017),
        'pres_trump':(2017, 2021),
    }

    annual_df = df[~df.term.isin(TERMS)]
    annual_df['term'] = annual_df['term'].apply(int)

    for era, (start, end) in eras.items():
        data[f'era_{era}'] = annual_df[annual_df.term.between(start, end)].facts.tolist()

    area_map = {
        'criminal':'Criminal Procedure', # roberts vs. rehnquist
        'civil':'Civil Rights', # warren vs burger on civil rights
        'economic':'Economic Activity',  # roberts vs. rehnquist
        'judicial':'Judicial Power', # burger vs rehnquist
        'due_process':'Due Process', # roberts vs. rehnquist
        'federalism':'Federalism',
        'privacy':'Privacy',
        'unions':'Unions',
        'tax':'Federal Taxation',
    }

    time_areas = [
        ('chief_roberts','chief_rehnquist','criminal'),
        ('chief_burger','chief_warren','civil'),
        ('chief_roberts','chief_rehnquist','economic'),
        ('chief_burger','chief_warren','judicial'),
        ('chief_roberts','chief_rehnquist','due_process'),
        ('chief_roberts','chief_rehnquist','federalism'),
        ('chief_roberts','chief_rehnquist','privacy'),
        ('chief_roberts','chief_rehnquist','unions'),
        ('chief_roberts','chief_rehnquist','tax'),
    ]

    for era1, era2, issue in time_areas:
        issue_area = area_map[issue]
        start1, end1 = eras[era1]
        data[f'issue_{issue}_era_{era1}'] = annual_df[annual_df.term.between(start1, end1) & (annual_df.issue_area==issue_area)].facts.tolist()
        start2, end2 = eras[era2]
        data[f'issue_{issue}_era_{era2}'] = annual_df[annual_df.term.between(start2, end2) & (annual_df.issue_area==issue_area)].facts.tolist()

    save_output_json(data, NAME)

def process_short_answer_scoring():
    """
    Short answers are downloaded from a GitHub mirror of the dataset. We consider
    only responses to essay set 1. The two scores are averaged and binned into
    good (>= 2.5), medium (1.5-2.5), and bad (<1.5).
    """

    NAME = 'short_answer_scoring'
    URL = 'https://raw.githubusercontent.com/abdelrahmanelnaka/AraScore-Dataset/master/Question%201DataSet.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.tsv'
    download_file(URL, directory, filename)

    df = pd.read_csv(join(directory, filename), encoding='utf-8')
    df = df[df.EssaySet == 1]
    df['average_score'] = df[['Score1', 'Score2']].mean(axis=1)
    good_answers = df[df.average_score >= 2.5].EssayText.tolist()
    medium_answers = df[(1.5 <= df.average_score) & (df.average_score < 2.5)].EssayText.tolist()
    bad_answers = df[df.average_score < 1.5].EssayText.tolist()

    data = {
        'good_answers':good_answers,
        'medium_answers':medium_answers,
        'bad_answers':bad_answers,
    }
    
    save_output_json(data, NAME)

def process_stock_news():
    """
    Headlines are downloaded from a GitHub mirror. We clean the text and divide
    the samples based on whether the DOW rose or fell that day.
    """

    NAME = 'stock_news'
    URL = 'https://raw.githubusercontent.com/ShravanChintha/Stock-Market-prediction-using-daily-news-headlines/master/Combined_News_DJIA.csv'
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)

    df = pd.read_csv(join(directory, filename))

    data = defaultdict(list)

    def clean(s):
        try:
            return str(eval(s)).lstrip('b')
        except:
            return str(s).lstrip('b')

    for col in df.columns:
        if "Top" in col:
            data['down'].extend(df[df['Label'] == 0][col].apply(clean).tolist())
            data['up'].extend(df[df['Label'] == 1][col].apply(clean).tolist())

    save_output_json(data, NAME)

def process_suicide_notes():
    """
    Reddit posts are downloaded from a GitHub repository. The post title and body
    are combined to form the text samples. Samples are split based on whether they
    were posted in a suicide-related Subreddit.
    """

    NAME = 'suicide_notes'
    URL = 'https://raw.githubusercontent.com/hesamuel/goodbye_world/master/data/data_for_model_2.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)
    
    df = pd.read_csv(join(directory, filename))
    df['all_text'] = df['title'] + ' ' + df['selftext']
    data = {
        'depression':df[df.is_suicide == 0].all_text.tolist(),
        'suicide':df[df.is_suicide == 1].all_text.tolist(),
    }

    save_output_json(data, NAME)

def process_times_india_headlines():
    """
    Headlines are downloaded from a Dataverse mirror. We use the first
    1000 headlines in each year as samples.
    """

    NAME = 'times_india_headlines'
    URL = 'https://dataverse.harvard.edu/api/access/datafile/6175512'
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.csv'
    download_file(URL, directory, filename)
    
    df = pd.read_csv(join(directory, filename))
    df['year'] = df['publish_date'].astype(str).str[:4].astype(int)
    
    data = {}
    for year in df['year'].unique():
        data[str(year)] = df[df['year']==year]['headline_text'].tolist()[:1000]
    
    save_output_json(data, NAME)

def process_trial_deception():
    """
    Trial testimonies are downloaded from the author's website. The testimonies are
    are divided based on whether they are considered truthful.
    """

    NAME = 'trial_deception'
    URL = 'http://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip'
    
    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip(URL, directory)

    files = glob.glob(f'{directory}/Real-life_Deception_Detection_2016/Transcription/**/*.txt')
    truth, lie = [], []
    for file in files:
        with open(file, 'r') as f:
            contents = encode_ascii(f.read())
            if 'truth' in file:
                truth.append(contents)
            else:
                lie.append(contents)
    
    data = {
        'truth':truth,
        'lie':lie
    }
    
    save_output_json(data, NAME)


def process_tweet_gender():
    """
    Tweets are downloaded from a GitHub mirror. We consider only Tweets which have a 
    100% rating for confidence. The tweets are split into male and female gender groupings.
    """

    NAME = 'tweet_gender'
    URL = 'https://raw.githubusercontent.com/tranctan/Gender-Classification-based-on-Twritter-textual-data/master/gender_dataset.csv'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'{NAME}.tsv'
    download_file(URL, directory, filename)

    df = pd.read_csv(join(directory, filename), encoding='latin-1')
    df = df[df['gender:confidence'] == 1.00]
    df['text'] = df['text'].apply(encode_ascii)
    male_tweets = df[df.gender=='male'].text.tolist()
    female_tweets = df[df.gender=='female'].text.tolist()

    data = {
        'male_tweets':male_tweets,
        'female_tweets':female_tweets
    }
    
    save_output_json(data, NAME)

def process_tweet_rumor():
    """
    Twitter IDs are downloaded from Zenodo archive and collects 300 Tweets for each
    rumor using the Twitter API. Tweets are evenly divided into early, middle, and
    late thirds based on the publication time.
    """

    NAME = 'tweet_rumor'

    import scrape_twitter_rumors

    data = scrape_twitter_rumors.scrape()
    
    save_output_json(data, NAME)

def process_twitter_bots():
    """
    Annotated Tweets are downloaded from an online repository. We filter out
    non-English Tweets using the guess_langauge package and exclude any Tweets
    that containt eh words "fake" or "bot." For Tweets from traditional bots, 
    social bots, and humans, we sample 20,000 of each.
    """

    NAME = 'twitter_bots'

    from guess_language import guess_language

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    download_zip('https://botometer.osome.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv.zip', directory)

    trad_bot_path = join(directory, 'datasets_full.csv', 'traditional_spambots_1.csv.zip')
    social_bot_path = join(directory, 'datasets_full.csv', 'social_spambots_2.csv.zip')
    human_path = join(directory, 'datasets_full.csv', 'genuine_accounts.csv.zip')

    extract_zip(trad_bot_path, directory)
    extract_zip(social_bot_path, directory)
    extract_zip(human_path, directory)

    df_trad_bot = pd.read_csv(join(directory, 'traditional_spambots_1.csv', 'tweets.csv'), encoding='latin-1', nrows=50000)
    df_trad_bot['type'] = 'trad_bot'
    df_social_bot = pd.read_csv(join(directory, 'social_spambots_2.csv', 'tweets.csv'), encoding='latin-1', nrows=50000)
    df_social_bot['type'] = 'social_bot'
    df_human = pd.read_csv(join(directory, 'genuine_accounts.csv', 'tweets.csv'), encoding='latin-1', nrows=50000)
    df_human['type'] = 'human'

    df = pd.concat([df_trad_bot, df_social_bot, df_human])
    exclude_words = ['fake', 'bot']
    for word in exclude_words:
        df = df[~df['text'].str.lower().str.contains(word).astype(bool)]
    df['language'] = df.text.apply(guess_language)

    df=df[df.language == 'en']

    def clean(text):
        try:
            return encode_ascii(codecs.unicode_escape_decode(text)[0])
        except:
            return np.nan
        
    df['text'] = df['text'].apply(clean)
    df = df[~df['text'].isna()]

    data = {
        'trad_bot':df[df.type == 'trad_bot'].sample(20000, random_state=0).text.tolist(),
        'social_bot':df[df.type == 'social_bot'].sample(20000, random_state=0).text.tolist(),
        'human':df[df.type == 'human'].sample(20000, random_state=0).text.tolist(),
    }
    
    save_output_json(data, NAME)

def process_twitter_misspellings():
    """
    Assorted Tweets are downloaded from a Gihtub mirro. We manually identify eight
    common mispellings of words ("your", "with", "that", "going", "know", "you", "what",
    "the") and divide samples based on whether they contain each mispelling.
    """
    
    from datasets import load_dataset

    NAME = 'twitter_misspellings'

    dataset = load_dataset("sentiment140")

    dfs = [dataset[split].to_pandas()
        for split in ['test','train']]
    df = pd.concat(dfs, axis=0)

    normalization_pairs = [
        ('your', [' ur '], [' your ', ' you\'re ']),
        ('with', [' wit '], [' with ']),
        ('that', [' dat ', ' dats '], [' that ']),   
        ('going', [' goin '], ['going ']),
        ('know', [' kno '], [' know ']),
        ('you', [' u '], [' you ']),
        ('what', [' wut ', ' wat '], [' what ']),
        ('the', [' da '], [' the '])
    ]

    data = {}
    for group, misspell, proper in normalization_pairs:
        data[group + '_misspell'] = df[df['text'].str.contains('|'.join(misspell))]['text'].tolist()
        data[group + '_proper'] = df[df['text'].str.contains('|'.join(proper))]['text'].tolist()
    

    save_output_json(data, NAME)

def process_twitter_sentiment140():
    """
    Assorted Tweets are downloaded from a mirror and the text is
    used as-is for clustering.
    """

    from datasets import load_dataset

    NAME = 'twitter_sentiment140'

    dataset = load_dataset("sentiment140")

    dfs = [dataset[split].to_pandas()
        for split in ['test','train']]
    df = pd.concat(dfs, axis=0)

    snippets = df['text'].tolist()
    sentences = sentence_tokenize(snippets)

    save_unlabeled_json(sentences, NAME)

def process_un_debates():
    """
    Debate transcripts are downloaded from the Dataverse reproduction package. Samples
    are divided based on the country and year of the snippet. First, we isolate samples
    from Russia, China, and the United States and specify 3 time periods of interest.
    Next, we divide all samples by the decade. Finally, we create distributions for
    19 countries of interest.
    """

    NAME = 'un_debates'

    files = glob.glob(f'{MANUAL_FOLDER}/{NAME}/TXT/**/*.txt')

    df = pd.DataFrame()
    for file in tqdm(files):
        country, year = re.findall('(\w+)_\d+_(\d{4}).txt', file)[0]
        with open(file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines if l.strip()]
        snippets = lines[0:1]
        for line in lines[1:]:
            if not line[0].isupper():
                snippets[-1] += ' ' + line
            else:
                snippets.append(line)
        df = df.append(pd.DataFrame({'country':country,'year':int(year),'snippets':snippets}))
    
    data = {}

    country_years = {
        'russia_2k_08':('RUS',2000,2007),
        'russia_08_12':('RUS',2008,2012),
        'russia_post_12':('RUS',2013,2020),
        'china_2k_13':('CHN',2000,2013),
        'china_13_16':('CHN',2014,2016),
        'china_post_16':('CHN',2017,2020),
        'us_bush':('USA',2001,2008),
        'us_obama':('USA',2009,2016),
        'us_trump':('USA',2017,2021),
    }

    for pair, (country, start_year, end_year) in country_years.items():
        data[pair] = df[(df.country==country)&df.year.between(start_year, end_year)].snippets.tolist()

    decades = {
        '80s':(1980,1989),
        '90s':(1990,1999),
        '2ks':(2000,2009),
        '10s':(2010,2019),
    }

    decade_countries = {
        'israel':'IRL',
        'pales':'PSE',
        'egypt':'EGY',
        'jordan':'JOR',
        'lebanon':'LBN',
        'iraq':'IRQ',
        'iran':'IRN',
        'syria':'SYR',
        'sudan':'SDN',
        'bahrain':'BHR',
        'kuwait':'KWT',
        'oman':'OMN',
        'qatar':'QAT',
        'saudi':'SAU',
        'uae':'ARE',
        'pak':'PAK',
        'india':'IND',
        'japan':'JPN',
        'korea':'KOR',
    }

    for decade, (start_year, end_year) in decades.items():
        for country, country_code in decade_countries.items():
            data[f'{country}_{decade}'] = df[(df.country==country_code)&df.year.between(start_year, end_year)].snippets.tolist()
   
    save_output_json(data, NAME)


def process_unhealthy_conversations():
    """
    Conversation transcripts are downloaded from the official Github repository. For each
    annotated attribute, we split the dataset based on whether that form of 
    unhealthy conversation is present in the sample.
    """

    NAME = 'unhealthy_conversations'
    URLS = {'test':'https://raw.githubusercontent.com/conversationai/unhealthy-conversations/main/corpus/test.csv',
            'train':'https://raw.githubusercontent.com/conversationai/unhealthy-conversations/main/corpus/train.csv',
            'val':'https://raw.githubusercontent.com/conversationai/unhealthy-conversations/main/corpus/val.csv'}

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    for dataset, url in URLS.items():
        filename = f'{dataset}.csv'
        download_file(url, directory, filename)
    
    files = glob.glob(f'{directory}/*.csv')
    data = defaultdict(list)
    
    attributes = ['antagonize', 'condescending', 'dismissive', 'generalisation', 'generalisation_unfair', 'healthy', 'hostile', 'sarcastic']
    for file in files:
        df = pd.read_csv(file)
        for attr in attributes:
            data[attr].extend(df[df[attr] == 1]['comment'].tolist())
            data['not_' + attr].extend(df[df[attr] == 0]['comment'].tolist())
    
    save_output_json(data, NAME)

def process_urban_dictionary():
    """"
    Urban Dictionary entries are downloaded from Kaggle. Definitions are split into groups
    representing the top 1, 5, and 10 percent of definitions ranked by both upotes and
    downvotes; we sample 10,000 from each and create a control distribution by
    randomly sampling 10,000 definitions from all entries.
    """

    NAME = 'urban_dictionary'
    FILE = 'manual/urban_dictionary/urbandict-word-defs.csv'

    df = pd.read_csv(FILE,error_bad_lines=False)
    top_1_up = np.percentile(df.up_votes, 99)
    top_5_up = np.percentile(df.up_votes, 95)
    top_10_up = np.percentile(df.up_votes, 90)
    top_1_down = np.percentile(df.down_votes, 99)
    top_5_down = np.percentile(df.down_votes, 95)
    top_10_down = np.percentile(df.down_votes, 90)

    df = df.dropna(subset=['definition'])

    data = {
        'control':df.definition.sample(10000, random_state=0).tolist(), # randomly sample 10k
        'top_1_up':df[df.up_votes>top_1_up].definition.sample(10000, random_state=0).tolist(),
        'top_5_up':df[(df.up_votes>top_5_up)&(df.up_votes<=top_1_up)].sample(10000, random_state=0).definition.tolist(),
        'top_10_up':df[(df.up_votes>top_10_up)&(df.up_votes<=top_5_up)].sample(10000, random_state=0).definition.tolist(),
        'top_1_down':df[df.down_votes>top_1_down].sample(10000, random_state=0).definition.tolist(),
        'top_5_down':df[(df.down_votes>top_5_down)&(df.down_votes<=top_1_down)].sample(10000, random_state=0).definition.tolist(),
        'top_10_down':df[(df.down_votes>top_10_down)&(df.down_votes<=top_5_down)].sample(10000, random_state=0).definition.tolist(),
    }

    save_output_json(data, NAME)

def process_wikitext():
    """
    The Wikipedia snippets are loaded from hugginface. We remove any samples that are
    empty or start with '=' (which represent headings); samples are tokenized at the
    sentence level and used for clustering.
    """

    NAME = 'wikitext'

    from datasets import load_dataset

    dataset = 'wikitext'
    config = 'wikitext-2-v1-raw'
    df = load_dataset(dataset, config)['train'].to_pandas()
    
    snippets = df['text'].tolist()
    
    # filter
    snippets = list(map(lambda x: x.strip(),
        filter(lambda x: not x.strip().startswith('=') and len(x.strip()) > 0,
            snippets)))

    sentences = sentence_tokenize(snippets)
    
    save_unlabeled_json(sentences, NAME)

def process_yc_startups():
    """
    YCombinator company descriptions are downloaded from a 2022 scrape on Github. Only
    companies are long descriptions are preserved. Companies are split according to
    founder characteristics, year, "top company" designation, operating status, and
    location.
    """

    NAME = 'yc_startups'
    URL = 'https://raw.githubusercontent.com/akshaybhalotia/yc_company_scraper/main/data/combined_companies_data.json'

    directory = f'{DOWNLOAD_FOLDER}/{NAME}'
    filename = f'startups.csv'
    download_file(URL, directory, filename)

    df = pd.read_json(join(directory, filename))
    df = df[df['long_description'].astype(bool)] # keep only dfs with long descriptions
    template = """{name} - {one_liner}

    {long_description}"""
    make_desc = lambda row: template.format(**row)
    df['all_text'] = df.apply(make_desc, axis=1)

    data = {}
    for group in ('black','latinx','women'):
        data[f'highlight_{group}_yes'] = df[df[f'highlight_{group}']]['all_text'].tolist()
        data[f'highlight_{group}_no']  = df[~df[f'highlight_{group}']]['all_text'].tolist()
    for n in range(6, 23):
        year = str(n).zfill(2)
        data[f'year_{year}'] = df[df['batch'].str.contains(year)]['all_text'].tolist()
    data['top_yes'] = df[df['top_company']]['all_text'].tolist()
    data['top_no']  = df[~df['top_company']]['all_text'].tolist()
    for status in ('Active','Inactive','Acquired','Public'):
        samples = df[df['status'] == status]['all_text'].tolist()
        data[f'status_{status.lower()}'] = samples
    loc_dc = df[~df['location'].isna()]
    data['location_usa'] = loc_dc[loc_dc['location'].str.contains('USA')]['all_text'].tolist()
    data['location_not_usa'] = loc_dc[~loc_dc['location'].str.contains('USA')]['all_text'].tolist()

    save_output_json(data, NAME)

"""
******
Driver
******
"""

def main():

    datasets = [func[8:] for func in globals().keys() if 'process_' in func]

    parser = argparse.ArgumentParser(
                    prog = 'Pull Data',
                    description = 'Processes the datasets comprising the D3 benchmark.')

    parser.add_argument('--dataset',choices=datasets,type=str)
    parser.add_argument('--cleanup',action='store_true')
    parser.add_argument('--access',action='store_true')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # make output folder if doesn't exist

    args = parser.parse_args()

    if args.dataset:
        dataset = args.dataset
        assert dataset in datasets, 'Not a valid dataset'
        process_func = globals().get(f'process_{dataset}')
        process_func()
        
    if args.access:
        schema = yaml.load(open('schema/datasets.yaml'), Loader=SafeLoader)
        datasets = [dataset for dataset, metadata in schema.items() if metadata['status']=='accessible']

        pbar = tqdm(datasets)
        for dataset in pbar:
            pbar.set_description(f'processing {dataset}')
            process_func = globals().get(f'process_{dataset}')
            process_func()
    
    if args.cleanup:
        delete_downloads()

if __name__ == '__main__':
    main()