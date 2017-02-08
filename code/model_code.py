"""Code for generating model.

Functions in this module will load training data and build the model.
"""

import os
import xml.etree.ElementTree as ET
from re import finditer
from unidecode import unidecode
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

DATA_DIR = 'data/pan15-english'


def load_pan_xml_tweets(file_dir=DATA_DIR):
    """Parse XML files to load tweets.

    Reads in tweets from XML files. One file per user. File names are user IDs.
    Each tweet is between 'document' tags and ends with two tabs which we skip.
    INPUT: str

    OUTPUT: list
    """
    files_e = os.listdir(file_dir)
    users = []
    for fil_e in files_e:
        if fil_e[-3:] == 'xml':
            tree = ET.parse(file_dir+'/'+fil_e)
            root = tree.getroot()
            users.append([fil_e[:-3], []])
            for cdata in root.findall('document'):
                users[-1][1].append(cdata.text[:-2])
    return users


def load_pan_y(file_dir=DATA_DIR):
    """Read labels in from text file and return a DataFrame."""
    ydf = pd.read_csv(file_dir + '/truth.txt', sep=':::', header=None, names=[
        'userid', 'gender', 'age_group', 'extroverted', 'stable', 'agreeable',
        'conscientious', 'open'])
    # Reduce the length of the user IDs.
    ydf['userid'] = ydf['userid'].map(lambda x: x[:8])
    return ydf


def camel_case_split(word):
    """Split words that are camel case."""
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                       word)
    return [m.group(0) for m in matches]


def featurize_tweet(tweet):
    """Pull a bunch of features out of the tweet."""
    tokens = unidecode(tweet).split()
    retweet = 0
    link_ct = 0
    at_ct = 0
    hashtags = []
    tag_ct = 0
    music_share = 0
    excl_ct = 0
    ques_ct = 0
    ellipsis = 0
    i_my = 0
    we_our = 0
    you_your = 0
    loud_words = 0
    prop_names = 0
    cap_ct = 0
    rem_tok = []
    add_tok = []
    for token in tokens:
        http_loc = token.find('http')
        if token[:2] == 'rt':
            retweet += 1
            rem_tok.append(token)
        elif token[0] == '@' or (len(token) > 1 and token[1] == '@'):
            at_ct += 1
            rem_tok.append(token)
        elif http_loc != -1:
            link_ct += 1
            rem_tok.append(token)
            if http_loc != 0:
                token = token[:http_loc]
                add_tok.append(token)
        if token in {'#NowPlaying', '(Courtesy', '(Feat.', '#Shazam'}:
            music_share += 1
        if token[0] == '#' and len(token) > 1:
            if not token[1:].isdigit():
                tag_ct += 1
                hashtags.append(token)
                rem_tok.append(token)
                add_tok.extend([token[1:]])
                if camel_case_split(token[1:])[0] != token[1:]:
                    add_tok.extend(camel_case_split(token[1:]))
        elif token.lower() in {'i', "i'm", "i've", "i'd", "i'll", 'me', 'my',
                               'mine', 'myself'}:
            i_my += 1
        elif token.lower() in {'we', 'our', 'us', "we'd", "we've", "we'll",
                               'ours', 'ourselves'}:
            we_our += 1
        elif token.lower() in {'you', 'your', "you've", "you'd", "you'll",
                               'yours', 'yourself'}:
            you_your += 1
        if token.isupper():
            loud_words += 1
        elif token[0].isupper():
            prop_names += 1
        for c in token:
            if c == '!':
                excl_ct += 1
            if c == '?':
                ques_ct += 1
            if c == '.':
                ellipsis += 1
    for token in add_tok:
        tokens.append(token)
    for token in rem_tok:
        tokens.remove(token)
    for token in tokens:
        for c in token:
            if c.isupper():
                cap_ct += 1
    word_ct = len(tokens)
    tw_text = ' '.join(tokens)
    features = [tweet, 1, retweet, word_ct, link_ct, at_ct, hashtags, tag_ct,
                music_share, excl_ct, ques_ct, ellipsis, i_my, we_our,
                you_your, loud_words, prop_names, cap_ct, tw_text]
    columns = ['tweet', 'tweet_ct', 'RT', 'word_ct', 'link_ct', 'at_ct',
               'hashtags', 'tag_ct', 'music_share', 'excl_ct', 'ques_ct',
               'ellipsis', 'I_my', 'We_our', 'you_your', 'loud_words',
               'prop_names', 'CAP_ct', 'tw_text']
    return features, columns


def get_x_and_y(data_dir=DATA_DIR):
    """Return X and y ready for training."""
    users = load_pan_xml_tweets(data_dir)
    df = pd.DataFrame(columns=['user']+featurize_tweet(users[0][1][0])[1])
    for user in users:
        for tweet in user[1]:
            features, cols = featurize_tweet(tweet)
            df = df.append(pd.DataFrame([[user[0][:8]]+features],
                                        columns=['user']+cols))
    keep_cols = list(df.columns)
    keep_cols.remove('tweet')
    keep_cols.remove('hashtags')
    keep_cols.remove('tw_text')
    compdf = df[keep_cols].groupby('user').sum()
    compdf2 = compdf.apply(lambda x: 100.0*x/x['tweet_ct'], axis=1)
    compdf2['tw_text'] = df[['user', 'tw_text']].groupby('user').agg(' '.join)
    ydf = load_pan_y()
    combineddf = compdf2.join(ydf, how='outer')
    X = combineddf[list(compdf2.columns)]
    y = combineddf[list(ydf.columns)]

    # Convert gender to -1.0 and +1.0 so we can use a regressor for prediction.
    # It will be converted back to categorical afterwards depending on sign.
    gender_dict = {'M': -1.0, 'F': 1.0}
    y = y.replace(gender_dict)

    # Convert age ranges to floats: [1.0, 2.0, 3.0, 4.0] for regression.
    age_dict = {'18-24': 1.0, '25-34': 2.0, '35-49': 3.0, '50-XX': 4.0}
    y = y.replace(age_dict)

    return X, y


def LSA_pipe(Xtrain, ytrains, lsa_n):
    """Train a model for each y label in ytrains."""
    pl_lsa = []
    vect = []
    tfidf = []
    lsa = []
    rfr_lsa = []
    for yt in ytrains:
        vect.append(CountVectorizer())
        tfidf.append(TfidfTransformer())
        lsa.append(TruncatedSVD())
        rfr_lsa.append(RandomForestRegressor())
        pl_lsa.append(Pipeline([
            ('vect', vect[-1]),
            ('tfidf', tfidf[-1]),
            ('lsa', lsa[-1]),
            ('rfr', rfr_lsa[-1]),
            ]))
        pl_lsa[-1].set_params(rfr__max_depth=6,
                              rfr__n_estimators=100,
                              lsa__n_components=lsa_n,
                              tfidf__norm='l1',
                              tfidf__use_idf=False,
                              vect__max_df=1.0,
                              vect__max_features=5000,
                              vect__ngram_range=(1, 3),
                              vect__stop_words=None).fit(Xtrain, yt)

    return pl_lsa, vect, tfidf, lsa, rfr_lsa
