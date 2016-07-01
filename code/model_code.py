import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from unidecode import unidecode
import matplotlib.pyplot as plt
from re import finditer
from pprint import pprint
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

dir_e = 'pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23'
files_e = os.listdir(dir_e)

tweets = []
for fil_e in files_e:
    if fil_e[-3:] == 'xml':
        tree = ET.parse(dir_e+'/'+fil_e)
        root = tree.getroot()
        tweets.append([fil_e[:-3], []])
        for CDATA in root.findall('document'):
            tweets[-1][1].append(CDATA.text[:-2])

ydf = pd.read_csv(dir_e + '/truth.txt', sep=':::', header=None, names=['userid', 'gender', 'age_group', 'extroverted', 'stable', 'agreeable', 'conscientious', 'open'])
ydf['userid'] = ydf['userid'].map(lambda x: x[:8])

def camel_case_split(word):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', word)
    return [m.group(0) for m in matches]

df = pd.DataFrame(columns=('user', 'tweet', 'tweet_ct', 'RT', 'word_ct', 'link_ct', 'at_ct', 'hashtags', 'tag_ct', 'music_share', 'excl_ct', 'ques_ct', 'ellipsis', 'I_my', 'We_our', 'you_your', 'loud_words', 'prop_names', 'CAP_ct', 'tw_text'))

for user in tweets:
    for tweet in user[1]:
        tokens = unidecode(tweet).split()
        RT = 0
        link_ct = 0
        at_ct = 0
        hashtags = []
        tag_ct = 0
        music_share = 0
        excl_ct = 0
        ques_ct = 0
        ellipsis = 0
        I_my = 0
        We_our = 0
        you_your = 0
        loud_words = 0
        prop_names = 0
        CAP_ct = 0
        rem_tok = []
        add_tok = []
        for token in tokens:
            http_loc = token.find('http')
            if token[:2] == 'RT':
                RT += 1
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
            if token in {'#NowPlaying', '(Courtesy', '(Feat.'}:
                music_share += 1
            if token[0] == '#' and len(token) > 1:
                if not token[1:].isdigit():
                    tag_ct += 1
                    hashtags.append(token)
                    rem_tok.append(token)
                    add_tok.extend(camel_case_split(token[1:]))
            elif token.lower() in {'i', "i'm", "i've", "i'd", "i'll", 'me', 'my', 'mine'}:
                I_my += 1
            elif token.lower() in {'we', 'our', 'us', "we'd", "we've", "we'll", 'ours'}:
                We_our +=1
            elif token.lower() in {'you', 'your', "you've", "you'd", "you'll", 'yours'}:
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
                    CAP_ct += 1
        word_ct = len(tokens)
        tw_text = ' '.join(tokens)
        df = df.append(pd.DataFrame([[user[0][:8], tweet, 1, RT, word_ct, link_ct, at_ct, hashtags, tag_ct, music_share, excl_ct, ques_ct, ellipsis, I_my, We_our, you_your, loud_words, prop_names, CAP_ct, tw_text]],
            columns=('user', 'tweet', 'tweet_ct', 'RT', 'word_ct', 'link_ct', 'at_ct', 'hashtags', 'tag_ct', 'music_share', 'excl_ct', 'ques_ct', 'ellipsis', 'I_my', 'We_our', 'you_your', 'loud_words', 'prop_names', 'CAP_ct', 'tw_text')))

compdf = df[['user', 'tweet_ct', 'RT', 'word_ct', 'link_ct', 'at_ct', 'tag_ct', 'music_share', 'excl_ct', 'ques_ct', 'ellipsis', 'I_my', 'We_our', 'you_your', 'loud_words', 'prop_names', 'CAP_ct']].groupby('user').sum()
