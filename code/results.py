"""Functions for producing model results.

Contains functions for producing the plots and other results of the model.
"""

import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from unidecode import unidecode
import matplotlib.pyplot as plt
from re import finditer
from pprint import pprint
from time import time
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_curve

def pers_pred_seq(tweet_list, step,  pipe, vect, tfidf, lsa, rfr):
    """Run model and return results.

    INPUT: list of strings, int, list of pipelines, list of vectorizors, list
            of tfidf tranformers list of lsa transformers, list of random
            forest regressors, 7 in each list.
    OUTPUT: list of values suitable for plotting plus results from all decision
            tree estimators in the forest with their standard deviation for
            increasing numbers of tweets.
    """
    l_results = []
    l_pred_std = []
    l_tpreds = []
    df = pd.DataFrame(columns=featurize_tweet(tweet_list[0])[1])
    for i in range(1+len(tweet_list)/step):
        add_tweets = tweet_list[step*i:step*(i+1)]
        for tweet in add_tweets:
            features, cols = featurize_tweet(tweet)
            df = df.append(pd.DataFrame([features], columns=cols))
        keep_cols = list(df.columns)
        keep_cols.remove('tweet')
        keep_cols.remove('hashtags')
        keep_cols.remove('tw_text')
        compdf = df[keep_cols].sum()
        compdf['tw_text'] = ' '.join(df['tw_text'].tolist())
        y_pred_l = pipe.predict(compdf['tw_text'])
        l_results.append(y_pred_l)
        tpreds = [[] for i in range(7)]
        pred_std = []
        for j in range(7):
            X = lsa[j].transform(tfidf[j].transform(vect[j].transform(compdf['tw_text'])))
            for tree in rfr[j].estimators_:
                tpreds[j].append(tree.predict(X))
            pred_std.append(np.array(tpreds[j]).std())
        l_pred_std.append(pred_std)
        l_tpreds.append()
    return l_results, l_pred_std, l_tpreds
