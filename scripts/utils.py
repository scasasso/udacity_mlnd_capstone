import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def get_data(path='../data', subset=True):

    # Read and merge the data
    df_fake = pd.read_csv(os.path.join(path, 'fake.csv'))
    df_real = pd.read_csv(os.path.join(path, 'real.csv'))
    df_fake['label'] = 1
    df_real['label'] = 0
    df_tot = pd.concat([df_fake, df_real])
    df_tot = skshuffle(df_tot, random_state=42)

    # Add text/title length
    df_tot['text_length'] = df_tot.apply(lambda x: len(x['text']) if not pd.isnull(x['text']) else -1, axis=1)
    df_tot['title_length'] = df_tot.apply(lambda x: len(x['title']) if not pd.isnull(x['title']) else -1, axis=1)

    if subset:
        # Subset the data
        df_tot = df_tot.dropna(subset=['text'])
        df_tot = df_tot[df_tot['text_length'] >= 10].copy()  # as they were nans
        df_tot = df_tot[df_tot['language'] == 'english']

    return df_tot


def plot_var(df_in, var_name, x_range, x_binwidth, x_tickwidth, y_range=(0.0001, 10.), ext='.pdf'):

    # Get fakes
    xfake = df_in.loc[df_in['label'] == 1, var_name]
    plt.hist(xfake, label='fake', log='y', histtype='step', color='red', linewidth=1.2, fill=False, alpha=0.75,
             bins=np.arange(x_range[0], x_range[1], x_binwidth), weights=np.ones_like(xfake) / len(xfake))

    # Get reals
    xreal = df_in.loc[df_in['label'] == 0, var_name]
    plt.hist(xreal, label='real', log='y', histtype='step', color='green', linewidth=1.2, fill=False, alpha=0.75,
             bins=np.arange(x_range[0], x_range[1], x_binwidth), weights=np.ones_like(xreal) / len(xreal))

    plt.legend(loc='upper right')
    plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
    plt.xlabel(var_name)
    plt.ylabel('Arbitrary Units')
    plt.xticks(np.arange(x_range[0], x_range[1], x_tickwidth))
    plt.savefig(var_name + ext)
    plt.close('all')


def get_hist(ax):
    n, bins = [], []
    x1 = None
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        bins.append(x0)  # left edge of each bin
    if x1 is not None:
        bins.append(x1)  # also get right edge of last bin

    return n, bins


def get_scores(y_pred, y_test):

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    return {'f1_score': f1, 'accuracy': acc, 'precision': pre, 'recall': rec}


def scores_str(score_dict):

    score_str = ''
    for name, score in score_dict.items():
        score_str += ' {0} = {1:.3f}\n'.format(name, score)

    return score_str


def chop_text_rnd(text, frac=1.):
    if random.random() > frac:
        return text

    sentence_list = text.split('.')

    if len(sentence_list) <= 2:
        return text

    sentence_list.pop(random.randint(0, len(sentence_list) - 1))

    return '.'.join(sentence_list)
