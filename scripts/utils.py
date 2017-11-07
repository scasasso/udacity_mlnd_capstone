import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as skshuffle


def get_data():

    # Read and merge the data
    df_fake = pd.read_csv('../data/fake.csv')
    df_real = pd.read_csv('../data/real.csv')
    df_fake['label'] = 1
    df_real['label'] = 0
    df_tot = pd.concat([df_fake, df_real])
    df_tot = skshuffle(df_tot)

    # Subset the data
    df_tot = df_tot.dropna(subset=['text'])
    df_tot = df_tot[df_tot['language'] == 'english']

    return df_tot


def plot_var(df_in, var_name, x_range, x_binwidth, x_tickwidth, y_range=(0.0001, 10.)):

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
    plt.savefig(var_name + '.png')
    plt.close('all')
