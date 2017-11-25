import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import utils

if __name__ == '__main__':

    # Get DataFrame
    df = utils.get_data(subset=False)

    # Plot histograms
    plots = [{'var_name': 'text_length', 'x_range': (0, 10000), 'x_binwidth': 100, 'x_tickwidth': 1000},
             {'var_name': 'title_length', 'x_range': (0, 200), 'x_binwidth': 10, 'x_tickwidth': 20},
             {'var_name': 'participants_count', 'x_range': (0, 20), 'x_binwidth': 1, 'x_tickwidth': 2},
             {'var_name': 'replies_count', 'x_range': (0, 20), 'x_binwidth': 1, 'x_tickwidth': 2},
             {'var_name': 'comments', 'x_range': (0, 10), 'x_binwidth': 1, 'x_tickwidth': 1},
             {'var_name': 'likes', 'x_range': (0, 20), 'x_binwidth': 1, 'x_tickwidth': 2}]

    for pl_specs in plots:
        utils.plot_var(df, **pl_specs)

    # Other plots
    # Language breakdown
    ax = df.loc[df['label'] == 1, 'language'].value_counts().plot(kind='bar', alpha=0.75,
                                                                  legend=False, log='y',
                                                                  title='Fake news data: language breakdown')
    plt.ylabel('Number of articles')
    plt.tight_layout()
    plt.savefig('fake_languages.pdf')
    plt.close('all')

    # Word clouds
    sw_file_path = '../data/stopwords.txt'
    sw_list = json.load(open(sw_file_path, 'r'))
    i_row = 56  # Random one

    wcloud_fake = WordCloud(stopwords=sw_list).generate(df.iloc[np.where((~pd.isnull(df['text'])) & (df['label'] == 1))[0][i_row],
                                                        df.columns.get_loc('text')])
    plt.imshow(wcloud_fake, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wcloud_fake.pdf')
    plt.close('all')

    wcloud_real = WordCloud(stopwords=sw_list).generate(df.iloc[np.where((~pd.isnull(df['text'])) & (df['label'] == 0))[0][i_row],
                                                        df.columns.get_loc('text')])
    plt.imshow(wcloud_real, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wcloud_real.pdf')
    plt.close('all')




