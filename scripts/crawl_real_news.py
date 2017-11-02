import os
import json
import glob
import optparse
import random
from operator import itemgetter
import pandas as pd

# Reliable sources
sources = ['nytimes', 'forbes', 'washingtonpost', 'theguardian', 'bbc', 'npr', 'reuters', 'wsj', 'bloomberg']


def parse_args():

    parser = optparse.OptionParser()
    parser.add_option('--path_to_json', type='str', help='Path to the directory with the json files')
    parser.add_option('--out_dir', type='str', default='.', help='Path to the output directory')
    parser.add_option('--max_tot', type='int', default=1000000, help='Max number of news articles')
    parser.add_option('--max_per_source', type='int', default=3000, help='Max number of news articles from each source')
    options, args = parser.parse_args()

    return options


# Same shape as the fake news dataset
def get_news_dict(in_dict):

    try:
        out_dict = {}
        out_dict['uuid'] = in_dict['uuid']
        out_dict['ord_in_thread'] = in_dict['ord_in_thread']
        out_dict['author'] = in_dict['author']
        out_dict['published'] = in_dict['published']
        out_dict['title'] = in_dict['title']
        out_dict['text'] = in_dict['text']        
        out_dict['language'] = in_dict['language']
        out_dict['crawled'] = in_dict['crawled']
        out_dict['site_url'] = in_dict['thread']['site']
        out_dict['country'] = in_dict['thread']['country']
        out_dict['domain_rank'] = in_dict['thread']['domain_rank']
        out_dict['thread_title'] = in_dict['thread']['title']
        out_dict['spam_score'] = in_dict['thread']['spam_score']
        out_dict['main_img_url'] = in_dict['thread']['main_image']
        out_dict['replies_count'] = in_dict['thread']['replies_count']
        out_dict['participants_count'] = in_dict['thread']['participants_count']
        out_dict['likes'] = in_dict['thread']['social']['facebook']['likes']
        out_dict['comments'] = in_dict['thread']['social']['facebook']['comments']
        out_dict['shares'] = in_dict['thread']['social']['facebook']['shares']
        out_dict['type'] = 'real'
    except KeyError:
        return None

    return out_dict
    

if __name__ == '__main__':

    # Get command line options
    opt = parse_args()

    # Get list of json files
    file_path_list = glob.glob(os.path.join(opt.path_to_json, '*.json'))
    random.shuffle(file_path_list)  # randomize

    # Loop
    source_dict = {}
    out_list = []
    for i, f_path in enumerate(file_path_list):

        if len(out_list) >= opt.max_tot:
            break

        if i % 10000 == 0:
            print('Progress {0}/{1}'.format(i, len(file_path_list)))

        news = json.load(open(f_path, 'r'))
        try:
            is_news = news['thread']['site_type'] == 'news'
            is_en = news['language'] == 'english'
            is_source = any([sou in news['thread']['site'] for sou in sources])
            is_ok = is_news and is_en and is_source
        except (KeyError, TypeError):
            is_ok = False

        if is_ok:
            site = news['thread']['site']
            source_dict.setdefault(site, 0)
            if source_dict[site] >= opt.max_per_source:
                continue
            news_out = get_news_dict(news)
            out_list.append(news_out)
            source_dict[site] += 1

    # Write DataFrame to output directory
    df = pd.DataFrame(out_list)
    df.to_csv(os.path.join(opt.out_dir, 'real.csv'), index=False)

    print('{} news have been selected'.format(len(df)))
    print('\nBreakdown:')
    for sou, count in sorted(source_dict.items(), key=itemgetter(1), reverse=True):
        print(' {0}: {1} news'.format(sou, count))
