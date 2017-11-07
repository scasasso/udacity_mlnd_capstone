import pandas as pd
import utils

if __name__ == '__main__':

    # Get DataFrame
    df = utils.get_data()
    df['text_length'] = df.apply(lambda x: len(x['text']), axis=1)
    df['title_length'] = df.apply(lambda x: len(x['title']) if not pd.isnull(x['title']) else -1, axis=1)

    # Plot
    plots = [{'var_name': 'text_length', 'x_range': (0, 10000), 'x_binwidth': 100, 'x_tickwidth': 1000},
             {'var_name': 'title_length', 'x_range': (0, 200), 'x_binwidth': 10, 'x_tickwidth': 20},
             {'var_name': 'participants_count', 'x_range': (0, 20), 'x_binwidth': 1, 'x_tickwidth': 2},
             {'var_name': 'replies_count', 'x_range': (0, 20), 'x_binwidth': 1, 'x_tickwidth': 2},
             {'var_name': 'comments', 'x_range': (0, 10), 'x_binwidth': 1, 'x_tickwidth': 1},
             {'var_name': 'likes', 'x_range': (0, 20), 'x_binwidth': 1, 'x_tickwidth': 2}]

    for pl_specs in plots:
        plot_var(df, **pl_specs)
