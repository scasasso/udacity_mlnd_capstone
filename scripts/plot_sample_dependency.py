import pickle
import numpy as np
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import OrderedDict as odict


def fit_samples(X_train, y_train, X_test, y_test):

    # Vectorize according to TFIDF
    print(' Transforming text to feature with TFIDF...')
    vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.7, stop_words='english')  # check sublinear_tf=True
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    f1_scorer = make_scorer(f1_score)

    print(' Loading models...')
    clf_lr = pickle.load(open('lr_opt.pkl', 'rb'))
    clf_rndf = pickle.load(open('rndf_opt.pkl', 'rb'))

    print(' Building EN2...')
    en2_estimators = [('lr', clf_lr), ('rndf', clf_rndf)]
    clf_en2 = VotingClassifier(en2_estimators, voting='soft', n_jobs=-1)
    clf_en2.fit(X_train_tfidf, y_train)
    pred_en2 = clf_en2.predict(X_test_tfidf)
    scores_en2 = get_scores(y_test, pred_en2)
    print(' EN2 scores:\n{0}'.format(scores_str(scores_en2)))

    return scores_en2

# Read the data
df_tot = get_data()

# Define feature and labels
X = df_tot['text']
y = df_tot['label']

# Split into train and test set
X_train_tot, X_test_tot, y_train_tot, y_test_tot = train_test_split(X, y, test_size=0.20, random_state=42)

score_dict = odict()
for n_samples in np.linspace(200, len(X_train_tot), 20):
    n_samples = int(np.floor(n_samples))
    print('Fitting {0} samples of the training set...'.format(n_samples))
    scores = fit_samples(X_train_tot[:n_samples, ], y_train_tot[:n_samples, ], X_test_tot, y_test_tot)
    score_dict[n_samples] = scores

# Dump the result to pkl
pickle.dump(score_dict, open('n_sample_dep.pkl', 'wb'))

# Plot
xs = [k for k, v in score_dict.items()]
ys_accuracy = [v['accuracy'] for k, v in score_dict.items()]
ys_precision = [v['precision'] for k, v in score_dict.items()]
ys_recall = [v['recall'] for k, v in score_dict.items()]
ys_f1_score = [v['f1_score'] for k, v in score_dict.items()]
acc = plt.plot(xs, ys_accuracy, '-', label='accuracy')
pre = plt.plot(xs, ys_precision, '-', label='precision')
rec = plt.plot(xs, ys_recall, '-', label='recall')
f1s = plt.plot(xs, ys_f1_score, '-', label='F1-score')
plt.legend()
plt.xlabel('N. train samples')
plt.ylabel('score')
plt.savefig('scores_vs_nsamples.pdf')

