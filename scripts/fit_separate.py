import numpy as np
import pickle
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer

# Read the data
df_tot = get_data()

# Define feature and labels
X = df_tot['text']
y = df_tot['label']

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Make F1 a scorer
f1_scorer = make_scorer(f1_score)

# Vectorize according to TFIDF
print('Transforming text to feature with TFIDF...')
vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.7, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Multinomial Naive Bayes
print('Evaluating MNB...')
clf_mnb = MultinomialNB()
clf_mnb.fit(X_train_tfidf, y_train)
pred_mnb = clf_mnb.predict(X_test_tfidf)
scores_mnb = get_scores(y_test, pred_mnb)
print('MNB scores:\n{0}'.format(scores_str(scores_mnb)))

print('Optimizing MNB with GridSearchCV...')
params_mnb = {'alpha': np.arange(0., 1., 0.1), 'fit_prior': [True, False]}
grid_mnb = GridSearchCV(clf_mnb, param_grid=params_mnb, scoring=f1_scorer, error_score=0., n_jobs=-1, verbose=1)
grid_mnb.fit(X_train_tfidf, y_train)
clf_mnb_opt = grid_mnb.best_estimator_
pred_mnb_opt = clf_mnb_opt.predict(X_test_tfidf)
scores_mnb_opt = get_scores(y_test, pred_mnb_opt)
print('MNB (optimised) scores:\n{0}'.format(scores_str(scores_mnb_opt)))
pickle.dump(clf_mnb_opt, open('mnb_opt.pkl', 'wb'))


# Logistic regression
print('Evaluating LRE...')
clf_lr = LogisticRegressionCV(scoring=f1_scorer, max_iter=200, class_weight='balanced', random_state=42)
clf_lr.fit(X_train_tfidf, y_train)
pred_lr = clf_lr.predict(X_test_tfidf)

scores_lr = get_scores(y_test, pred_lr)
print('LRE scores:\n{0}'.format(scores_str(scores_lr)))

reg_C = clf_lr.C_
Cs = np.arange(np.round(reg_C, 1) - 2., np.round(reg_C, 1) + 2., 0.1)
clf_lr = LogisticRegressionCV(scoring=f1_scorer, max_iter=200, Cs=Cs, class_weight='balanced', random_state=42)
print('Optimizing LRE with GridSearchCV...')
params_lr = {'dual': [True, False],
             'fit_intercept': [True, False],
             'penalty': ['l1', 'l2'],
             'solver': ['sag', 'liblinear']}

grid_lr = GridSearchCV(clf_lr, param_grid=params_lr, scoring=f1_scorer, error_score=0., n_jobs=-1, verbose=0)
grid_lr.fit(X_train_tfidf, y_train)
clf_lr_opt = grid_lr.best_estimator_
pred_lr_opt = clf_lr_opt.predict(X_test_tfidf)
scores_lr_opt = get_scores(y_test, pred_lr_opt)
print('LRE (optimised) scores:\n{0}'.format(scores_str(scores_lr_opt)))
pickle.dump(clf_lr_opt, open('lr_opt.pkl', 'wb'))

# Random forest
print('Evaluating RFO...')
clf_rndf = RandomForestClassifier(class_weight='balanced_subsample',
                                  n_estimators=30, max_depth=20,
                                  min_samples_split=20, max_features=0.01,
                                  random_state=42)
clf_rndf.fit(X_train_tfidf, y_train)
pred_rndf = clf_rndf.predict(X_test_tfidf)

scores_rndf = get_scores(y_test, pred_rndf)
print('RFO scores:\n{0}'.format(scores_str(scores_rndf)))

print('Optimizing RFO with GridSearchCV...')
params_rndf = {'n_estimators': [50, 100],
               'max_depth': [20, 50, 100],
               'min_samples_split': [20, 50],
               'max_features': [0.02, 0.05]}

grid_rndf = GridSearchCV(clf_rndf, param_grid=params_rndf, scoring=f1_scorer, error_score=0., n_jobs=-1, verbose=10)
grid_rndf.fit(X_train_tfidf, y_train)
clf_rndf_opt = grid_rndf.best_estimator_
pred_rndf_opt = clf_rndf_opt.predict(X_test_tfidf)
scores_rndf_opt = get_scores(y_test, pred_rndf_opt)
print('RFO (optimised) scores:\n{0}'.format(scores_str(scores_rndf_opt)))
pickle.dump(clf_rndf_opt, open('rndf_opt.pkl', 'wb'))
