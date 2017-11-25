import pickle
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split

# Read the data
df_tot = get_data()

# Define feature and labels
X = df_tot['text']
y = df_tot['label']

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Vectorize according to TFIDF
print('Transforming text to feature with TFIDF...')
vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.7, stop_words='english')  # check sublinear_tf=True
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

f1_scorer = make_scorer(f1_score)

print('Loading models...')
clf_mnb = pickle.load(open('mnb_opt.pkl', 'rb'))
clf_lr = pickle.load(open('lr_opt.pkl', 'rb'))
clf_rndf = pickle.load(open('rndf_opt.pkl', 'rb'))

# Ensemble 1
print('Building EN1...')
en1_estimators = [('mnb', clf_mnb), ('lr', clf_lr), ('rndf', clf_rndf)]
clf_en1 = VotingClassifier(en1_estimators, voting='hard', n_jobs=-1)
clf_en1.fit(X_train_tfidf, y_train)
pred_en1 = clf_en1.predict(X_test_tfidf)
scores_en1 = get_scores(y_test, pred_en1)
print('EN1 scores:\n{0}'.format(scores_str(scores_en1)))
pickle.dump(clf_en1, open('en1_opt.pkl', 'wb'))


# Ensemble 2
print('Building EN2...')
en2_estimators = [('lr', clf_lr), ('rndf', clf_rndf)]
clf_en2 = VotingClassifier(en2_estimators, voting='soft', n_jobs=-1)
clf_en2.fit(X_train_tfidf, y_train)
pred_en2 = clf_en2.predict(X_test_tfidf)
scores_en2 = get_scores(y_test, pred_en2)
print('EN2 scores:\n{0}'.format(scores_str(scores_en2)))
pickle.dump(clf_en2, open('en2_opt.pkl', 'wb'))
