from utils import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

# Read the data
df_tot = get_data()

# Perturbate the data
print('Perturbating the dataset...')
df_tot['text'] = df_tot.apply(lambda x: chop_text_rnd(x['text']), axis=1)

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

# Load models for the ensemble
print('Loading models...')

# Logistic regression
clf_lr = pickle.load(open('lr_opt.pkl', 'rb'))
clf_lr.fit(X_train_tfidf, y_train)
pred_lr = clf_lr.predict(X_test_tfidf)
scores_lr = get_scores(y_test, pred_lr)
print('LRE scores:\n{0}'.format(scores_str(scores_lr)))

# Random forest
clf_rndf = pickle.load(open('rndf_opt.pkl', 'rb'))

# Train the model
print('Building EN2...')
en2_estimators = [('lr', clf_lr), ('rndf', clf_rndf)]
clf_en2 = VotingClassifier(en2_estimators, voting='soft', n_jobs=-1)
clf_en2.fit(X_train_tfidf, y_train)
pred_en2 = clf_en2.predict(X_test_tfidf)
scores_en2 = get_scores(y_test, pred_en2)
print('EN2 scores:\n{0}'.format(scores_str(scores_en2)))
