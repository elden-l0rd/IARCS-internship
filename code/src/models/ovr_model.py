from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

tfidf_vectorizer = TfidfVectorizer()
base_classifier = LogisticRegression()

clf = OneVsRestClassifier(base_classifier)

def vectorize(df_train, df_test):
    df_train['NameDesc'] = df_train['NameDesc'].apply(lambda x: ' '.join(x))
    df_test['NameDesc'] = df_test['NameDesc'].apply(lambda x: ' '.join(x))

    X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['NameDesc']).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(df_test['NameDesc']).toarray()

    y_train = df_train['STRIDE'].values
    y_test = df_test['STRIDE'].values

    return X_train_tfidf, X_test_tfidf, y_train, y_test

def train_loop(X_train_tfidf, y_train, y_test):
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_train_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, acc, report, cm
