import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle


df = pd.read_csv("news.csv")

labels = df.label

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, stratify=labels, random_state=42)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and Transform train and test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize PassiveAgressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

pickle.dump(pac, open("model.sav", 'wb'))

# Prediction and Accuracy

# y_pred = pac.predict(tfidf_test)
# print("Accuracy score: {:.2f}".format(pac.score(tfidf_test, y_test)))
# confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
