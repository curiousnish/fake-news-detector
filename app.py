import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


st.title('Fake News Detector')
df = pd.read_csv("news.csv")

labels = df.label

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, stratify=labels, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and Transform train and test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize PassiveAgressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# y_pred = pac.predict(tfidf_test)
y_pred = pac.predict(tfidf_test[0])

st.write(y_pred)
