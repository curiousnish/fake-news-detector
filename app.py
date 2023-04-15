import streamlit as st
import pickle


st.title('Fake News Detector')


loaded_model = pickle.load(open("model.sav", 'rb'))

pred = loaded_model.predict()
