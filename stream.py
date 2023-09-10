import streamlit as st
import pickle
import pandas as pd
import sqlite3
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request, redirect
from cleansing import processing_text
from wnr import create_table,insert_to_table

from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Load model and CountVectorizer
with open('Pickle/modelRegressi.pickle', 'rb') as model_file:
    modelr = pickle.load(model_file)

with open('Pickle/cvregressi.pickle', 'rb') as cv_file:
    cvr = pickle.load(cv_file)

def loading_all_files():
    tokenizer = pickle.load(open('Pickle/tokenizer.pkl','rb'))
    onehot = pickle.load(open('Pickle/onehot.pkl','rb'))
    input_len = pickle.load(open('Pickle/input_len.pkl','rb'))
    
    return tokenizer, onehot, input_len

tokenizer, onehot, input_len = loading_all_files()


def predict_sentiment(text):
    cleaned_text = processing_text(text)
    result = modelr.predict(cvr.transform([cleaned_text]))
    return result[0]

def predict_sentiment_nn(text):
    cleaned_text = processing_text(text)
    model_cnn = pickle.load(open('Model/model_CNN.h5','rb'))
    paragraph = tokenizer.texts_to_sequences(cleaned_text)
    padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)

    y_pred = model_cnn.predict(padded_paragraph, batch_size=1)
    sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
    return sentiment

def predict_sentiment_lstm(text):
    cleaned_text = processing_text(text)
    model_lstm = pickle.load(open('Model/model_LSTM.h5','rb'))
    paragraph = tokenizer.texts_to_sequences(cleaned_text)
    padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)

    y_pred = model_lstm.predict(padded_paragraph, batch_size=1)

    sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
    return sentiment

def main():
    st.sidebar.title('Sentiment Analysis')
    option = st.sidebar.radio('Choose an option', ('Using NN', 'Using LSTM', 'Using Regression'))

    if option == 'Using NN':
        st.title('Sentiment Analysis App - Using NN')

        optionreg = st.selectbox('Choose an option', ('Input Processing', 'File Processing'))

        if optionreg == 'Input Processing':
            text_input = st.text_area('Enter text')
            if st.button('Predict'):
                sentiment = predict_sentiment_nn(text_input)
                st.write(f'Sentiment: {sentiment}')

        elif optionreg == 'File Processing':
            file = st.file_uploader('Upload a CSV file', type=['csv'])
            if file is not None:
                df = pd.read_csv(file)
                if 'data_text' in df.columns:
                    create_table()
                    for idx, row in df.iterrows():
                        text = row['data_text']
                        sentiment = predict_sentiment_nn(text)
                        insert_to_table(text, sentiment)
                    st.success('Prediction and storage complete')
                else:
                    st.error('No "data_text" column in the CSV file')

    elif option == 'Using LSTM':
        st.title('Sentiment Analysis App - Using LSTM')

        optionreg = st.selectbox('Choose an option', ('Input Processing', 'File Processing'))

        if optionreg == 'Input Processing':
            text_input = st.text_area('Enter text')
            if st.button('Predict'):
                sentiment = predict_sentiment_lstm(text_input)
                st.write(f'Sentiment: {sentiment}')

        elif optionreg == 'File Processing':
            file = st.file_uploader('Upload a CSV file', type=['csv'])
            if file is not None:
                df = pd.read_csv(file)
                if 'data_text' in df.columns:
                    create_table()
                    for idx, row in df.iterrows():
                        text = row['data_text']
                        sentiment = predict_sentiment_lstm(text)
                        insert_to_table(text, sentiment)
                    st.success('Prediction and storage complete')
                else:
                    st.error('No "data_text" column in the CSV file')
 
    elif option == 'Using Regression':
        st.title('Sentiment Analysis App - Using Regression')

        optionreg = st.selectbox('Choose an option', ('Input Processing', 'File Processing'))

        if optionreg == 'Input Processing':
            text_input = st.text_area('Enter text')
            if st.button('Predict'):
                sentiment = predict_sentiment(text_input)
                st.write(f'Sentiment: {sentiment}')

        elif optionreg == 'File Processing':
            file = st.file_uploader('Upload a CSV file', type=['csv'])
            if file is not None:
                df = pd.read_csv(file)
                if 'data_text' in df.columns:
                    create_table()
                    for idx, row in df.iterrows():
                        text = row['data_text']
                        sentiment = predict_sentiment(text)
                        insert_to_table(text, sentiment)
                    st.success('Prediction and storage complete')
                else:
                    st.error('No "data_text" column in the CSV file')
                    
if __name__ == '__main__':
    main()
