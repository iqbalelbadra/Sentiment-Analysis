import streamlit as st
import pickle
import pandas as pd
from cleansing import processing_text
from wnr import create_table, insert_to_table

# Load model and CountVectorizer
with open('Pickle/modelRegressi.pickle', 'rb') as model_file:
    modelr = pickle.load(model_file)

with open('Pickle/cvregressi.pickle', 'rb') as cv_file:
    cvr = pickle.load(cv_file)

def predict_sentiment(text):
    cleaned_text = processing_text(text)
    result = modelr.predict(cvr.transform([cleaned_text]))
    return result[0]

def main():
    st.sidebar.title('Sentiment Analysis')
    option = st.sidebar.radio('Choose an option', ('Using RNN', 'Using LSTM', 'Using Regression'))

    if option == 'Using RNN':
        st.title('Sentiment Analysis App - Using RNN')

    elif option == 'Using LSTM':
        st.title('Sentiment Analysis App - Using LSTM')
 
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
