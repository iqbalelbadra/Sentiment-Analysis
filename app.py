import re
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

app = Flask(__name__)
app.json_provider_class	 = LazyJSONEncoder


title = str(LazyString(lambda: 'API Documentation for Sentiment Analysis'))
version = str(LazyString(lambda: '1.0.0'))
description = str(LazyString(lambda: 'API for Sentiment Analysis on Bahasa Indonesia Text'))
host = LazyString(lambda: request.host)

swagger_config = {
    "headers": [],
    "specs": [{"endpoint": "docs", "route": '/docs.json'}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger_config["info"] = {
    "title": "API Documentation for Sentiment Analysis",
    "version": "1.0.0",
    "description": "This API provides sentiment analysis capabilities for Bahasa Indonesia text data. It can classify text into positive, neutral, or negative sentiments based on the text and file."
}


swagger = Swagger(app,
#				  template = swagger_template,
				  config = swagger_config
				 )

def loading_all_files():
    tokenizer = pickle.load(open('Pickle/tokenizer.pkl','rb'))
    onehot = pickle.load(open('Pickle/onehot.pkl','rb'))
    input_len = pickle.load(open('Pickle/input_len.pkl','rb'))
    
    return tokenizer, onehot, input_len

tokenizer, onehot, input_len = loading_all_files()


@app.route('/', methods= ['GET'])
def hello():
	return redirect('/docs/')


## CNN

@swag_from("docs/input_processing.yml", methods=['POST'])
@app.route('/input-processing',methods=['POST'])
def input_processing():
    text = request.form.get('text')
    cleaned = processing_text(text)
    model_cnn= load_model('Model/model_CNN.h5')
    paragraph = tokenizer.texts_to_sequences(cleaned)
    padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)

    y_pred = model_cnn.predict(padded_paragraph, batch_size=1)

    sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
    probability = np.max(y_pred, axis=1)[0]
    probability = float(probability)
    json_response = {'Description':'Sentiment Analysis using CNN',
                    'Data':{
                        'Sentiment':sentiment,
                        'Text':text,
                        'Probability':probability,
                    },
                    }
    response_data = jsonify(json_response)

    return response_data

@swag_from("docs/file_processing.yml", methods=['POST'])
@app.route('/file-processing', methods=['POST'])
def file_processing():
    file = request.files['file']
    if file:
        df = pd.read_csv(file, encoding='latin1')
        if("data_text" in df.columns):
            create_table()
            for idx, row in df.iterrows():
                text = row['data_text']
                cleaned = processing_text(text)
                model_cnn= tf.keras.models.load_model('Model/model_CNN.h5')
                paragraph = tokenizer.texts_to_sequences(cleaned)
                padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)

                y_pred = model_cnn.predict(padded_paragraph, batch_size=1)

                sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
                probability = np.max(y_pred, axis=1)[0]
                probability = float(probability)
                insert_to_table(text, sentiment, probability)

            response_data = jsonify({'response': 'SUCCESS PREDICT'})
            return response_data
        else:
            response_data = jsonify({'ERROR_WARNING': "No COLUMNS data_text APPEAR ON THE UPLOADED FILE"})
            return response_data
    else:
        response_data = jsonify({'response': 'No file uploaded'})
        return response_data

## NN

@swag_from("docs/input_processing_nn.yml", methods=['POST'])
@app.route('/input-processing-nn',methods=['POST'])
def input_processing_nn():
    text = request.form.get('text')
    cleaned = processing_text(text)
    model_nn = pickle.load(open('Model/model_NN_sklearn.h5','rb'))
    
    paragraph = tokenizer.texts_to_sequences([cleaned])
    padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)
    y_pred = model_nn.predict(padded_paragraph)
    sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
    probability = np.max(y_pred, axis=1)[0]
    probability = float(probability)
    json_response = {'Description':'Sentiment Analysis using CNN',
                    'Data':{
                        'Sentiment':sentiment,
                        'Text':text,
                        'Probability':probability,
                    },
                    }
    response_data = jsonify(json_response)

    return response_data

@swag_from("docs/file_processing_nn.yml", methods=['POST'])
@app.route('/file-processing-nn', methods=['POST'])
def file_processing_nn():
    file = request.files['file']
    if file:
        df = pd.read_csv(file, encoding='latin1')
        if("data_text" in df.columns):
            create_table()
            for idx, row in df.iterrows():
                text = row['data_text']
                cleaned = processing_text(text)
                model_cnn= tf.keras.models.load_model('Model/model_CNN.h5')
                paragraph = tokenizer.texts_to_sequences(cleaned)
                padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)

                y_pred = model_cnn.predict(padded_paragraph, batch_size=1)

                sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
                probability = np.max(y_pred, axis=1)[0]
                probability = float(probability)
                insert_to_table(text, sentiment, probability)

            response_data = jsonify({'response': 'SUCCESS PREDICT'})
            return response_data
        else:
            response_data = jsonify({'ERROR_WARNING': "No COLUMNS data_text APPEAR ON THE UPLOADED FILE"})
            return response_data
    else:
        response_data = jsonify({'response': 'No file uploaded'})
        return response_data

## LSTM

@swag_from("docs/input_processing_lstm.yml", methods=['POST'])
@app.route('/input_processing_lstm',methods=['POST'])
def input_processing_lstm():
    text = request.form.get('text')
    cleaned = processing_text(text)
    model_lstm = pickle.load(open('Model/model_LSTM.h5','rb'))
    paragraph = tokenizer.texts_to_sequences(cleaned)
    padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)

    y_pred = model_lstm.predict(padded_paragraph, batch_size=1)

    sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
    probability = np.max(y_pred, axis=1)[0]
    probability = float(probability)
    probability = '{:.0%}'.format(probability)

    json_response = {'Description':'Sentiment Analysis using LSTM',
                    'Data':{
                        'Sentiment': sentiment,
                        'Text': text,
                        'Probability': probability,
                    },
                    }
    response_data = jsonify(json_response)

    return response_data

@swag_from("docs/file_processing_lstm.yml", methods=['POST'])
@app.route('/file_processing_lstm', methods=['POST'])
def file_processing_lstm():
    file = request.files['file']
    if file:
        df = pd.read_csv(file, encoding='latin1')
        if("data_text" in df.columns):
            create_table()
            for idx, row in df.iterrows():
                text = row['data_text']
                cleaned = processing_text(text)
                model_lstm = pickle.load(open('Model/model_LSTM.h5','rb'))
                paragraph = tokenizer.texts_to_sequences(cleaned)
                padded_paragraph = pad_sequences(paragraph, padding='post', maxlen=input_len)

                y_pred = model_lstm.predict(padded_paragraph, batch_size=1)

                sentiment = onehot.inverse_transform(y_pred).reshape(-1)[0]
                probability = np.max(y_pred, axis=1)[0]
                probability = float(probability)
                insert_to_table(text, sentiment, probability)

            response_data = jsonify({'response': 'SUCCESS PREDICT'})

            return response_data
        else:
            response_data = jsonify({'ERROR_WARNING': "No COLUMNS data_text APPEAR ON THE UPLOADED FILE"})
            return response_data
    else:
        response_data = jsonify({'response': 'No file uploaded'})
        return response_data

## Regression

@swag_from("docs/input_processing_reg.yml", methods=['POST'])
@app.route('/input_processing_reg',methods=['POST'])
def input_processing_reg():
    text = request.form.get('text')
    cleaned = processing_text(text)
    with open('Pickle/cvregressi.pickle', 'rb') as file:
        cx = pickle.load(file)
    with open('Pickle/modelRegressi.pickle', 'rb') as file:
        mp = pickle.load(file)
    
    result = mp.predict(X=cx.transform([cleaned]))
    accuracy = 0.0
    
    json_response = {'Description':'Sentiment Analysis using Regression',
                    'Data':{
                        'Sentiment':result[0],
                        'Text':text,
                        'Probability':accuracy,
                    },
                    }
    response_data = jsonify(json_response)

    return response_data

@swag_from("docs/file_processing_reg.yml", methods=['POST'])
@app.route('/file_processing_reg', methods=['POST'])
def file_processing_reg():
    file = request.files['file']
    if file:
        df = pd.read_csv(file, encoding='latin1')
        if("data_text" in df.columns):
            create_table()
            for idx, row in df.iterrows():
                text = row['data_text']
                cleaned = processing_text(text)
                with open('Pickle/cvregressi.pickle', 'rb') as file:
                    cx = pickle.load(file)
                with open('Pickle/modelRegressi.pickle', 'rb') as file:
                    mp = pickle.load(file)
                result = mp.predict(X=cx.transform([cleaned]))
                accuracy = 0.0
                result = result[0]
                insert_to_table(text, result, accuracy)

            response_data = jsonify({'response': 'SUCCESS PREDICT'})
            
            return response_data
        else:
            response_data = jsonify({'ERROR_WARNING': "No COLUMNS data_text APPEAR ON THE UPLOADED FILE"})
            return response_data
    else:
        response_data = jsonify({'response': 'No file uploaded'})
        return response_data


if __name__ == '__main__':
	app.run(debug=True)
