import re
import pandas as pd
import pickle
import numpy as np

from flask import Flask, jsonify, request, render_template, redirect, url_for

from data_cleansing import text_normalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__, template_folder='templates')

def loading_all_files():
    tokenizer = pickle.load(open('data/tokenizer.pkl','rb'))
    onehot = pickle.load(open('data/onehot.pkl','rb'))
    model_lstm = pickle.load(open('data/model_LSTM.h5','rb'))
    # model_cnn = pickle.load(open('data/model_CNN.h5','rb'))
    input_len = pickle.load(open('data/input_len.pkl','rb'))

    return tokenizer, onehot, model_lstm, input_len

tokenizer, onehot, model_lstm, input_len = loading_all_files()


@app.route('/', methods=['GET', "POST"])
def homepage():
    if request.method == 'POST':
        go_to_page = request.form['inputText']
        if go_to_page == "1":
            return redirect(url_for("model_LSTM"))
        elif go_to_page == "2":
            return redirect(url_for("model_CNN"))
        elif go_to_page == "3":
            return redirect(url_for("read_database"))
    else:
        return render_template("homepage.html")

@app.route('/model-LSTM',methods=['GET', 'POST'])
def model_LSTM():
    if request.method == 'POST':
        paragraph=request.form['inputText']
        paragraph = text_normalization(paragraph)
        paragraph = tokenizer.texts_to_sequences([paragraph])
        padded_paragraph = pad_sequences(paragraph,padding='post',maxlen=input_len)
        
        y_pred = model_lstm.predict(padded_paragraph, batch_size=1)


        probability = np.max(y_pred, axis=1)
        print(max(y_pred[0]))
        print(probability)
        y_pred = onehot.inverse_transform(y_pred)
        # return y_pred[0], probability[0]
        json_response={'response':"SUCCESS",
                       'prediction': print(max(y_pred[0])),
                      }
        json_response=jsonify(json_response)
        return json_response
    else:
        return render_template("model_lstm.html")


@app.route('/model-CNN',methods=['GET', 'POST'])
def model_CNN():
    if request.method == 'POST':
        paragraph=request.form['inputText']
        paragraph = text_normalization(paragraph)
        paragraph = tokenizer.texts_to_sequences([paragraph])
        padded_paragraph = pad_sequences(paragraph,padding='post',maxlen=input_len)
        
        y_pred = model_cnn.predict(padded_paragraph, batch_size=1)
        

        probability = np.max(y_pred, axis=1)
        print(max(y_pred[0]))
        print(probability)
        y_pred = onehot.inverse_transform(y_pred)
        # return y_pred[0], probability[0]
        json_response={'response':"SUCCESS",
                       'prediction': print(max(y_pred[0])),
                      }
        json_response=jsonify(json_response)
        return json_response
    else:
        return render_template("model_cnn.html")

if __name__ == '__main__':
    app.run(debug=True)
