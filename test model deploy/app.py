import re
import pandas as pd
import pickle
import numpy as np
import sqlite3

from flask import Flask, jsonify, request, render_template, redirect, url_for
from data_reading_and_writing import create_table, insert_to_table

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

TABLE_NAME = "text_record"

@app.route('/', methods=['GET', "POST"])
def homepage():
    if request.method == 'POST':
        go_to_page = request.form['inputText']
        if go_to_page == "1":
            return redirect(url_for("model_LSTM"))
        elif go_to_page == "2":
            return redirect(url_for("input_file"))
        elif go_to_page == "3":
            return redirect(url_for("read_database"))
    else:
        return render_template("homepage.html")

@app.route('/model-LSTM',methods=['GET', 'POST'])
def model_LSTM():
    if request.method == 'POST':
        paragraph1 = request.form['inputText']
        paragraph = text_normalization(paragraph1)
        paragraph = tokenizer.texts_to_sequences([paragraph])
        padded_paragraph = pad_sequences(paragraph,padding='post',maxlen=input_len)
        y_pred = model_lstm.predict(padded_paragraph, batch_size=1)
        probability = np.max(y_pred, axis=1)
        # print(max(y_pred[0]))
        # print(probability)

        y_pred = onehot.inverse_transform(y_pred)
        create_table()
        insert_to_table(value_1=paragraph1, 
                        value_2=y_pred[0][0]) 
        print(y_pred[0])
        # return y_pred[0], probability[0]
        json_response={'response':"SUCCESS",
                       'prediction': y_pred[0][0],
                      }
        json_response=jsonify(json_response)
        return json_response
    else:
        return render_template("model_lstm.html")

@app.route('/file-processing',methods=['GET', 'POST'])
def input_file():
    if request.method == 'POST':
        input_file = request.files['inputFile']
        df = pd.read_csv(input_file, encoding='latin1')
        if("Tweet" in df.columns):
            list_of_input_text = df['Tweet'] #yang dari CSV
            list_of_status = df['Tweet'].apply(lambda x: text_normalization(x)) #ini yang hasil cleaning-an

            create_table()
            for paragraph1, cleaned_text in zip(list_of_tweets, list_of_cleaned_tweet): # disini di-looping barengan
                insert_to_table(value_1=previous_text, value_2=cleaned_text)
            
            json_response={'response':"SUCCESS",
                           'list_of_tweets': list_of_tweets[0],
                           'list_of_cleaned_tweet': list_of_cleaned_tweet[0]
                          }
            json_response=jsonify(json_response)
            return json_response
        else:
            json_response={'ERROR_WARNING': "NO COLUMNS 'Tweet' APPEAR ON THE UPLOADED FILE"}
            json_response = jsonify(json_response)
            return json_response
        return json_response
    else:
        return render_template("test_file_input.html")

# @app.route('/model-CNN',methods=['GET', 'POST'])
# def model_CNN():
#     if request.method == 'POST':
#         paragraph=request.form['inputText']
#         paragraph = text_normalization(paragraph)
#         paragraph = tokenizer.texts_to_sequences([paragraph])
#         padded_paragraph = pad_sequences(paragraph,padding='post',maxlen=input_len)
        
#         y_pred = model_cnn.predict(padded_paragraph, batch_size=1)
        

#         probability = np.max(y_pred, axis=1)
#         print(max(y_pred[0]))
#         print(probability)
#         y_pred = onehot.inverse_transform(y_pred)
#         # return y_pred[0], probability[0]
#         json_response={'response':"SUCCESS",
#                        'prediction': print(max(y_pred[0])),
#                       }
#         json_response=jsonify(json_response)
#         return json_response
#     else:
#         return render_template("model_cnn.html")

if __name__ == '__main__':
    app.run(debug=True)
