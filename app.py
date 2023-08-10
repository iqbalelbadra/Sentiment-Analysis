import re
import pandas as pd
import sqlite3

from flask import Flask, jsonify, request, redirect
from cleansing import processing_text

from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

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



@app.route('/', methods= ['GET'])
def hello():
	return redirect('/docs/')

## RNN

@swag_from("docs/input_processing.yml", methods=['POST'])
@app.route('/input-processing',methods=['POST'])
def input_processing():
    text = request.form.get('text')
    cleaned = processing_text(text)
    
    json_response = {'Description':'Sentiment Analysis using RNN',
                    'Data':{
                        'Sentiment':'Positive',
                        'Text':text,
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
            json_response = {'Description':'Sentiment Analysis using RNN',
                    'Data':{
                        'Sentiment':'Positive',
                        'Text':'Text',
                    },
                    }
            response_data = jsonify(json_response)
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

    json_response = {'Description':'Sentiment Analysis using LSTM',
                    'Data':{
                        'Sentiment':'Positive',
                        'Text':text,
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
            json_response = {'Description':'Sentiment Analysis using LSTM',
                    'Data':{
                        'Sentiment':'Positive',
                        'Text':'Text',
                    },
                    }
            response_data = jsonify(json_response)
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

    json_response = {'Description':'Sentiment Analysis using Regression',
                    'Data':{
                        'Sentiment':'Positive',
                        'Text':text,
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
            json_response = {'Description':'Sentiment Analysis using Regression',
                    'Data':{
                        'Sentiment':'Positive',
                        'Text':'Text',
                    },
                    }
            response_data = jsonify(json_response)
            return response_data
        else:
            response_data = jsonify({'ERROR_WARNING': "No COLUMNS data_text APPEAR ON THE UPLOADED FILE"})
            return response_data
    else:
        response_data = jsonify({'response': 'No file uploaded'})
        return response_data


if __name__ == '__main__':
	app.run(debug=True)
