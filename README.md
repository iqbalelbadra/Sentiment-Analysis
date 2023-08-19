# Sentiment Analysis Project

This project implements a sentiment analysis system for Bahasa Indonesia text data. The system uses different approaches, including RNN, LSTM, and Regression, to classify text into positive, neutral, or negative sentiments.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Author](#author)
- [License](#license)

## Description

This project provides an API and a Streamlit app for sentiment analysis of Bahasa Indonesia text data. It includes the following components:

- API for Sentiment Analysis:
  - Endpoints for RNN-based, LSTM-based, and Regression-based sentiment analysis
  - Input processing for text data
  - File processing for bulk sentiment analysis from CSV files

- Streamlit App:
  - User-friendly interface to interact with different sentiment analysis methods
  - Option to enter text for analysis or upload a CSV file for batch analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iqbalelbadra/Sentiment-Analysis
   cd Sentiment-Analysis
   ```

2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### API

1. Run the Flask API:
   ```bash
   python app.py
   ```

2. Access the Swagger documentation at `http://localhost:5000/docs` to explore the available API endpoints.

### Streamlit App

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. The app will open in your browser, providing options to perform sentiment analysis using different methods.

## Dependencies

- Flask
- Flask-Cors
- flasgger
- numpy
- pandas
- scikit-learn
- streamlit

Refer to the `requirements.txt` file for specific versions of the dependencies.

## Demo on huggingface space

https://huggingface.co/spaces/iqbalelbadra/sentiment-analysis


## Author

- Iqbal Ahdagita Elbadra
- GitHub: [iqbalelbadra](https://github.com/iqbalelbadra)

- Brain Dior
- GitHub: [braindior01](https://github.com/braindior01)

- Abed Nigo
- GitHub: [myniggname](https://github.com/myniggname)

## License

This project is licensed under the [MIT License](LICENSE).
```
