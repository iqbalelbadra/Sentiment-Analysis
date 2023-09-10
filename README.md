
# Sentiment Analysis API and Streamlit App

This project provides a Sentiment Analysis API and a Streamlit web app for sentiment analysis on Bahasa Indonesia text data. It uses different machine learning models, including Convolutional Neural Networks (CNN), Neural Networks (NN), Long Short-Term Memory (LSTM), and Regression.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [API Usage](#api-usage)
  - [Streamlit Web App](#streamlit-web-app)
- [Models](#models)
- [File Structure](#file-structure)
- [Author](#author)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project consists of two main components:

1. **Sentiment Analysis API**: This Flask-based API provides endpoints for text and file-based sentiment analysis using various machine learning models. The available models include CNN, NN, LSTM, and Regression.

2. **Streamlit Web App**: The Streamlit app offers a user-friendly interface for sentiment analysis. Users can input text for analysis or upload CSV files for batch processing. It uses the Regression model for analysis.

## Getting Started

### Prerequisites

Before running the project, make sure you have the following prerequisites installed:

- Python 3.9.x or Above
- Flask
- Tensorflow
- scikit-learn
- Streamlit
- Other necessary libraries (install using `pip install -r requirements.txt`)

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/iqbalelbadra/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### API Usage

To use the Sentiment Analysis API, start the Flask app:

```bash
python app.py
```

You can access the API documentation at `http://localhost:5000/docs/`. It provides detailed information about available endpoints and how to use them.

### Streamlit Web App

To use the Streamlit web app, run the following command:

```bash
streamlit run stream.py
```

This will start the Streamlit app, and you can access it via your web browser. Follow the on-screen instructions to perform sentiment analysis.

## Models

- **CNN Model**: Convolutional Neural Network for text classification.
- **NN Model**: Neural Network (MLPClassifier) for text classification.
- **LSTM Model**: Long Short-Term Memory network for text classification.
- **Regression Model**: Regression-based sentiment analysis.

## File Structure

- `app.py`: Main Flask application for the Sentiment Analysis API.
- `stream.py`: Streamlit web app for sentiment analysis.
- `Pickle/`: Directory containing saved models and other necessary files.
- `Model/`: Directory containing saved Keras models.
- `cleansing.py`: Text preprocessing functions.
- `wnr.py`: Functions for working with SQLite database.
- `docs/`: Swagger API documentation files.

## Demo on huggingface space

[Demo on Hugging Face Space](https://huggingface.co/spaces/iqbalelbadra/sentiment-analysis)

## Demo on streamlit share

[Demo on Streamlit Share](https://sentiment-indonesia.streamlit.app/)

## Author

- Iqbal Ahdagita Elbadra
- GitHub: [iqbalelbadra](https://github.com/iqbalelbadra)

- Brain Dior
- GitHub: [braindior01](https://github.com/braindior01)

- Abed Nigo
- GitHub: [myniggname](https://github.com/myniggname)

## Contributing

Contributions are welcome! If you'd like to improve this project or add new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can copy and paste this README template into a file named `README.md` in your GitHub repository. Be sure to customize it to include any additional information specific to your project.