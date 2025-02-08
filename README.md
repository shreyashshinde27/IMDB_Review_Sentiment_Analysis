# AI IMDb Review Analysis

## Project Description
The AI IMDb Review Analysis project is a machine learning application that performs sentiment analysis on IMDb movie reviews. This project leverages natural language processing (NLP) techniques to classify reviews as positive or negative. It includes a user-friendly web interface built with Flask, enabling users to upload reviews and see real-time sentiment predictions.

## Features
- Sentiment Analysis using Logistic Regression
- Preprocessing pipeline for text normalization
- Web-based interface built with Flask
- Model and vectorizer persistence with joblib

## Setup Instructions

### Step 1: Clone the Repository
```bash
$ git clone https://github.com/shreyashshinde27/AI-IMDB_Review_Analysis.git
$ cd AI-IMDB_Review_Analysis
```

### Step 2: Create and Activate a Virtual Environment
#### On Windows:
```bash
$ python -m venv venv
$ venv\Scripts\activate
```
#### On macOS/Linux:
```bash
$ python3 -m venv venv
$ source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
$ pip install -r requirements.txt
```

### Step 4: Run the Flask Application
```bash
$ python app.py
```
Open `http://127.0.0.1:5000/` in your browser to access the web app.

## Folder Structure
```
project/
│
├── static/
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── app.py
├── logistic_regression_model.pkl
└── text_preprocessing.pkl
```

## Usage
1. Launch the Flask application using the above steps.
2. Access the web interface through `http://127.0.0.1:5000/`.
3. Upload a text file containing IMDb reviews or enter text manually.
4. View the sentiment analysis results instantly.

## Model Details
- **Algorithm**: Logistic Regression
- **Libraries Used**: scikit-learn, pandas, numpy, Flask
- **Model Persistence**: Models are saved using joblib for efficient loading.

## Future Enhancements
- Integrate deep learning models for improved accuracy.
- Add multilingual support for reviews in different languages.
- Implement user authentication for personalized experiences.

## Contributing
Feel free to fork this repository, open issues, and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


