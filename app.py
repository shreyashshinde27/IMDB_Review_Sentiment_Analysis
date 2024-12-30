from flask import Flask, request, jsonify, render_template
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load preprocessing objects and model
with open('text_preprocessing.pkl', 'rb') as file:
    preprocessing_data = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

tfidf_vectorizer = preprocessing_data['tfidf_vectorizer']
stop_words = set(preprocessing_data['stop_words'])
lemmatizer = WordNetLemmatizer()

# Preprocessing functions
def clean_text(text):
    """Cleans the input text by removing HTML, punctuation, and stopwords."""
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove punctuation and keep only alphabets
    text = text.lower().strip()  # Convert to lowercase and strip extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

def lemmatize_tokens(tokens):
    """Lemmatizes a list of tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]

# Flask routes
@app.route('/')
def home():
    """Serves the HTML form."""
    return render_template('index.html')  # Ensure index.html is in the templates folder

@app.route('/classify', methods=['POST'])
def classify():
    """Classifies the given review as Positive or Negative."""
    data = request.json
    review = data.get('review', '')

    if not review:
        return jsonify({'error': 'No review provided'}), 400

    try:
        # Preprocess the review
        cleaned_review = clean_text(review)
        lemmatized_review = lemmatize_tokens(word_tokenize(cleaned_review))
        lemmatized_review_string = ' '.join(lemmatized_review)

        # Transform the review using TF-IDF
        review_vector = tfidf_vectorizer.transform([lemmatized_review_string])

        # Predict using the logistic regression model
        prediction = model.predict(review_vector)
        prediction_label = 'Positive' if prediction[0] == 1 else 'Negative'

        return jsonify({'classification': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
