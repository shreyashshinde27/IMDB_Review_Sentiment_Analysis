import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

model_file_path = "C:/Users/vishn/Downloads/Pickle_Files/best_model.pkl"
vectorizer_file_path = "C:/Users/vishn/Downloads/Pickle_Files/vectorizer.pkl"

# Load the model and vectorizer
try:
    with open(model_file_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_file_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

except FileNotFoundError:
    st.error("Required files not found. Ensure 'best_model.pkl' and 'vectorizer.pkl' are in the correct directory.")
    st.stop()
except ValueError as ve:
    st.error(f"File content error: {ve}")
    st.stop()

# Function to preprocess and predict sentiment
def predict_sentiment(user_input):
    preprocessed_input = vectorizer.transform([user_input])  # Vectorize the input
    prediction = model.predict(preprocessed_input)  # Predict
    return "Positive" if prediction == 1 else "Negative"

# Streamlit interface
st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 Sentiment Analysis for Movie Reviews")
st.markdown("""
Enter a movie review below, and the app will predict whether the sentiment is **Positive** or **Negative**.
Please make sure to **only enter movie-related reviews**.
""")
st.info("Example: 'The movie was fantastic! Loved the plot and the acting.'")

# User input
user_input = st.text_area("Enter your Movie Review:", "", height=200)

# Function to validate movie-related input
def is_movie_related(review):
    # Simple check to ensure the review contains movie-related terms (can be expanded)
    keywords = ['movie', 'film', 'actor', 'actress', 'director', 'plot', 'cinema', 'scene', 'character']
    return any(keyword in review.lower() for keyword in keywords)

# Make predictions
if st.button("Analyze Sentiment"):
    if user_input.strip():
        if is_movie_related(user_input):
            try:
                result = predict_sentiment(user_input)
                st.subheader(f"Prediction: {result}")
                if result == "Positive":
                    st.success("The sentiment is Positive! 🎉")
                else:
                    st.error("The sentiment is Negative. 😞")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter a movie-related review. The app only works for movie reviews.")
    else:
        st.error("Please enter a valid review.")

# Additional enhancements (styling)
st.markdown("""
    <style>
        .stTextArea textarea {
            font-size: 18px;
            padding: 10px;
        }
        .stButton button {
            background-color: #FF6347;
            color: white;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color:rgb(255, 255, 255);
        }
    </style>
""", unsafe_allow_html=True)