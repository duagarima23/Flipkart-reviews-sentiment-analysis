import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Flipkart Review Sentiment Analyzer", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: gray;
        }
        .result {
            text-align: center;
            font-size: 22px;
            padding: 10px;
            border-radius: 8px;
        }
        .emoji {
            font-size: 26px;
        }
    </style>
""", unsafe_allow_html=True)

# Centered Flipkart logo
st.markdown(
    """
    <div style='text-align: center;'>
        <img src="https://1000logos.net/wp-content/uploads/2021/02/Flipkart-logo.png" width="200"/>
    </div>
    """, unsafe_allow_html=True
)

# Title
st.markdown("<div class='title'>Flipkart Review Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether a Flipkart product review is Positive or Negative</div>", unsafe_allow_html=True)
st.write("")

# Input
review = st.text_area("✍️ Enter a product review:")

# Prediction
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        transformed = vectorizer.transform([review])
        prediction = model.predict(transformed)[0]

        # Get confidence score
        try:
            probas = model.predict_proba(transformed)
            confidence = round(probas[0][prediction] * 100, 2)
        except AttributeError:
            confidence = None

        # Display results
        if prediction == 1:
            st.success("Predicted Sentiment: Positive 😊")
        else:
            st.error("Predicted Sentiment: Negative 😞")

        if confidence is not None:
            st.markdown(f"<div class='result'>🧠 <b>Model Confidence:</b> {confidence} %</div>", unsafe_allow_html=True)

