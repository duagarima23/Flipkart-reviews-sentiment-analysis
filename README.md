# Flipkart Reviews Sentiment Analysis 🛍️

This project performs sentiment analysis on Flipkart product reviews using a Machine Learning model. It includes a user-friendly web application built with **Streamlit** that allows users to input product reviews and instantly get sentiment predictions — Positive or Negative.

---

## 🚀 Features

- Preprocessing of review text (stopword removal, tokenization)
- Sentiment classification using Decision Tree Classifier
- Web interface for live predictions using Streamlit
- Simple and responsive user interface

---

## 🧠 Model Details

- **Algorithm:** Decision Tree Classifier  
- **Text Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency)  
- **Tools Used:**  
  - `scikit-learn` for ML  
  - `nltk` for text preprocessing  
  - `streamlit` for frontend

---

## 📁 Project Structure
<pre> ## 📁 Project Structure ``` Flipkart_Reviews_Sentiment_Analysis/ │ ├── app.py # Streamlit web application ├── Flipkart_Reviews_Sentiment_Analysis_dt.ipynb # Notebook for training the model ├── model.pkl # Saved trained model ├── vectorizer.pkl # Saved TF-IDF vectorizer ├── requirements.txt # Project dependencies ├── README.md # Project documentation └── resources/ └── flipkart_data.csv #dataset used ``` </pre>

