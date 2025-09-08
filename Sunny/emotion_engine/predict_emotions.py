# File: /sanchi_nlp/emotion_engine/predict_emotions.py

import joblib

# Load model components
model = joblib.load("model/emotion_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
binarizer = joblib.load("model/emotion_binarizer.pkl")

def predict_emotions(text):
    # Clean & vectorize
    vec = vectorizer.transform([text])
    
    # Predict probabilities
    probabilities = model.predict_proba(vec)[0]
    
    # Get emotion names from binarizer
    emotion_names = binarizer.classes_

    # Combine emotions with their confidence scores
    predictions = list(zip(emotion_names, probabilities))
    
    # Filter out low-confidence emotions (threshold = 0.2)
    filtered = [(emo, round(score, 2)) for emo, score in predictions if score > 0.2]

    # Sort by score descending
    sorted_emotions = sorted(filtered, key=lambda x: x[1], reverse=True)

    return sorted_emotions
