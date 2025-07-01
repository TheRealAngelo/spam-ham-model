import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SpamDetector:
    """SMS Spam Detector class for easy model loading and prediction"""
    
    def __init__(self, model_path='models/spam_detector_model.joblib', 
                 vectorizer_path='models/vectorizer.joblib'):
        """
        Initialize the spam detector with model paths
        
        Args:
            model_path (str): Path to the trained model file
            vectorizer_path (str): Path to the vectorizer file
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
    
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.is_loaded = True
            print("✅ Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("Please make sure you have run train.py first to create the model files.")
            raise
    
    def preprocess_text(self, text):
        """
        Preprocess text message (same preprocessing as training)
        
        Args:
            text (str): Input text message
            
        Returns:
            str: Preprocessed text
        """
        return text.lower().strip()
    
    def predict(self, text):
        """
        Predict if a text message is spam or ham
        
        Args:
            text (str): Input text message
            
        Returns:
            tuple: (prediction, probabilities)
                - prediction (str): 'spam' or 'ham'
                - probabilities (np.array): [ham_prob, spam_prob]
        """
        if not self.is_loaded:
            self.load_models()
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the text
        text_vec = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        
        return prediction, probabilities
    
    def predict_batch(self, messages):
        """
        Predict multiple messages at once
        
        Args:
            messages (list): List of text messages
            
        Returns:
            tuple: (predictions, probabilities)
                - predictions (list): List of predictions
                - probabilities (np.array): Array of probability scores
        """
        if not self.is_loaded:
            self.load_models()
        
        # Preprocess all messages
        processed_messages = [self.preprocess_text(msg) for msg in messages]
        
        # Vectorize all messages
        messages_vec = self.vectorizer.transform(processed_messages)
        
        # Make predictions
        predictions = self.model.predict(messages_vec)
        probabilities = self.model.predict_proba(messages_vec)
        
        return predictions.tolist(), probabilities
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            self.load_models()
        
        info = {
            'model_type': type(self.model).__name__,
            'vectorizer_type': type(self.vectorizer).__name__,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'feature_names_sample': list(self.vectorizer.get_feature_names_out())[:10]
        }
        
        return info

def test_detector():
    """Test function to demonstrate the spam detector"""
    detector = SpamDetector()
    
    # Test messages
    test_messages = [
        "Congratulations! You've won $1000! Call now!",
        "Hey, are we still meeting for lunch today?",
        "FREE entry in 2 a wkly comp to win FA Cup final tkts",
        "Thanks for the birthday wishes! See you soon.",
        "URGENT! You have won a prize. Call 08000123456 now!"
    ]
    
    print("🧪 Testing SMS Spam Detector")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        prediction, probabilities = detector.predict(message)
        spam_confidence = probabilities[1] * 100
        
        print(f"\n📧 Test {i}:")
        print(f"Message: {message}")
        print(f"Prediction: {prediction.upper()}")
        print(f"Spam Confidence: {spam_confidence:.2f}%")
        print(f"Ham Probability: {probabilities[0]:.4f}")
        print(f"Spam Probability: {probabilities[1]:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    test_detector()
