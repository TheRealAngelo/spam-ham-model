#AWAG HILABTA
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train_spam_detector():
    """Train the SMS Spam Detector model and save it using joblib"""
    
    print("Loading and preprocessing data...")
    # Load and clean the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    # Simple normalization
    df['text'] = df['text'].str.lower()
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Distribution:\n{df['label'].value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    # Convert text to numeric features
    print("\nVectorizing text data...")
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train the model
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    # Make predictions
    y_pred = model.predict(X_test_vec)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].round(4)
    print(report_df)
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title('Confusion Matrix - SMS Spam Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    # Save the trained model and vectorizer using joblib
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'models/spam_detector_model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    
    print(" Model and vectorizer saved successfully!")
    print(" Files saved:")
    print("   - models/spam_detector_model.joblib")
    print("   - models/vectorizer.joblib")
    
    return model, vectorizer, accuracy

if __name__ == "__main__":
    train_spam_detector()