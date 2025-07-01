#  SMS Spam Detector

A machine learning-powered SMS spam detection application built with Python, Scikit-learn, and Streamlit.

##  Features

- **Real-time Spam Detection**: Analyze SMS messages instantly
- **Interactive Web Interface**: Beautiful and user-friendly Streamlit dashboard
- **Probability Visualization**: See confidence scores with interactive charts
- **Model Persistence**: Trained models saved using joblib for fast loading
- **Batch Processing**: Analyze multiple messages at once
- **Performance Metrics**: Detailed model evaluation with confusion matrix

##  Project Structure

```
spam-ham-model/
├── main.py              # Streamlit web application
├── train.py             # Model training script
├── utils.py             # Utility functions and SpamDetector class
├── deploy.py            # Deployment automation script
├── requirements.txt     # Python dependencies
├── spam.csv            # Training dataset
├── models/             # Saved model files
│   ├── spam_detector_model.joblib
│   └── vectorizer.joblib
└── README.md           # This file
```

##  Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Run the automated deployment script
python deploy.py
```

This script will:
1.  Install all dependencies
2.  Train the machine learning model
3.  Test the model
4.  Launch the Streamlit application

### Option 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   ```bash
   python train.py
   ```

3. **Test the Model (Optional)**
   ```bash
   python utils.py
   ```

4. **Launch the Application**
   ```bash
   streamlit run main.py
   ```

##  Dependencies

- **streamlit** >= 1.28.0 - Web application framework
- **pandas** >= 1.5.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **scikit-learn** >= 1.3.0 - Machine learning library
- **joblib** >= 1.3.0 - Model serialization
- **plotly** >= 5.15.0 - Interactive visualizations
- **seaborn** >= 0.12.0 - Statistical plotting
- **matplotlib** >= 3.7.0 - Basic plotting

## 🤖 Model Details

### Algorithm
- **Multinomial Naive Bayes**: Excellent for text classification tasks
- **Count Vectorization**: Converts text to numerical features
- **Text Preprocessing**: Lowercase normalization

### Performance
The model achieves high accuracy on the SMS Spam Collection dataset with:
- High precision for spam detection
- Low false positive rate
- Fast prediction times

### Dataset
Uses the SMS Spam Collection v.1 dataset containing:
- 5,574 SMS messages
- Binary classification (ham/spam)
- Real SMS messages in English

## 🎮 How to Use

### Web Interface
1. Open the Streamlit application in your browser
2. Enter an SMS message in the text area
3. Click "Analyze Message"
4. View the prediction results with probability scores

### Example Messages to Try

**Spam Examples:**
- "Congratulations! You've won $1000! Call now to claim your prize!"
- "FREE entry in 2 a wkly comp to win FA Cup final tkts"
- "URGENT! You have won a prize. Call 08000123456 now!"

**Ham Examples:**
- "Hey, are we still meeting for lunch today?"
- "Thanks for the birthday wishes! See you soon."
- "Can you pick up milk on your way home?"

### Programmatic Usage

```python
from utils import SpamDetector

# Initialize detector
detector = SpamDetector()

# Predict single message
prediction, probabilities = detector.predict("Your message here")
print(f"Prediction: {prediction}")
print(f"Spam probability: {probabilities[1]:.4f}")

# Predict multiple messages
messages = ["Message 1", "Message 2"]
predictions, probabilities = detector.predict_batch(messages)
```

## 🛠️ Development

### Training a New Model

To retrain the model with different parameters:

1. Edit the training parameters in `train.py`
2. Run the training script:
   ```bash
   python train.py
   ```
3. The new model will be saved in the `models/` directory

### Customizing the Interface

The Streamlit interface can be customized by editing `main.py`:
- Modify the CSS styling
- Add new visualization components
- Change the layout and colors

##  Model Evaluation

The training script provides comprehensive evaluation metrics:
- **Accuracy Score**: Overall classification accuracy
- **Classification Report**: Precision, recall, and F1-score per class
- **Confusion Matrix**: Visual representation of prediction results
- **Probability Distributions**: Confidence analysis

##  Deployment Options

### Local Deployment
```bash
streamlit run main.py
```

### Cloud Deployment

**Streamlit Cloud:**
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy with one click

**Heroku:**
1. Create a `Procfile`:
   ```
   web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI

**Docker:**
1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "main.py"]
   ```
2. Build and run the container

## 🔧 Command Line Options

The deployment script supports various options:

```bash
# Skip dependency installation
python deploy.py --skip-install

# Skip model training
python deploy.py --skip-train

# Skip model testing
python deploy.py --skip-test

# Only deploy (skip all other steps)
python deploy.py --only-deploy
```

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


##  Acknowledgments

- SMS Spam Collection Dataset by Tiago A. Almeida and José María Gómez Hidalgo
- Scikit-learn community for the excellent machine learning library
- Streamlit team for the amazing web app framework

##  Support

If you encounter any issues or have questions:
1. Check the troubleshooting section below
2. Search existing issues on GitHub
3. Create a new issue with detailed information

##  Troubleshooting

### Common Issues

**Model files not found:**
```
 Model files not found! Please run train.py first to train the model.
```
**Solution:** Run `python train.py` to create the model files.

**Dependencies not installed:**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution:** Run `pip install -r requirements.txt`.

**Dataset not found:**
```
FileNotFoundError: spam.csv
```
**Solution:** Ensure the `spam.csv` file is in the project root directory.

### Performance Tips

- For better performance, ensure you have sufficient RAM (at least 4GB)
- Use SSD storage for faster model loading
- Consider using GPU acceleration for large-scale training

---

**By: Angelo Morales © 2025**
