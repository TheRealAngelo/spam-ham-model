import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="🛡️ SMS Guardian AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Modern CSS with advanced styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }
    
    /* Header styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Card styles */
    .analysis-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .spam-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #dc2626;
        border: 2px solid #f87171;
    }
    
    .ham-box {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        color: #16a34a;
        border: 2px solid #4ade80;
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 1rem;
        padding: 1rem;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Text area styles */
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric styles */
    .metric-container {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    /* Example button styles */
    .example-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        transition: all 0.3s ease;
    }
    
    .example-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4);
    }
    
    /* Footer styles */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 2rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 15px;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .analysis-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('models/spam_detector_model.joblib')
        vectorizer = joblib.load('models/vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please run train.py first to train the model.")
        st.stop()

def predict_spam(text, model, vectorizer):
    """Predict if a text message is spam or ham"""
    # Preprocess the text (same as training)
    text = text.lower()
    
    # Vectorize the text
    text_vec = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    return prediction, probability

def main():
    # Header
    st.markdown('<h1 class="main-header"> SMS Spam Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title(" Controls")
    st.sidebar.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        model, vectorizer = load_models()
    
    st.sidebar.success(" Models loaded successfully!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(" Analyze Your Message")
        
        # Text input
        message = st.text_area(
            "Enter your SMS message:",
            placeholder="Type your message here...",
            height=150,
            help="Enter the SMS message you want to analyze for spam detection."
        )
        
        # Prediction button
        if st.button("🚀 Analyze Message", type="primary", use_container_width=True):
            if message.strip():
                with st.spinner("Analyzing message..."):
                    prediction, probabilities = predict_spam(message, model, vectorizer)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader(" Analysis Results")
                    
                    # Prediction result
                    if prediction == 'spam':
                        st.markdown(
                            f'<div class="prediction-box spam-box"> SPAM DETECTED!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box ham-box"> LEGITIMATE MESSAGE</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Probability visualization
                    col_prob1, col_prob2 = st.columns(2)
                    
                    with col_prob1:
                        # Probability bars
                        prob_df = pd.DataFrame({
                            'Category': ['Ham (Legitimate)', 'Spam'],
                            'Probability': [probabilities[0], probabilities[1]],
                            'Color': ['#2e7d32', '#c62828']
                        })
                        
                        fig = px.bar(
                            prob_df, 
                            x='Category', 
                            y='Probability',
                            color='Color',
                            color_discrete_map={'#2e7d32': '#2e7d32', '#c62828': '#c62828'},
                            title="Prediction Probabilities"
                        )
                        fig.update_layout(showlegend=False, height=400)
                        fig.update_yaxes(range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_prob2:
                        # Gauge chart
                        spam_prob = probabilities[1] * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = spam_prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Spam Probability (%)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.markdown("###  Detailed Probabilities")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.metric(
                            label="Ham (Legitimate)",
                            value=f"{probabilities[0]:.4f}",
                            delta=f"{(probabilities[0] - 0.5):.4f}"
                        )
                    
                    with prob_col2:
                        st.metric(
                            label="Spam",
                            value=f"{probabilities[1]:.4f}",
                            delta=f"{(probabilities[1] - 0.5):.4f}"
                        )
            else:
                st.warning(" Please enter a message to analyze.")
    
    with col2:
        st.header(" Information")
        
        st.markdown("""
        <div class="info-box" style="color: black;">
        <h4> How it works:</h4>
        <ul>
        <li>Enter your SMS message in the text area</li>
        <li>Click "Analyze Message" to get predictions</li>
        <li>View the results with probability scores</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model information
        st.subheader(" Model Information")
        st.info("""
        **Algorithm:** Multinomial Naive Bayes  
        **Features:** Count Vectorization  
        **Training Data:** SMS Spam Collection Dataset  
        """)
        
        # Example messages
        st.subheader(" Try These Examples")
        
        example_spam = "Congratulations! You've won $1000! Call now to claim your prize!"
        example_ham = "Hey, are we still meeting for lunch today?"
        
        if st.button(" Example Spam", use_container_width=True):
            st.text_area("Example message:", value=example_spam, key="example_spam", height=100)
        
        if st.button(" Example Ham", use_container_width=True):
            st.text_area("Example message:", value=example_ham, key="example_ham", height=100)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>By: Angelo Morales © 2025</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()