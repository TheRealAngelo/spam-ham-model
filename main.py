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
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 3rem;
        background: linear-gradient(135deg, #1f77b4, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e3f2fd;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    .spam-box {
        background: linear-gradient(135deg, #ffebee, #fce4ec);
        color: #c62828;
        border: 2px solid #e57373;
    }
    
    .ham-box {
        background: linear-gradient(135deg, #e8f5e8, #f1f8e9);
        color: #2e7d32;
        border: 2px solid #81c784;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f8fafe, #f0f2f6);
        padding: 2rem;
        border-radius: 16px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 20px rgba(31, 119, 180, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(31, 119, 180, 0.15);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 16px rgba(0, 0, 0, 0.08);
        border: 1px solid #e3f2fd;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);
    }
    
    .example-button {
        background: linear-gradient(135deg, #1f77b4, #42a5f5);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .example-button:hover {
        background: linear-gradient(135deg, #1565c0, #1976d2);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(31, 119, 180, 0.3);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border: 1px solid #e3f2fd;
    }
    
    .footer {
        text-align: center;
        color: #666;
        padding: 3rem 0;
        border-top: 1px solid #e3f2fd;
        margin-top: 3rem;
    }
    
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e3f2fd !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #1f77b4 !important;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1) !important;
    }
    
    .stButton button {
        border-radius: 12px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
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
    st.markdown('<h1 class="main-header">SMS Spam Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("Control Panel")
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        model, vectorizer = load_models()
    
    st.sidebar.success("Models loaded successfully!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Analyze Your Message</h2>', unsafe_allow_html=True)
        
        # Text input
        message = st.text_area(
            "Enter your SMS message:",
            placeholder="Type your message here...",
            height=150,
            help="Enter the SMS message you want to analyze for spam detection."
        )
        
        # Prediction button
        if st.button("Analyze Message", type="primary", use_container_width=True):
            if message.strip():
                with st.spinner("Analyzing message..."):
                    prediction, probabilities = predict_spam(message, model, vectorizer)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<h3 class="section-header">Analysis Results</h3>', unsafe_allow_html=True)
                    
                    # Prediction result
                    if prediction == 'spam':
                        st.markdown(
                            f'<div class="prediction-box spam-box">SPAM DETECTED</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box ham-box">LEGITIMATE MESSAGE</div>',
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
                                'bar': {'color': "#1f77b4"},
                                'steps': [
                                    {'range': [0, 25], 'color': "#e8f5e8"},
                                    {'range': [25, 50], 'color': "#fff3cd"},
                                    {'range': [50, 75], 'color': "#ffeeba"},
                                    {'range': [75, 100], 'color': "#ffebee"}
                                ],
                                'threshold': {
                                    'line': {'color': "#c62828", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.markdown('<h4 class="section-header">Detailed Probabilities</h4>', unsafe_allow_html=True)
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h4 style="color: #2e7d32; margin-bottom: 0.5rem;">Ham (Legitimate)</h4>
                            <h2 style="color: #1f77b4; margin: 0;">{probabilities[0]:.4f}</h2>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with prob_col2:
                        st.markdown(f'''
                        <div class="metric-card">
                            <h4 style="color: #c62828; margin-bottom: 0.5rem;">Spam</h4>
                            <h2 style="color: #1f77b4; margin: 0;">{probabilities[1]:.4f}</h2>
                        </div>
                        ''', unsafe_allow_html=True)
            else:
                st.warning("Please enter a message to analyze.")
    
    with col2:
        st.markdown('<h2 class="section-header">Information</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
        <h4 style="color: #1f77b4; margin-bottom: 1rem;">How it works:</h4>
        <ul style="color: #666; line-height: 1.6;">
        <li>Enter your SMS message in the text area</li>
        <li>Click "Analyze Message" to get predictions</li>
        <li>View the results with probability scores</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Model information
        st.markdown('<h3 class="section-header">Model Information</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <p style="color: #666; margin-bottom: 0.5rem;"><strong>Algorithm:</strong> Multinomial Naive Bayes</p>
        <p style="color: #666; margin-bottom: 0.5rem;"><strong>Features:</strong> Count Vectorization</p>
        <p style="color: #666; margin: 0;"><strong>Training Data:</strong> SMS Spam Collection Dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example messages
        st.markdown('<h3 class="section-header">Try These Examples</h3>', unsafe_allow_html=True)
        
        example_spam = "Congratulations! You've won $1000! Call now to claim your prize!"
        example_ham = "Hey, are we still meeting for lunch today?"
        
        if st.button("Example Spam Message", use_container_width=True):
            st.text_area("Example message:", value=example_spam, key="example_spam", height=100)
        
        if st.button("Example Legitimate Message", use_container_width=True):
            st.text_area("Example message:", value=example_ham, key="example_ham", height=100)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>By: Angelo Morales © 2025</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()