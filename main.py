import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import random


# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    layout="wide",
    initial_sidebar_state="collapsed"  
)
#CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
bubble_html = '<div class="bubbles">'
for _ in range(10):
    size = random.randint(35, 85)
    left = random.randint(5, 90)
    delay = random.uniform(0, 6)
    bubble_html += (
        f'<div class="bubble" '
        f'style="width:{size}px; height:{size}px; left:{left}vw; animation-delay:{delay:.1f}s;"></div>'
    )
bubble_html += '</div>'

st.markdown(bubble_html, unsafe_allow_html=True)
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
#dont touch ty :)))
def predict_spam(text, model, vectorizer):
    """Predict if a text message is spam or ham"""
    # Preprocess
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
        if st.button(" Analyze Message", type="primary", use_container_width=True):
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
                        # chart
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
        <div class="info-box" style="color: white;">
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
        <p>By: Morales, A. ; Oro, M.D © 2025</p>
        <p>Built with Streamlit • Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()