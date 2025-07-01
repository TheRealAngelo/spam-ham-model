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
    # Modern Header with gradient
    st.markdown('''
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">🛡️ SMS Guardian AI</h1>
        <p class="subtitle">Advanced Machine Learning-Powered SMS Spam Detection</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #667eea; font-weight: 600;">🎛️ Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Load models with modern spinner
        with st.spinner("🚀 Loading AI Models..."):
            model, vectorizer = load_models()
        
        st.success("✅ Models loaded successfully!")
        
        # Model stats in sidebar
        st.markdown("""
        <div class="info-card">
            <h4 style="margin-bottom: 1rem; color: #374151;">📊 Model Statistics</h4>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 500;">Algorithm:</span>
                <span style="color: #667eea;">Naive Bayes</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 500;">Accuracy:</span>
                <span style="color: #16a34a;">98.2%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight: 500;">Features:</span>
                <span style="color: #667eea;">Text Vectors</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content with modern layout
    col1, col2 = st.columns([2.5, 1.5], gap="large")
    
    with col1:
        st.markdown("""
        <div class="analysis-card">
            <h2 style="color: #374151; font-weight: 600; margin-bottom: 1.5rem;">
                📝 Message Analysis Center
            </h2>
        """, unsafe_allow_html=True)
        
        # Modern text input
        message = st.text_area(
            "✍️ Enter your SMS message:",
            placeholder="Type or paste your message here for instant AI analysis...",
            height=120,
            help="Our AI will analyze the message and determine if it's spam or legitimate.",
            key="message_input"
        )
        
        # Modern analyze button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            analyze_btn = st.button(
                "� Analyze with AI", 
                type="primary", 
                use_container_width=True,
                key="analyze_button"
            )
        
        if analyze_btn:
            if message.strip():
                with st.spinner("🤖 AI is analyzing your message..."):
                    prediction, probabilities = predict_spam(message, model, vectorizer)
                    
                    # Modern results section
                    st.markdown("</div>", unsafe_allow_html=True)  # Close analysis card
                    
                    # Results header
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="color: #374151; font-weight: 600;">🎯 Analysis Results</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction result with modern styling
                    if prediction == 'spam':
                        st.markdown("""
                        <div class="prediction-box spam-box">
                            🚨 SPAM DETECTED!
                            <div style="font-size: 0.9rem; margin-top: 0.5rem; font-weight: 400;">
                                This message appears to be spam
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="prediction-box ham-box">
                            ✅ LEGITIMATE MESSAGE
                            <div style="font-size: 0.9rem; margin-top: 0.5rem; font-weight: 400;">
                                This message appears to be safe
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Modern probability visualization
                    col_prob1, col_prob2 = st.columns(2, gap="large")
                    
                    with col_prob1:
                        st.markdown("### 📊 Probability Distribution")
                        # Enhanced probability bars
                        prob_df = pd.DataFrame({
                            'Category': ['Legitimate', 'Spam'],
                            'Probability': [probabilities[0], probabilities[1]],
                            'Color': ['#16a34a', '#dc2626']
                        })
                        
                        fig = px.bar(
                            prob_df, 
                            x='Category', 
                            y='Probability',
                            color='Color',
                            color_discrete_map={'#16a34a': '#16a34a', '#dc2626': '#dc2626'},
                            title="AI Confidence Levels"
                        )
                        fig.update_layout(
                            showlegend=False, 
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter", size=12),
                            title_font_size=16
                        )
                        fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)')
                        fig.update_yaxes(range=[0, 1], gridcolor='rgba(0,0,0,0.1)')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_prob2:
                        st.markdown("### 🎯 Spam Risk Meter")
                        # Enhanced gauge chart
                        spam_prob = probabilities[1] * 100
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = spam_prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Risk Level (%)", 'font': {'size': 16, 'family': 'Inter'}},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickfont': {'size': 12}},
                                'bar': {'color': "#667eea", 'thickness': 0.3},
                                'steps': [
                                    {'range': [0, 25], 'color': "#dcfce7"},
                                    {'range': [25, 50], 'color': "#fef3c7"},
                                    {'range': [50, 75], 'color': "#fed7aa"},
                                    {'range': [75, 100], 'color': "#fecaca"}
                                ],
                                'threshold': {
                                    'line': {'color': "#dc2626", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 80
                                }
                            }
                        ))
                        fig.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced detailed metrics
                    st.markdown("### 📈 Detailed Analysis")
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        confidence_ham = probabilities[0] * 100
                        st.metric(
                            label="✅ Legitimate Confidence",
                            value=f"{confidence_ham:.1f}%",
                            delta=f"{confidence_ham - 50:.1f}%" if confidence_ham > 50 else None
                        )
                    
                    with metric_col2:
                        confidence_spam = probabilities[1] * 100
                        st.metric(
                            label="🚨 Spam Confidence", 
                            value=f"{confidence_spam:.1f}%",
                            delta=f"{confidence_spam - 50:.1f}%" if confidence_spam > 50 else None
                        )
            else:
                st.warning("⚠️ Please enter a message to analyze.")
        else:
            st.markdown("</div>", unsafe_allow_html=True)  # Close analysis card if no analysis
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #374151; margin-bottom: 1rem;">ℹ️ How It Works</h3>
            <div style="color: #6b7280; line-height: 1.6;">
                <p><strong>1.</strong> Enter your SMS message</p>
                <p><strong>2.</strong> AI analyzes text patterns</p>
                <p><strong>3.</strong> Get instant results with confidence scores</p>
                <p><strong>4.</strong> Make informed decisions about your messages</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Example messages with modern styling
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #374151; margin-bottom: 1rem;">💡 Try These Examples</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Example buttons with modern styling
        example_spam = "🎉 CONGRATULATIONS! You've WON $5000 CASH! Call 555-SCAM NOW to claim your FREE prize! Limited time offer!"
        example_ham = "Hey! Just wanted to confirm our meeting tomorrow at 3 PM. Should I bring the documents?"
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("🚨 Spam Example", use_container_width=True, key="spam_ex"):
                st.session_state.message_input = example_spam
                st.rerun()
        
        with col_ex2:
            if st.button("✅ Safe Example", use_container_width=True, key="ham_ex"):
                st.session_state.message_input = example_ham
                st.rerun()
        
        # Additional info
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #374151; margin-bottom: 0.5rem;">🔬 Technology Stack</h4>
            <div style="color: #6b7280; font-size: 0.9rem;">
                <p>• <strong>Machine Learning:</strong> Scikit-learn</p>
                <p>• <strong>Algorithm:</strong> Multinomial Naive Bayes</p>
                <p>• <strong>Feature Engineering:</strong> Count Vectorization</p>
                <p>• <strong>Dataset:</strong> SMS Spam Collection</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Modern footer
    st.markdown("""
    <div class="footer">
        <h4 style="margin-bottom: 1rem; color: #374151;">🛡️ SMS Guardian AI</h4>
        <p style="margin-bottom: 0.5rem;">Created with ❤️ by <strong>Angelo Morales</strong></p>
        <p style="margin-bottom: 1rem;">Powered by Advanced Machine Learning & Streamlit</p>
        <div style="font-size: 0.8rem; color: #9ca3af;">
            © 2025 SMS Guardian AI • Protecting your digital communication
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()