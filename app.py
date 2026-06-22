import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from doodles import get_doodles_html_css

# Page Configuration
st.set_page_config(
    page_title="Cricket Score Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load metadata
@st.cache_resource
def load_meta():
    with open('meta_info.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Load trained pipeline
@st.cache_resource
def load_model():
    with open('cricket_score_predictor.pkl', 'rb') as f:
        return pickle.load(f)

try:
    meta = load_meta()
    model = load_model()
except Exception as e:
    st.error("Failed to load model or metadata. Please run train_model.py first.")
    st.stop()

# Inject background doodles & custom CSS
st.markdown(get_doodles_html_css(), unsafe_allow_html=True)

# Custom CSS for styling the app components
st.markdown("""
<style>
/* Overall background and styling */
.stApp {
    background-color: #0A2A66 !important;
    background-image: radial-gradient(circle at 50% 50%, #153e8a 0%, #0A2A66 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Inter', 'Roboto', sans-serif !important;
}

/* Header style */
.main-header {
    font-size: 3rem;
    font-weight: 800;
    color: #FF9933;
    text-align: center;
    margin-bottom: 5px;
    text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.4);
    letter-spacing: 1px;
}

.sub-header {
    font-size: 1.1rem;
    color: #FFCC80;
    text-align: center;
    margin-bottom: 30px;
    font-weight: 400;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
}

/* Glassmorphism Card for Form */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.11) !important;
    backdrop-filter: blur(15px) !important;
    -webkit-backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 20px !important;
    padding: 35px !important;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4) !important;
    margin-bottom: 25px !important;
}

/* Style Form fields */
div[data-baseweb="select"] > div {
    background-color: rgba(255, 255, 255, 0.95) !important;
    color: #0A2A66 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 204, 128, 0.4) !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] svg {
    fill: #0A2A66 !important;
}

div[data-testid="stForm"] label {
    color: #FFCC80 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin-bottom: 5px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Slider Custom Styling */
div[data-testid="stSlider"] label {
    color: #FFCC80 !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #FF9933 !important;
    border: 2px solid #FFCC80 !important;
}

div[data-testid="stSlider"] div[data-key] {
    color: #FFFFFF !important;
}

/* Custom Predict Button */
div.stButton > button {
    background: linear-gradient(135deg, #FF9933 0%, #FF5500 100%) !important;
    color: white !important;
    border: none !important;
    padding: 14px 28px !important;
    border-radius: 30px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(255, 85, 0, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    width: 100% !important;
    cursor: pointer !important;
    margin-top: 15px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #FFBB55 0%, #FF7700 100%) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(255, 85, 0, 0.6) !important;
}

div.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Result Card styling */
.result-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    border: 3px solid #FF9933;
    margin-top: 25px;
    color: #0A2A66 !important;
    animation: fadeInUp 0.6s cubic-bezier(0.23, 1, 0.32, 1) both;
}

.result-card h3 {
    color: #0A2A66 !important;
    font-size: 20px;
    margin-top: 10px;
    margin-bottom: 5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.score-display {
    color: #FF5500 !important;
    font-size: 64px;
    font-weight: 900;
    margin: 15px 0;
    text-shadow: 2px 2px 4px rgba(255, 85, 0, 0.15);
}

.range-display {
    color: #444444;
    font-size: 18px;
    margin-bottom: 0;
}

.cricket-icon-large {
    font-size: 55px;
    animation: pulse 2s infinite ease-in-out;
    display: inline-block;
}

/* Custom Warning box */
.warning-box {
    background: rgba(255, 85, 0, 0.15);
    border-left: 5px solid #FF5500;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
    color: #FFCC80;
    font-weight: 500;
}

/* Keyframes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Layout Title
st.markdown('<div class="main-header">Cricket Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Premium IPL Score Prediction Engine</div>', unsafe_allow_html=True)

# Create input form inside card
with st.form(key="predict_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Default to a popular team (Kolkata Knight Riders)
        default_bat = meta['teams'].index("Kolkata Knight Riders") if "Kolkata Knight Riders" in meta['teams'] else 0
        bat_team = st.selectbox(
            "Batting Team",
            options=meta['teams'],
            index=default_bat
        )
        
    with col2:
        # Default to a popular team (Royal Challengers Bangalore)
        default_bowl = meta['teams'].index("Royal Challengers Bangalore") if "Royal Challengers Bangalore" in meta['teams'] else 1
        bowl_team = st.selectbox(
            "Bowling Team",
            options=meta['teams'],
            index=default_bowl
        )
        
    # Venue selector
    default_venue = meta['venues'].index("M Chinnaswamy Stadium") if "M Chinnaswamy Stadium" in meta['venues'] else 0
    venue = st.selectbox(
        "Venue / Stadium",
        options=meta['venues'],
        index=default_venue
    )
    
    st.write("---")
    st.write("**Match State Inputs**")
    
    col3, col4 = st.columns(2)
    
    with col3:
        runs = st.slider(
            "Current Runs Scored",
            min_value=0,
            max_value=250,
            value=80,
            step=1
        )
        
        overs = st.slider(
            "Overs Bowled (Must be >= 5.0)",
            min_value=5.0,
            max_value=20.0,
            value=10.0,
            step=0.1
        )
        
    with col4:
        wickets = st.slider(
            "Wickets Fallen",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )
        
        runs_last_5 = st.slider(
            "Runs in Last 5 Overs",
            min_value=0,
            max_value=100,
            value=40,
            step=1
        )
        
    wickets_last_5 = st.slider(
        "Wickets in Last 5 Overs",
        min_value=0,
        max_value=7,
        value=1,
        step=1
    )
    
    # Form submission button
    submit_button = st.form_submit_button(label="Predict Score")

# Prediction logic & validation
if submit_button:
    # Validations
    has_errors = False
    
    if bat_team == bowl_team:
        st.markdown(
            f'<div class="warning-box">⚠️ Batting Team ({bat_team}) and Bowling Team ({bowl_team}) cannot be the same. Please select different teams.</div>', 
            unsafe_allow_html=True
        )
        has_errors = True
        
    if runs_last_5 > runs:
        st.markdown(
            f'<div class="warning-box">⚠️ Runs in the last 5 overs ({runs_last_5}) cannot exceed total runs ({runs}).</div>',
            unsafe_allow_html=True
        )
        has_errors = True
        
    if wickets_last_5 > wickets:
        st.markdown(
            f'<div class="warning-box">⚠️ Wickets in the last 5 overs ({wickets_last_5}) cannot exceed total wickets ({wickets}).</div>',
            unsafe_allow_html=True
        )
        has_errors = True

    if not has_errors:
        # Build input dataframe for prediction
        # Ensure column order matches training data:
        # ['bat_team', 'bowl_team', 'venue', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']
        input_data = pd.DataFrame({
            'bat_team': [bat_team],
            'bowl_team': [bowl_team],
            'venue': [venue],
            'runs': [runs],
            'wickets': [wickets],
            'overs': [overs],
            'runs_last_5': [runs_last_5],
            'wickets_last_5': [wickets_last_5]
        })
        
        try:
            # Predict
            pred = model.predict(input_data)[0]
            predicted_score = int(round(pred))
            
            # Bound prediction reasonably
            if wickets == 10:
                # If all out, final score is the current score
                predicted_score = runs
                min_score = runs
                max_score = runs
            else:
                # Ensure predicted score is at least the current score
                predicted_score = max(runs, predicted_score)
                min_score = max(runs, predicted_score - 5)
                max_score = predicted_score + 5
            
            # Display results in premium styled layout
            st.markdown(f"""
            <div class="result-card">
                <div class="cricket-icon-large">🏏</div>
                <h3>Predicted Final Score</h3>
                <div class="score-display">{predicted_score}</div>
                <div class="range-display">Predicted Range: <strong>{min_score} - {max_score}</strong> runs</div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
