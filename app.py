import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import tempfile
import os
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="BioPulse AI | Premium", page_icon="🫀", layout="wide")

# --- ULTRA PREMIUM CSS WITH ANIMATIONS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

:root {
    --primary: #FF2E63;
    --secondary: #08D9D6;
    --bg: #0B0C10;
    --card: rgba(30, 33, 40, 0.6);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Animated Background Gradient */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(255, 46, 99, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(8, 217, 214, 0.05) 0%, transparent 40%),
                #0B0C10;
    color: #EAEAEA;
    overflow-x: hidden;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1100px; padding-top: 3rem; }

/* Floating Background Animation */
@keyframes float {
    0% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-20px) scale(1.05); }
    100% { transform: translateY(0px) scale(1); }
}
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    z-index: -1;
    animation: float 8s infinite ease-in-out alternate;
}
.orb1 { top: 10%; left: 5%; width: 300px; height: 300px; background: rgba(255, 46, 99, 0.15); }
.orb2 { bottom: 10%; right: 5%; width: 350px; height: 350px; background: rgba(8, 217, 214, 0.15); animation-delay: -4s; }

/* Hero Text Gradient Animation */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-title {
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(270deg, #FF2E63, #08D9D6, #FF2E63);
    background-size: 200% 200%;
    animation: gradientShift 5s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    letter-spacing: -2px;
}
.hero-sub {
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 50px;
    font-weight: 300;
}

/* Custom File Uploader */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(255, 46, 99, 0.3) !important;
    border-radius: 20px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--primary) !important;
    background: rgba(255, 46, 99, 0.05) !important;
    box-shadow: 0 0 20px rgba(255, 46, 99, 0.2);
}

/* Button Styling */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #FF2E63 0%, #08D9D6 100%);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 12px 0;
    border: none;
    border-radius: 12px;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 15px rgba(255, 46, 99, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(8, 217, 214, 0.4);
}

/* Glassmorphism Metric Cards */
.glass-card {
    background: var(--card);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    transition: transform 0.3s ease;
    margin-bottom: 20px;
}
.glass-card:hover {
    transform: translateY(-5px);
    border-color: rgba(255, 46, 99, 0.3);
}

/* Heartbeat Animation */
@keyframes heartbeat {
    0% { transform: scale(1); }
    14% { transform: scale(1.1); }
    28% { transform: scale(1); }
    42% { transform: scale(1.1); }
    70% { transform: scale(1); }
}
.heart-icon {
    display: inline-block;
    animation: heartbeat 1.5s infinite;
    font-size: 2rem;
}
.metric-value {
    font-size: 4rem;
    font-weight: 700;
    color: #fff;
    margin: 10px 0;
    text-shadow: 0 0 20px rgba(255, 46, 99, 0.5);
}
.metric-label {
    color: #888;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 3px;
}
</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="hero-title">BioPulse AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Next-Generation Contactless Health Vitals Monitor</div>', unsafe_allow_html=True)

# --- FUNCTION TO PROCESS VIDEO ---
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 30 

    green_signal = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        
        # Center-ROI extraction (Face must be centered)
        x_min, x_max = int(w * 0.35), int(w * 0.65)
        y_min, y_max = int(h * 0.20), int(h * 0.40)
        
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size > 0:
            mean_green = np.mean(roi[:, :, 1])
            green_signal.append(mean_green)

    cap.release()

    if len(green_signal) < fps * 3:
        return None, None, None

    signal = np.array(green_signal)
    if np.std(signal) < 0.1:
        return None, None, None

    signal = signal - np.mean(signal)
    fs = fps
    nyq = 0.5 * fs
    low, high = 0.7 / nyq, 3.5 / nyq
    b, a = butter(1, [low, high], btype='band')
    
    if len(signal) <= 15:
         return None, None, None
         
    filtered_signal = filtfilt(b, a, signal)
    N = len(filtered_signal)
    yf = fft(filtered_signal)
    xf = np.linspace(0.0, 1.0/(2.0*(1/fs)), N//2)
    
    peak_freq = xf[np.argmax(np.abs(yf[0:N//2]))]
    heart_rate = peak_freq * 60
    
    if heart_rate < 40 or heart_rate > 200:
        return None, None, None
        
    hrv = np.std(np.diff(filtered_signal)) * 1000 
    stress_level = "High" if hrv < 10 else ("Moderate" if hrv < 25 else "Low")
    
    return int(heart_rate), stress_level, filtered_signal[:200]

# --- UI LOGIC ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📹 Video Input")
    st.markdown("<p style='color: #888; font-size: 0.9rem;'>Record a 10s video in good lighting. Keep your face exactly in the center of the frame.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drop your video here", type=["mp4", "mov", "avi"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Initialize Scan"):
            with st.spinner("Extracting blood flow signals..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                try:
                    hr, stress, chart_data = process_video(tfile.name)
                except Exception as e:
                    hr, stress, chart_data = None, None, None
                finally:
                    os.unlink(tfile.name) 
                
                if hr is None:
                    st.error("Signal lost. Ensure good lighting and keep your face centered.")
                else:
                    st.session_state['hr'] = hr
                    st.session_state['stress'] = stress
                    st.session_state['chart'] = chart_data

with col2:
    st.markdown("### 📊 Vitals Dashboard")
    
    if 'hr' in st.session_state:
        # Heart Rate Card
        st.markdown(f"""
        <div class="glass-card">
            <div class="heart-icon">🫀</div>
            <div class="metric-label">Heart Rate</div>
            <div class="metric-value">{st.session_state['hr']} <span style="font-size: 1.5rem; color: #888;">BPM</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Stress Card
        stress_color = '#08D9D6' if st.session_state['stress']=='Low' else '#F59E0B' if st.session_state['stress']=='Moderate' else '#FF2E63'
        st.markdown(f"""
        <div class="glass-card" style="padding: 20px;">
            <div class="metric-label">Stress Level</div>
            <div style="font-size: 2rem; font-weight: 700; color: {stress_color}; text-shadow: 0 0 15px {stress_color}80;">{st.session_state['stress']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Waveform Chart
        st.markdown("<h4 style='color: #888; margin-top: 20px;'>Blood Flow Waveform</h4>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state['chart'], 
            mode='lines', 
            line=dict(color='#FF2E63', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 46, 99, 0.1)'
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#888", 
            height=200, 
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.markdown("""
        <div class="glass-card" style="height: 350px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-size: 4rem; opacity: 0.2;">📡</div>
            <h3 style="color: #888; font-weight: 400;">Awaiting Scan</h3>
            <p style="color: #555; font-size: 0.9rem;">Upload a video and click Initialize Scan.</p>
        </div>
        """, unsafe_allow_html=True)
