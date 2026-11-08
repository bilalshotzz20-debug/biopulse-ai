import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import tempfile
import os
import plotly.graph_objects as go
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="BioPulse AI", page_icon="🫀", layout="wide")

# --- PREMIUM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.stApp { background: radial-gradient(circle at 10% 20%, #1E1B4B 0%, #020617 100%); color: #F8FAFC; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1000px; padding-top: 3rem; }
.hero h1 { font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #F43F5E 0%, #8B5CF6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0; }
.hero p { text-align: center; color: #94A3B8; font-size: 1.1rem; margin-bottom: 40px; }
.metric-box { background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; }
.metric-val { font-size: 3rem; font-weight: 800; color: #F43F5E; }
.metric-label { font-size: 1rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>BioPulse AI</h1><p>Contactless Heart Rate & Stress Monitor using Computer Vision</p></div>', unsafe_allow_html=True)

# --- FUNCTION TO PROCESS VIDEO ---
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 30 

    # Download XML file safely (Streamlit fix)
    xml_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            response = requests.get(url, timeout=10)
            with open(xml_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"Failed to download AI model: {e}")
            return None, None, None

    face_cascade = cv2.CascadeClassifier(xml_path)
    green_signal = []

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Forehead ROI
            roi = frame[y + 15 : y + int(h * 0.35), x + 10 : x + w - 10]
            if roi.size > 0:
                mean_green = np.mean(roi[:, :, 1])
                green_signal.append(mean_green)
            break 

    cap.release()

    if len(green_signal) < fps * 3:
        return None, None, None

    signal = np.array(green_signal)
    signal = signal - np.mean(signal)
    
    fs = fps
    nyq = 0.5 * fs
    low = 0.7 / nyq
    high = 3.5 / nyq
    b, a = butter(1, [low, high], btype='band')
    
    if len(signal) <= 15:
         return None, None, None
         
    filtered_signal = filtfilt(b, a, signal)
    
    N = len(filtered_signal)
    yf = fft(filtered_signal)
    xf = np.linspace(0.0, 1.0/(2.0*(1/fs)), N//2)
    
    peak_freq = xf[np.argmax(np.abs(yf[0:N//2]))]
    heart_rate = peak_freq * 60
    
    hrv = np.std(np.diff(filtered_signal)) * 1000 
    stress_level = "High" if hrv < 10 else ("Moderate" if hrv < 25 else "Low")
    
    return int(heart_rate), stress_level, filtered_signal[:200]

# --- UI LOGIC ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📹 Upload Video")
    st.markdown("Please upload a **10 to 15-second** still video of your face. Ensure good lighting and minimal movement.")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        if st.button("Analyze Heart Rate", use_container_width=True):
            with st.spinner("Processing video... Please wait."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                st.info("Video uploaded. Extracting signals...")
                
                try:
                    hr, stress, chart_data = process_video(tfile.name)
                except Exception as e:
                    st.error(f"Processing Error: {e}")
                    hr, stress, chart_data = None, None, None
                finally:
                    os.unlink(tfile.name) 
                
                if hr is None:
                    st.error("Could not detect a face. Ensure good lighting and keep your face still.")
                else:
                    st.session_state['hr'] = hr
                    st.session_state['stress'] = stress
                    st.session_state['chart'] = chart_data
                    st.success("Analysis Complete!")

with col2:
    st.markdown("### 📊 Results")
    if 'hr' in st.session_state:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Heart Rate</div>
            <div class="metric-val">{st.session_state['hr']} <span style="font-size:1.5rem;">BPM</span></div>
        </div>
        <br>
        <div class="metric-box">
            <div class="metric-label">Stress Level</div>
            <div class="metric-val" style="color: {'#10B981' if st.session_state['stress']=='Low' else '#F59E0B' if st.session_state['stress']=='Moderate' else '#F43F5E'}">{st.session_state['stress']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>#### 🌊 Extracted Blood Flow Signal", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state['chart'], mode='lines', line=dict(color='#F43F5E', width=2)))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=200, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting video upload...")