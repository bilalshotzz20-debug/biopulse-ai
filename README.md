# 🫀 BioPulse AI: Contactless Heart Rate Monitor

An advanced Computer Vision application that detects Heart Rate and Stress Levels using just a smartphone camera. Built using Remote Photoplethysmography (rPPG) technology.

🔗 **Live Demo:** [https://biopulseai-fcdvntwwhyvcrx7qp7hmvn.streamlit.app/]

## 💡 How it Works
The app analyzes microscopic changes in the green color channel of your face (forehead) caused by blood flow. Using Fast Fourier Transform (FFT) and Bandpass filters, it calculates the Heart Rate (BPM) and Heart Rate Variability (HRV) to estimate stress levels.

## 🛠️ Tech Stack
- **Computer Vision:** OpenCV
- **Signal Processing:** SciPy, NumPy
- **Frontend:** Streamlit, Plotly
- **Deployment:** Streamlit Cloud (Python 3.11)

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
