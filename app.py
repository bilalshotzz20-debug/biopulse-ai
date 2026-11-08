    # Download XML file safely
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

    # --- FIX FOR OPENCV VERSION ERRORS ---
    try:
        # Purana OpenCV tareeqa (4.8.1 ke liye)
        face_cascade = cv2.CascadeClassifier(xml_path)
    except AttributeError:
        try:
            # Naya OpenCV tareeqa (5.x ke liye)
            face_cascade = cv2.objdetect.CascadeClassifier(xml_path)
        except Exception as e:
            st.error(f"Face detector initialization failed: {e}")
            return None, None, None

    green_signal = []
