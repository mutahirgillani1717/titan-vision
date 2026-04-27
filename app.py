# Author: Mutahir Hussainn
# Project: Titan Vision - High-Speed Object Tracking Pipeline

import streamlit as st

import tempfile
import time
from ultralytics import YOLO
import cv2
# --- Page Configuration ---
st.set_page_config(page_title="Titan Vision developed by Mutahir Hussain | AI Tracker", layout="wide", initial_sidebar_state="expanded")

# --- Modern Dashboard CSS ---
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #151A23;
        border-right: 1px solid #2D3748;
    }
    
    /* Typography & Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Highlight Accents */
    .highlight {
        color: #3A86FF; /* Electric Blue Accent */
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #3A86FF !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3A86FF;
        color: #FFFFFF;
        border-radius: 6px;
        border: none;
        width: 100%;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 12px rgba(58, 134, 255, 0.3);
    }
    
    /* File Uploader Container */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #2D3748;
        border-radius: 8px;
        background-color: #1A202C;
    }
    </style>
""", unsafe_allow_html=True)

# --- AI Model Initialization ---
@st.cache_resource
def load_model():
    # Loading the nano model for high-speed portfolio performance
    return YOLO("yolov10n.pt")

model = load_model()

# --- Dictionary for COCO Classes ---
# This makes the UI readable for users instead of showing them raw ID numbers
TARGET_CLASSES = {
    "Person": 0,
    "Bicycle": 1,
    "Car": 2,
    "Motorcycle": 3,
    "Bus": 5,
    "Train": 6,
    "Truck": 7
}

# --- Header Section ---
st.markdown("<h1>Titan Vision <span class='highlight'>Pipeline</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #A0AEC0; font-size: 1.1rem;'>High-speed computer vision pipeline optimized for real-time object tracking, developed by Mutahir Hussain.</p>", unsafe_allow_html=True)
st.divider()

# --- Sidebar UI ---
st.sidebar.markdown("<h3>Control Panel</h3>", unsafe_allow_html=True)

uploaded_video = st.sidebar.file_uploader("Upload Video Feed", type=['mp4', 'avi', 'mov'])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
confidence = st.sidebar.slider("AI Confidence Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
selected_classes = st.sidebar.multiselect(
    "Target Objects to Track",
    options=list(TARGET_CLASSES.keys()),
    default=["Person", "Car", "Motorcycle", "Truck"] # Default safe classes
)

# Convert selected text classes back to YOLO ID numbers
class_ids = [TARGET_CLASSES[cls] for cls in selected_classes]

# --- Main Interface Layout ---
# Top Row: Live Metrics
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    fps_metric = st.empty()
    fps_metric.metric(label="Live FPS", value="0")
with metric_col2:
    frame_metric = st.empty()
    frame_metric.metric(label="Frames Processed", value="0")
with metric_col3:
    object_metric = st.empty()
    object_metric.metric(label="Active Targets", value="0")

st.markdown("<br>", unsafe_allow_html=True)

# Bottom Row: Video Feed and Controls
video_col, control_col = st.columns([8, 2])

with control_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    stop_button = st.button("Stop Processing")

with video_col:
    video_placeholder = st.empty()
    video_placeholder.markdown("""
        <div style='height: 450px; border-radius: 8px; background-color: #151A23; border: 1px solid #2D3748; display: flex; align-items: center; justify-content: center; color: #718096;'>
        Awaiting Video Upload...
        </div>
    """, unsafe_allow_html=True)

# --- Core Processing Loop ---
if uploaded_video is not None and len(class_ids) > 0:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    
    # Variables for FPS calculation
    prev_time = time.time()

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.toast("Processing complete!", icon="✅")
            break

        # Run YOLOv10 Tracking
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=confidence,
            classes=class_ids, 
            verbose=False
        )
        
        # Calculate Metrics
        frame_count += 1
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Safely count detected objects
        objects_on_screen = len(results[0].boxes) if results[0].boxes is not None else 0

        # Update Live Dashboard
        fps_metric.metric(label="Live FPS", value=f"{int(fps)}")
        frame_metric.metric(label="Frames Processed", value=f"{frame_count}")
        object_metric.metric(label="Active Targets", value=f"{objects_on_screen}")

        # Render Video
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
elif uploaded_video is not None and len(class_ids) == 0:
    st.warning("Please select at least one Target Object to track from the sidebar.")
