"""
app.py - AI Traffic Monitoring System Dashboard
Main Streamlit application integrating YOLO vehicle detection, multi-object tracking,
virtual line counting, traffic density classification, real-time Plotly charts,
and CSV analytics reporting.
"""

import time
import tempfile
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from detector import VehicleDetector
from tracker import VehicleTracker, LineCounter
from analytics import TrafficAnalytics
from utils.drawing_utils import (
    draw_bounding_boxes,
    draw_virtual_line,
    draw_hud_overlay,
)
from utils.video_utils import (
    FPSCalculator,
    ensure_sample_video_exists,
    get_video_properties,
)
from utils.max_accelerator import accelerator

# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Monitoring System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern dashboard aesthetics
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748b;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    .badge-medium {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #fdba74;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #fca5a5;
    }
    .engine-pill {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        background-color: #e0e7ff;
        color: #3730a3;
        display: inline-block;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "analytics" not in st.session_state:
    st.session_state.analytics = TrafficAnalytics()

if "tracker" not in st.session_state:
    st.session_state.tracker = VehicleTracker()

if "line_counter" not in st.session_state:
    st.session_state.line_counter = LineCounter()

if "detector" not in st.session_state:
    st.session_state.detector = VehicleDetector()
# Default performance settings (can be overridden in sidebar)
if "processing_width" not in st.session_state:
    st.session_state.processing_width = 640  # width in pixels
if "frame_skip" not in st.session_state:
    st.session_state.frame_skip = 1  # skip every other frame by default


# ---------------------------------------------------------------------------
# Sidebar Configuration Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🚦 System Controls")

    # Acceleration & Engine Status Info
    accel_status = accelerator.get_status()
    st.markdown(
        f"""
        <div class="engine-pill">
            ⚡ Backend: {accel_status['engine_name']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Video Source")
    video_source_type = st.radio(
        "Choose Input Feed:",
        options=["Sample Traffic Video", "Upload Video File", "Live Webcam"],
        index=0,
    )

    video_path = None
    webcam_index = 0

    if video_source_type == "Sample Traffic Video":
        sample_file = ensure_sample_video_exists()
        video_path = str(sample_file)
        st.info("Using high-definition test traffic stream.")

    elif video_source_type == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload MP4/AVI/MOV traffic video",
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded_file is not None:
            # Save uploaded bytes to temporary file for OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            st.warning("Please upload a video file to proceed.")

    elif video_source_type == "Live Webcam":
        webcam_index = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)
        st.info("Live webcam mode active.")

    st.markdown("---")
    st.subheader("2. AI Model Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, config.DEFAULT_CONF_THRESHOLD, 0.05)
    iou_threshold = st.slider("IoU NMS Threshold", 0.10, 0.90, config.DEFAULT_IOU_THRESHOLD, 0.05)
    st.session_state.detector.set_thresholds(conf_threshold, iou_threshold)

    # Class filter checkboxes
    selected_classes = st.multiselect(
        "Vehicles to Monitor:",
        options=list(config.TARGET_VEHICLE_CLASSES.values()),
        default=list(config.TARGET_VEHICLE_CLASSES.values()),
    )
    st.session_state.detector.set_allowed_classes(set(selected_classes))

    st.markdown("---")
    st.subheader("3. Counting Sensor Line")
    line_y_ratio = st.slider(
        "Virtual Line Position (Height %)",
        min_value=0.20,
        max_value=0.85,
        value=config.DEFAULT_LINE_POSITION_RATIO,
        step=0.05,
        help="Adjust the vertical height of the virtual tripwire counting sensor.",
    )
    st.session_state.line_counter.set_line_position(line_y_ratio)
    # --- Performance tuning controls ---
    proc_width = st.slider(
        "Processing Resolution (Width px)",
        min_value=320, max_value=960, value=640, step=80,
        help="Resize frames before YOLO inference. Smaller width = faster processing.",
    )
    st.session_state.processing_width = proc_width
    frame_skip = st.slider(
        "Frame Skip", min_value=0, max_value=3, value=0, step=1,
        help="Process every N+1 frames. 0 = every frame.",
    )
    st.session_state.frame_skip = frame_skip

    st.markdown("---")
    st.subheader("4. Traffic Density Thresholds")
    col_th1, col_th2 = st.columns(2)
    with col_th1:
        low_th = st.number_input("Low Max (<)", min_value=1, max_value=20, value=config.DEFAULT_LOW_DENSITY_THRESHOLD)
    with col_th2:
        med_th = st.number_input("Medium Max (<=)", min_value=low_th + 1, max_value=50, value=config.DEFAULT_MEDIUM_DENSITY_THRESHOLD)
    st.session_state.analytics.density_estimator.set_thresholds(low_th, med_th)

    st.markdown("---")
    st.subheader("5. Actions")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if not st.session_state.is_running:
            if st.button("▶ Start Monitoring", type="primary", use_container_width=True):
                st.session_state.is_running = True
                st.rerun()
        else:
            if st.button("⏸ Pause / Stop", type="secondary", use_container_width=True):
                st.session_state.is_running = False
                st.rerun()

    with col_btn2:
        if st.button("🔄 Reset Counts", use_container_width=True):
            st.session_state.tracker.reset()
            st.session_state.line_counter.reset()
            st.session_state.analytics.reset()
            st.success("All counters and history reset.")
            st.rerun()

    st.markdown("---")
    st.subheader("Debug Overlay")
    st.session_state.show_debug = st.checkbox("Show debug overlay (track ID, class, confidence)", value=False)

    # CSV Download Button
    st.markdown("---")
    csv_data = st.session_state.analytics.to_csv_string()
    st.download_button(
        label="📥 Download Analytics (CSV)",
        data=csv_data,
        file_name=f"traffic_report_{int(time.time())}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    # Benchmark toggle (optional per‑frame timing output)
    if 'benchmark_enabled' not in st.session_state:
        st.session_state.benchmark_enabled = False
    st.sidebar.checkbox('Enable benchmark (show per‑frame timings)', key='benchmark_enabled')


# ---------------------------------------------------------------------------
# Main Dashboard Header & KPI Metrics
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🚦 Smart AI Traffic Monitoring & Analytics System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real-time deep learning vehicle detection, multi-object tracking, virtual tripwire counting, and congestion density analytics.</div>',
    unsafe_allow_html=True,
)

# Top KPI Metric Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns([1.2, 1, 1, 1, 1.2])

with kpi_col1:
    total_metric_ph = st.empty()
    total_metric_ph.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Total Counted</div>
            <div class="metric-value">0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    cars_metric_ph = st.empty()
    cars_metric_ph.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Cars</div>
            <div class="metric-value" style="color: #FFA500;">0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    motorcycles_metric_ph = st.empty()
    motorcycles_metric_ph.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Motorcycles</div>
            <div class="metric-value" style="color: #FF69B4;">0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    heavy_metric_ph = st.empty()
    heavy_metric_ph.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Buses & Trucks</div>
            <div class="metric-value" style="color: #1E90FF;">0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col5:
    density_metric_ph = st.empty()
    density_metric_ph.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Traffic Density</div>
            <div style="margin-top: 8px;"><span class="badge-low">LOW</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Layout: Video Feed (Left) & Live Charts (Right)
# ---------------------------------------------------------------------------
feed_col, charts_col = st.columns([1.35, 1.0])

with feed_col:
    st.subheader("📹 Live Video Stream & AI Vision HUD")
    video_placeholder = st.empty()
    status_msg_ph = st.empty()

with charts_col:
    st.subheader("📊 Traffic Analytics")
    st.markdown("**Cumulative Vehicle Flow**")
    line_chart_ph = st.empty()
    st.markdown("**Vehicle Classification Breakdown**")
    bar_chart_ph = st.empty()

# Historical Data Table section below
st.markdown("---")
st.subheader("📋 Real-Time Traffic Analytics Log")
table_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Video Processing Execution Loop
# ---------------------------------------------------------------------------
def run_traffic_monitoring():
    """Main processing loop streaming frames, running AI detection, and updating UI."""
    cap = None

    if video_source_type == "Live Webcam":
        cap = cv2.VideoCapture(webcam_index)
    else:
        if not video_path or not Path(video_path).exists():
            status_msg_ph.error("⚠️ Selected video source file is unavailable.")
            st.session_state.is_running = False
            return
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        status_msg_ph.error("❌ Failed to open video source or camera feed.")
        st.session_state.is_running = False
        return

    fps_calc = FPSCalculator()
    frame_count = 0
    # Benchmark stats initialization
    if 'benchmark_enabled' not in st.session_state:
        st.session_state.benchmark_enabled = False
    if st.session_state.benchmark_enabled:
        st.session_state.bench_stats = {
            'read': 0.0,
            'resize': 0.0,
            'inference': 0.0,
            'tracking': 0.0,
            'counting': 0.0,
            'render': 0.0,
            'total': 0.0,
            'frames': 0,
        }

    while st.session_state.is_running:
        ret, frame = cap.read()
        start_time = time.time()
        read_time = start_time

        # Handle video looping for continuous demonstration
        if not ret:
            if video_source_type != "Live Webcam":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                status_msg_ph.warning("Webcam stream disconnected.")
                break

        frame_count += 1
        h, w = frame.shape[:2]

        # 1. AI Vehicle Detection (optimized)
        # Determine if detection should run based on frame_skip setting
        if frame_count % (st.session_state.frame_skip + 1) == 0:
            # Resize frame for efficient inference
            proc_width = st.session_state.processing_width
            scale_factor = proc_width / w
            proc_height = int(h * scale_factor)
            frame_resized = cv2.resize(frame, (proc_width, proc_height))
            # Run detection on resized frame with inference mode
            with torch.inference_mode():
                detections_resized = st.session_state.detector.detect(frame_resized)
            # Scale detections back to original resolution
            if detections_resized:
                inv_scale = w / proc_width
                for d in detections_resized:
                    d["bbox"] = [int(coord * inv_scale) for coord in d["bbox"]]
                    cx, cy = d["centroid"]
                    d["centroid"] = (int(cx * inv_scale), int(cy * inv_scale))
                detections = detections_resized
            else:
                detections = []
        else:
            # Skip detection this frame; provide empty list to tracker for age handling
            detections = []

        # 2. Multi-Object Tracking
        tracked_objects = st.session_state.tracker.update(detections)

        # 3. Virtual Line Crossing & Counting
        # 3. Virtual Line Crossing & Counting
        total_counted, class_counts, just_crossed = st.session_state.line_counter.update(
            tracked_objects, frame_height=h
        )
        # Retrieve synchronized total count
        total, class_counts = st.session_state.line_counter.get_counts()

        if just_crossed:
            st.session_state.analytics.record_crossing()

        # 4. Traffic Density & Metrics
        active_vehicles = len(tracked_objects)
        density_status = st.session_state.analytics.density_estimator.classify(active_vehicles)
        fps = fps_calc.update()

        # Log time-series snapshot
        snapshot = st.session_state.analytics.log_snapshot(
            total_vehicles=total,
            class_counts=class_counts,
            active_vehicles=active_vehicles,
        )

        # 5. Visual Annotations on Frame
        # A. Bounding boxes, labels, and tracking trails
        annotated_frame = draw_bounding_boxes(frame, tracked_objects, draw_trails=True, show_debug=st.session_state.get("show_debug", False))

        # B. Virtual Counting Sensor Line
        line_recently_triggered = st.session_state.line_counter.is_line_recently_triggered()
        annotated_frame = draw_virtual_line(
            annotated_frame,
            line_y_ratio=st.session_state.line_counter.line_y_ratio,
            crossed_recently=line_recently_triggered,
        )

        # C. Heads-Up Display (HUD) Overlay
        annotated_frame = draw_hud_overlay(
            annotated_frame,
            total_count=total,
            active_count=active_vehicles,
            density_status=density_status,
            fps=fps,
            class_counts=class_counts,
        )

        # Convert OpenCV BGR to RGB for Streamlit rendering
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # 6. Periodic UI Updates (every 3 frames to maintain high video throughput)
        if frame_count % 3 == 0:
            # Update KPI Cards
            total_metric_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Counted</div>
                    <div class="metric-value">{total}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            cars_metric_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Cars</div>
                    <div class="metric-value" style="color: #FFA500;">{class_counts.get('car', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            motorcycles_metric_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Motorcycles</div>
                    <div class="metric-value" style="color: #FF69B4;">{class_counts.get('motorcycle', 0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            buses_trucks = class_counts.get("bus", 0) + class_counts.get("truck", 0)
            heavy_metric_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Buses & Trucks</div>
                    <div class="metric-value" style="color: #1E90FF;">{buses_trucks}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            badge_class = f"badge-{density_status.lower()}"
            density_metric_ph.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Traffic Density</div>
                    <div style="margin-top: 8px;"><span class="{badge_class}">{density_status}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Update Charts and Table
            df_analytics = st.session_state.analytics.get_dataframe()
            if not df_analytics.empty:
                # Cumulative flow line chart
                chart_df = df_analytics[["Timestamp", "Total_Vehicles"]].copy()
                chart_df.set_index("Timestamp", inplace=True)
                line_chart_ph.line_chart(chart_df, height=200)

                # Vehicle classification breakdown bar chart
                dist_df = pd.DataFrame(
                    {
                        "Count": [
                            class_counts.get("car", 0),
                            class_counts.get("motorcycle", 0),
                            class_counts.get("bus", 0),
                            class_counts.get("truck", 0),
                            class_counts.get("bicycle", 0),
                        ]
                    },
                    index=["Car", "Motorcycle", "Bus", "Truck", "Bicycle"],
                )
                bar_chart_ph.bar_chart(dist_df, height=200, color="#0284c7")

                # Update live data table (latest 10 entries)
                table_placeholder.dataframe(
                    df_analytics.tail(10).iloc[::-1],
                    use_container_width=True,
                    hide_index=True,
                )

        # Brief sleep for cooperative multitasking
        time.sleep(0.01)

    cap.release()
    status_msg_ph.info("Monitoring paused. Click '▶ Start Monitoring' to resume.")
    # If benchmark enabled, compute and print average timings
    if st.session_state.get('benchmark_enabled') and st.session_state.get('bench_stats'):
        stats = st.session_state.bench_stats
        frames = stats.get('frames', 1)
        avg = {k: (v / frames) * 1000 for k, v in stats.items() if k != 'frames'}  # ms per frame
        print("\n=== Benchmark Summary ===")
        for stage in ['read', 'resize', 'inference', 'tracking', 'counting', 'render', 'total']:
            print(f"{stage.capitalize():<10}: {avg.get(stage, 0):.2f} ms")
        fps = 1000.0 / avg.get('total', 1) if avg.get('total', 0) > 0 else 0
        print(f"Estimated FPS: {fps:.2f}\n=== End Summary ===\n")


# ---------------------------------------------------------------------------
# Initial Render & Event Triggering
# ---------------------------------------------------------------------------
if st.session_state.is_running:
    run_traffic_monitoring()
else:
    # Display placeholder instruction banner when idle
    video_placeholder.info(
        "👋 Welcome to the AI Traffic Monitoring System!\n\n"
        "1. Select your video source from the left sidebar.\n"
        "2. Adjust detection parameters or the virtual tripwire sensor if desired.\n"
        "3. Click **'▶ Start Monitoring'** to begin real-time vehicle detection and counting."
    )

    # Render static charts and table if previous data exists
    df_analytics = st.session_state.analytics.get_dataframe()
    if not df_analytics.empty:
        chart_df = df_analytics[["Timestamp", "Total_Vehicles"]].copy()
        chart_df.set_index("Timestamp", inplace=True)
        line_chart_ph.line_chart(chart_df, height=200)

        class_counts = st.session_state.line_counter.class_counts
        dist_df = pd.DataFrame(
            {
                "Count": [
                    class_counts.get("car", 0),
                    class_counts.get("motorcycle", 0),
                    class_counts.get("bus", 0),
                    class_counts.get("truck", 0),
                    class_counts.get("bicycle", 0),
                ]
            },
            index=["Car", "Motorcycle", "Bus", "Truck", "Bicycle"],
        )
        bar_chart_ph.bar_chart(dist_df, height=200, color="#0284c7")

        table_placeholder.dataframe(
            df_analytics.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
