"""
config.py - Central Configuration for AI Traffic Monitoring System
Contains constants, class mappings, colors, threshold defaults, and file paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
UTILS_DIR = BASE_DIR / "utils"

# Ensure essential directories exist
for directory in [MODELS_DIR, VIDEOS_DIR, OUTPUT_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model Settings
# ---------------------------------------------------------------------------
# Default YOLO model weights (YOLOv8 Nano is lightweight, fast, and CPU-friendly)
DEFAULT_MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# Target vehicle classes from standard 80-class COCO dataset
# ID -> Name
TARGET_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Reverse mapping: Name -> ID
NAME_TO_CLASS_ID = {name: cid for cid, name in TARGET_VEHICLE_CLASSES.items()}

# Colors for bounding boxes and labels (BGR format for OpenCV)
CLASS_COLORS = {
    "car": (0, 165, 255),         # Orange
    "motorcycle": (255, 105, 180), # Hot Pink
    "bus": (30, 144, 255),        # Dodger Blue
    "truck": (50, 205, 50),       # Lime Green
    "bicycle": (238, 130, 238),   # Violet
    "default": (0, 255, 255),     # Yellow
}

# Hex colors for Streamlit charts & badges
CLASS_HEX_COLORS = {
    "car": "#FFA500",
    "motorcycle": "#FF69B4",
    "bus": "#1E90FF",
    "truck": "#32CD32",
    "bicycle": "#EE82EE",
    "total": "#00CED1",
}

# ---------------------------------------------------------------------------
# Detection & Tracking Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45

# Maximum frames a tracked object can be lost before removing from memory
TRACKER_MAX_LOST_FRAMES = 25
# Distance threshold in pixels to associate centroids
TRACKER_MAX_DISTANCE = 80

# Virtual Counting Line Default (percentage of frame height from top: 0.0 to 1.0)
DEFAULT_LINE_POSITION_RATIO = 0.60
# Crossing tolerance in pixels around the virtual line
CROSSING_TOLERANCE_PX = 25

# ---------------------------------------------------------------------------
# Traffic Density Classification Defaults
# ---------------------------------------------------------------------------
# Density is determined by the number of active vehicles currently in the frame
DEFAULT_LOW_DENSITY_THRESHOLD = 3    # <= 3 is LOW
DEFAULT_MEDIUM_DENSITY_THRESHOLD = 7 # 4 to 7 is MEDIUM, >= 8 is HIGH

DENSITY_LEVELS = {
    "LOW": {
        "color": "#28a745",
        "badge_class": "badge-success",
        "description": "Smooth flowing traffic",
    },
    "MEDIUM": {
        "color": "#fd7e14",
        "badge_class": "badge-warning",
        "description": "Moderate vehicle congestion",
    },
    "HIGH": {
        "color": "#dc3545",
        "badge_class": "badge-danger",
        "description": "Heavy traffic jam / slowdown",
    },
}

# ---------------------------------------------------------------------------
# Analytics & Logging
# ---------------------------------------------------------------------------
CSV_LOG_INTERVAL_SECONDS = 1.0  # Log aggregate metrics every 1 second
SAMPLE_VIDEO_FILENAME = "sample_traffic.mp4"
SAMPLE_VIDEO_PATH = VIDEOS_DIR / SAMPLE_VIDEO_FILENAME
DEFAULT_LOG_PATH = DATA_DIR / "traffic_analytics_log.csv"
