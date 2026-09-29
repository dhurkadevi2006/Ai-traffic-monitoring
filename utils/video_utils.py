"""
video_utils.py - Video Stream Handlers and Sample Traffic Generator
Provides utilities for video source loading, FPS calculation,
and generating a realistic sample traffic video for instant testing.
"""

import time
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, Generator
import cv2
import numpy as np
import config


class FPSCalculator:
    """Calculates smoothed exponential moving average Frames Per Second."""

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.prev_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        """Call each frame to update and return current smoothed FPS."""
        curr_time = time.perf_counter()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time

        if delta > 0:
            instant_fps = 1.0 / delta
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                self.fps = (self.alpha * self.fps) + ((1.0 - self.alpha) * instant_fps)
        return self.fps


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """Extract width, height, total frames, and FPS from a VideoCapture object."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def download_sample_traffic_video(destination_path: Optional[Path] = None) -> bool:
    """
    Downloads a lightweight, high-quality public domain traffic CCTV video clip.
    Returns True if download succeeded, False otherwise.
    """
    dest = Path(destination_path or config.SAMPLE_VIDEO_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Reliable public sample video URLs for traffic monitoring
    urls = [
        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/traffic.mp4",
    ]

    for url in urls:
        try:
            print(f"[VideoUtils] Downloading sample traffic clip from: {url}")
            urllib.request.urlretrieve(url, str(dest))
            if dest.exists() and dest.stat().st_size > 10000:
                print(f"[VideoUtils] Download complete: {dest} ({dest.stat().st_size // 1024} KB)")
                return True
        except Exception as e:
            print(f"[VideoUtils] Download failed from {url}: {e}")

    return False


def generate_synthetic_traffic_video(
    output_path: Optional[Path] = None,
    duration_sec: int = 15,
    fps: int = 25,
    resolution: Tuple[int, int] = (854, 480),
) -> Path:
    """
    Generates a realistic synthetic traffic video with animated vehicles,
    road lanes, dashed lines, and varied vehicle types (cars, buses, bikes, trucks).
    Guarantees the system has a valid test video without internet connection.
    """
    dest = Path(output_path or config.SAMPLE_VIDEO_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)

    w, h = resolution
    total_frames = duration_sec * fps

    # Define video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))

    # Lane boundaries (4-lane road: 2 northbound, 2 southbound)
    road_left = int(w * 0.18)
    road_right = int(w * 0.82)
    road_width = road_right - road_left
    lane_width = road_width // 4
    lanes = [
        road_left + lane_width * 0 + lane_width // 2,
        road_left + lane_width * 1 + lane_width // 2,
        road_left + lane_width * 2 + lane_width // 2,
        road_left + lane_width * 3 + lane_width // 2,
    ]

    # Vehicle types with visual attributes
    vehicle_templates = [
        {"type": "car", "w": 46, "h": 78, "colors": [(30, 30, 200), (200, 30, 30), (180, 180, 180), (40, 180, 40)]},
        {"type": "bus", "w": 62, "h": 140, "colors": [(220, 160, 20), (30, 120, 220)]},
        {"type": "truck", "w": 66, "h": 160, "colors": [(60, 140, 60), (100, 100, 100)]},
        {"type": "motorcycle", "w": 26, "h": 50, "colors": [(180, 50, 180), (30, 180, 220)]},
    ]

    # Spawn schedule for simulated vehicles
    vehicles = []
    np.random.seed(42)

    # Pre-generate 30 vehicles with arrival frames
    for i in range(28):
        lane_idx = int(np.random.randint(0, 4))
        lane_x = lanes[lane_idx]
        is_downward = lane_idx in [0, 1]  # Left lanes go down, right lanes go up

        v_type_prob = np.random.rand()
        if v_type_prob < 0.55:
            template = vehicle_templates[0]  # Car
        elif v_type_prob < 0.70:
            template = vehicle_templates[1]  # Bus
        elif v_type_prob < 0.85:
            template = vehicle_templates[2]  # Truck
        else:
            template = vehicle_templates[3]  # Motorcycle

        color = template["colors"][np.random.randint(0, len(template["colors"]))]
        speed = np.random.uniform(4.0, 7.5) if is_downward else np.random.uniform(-7.5, -4.0)
        start_frame = int(np.random.randint(0, max(1, total_frames - 80)))
        start_y = -template["h"] if is_downward else h + template["h"]

        vehicles.append({
            "id": i + 1,
            "type": template["type"],
            "w": template["w"],
            "h": template["h"],
            "color": color,
            "x": lane_x,
            "y": start_y,
            "speed": speed,
            "is_downward": is_downward,
            "start_frame": start_frame,
        })

    dash_offset = 0

    for frame_idx in range(total_frames):
        # 1. Background: Grass / Terrain borders
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (35, 95, 45)  # Natural green grass

        # 2. Road surface (asphalt gray)
        cv2.rectangle(frame, (road_left, 0), (road_right, h), (55, 55, 55), -1)

        # 3. Shoulder lines (solid white)
        cv2.line(frame, (road_left, 0), (road_left, h), (255, 255, 255), 3)
        cv2.line(frame, (road_right, 0), (road_right, h), (255, 255, 255), 3)

        # 4. Center dividing double-yellow lines
        center_x = (road_left + road_right) // 2
        cv2.line(frame, (center_x - 3, 0), (center_x - 3, h), (0, 215, 255), 2)
        cv2.line(frame, (center_x + 3, 0), (center_x + 3, h), (0, 215, 255), 2)

        # 5. Lane dashed white dividers (animated motion)
        dash_offset = (dash_offset + 3) % 40
        for lane_sep_x in [road_left + lane_width, center_x + lane_width]:
            for y_seg in range(-40 + dash_offset, h + 40, 40):
                cv2.line(frame, (lane_sep_x, y_seg), (lane_sep_x, y_seg + 22), (230, 230, 230), 2)

        # 6. Render active vehicles
        for v in vehicles:
            if frame_idx >= v["start_frame"]:
                v["y"] += v["speed"]

                # Only draw if visible on screen
                if -v["h"] <= v["y"] <= h + v["h"]:
                    vx = int(v["x"] - v["w"] // 2)
                    vy = int(v["y"] - v["h"] // 2)

                    # Vehicle body
                    cv2.rectangle(frame, (vx, vy), (vx + v["w"], vy + v["h"]), v["color"], -1)
                    cv2.rectangle(frame, (vx, vy), (vx + v["w"], vy + v["h"]), (20, 20, 20), 2)

                    # Windshields & details
                    windshield_h = max(6, int(v["h"] * 0.18))
                    if v["is_downward"]:
                        # Front windshield near bottom
                        cv2.rectangle(
                            frame,
                            (vx + 4, vy + v["h"] - windshield_h - 6),
                            (vx + v["w"] - 4, vy + v["h"] - 6),
                            (180, 230, 240),
                            -1,
                        )
                        # Headlights
                        cv2.circle(frame, (vx + 6, vy + v["h"] - 2), 3, (220, 255, 255), -1)
                        cv2.circle(frame, (vx + v["w"] - 6, vy + v["h"] - 2), 3, (220, 255, 255), -1)
                    else:
                        # Front windshield near top
                        cv2.rectangle(
                            frame,
                            (vx + 4, vy + 6),
                            (vx + v["w"] - 4, vy + 6 + windshield_h),
                            (180, 230, 240),
                            -1,
                        )
                        # Headlights
                        cv2.circle(frame, (vx + 6, vy + 2), 3, (220, 255, 255), -1)
                        cv2.circle(frame, (vx + v["w"] - 6, vy + 2), 3, (220, 255, 255), -1)

        writer.write(frame)

    writer.release()
    print(f"[VideoUtils] Successfully generated synthetic traffic video: {dest}")
    return dest


def ensure_sample_video_exists() -> Path:
    """Checks if a sample video exists; if not, attempts download or generates synthetic video."""
    if config.SAMPLE_VIDEO_PATH.exists() and config.SAMPLE_VIDEO_PATH.stat().st_size > 5000:
        return config.SAMPLE_VIDEO_PATH

    # Attempt download first for authentic CCTV look
    if download_sample_traffic_video(config.SAMPLE_VIDEO_PATH):
        return config.SAMPLE_VIDEO_PATH

    # Fallback to high-quality synthetic generator
    return generate_synthetic_traffic_video(config.SAMPLE_VIDEO_PATH)
