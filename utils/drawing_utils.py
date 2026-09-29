"""
drawing_utils.py - Visualization and Annotation Utilities
Provides methods to render bounding boxes, object IDs, tracking trails,
virtual counting lines, and real-time Heads-Up Displays (HUD) onto video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import config


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Return BGR color tuple for a vehicle class."""
    return config.CLASS_COLORS.get(class_name.lower(), config.CLASS_COLORS["default"])


def draw_bounding_boxes(
    frame: np.ndarray,
    tracked_objects: List[Dict[str, Any]],
    draw_trails: bool = True,
    show_debug: bool = False,
) -> np.ndarray:
    """
    Renders bounding boxes, class labels, tracking IDs, and centroid points.

    Args:
        frame: OpenCV BGR image
        tracked_objects: List of dicts containing:
            - 'track_id': int
            - 'bbox': [x1, y1, x2, y2]
            - 'class_name': str
            - 'confidence': float
            - 'centroid': (cx, cy)
            - 'history': list of past centroids [(x, y), ...]
            - 'has_crossed': bool
    """
    annotated = frame.copy()

    for obj in tracked_objects:
        bbox = obj.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        class_name = obj.get("class_name", "vehicle")
        conf = obj.get("confidence", 0.0)
        track_id = obj.get("track_id", -1)
        has_crossed = obj.get("has_crossed", False)
        centroid = obj.get("centroid", ((x1 + x2) // 2, (y1 + y2) // 2))

        color = get_class_color(class_name)

        # Draw bounding box (thicker if just crossed)
        thickness = 3 if has_crossed else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label content: e.g. "#4 Car 91%"
        label = f"#{track_id} {class_name.capitalize()} {int(conf * 100)}%"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw filled background rectangle for label
        label_y1 = max(0, y1 - label_h - 8)
        label_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + label_w + 6, label_y2),
            color,
            -1,
        )

        # Draw label text in black or white depending on contrast
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        # Optional debug overlay with track ID, class, confidence
        if show_debug:
            debug_text = f"ID:{track_id} {class_name.capitalize()} {int(conf * 100)}%"
            cv2.putText(
                annotated,
                debug_text,
                (x1 + 3, label_y2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


        # Draw centroid point
        cx, cy = int(centroid[0]), int(centroid[1])
        centroid_color = (0, 255, 0) if has_crossed else (0, 0, 255)
        cv2.circle(annotated, (cx, cy), 4, centroid_color, -1)

        # Draw motion trail (past centroids)
        if draw_trails and "history" in obj and len(obj["history"]) > 1:
            history = obj["history"]
            for i in range(1, len(history)):
                pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                alpha = i / len(history)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(annotated, pt1, pt2, trail_color, 2)

    return annotated


def draw_virtual_line(
    frame: np.ndarray,
    line_y_ratio: float,
    crossed_recently: bool = False,
) -> np.ndarray:
    """
    Renders the virtual counting trigger line across the frame.

    Args:
        frame: OpenCV BGR image
        line_y_ratio: Normalized vertical coordinate (0.0 to 1.0)
        crossed_recently: If True, flashes green to signal active counting trigger
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    line_y = int(h * line_y_ratio)

    # Line color: Green pulse if triggered, else Electric Cyan
    line_color = (0, 255, 0) if crossed_recently else (255, 255, 0)  # BGR
    thickness = 3 if crossed_recently else 2

    # Draw the horizontal virtual counting line
    cv2.line(annotated, (0, line_y), (w, line_y), line_color, thickness)

    # Add text label along the line
    line_label = "--- VIRTUAL COUNTING LINE ---"
    if crossed_recently:
        line_label += " [COUNT TRIGGERED!]"

    cv2.putText(
        annotated,
        line_label,
        (20, max(25, line_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        line_color,
        2,
        cv2.LINE_AA,
    )

    return annotated


def draw_hud_overlay(
    frame: np.ndarray,
    total_count: int,
    active_count: int,
    density_status: str,
    fps: float,
    class_counts: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Draws a clean, professional semi-transparent Heads-Up Display (HUD)
    showing real-time metrics directly on the video frame.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Create semi-transparent top banner for metrics
    banner_height = 42
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Density status color
    if density_status == "LOW":
        density_bgr = (0, 200, 0)     # Green
    elif density_status == "MEDIUM":
        density_bgr = (0, 165, 255)   # Orange
    else:
        density_bgr = (0, 0, 220)     # Red

    # Metrics text on banner
    metrics_str = f"TOTAL COUNTED: {total_count}  |  ACTIVE: {active_count}  |  DENSITY: "
    cv2.putText(
        annotated,
        metrics_str,
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    (metrics_w, _), _ = cv2.getTextSize(
        metrics_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    # Draw colored density badge text
    cv2.putText(
        annotated,
        density_status,
        (15 + metrics_w, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        density_bgr,
        2,
        cv2.LINE_AA,
    )

    # FPS counter on top right
    fps_str = f"FPS: {fps:.1f}"
    (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(
        annotated,
        fps_str,
        (w - fps_w - 20, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Bottom sub-banner with vehicle breakdowns if room permits
    if class_counts and w >= 640:
        sub_banner_h = 28
        sub_overlay = annotated.copy()
        cv2.rectangle(
            sub_overlay, (0, h - sub_banner_h), (w, h), (15, 15, 15), -1
        )
        cv2.addWeighted(sub_overlay, 0.70, annotated, 0.30, 0, annotated)

        breakdown_text = (
            f"Cars: {class_counts.get('car', 0)} | "
            f"Bikes: {class_counts.get('motorcycle', 0)} | "
            f"Buses: {class_counts.get('bus', 0)} | "
            f"Trucks: {class_counts.get('truck', 0)} | "
            f"Bicycles: {class_counts.get('bicycle', 0)}"
        )
        cv2.putText(
            annotated,
            breakdown_text,
            (15, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return annotated
