"""
tracker.py - Multi-Object Tracking & Virtual Line Counting Module
Tracks vehicles across video frames using persistent ID association and
counts unique vehicles crossing a virtual counting line without double-counting.
"""

import time
import math
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
import config


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / float(unionArea)


class TrackedVehicle:
    """Represents a single tracked vehicle state across multiple frames."""

    def __init__(self, track_id: int, bbox: List[float], class_name: str, confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence

        # Centroid coordinates
        cx = int((bbox[0] + bbox[2]) / 2.0)
        cy = int((bbox[1] + bbox[3]) / 2.0)
        self.centroid = (cx, cy)
        self.prev_centroid = (cx, cy)

        # History of centroids for trajectory and trail rendering
        self.history: List[Tuple[int, int]] = [(cx, cy)]

        self.lost_frames = 0
        self.has_crossed = False
        self.direction = "unknown"  # 'down', 'up', or 'unknown'
        self.first_seen = time.time()

    def update(self, bbox: List[float], class_name: str, confidence: float):
        """Update track with newly matched detection."""
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence

        self.prev_centroid = self.centroid
        cx = int((bbox[0] + bbox[2]) / 2.0)
        cy = int((bbox[1] + bbox[3]) / 2.0)
        self.centroid = (cx, cy)

        # Maintain last 30 positions for trail
        self.history.append((cx, cy))
        if len(self.history) > 30:
            self.history.pop(0)

        # Estimate motion direction
        if cy > self.prev_centroid[1] + 2:
            self.direction = "down"
        elif cy < self.prev_centroid[1] - 2:
            self.direction = "up"

        self.lost_frames = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize track state for rendering and analytics."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "centroid": self.centroid,
            "prev_centroid": self.prev_centroid,
            "history": self.history,
            "lost_frames": self.lost_frames,
            "has_crossed": self.has_crossed,
            "direction": self.direction,
        }


class VehicleTracker:
    """
    Multi-Object Tracker with persistent ID assignment.
    Combines Euclidean centroid distance and Bounding Box IoU.
    """

    def __init__(
        self,
        max_lost_frames: int = config.TRACKER_MAX_LOST_FRAMES,
        max_distance: float = config.TRACKER_MAX_DISTANCE,
    ):
        self.max_lost_frames = max_lost_frames
        self.max_distance = max_distance
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedVehicle] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Associate detections with existing tracks or register new tracks.

        Args:
            detections: List of detection dicts from VehicleDetector

        Returns:
            List of active tracked objects as dicts.
        """
        # If no active tracks, register all detections as new tracks
        if len(self.tracks) == 0:
            for det in detections:
                self._create_track(det)
            return [track.to_dict() for track in self.tracks.values()]

        # If no detections in current frame, age all existing tracks
        if len(detections) == 0:
            lost_ids = []
            for tid, track in self.tracks.items():
                track.lost_frames += 1
                if track.lost_frames > self.max_lost_frames:
                    lost_ids.append(tid)
            for tid in lost_ids:
                del self.tracks[tid]
            return [track.to_dict() for track in self.tracks.values()]

        # Form cost matrix between active tracks and current detections
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]
        det_centroids = [det["centroid"] for det in detections]

        num_tracks = len(track_centroids)
        num_dets = len(det_centroids)

        # Compute Euclidean distance matrix
        dist_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i in range(num_tracks):
            for j in range(num_dets):
                dx = track_centroids[i][0] - det_centroids[j][0]
                dy = track_centroids[i][1] - det_centroids[j][1]
                dist_matrix[i, j] = math.hypot(dx, dy)

        # Greedy matching by minimum distance
        matched_tracks = set()
        matched_dets = set()

        # Sort all (distance, track_idx, det_idx) pairs
        matches = []
        for i in range(num_tracks):
            for j in range(num_dets):
                if dist_matrix[i, j] <= self.max_distance:
                    matches.append((dist_matrix[i, j], i, j))
        matches.sort(key=lambda x: x[0])

        for dist, t_idx, d_idx in matches:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            tid = track_ids[t_idx]
            det = detections[d_idx]
            self.tracks[tid].update(det["bbox"], det["class_name"], det["confidence"])
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        # Unmatched existing tracks: increment lost_frames
        lost_ids = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid].lost_frames += 1
                if self.tracks[tid].lost_frames > self.max_lost_frames:
                    lost_ids.append(tid)
        for tid in lost_ids:
            del self.tracks[tid]

        # Unmatched detections: create new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det)

        return [track.to_dict() for track in self.tracks.values()]

    def _create_track(self, det: Dict[str, Any]):
        """Register a new vehicle track."""
        track = TrackedVehicle(
            track_id=self.next_track_id,
            bbox=det["bbox"],
            class_name=det["class_name"],
            confidence=det["confidence"],
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def reset(self):
        """Clear all active tracks and reset ID counter."""
        self.tracks.clear()
        self.next_track_id = 1


class LineCounter:
    """
    Virtual Line Counting Sensor.
    Monitors tracked vehicle centroids and counts unique crossings.
    Guarantees each vehicle ID is counted only once.
    """

    def __init__(
        self,
        line_y_ratio: float = config.DEFAULT_LINE_POSITION_RATIO,
        tolerance_px: int = config.CROSSING_TOLERANCE_PX,
    ):
        self.line_y_ratio = line_y_ratio
        self.tolerance_px = tolerance_px

        # Counting state
        self.counted_ids: Set[int] = set()
        self.total_count: int = 0
        self.class_counts: Dict[str, int] = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
            "bicycle": 0,
        }
        # Store the most recent resolved class (excluding generic "vehicle") for each track
        self.track_latest_class: Dict[int, str] = {}
        self.recent_crossing_timestamp = 0.0

    def get_counts(self) -> Tuple[int, Dict[str, int]]:
        """Return total counted vehicles and a copy of per‑class counters.
        The total is maintained directly in self.total_count.
        """
        return self.total_count, self.class_counts.copy()

    def set_line_position(self, ratio: float):
        """Dynamically update virtual line height position."""
        self.line_y_ratio = max(0.1, min(0.95, ratio))

    def update(
        self,
        tracked_objects: List[Dict[str, Any]],
        frame_height: int,
    ) -> Tuple[int, Dict[str, int], bool]:
        """
        Check tracked vehicles against the virtual counting line.

        Args:
            tracked_objects: Output from VehicleTracker.update()
            frame_height: Height of the video frame in pixels

        Returns:
            Tuple of:
                - total_count (int)
                - class_counts (dict)
                - just_crossed (bool: True if a vehicle crossed in this frame)
        """
        line_y = int(frame_height * self.line_y_ratio)
        just_crossed = False

        for obj in tracked_objects:
            tid = obj["track_id"]
            if tid in self.counted_ids:
                obj["has_crossed"] = True
                continue

            # Keep the most specific class name seen for this track (ignore generic "vehicle")
            current_name = obj["class_name"].lower()
            if current_name != "vehicle":
                self.track_latest_class[tid] = current_name

            curr_y = obj["centroid"][1]
            prev_y = obj["prev_centroid"][1]

            # Condition 1: Downward crossing (prev_y <= line_y and curr_y >= line_y)
            # Condition 2: Upward crossing (prev_y >= line_y and curr_y <= line_y)
            # Condition 3: Centroid within tolerance band around line_y
            downward_cross = (prev_y <= line_y and curr_y >= line_y)
            upward_cross = (prev_y >= line_y and curr_y <= line_y)
            within_band = abs(curr_y - line_y) <= (self.tolerance_px // 2)

            if downward_cross or upward_cross or (within_band and abs(curr_y - prev_y) > 0):
                self.counted_ids.add(tid)
                obj["has_crossed"] = True
                self.total_count += 1

                # Determine the class to tally: use the most recent specific class if available
                c_name = self.track_latest_class.get(tid, current_name)
                if c_name in self.class_counts:
                    self.class_counts[c_name] += 1
                else:
                    # Fallback for genuinely unknown classes
                    self.class_counts.setdefault(c_name, 0)
                    self.class_counts[c_name] += 1

                just_crossed = True
                self.recent_crossing_timestamp = time.time()

        return self.total_count, self.class_counts, just_crossed


# Original LineCounter methods removed – custom implementation above

    def is_line_recently_triggered(self, decay_seconds: float = 0.5) -> bool:
        """Returns True if a crossing event happened within decay_seconds."""
        return (time.time() - self.recent_crossing_timestamp) < decay_seconds

    def reset(self):
        """Reset counting memory and tallies."""
        self.counted_ids.clear()
        self.total_count = 0
        for k in self.class_counts:
            self.class_counts[k] = 0
        self.recent_crossing_timestamp = 0.0
