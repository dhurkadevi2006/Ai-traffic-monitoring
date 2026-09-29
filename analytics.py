"""
analytics.py - Traffic Density Estimation & Analytics Engine
Calculates traffic density levels (LOW, MEDIUM, HIGH), flow rate (Vehicles Per Minute),
aggregates real-time statistics, and manages CSV logging and data export.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import config


class TrafficDensityEstimator:
    """Classifies real-time traffic congestion based on active vehicle counts."""

    def __init__(
        self,
        low_threshold: int = config.DEFAULT_LOW_DENSITY_THRESHOLD,
        medium_threshold: int = config.DEFAULT_MEDIUM_DENSITY_THRESHOLD,
    ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def set_thresholds(self, low: int, medium: int):
        """Update density thresholds dynamically from UI sliders."""
        self.low_threshold = max(1, low)
        self.medium_threshold = max(self.low_threshold + 1, medium)

    def classify(self, active_count: int) -> str:
        """
        Determine traffic density status.

        Returns:
            'LOW', 'MEDIUM', or 'HIGH'
        """
        if active_count <= self.low_threshold:
            return "LOW"
        elif active_count <= self.medium_threshold:
            return "MEDIUM"
        else:
            return "HIGH"


class TrafficAnalytics:
    """
    Traffic Monitoring Analytics & Data Storage Manager.
    Maintains time-series metrics, computes flow rates, and handles CSV exporting.
    """

    def __init__(
        self,
        low_threshold: int = config.DEFAULT_LOW_DENSITY_THRESHOLD,
        medium_threshold: int = config.DEFAULT_MEDIUM_DENSITY_THRESHOLD,
        log_filepath: Optional[Path] = None,
    ):
        self.density_estimator = TrafficDensityEstimator(low_threshold, medium_threshold)
        self.log_filepath = Path(log_filepath or config.DEFAULT_LOG_PATH)
        self.log_filepath.parent.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.last_log_time = 0.0

        # Memory buffer of time-series records for charts & CSV
        self.records: List[Dict[str, Any]] = []

        # Sliding window for flow rate (timestamps of crossed vehicles)
        self.crossing_timestamps: List[float] = []

    def reset(self):
        """Reset all analytics history and timers."""
        self.start_time = time.time()
        self.last_log_time = 0.0
        self.records.clear()
        self.crossing_timestamps.clear()

    def record_crossing(self):
        """Register a vehicle crossing event timestamp for flow rate calculation."""
        self.crossing_timestamps.append(time.time())

    def compute_vehicles_per_minute(self, total_counted: int) -> float:
        """
        Calculate vehicle detection rate per minute.
        Uses recent 60-second window if data exists, otherwise total session rate.
        """
        now = time.time()
        # Clean timestamps older than 60 seconds
        self.crossing_timestamps = [t for t in self.crossing_timestamps if (now - t) <= 60.0]

        if len(self.crossing_timestamps) > 0:
            return float(len(self.crossing_timestamps))  # Already in last 60 seconds

        elapsed = max(1.0, now - self.start_time)
        return (total_counted / elapsed) * 60.0

    def log_snapshot(
        self,
        total_vehicles: int,
        class_counts: Dict[str, int],
        active_vehicles: int,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Periodically records a structured traffic snapshot.

        Returns:
            The recorded snapshot dict if logged, else None.
        """
        now = time.time()
        if not force and (now - self.last_log_time < config.CSV_LOG_INTERVAL_SECONDS):
            return None

        self.last_log_time = now
        density = self.density_estimator.classify(active_vehicles)
        vpm = self.compute_vehicles_per_minute(total_vehicles)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "Timestamp": timestamp_str,
            "Total_Vehicles": total_vehicles,
            "Cars": class_counts.get("car", 0),
            "Motorcycles": class_counts.get("motorcycle", 0),
            "Buses": class_counts.get("bus", 0),
            "Trucks": class_counts.get("truck", 0),
            "Bicycles": class_counts.get("bicycle", 0),
            "Active_Vehicles": active_vehicles,
            "Traffic_Density": density,
            "Flow_Rate_VPM": round(vpm, 1),
        }

        self.records.append(record)
        return record

    def save_to_csv(self, filepath: Optional[Path] = None) -> Path:
        """Saves accumulated traffic records to a CSV file."""
        target = Path(filepath or self.log_filepath)
        df = self.get_dataframe()
        df.to_csv(target, index=False)
        return target

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the time-series logs as a Pandas DataFrame."""
        if not self.records:
            columns = [
                "Timestamp",
                "Total_Vehicles",
                "Cars",
                "Motorcycles",
                "Buses",
                "Trucks",
                "Bicycles",
                "Active_Vehicles",
                "Traffic_Density",
                "Flow_Rate_VPM",
            ]
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.records)

    def to_csv_string(self) -> str:
        """Returns the CSV content as string for Streamlit download."""
        df = self.get_dataframe()
        return df.to_csv(index=False)
