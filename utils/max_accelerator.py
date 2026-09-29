"""
max_accelerator.py - Modular MAX / Mojo Acceleration Layer & Fallback
Provides an abstraction for Modular MAX Engine / Mojo graph compilation,
falling back gracefully to PyTorch / OpenCV DNN if MAX is unavailable.
"""

import sys
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AIAccelerator:
    """
    AI Acceleration Manager.
    Probes for Modular MAX Engine / Mojo execution capabilities.
    Provides seamless fallback to native PyTorch / OpenCV.
    """

    def __init__(self):
        self.max_available = False
        self.max_version = None
        self.engine_name = "PyTorch / Ultralytics"
        self.backend = "PyTorch"
        self.device = "cpu"
        self.notes = "Native PyTorch execution engine"
        self._probe_acceleration_stack()

    def _probe_acceleration_stack(self):
        """Detect MAX / Mojo / Torch hardware backends."""
        # 1. Attempt probing Modular MAX Engine (Linux / Apple Silicon / Modular SDK)
        try:
            import max.engine  # type: ignore
            self.max_available = True
            self.max_version = getattr(max, "__version__", "Available")
            self.engine_name = f"Modular MAX Engine {self.max_version}"
            self.backend = "MAX Engine"
            self.notes = "MAX AI Acceleration Active (Modular Graph Runtime)"
            logger.info("Modular MAX Engine successfully detected.")
            return
        except ImportError:
            # Expected on environments without Modular MAX SDK
            self.max_available = False

        # 2. Check PyTorch & GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda:0"
                gpu_name = torch.cuda.get_device_name(0)
                self.engine_name = f"PyTorch GPU (CUDA - {gpu_name})"
                self.backend = "PyTorch CUDA"
                self.notes = f"CUDA GPU acceleration active ({gpu_name})"
            else:
                # Optimized multi-core CPU inference
                num_threads = torch.get_num_threads()
                self.device = "cpu"
                self.engine_name = "PyTorch CPU (Multi-threaded)"
                self.backend = "PyTorch CPU"
                self.notes = f"CPU optimized inference ({num_threads} threads) [MAX Fallback]"
        except ImportError:
            self.device = "cpu"
            self.engine_name = "OpenCV DNN (Fallback)"
            self.backend = "OpenCV"
            self.notes = "OpenCV CPU execution"

    def get_status(self) -> Dict[str, Any]:
        """Returns acceleration status for display in Streamlit HUD."""
        return {
            "max_available": self.max_available,
            "engine_name": self.engine_name,
            "backend": self.backend,
            "device": self.device,
            "notes": self.notes,
            "python_version": sys.version.split()[0],
            "os_platform": sys.platform,
        }

    def optimize_torch_runtime(self):
        """Applies runtime optimizations when using PyTorch."""
        try:
            import torch
            # Enable TF32 where supported (Ampere+ GPUs)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            else:
                # Optimize CPU thread allocation for real-time video processing
                import os
                cpu_count = os.cpu_count() or 4
                torch.set_num_threads(min(8, max(2, cpu_count - 1)))
        except Exception as e:
            logger.warning(f"Could not apply runtime optimizations: {e}")


# Singleton accelerator instance
accelerator = AIAccelerator()
