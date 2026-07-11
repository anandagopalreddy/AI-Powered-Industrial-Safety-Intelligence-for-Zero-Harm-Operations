"""
Zero-Harm — PPE / Restricted-Area Vision Agent
==================================================
HONEST SCOPE NOTE — READ THIS BEFORE DEMOING IT TO A JUDGE
--------------------------------------------------------------
This module is a REAL integration point for YOLOv8-based PPE detection, but it
ships in SIMULATED mode by default, and you should say so if asked. Two
concrete blockers stopped this from being a trained, working detector today:

  1. Stock YOLOv8 checkpoints (yolov8n/s/m, pretrained on COCO) do not have
     "helmet" / "safety vest" classes at all — COCO's closest classes are
     generic "person". Genuine helmet/vest/restricted-zone detection needs a
     model fine-tuned on a labelled PPE dataset (several public ones exist,
     e.g. on Roboflow Universe), which means a training pass this environment
     cannot do without that data and GPU time.
  2. This build environment has no network access, so it can't even download
     a stock YOLOv8 checkpoint to prove the plumbing end-to-end.

What IS real: the integration code below calls the actual `ultralytics`
YOLO API when the package and a model file are present, exactly the way you'd
call it in production. If you train (or download) a PPE-fine-tuned .pt model
and place it at the path below, this module starts returning genuine
detections with no code changes — only MODEL_PATH needs to point at it, and
`pip install ultralytics` needs to be run.

Until then, `detect_ppe()` returns a clearly-labelled SIMULATED compliance
signal (`"source": "simulated"`) so the dashboard, the API response, and this
docstring never let a simulated number pass as a measured one. Present this
to judges as "the integration is built and tested against the real API
surface; the trained detection model is the next step," not as a working PPE
detector today.
"""

import hashlib
import os
import random
import time
from typing import Optional

MODEL_PATH = os.environ.get("ZEROHARM_PPE_MODEL_PATH", "models/ppe_yolov8.pt")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

PPE_CLASSES = ["helmet", "safety_vest", "restricted_area_entry"]


class VisionAgent:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.real_model_loaded = False

        if ULTRALYTICS_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.real_model_loaded = True
            except Exception:
                # Fall through to simulated mode rather than crashing the API
                # if the weights file is corrupt/incompatible.
                self.model = None
                self.real_model_loaded = False

    def detect_ppe(self, zone_id: str, image_bytes: Optional[bytes] = None) -> dict:
        """
        Returns a structured PPE/restricted-area compliance result for one
        zone. If a real YOLOv8 PPE model is loaded AND an image was supplied,
        runs genuine inference. Otherwise returns a clearly labelled
        simulated result so it is never mistaken for a measured detection.
        """
        if self.real_model_loaded and image_bytes is not None:
            return self._run_real_inference(zone_id, image_bytes)
        return self._simulate_detection(zone_id)

    def _run_real_inference(self, zone_id: str, image_bytes: bytes) -> dict:
        import io
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        results = self.model.predict(image, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_name = self.model.names.get(int(box.cls[0]), "unknown")
                detections.append({
                    "class": cls_name,
                    "confidence": round(float(box.conf[0]) * 100, 1),
                })
        helmet_seen = any(d["class"] == "helmet" for d in detections)
        vest_seen = any(d["class"] == "safety_vest" for d in detections)
        restricted_entry = any(d["class"] == "restricted_area_entry" for d in detections)
        return {
            "zone_id": zone_id,
            "source": "real_yolov8_inference",
            "model_path": self.model_path,
            "helmet_detected": helmet_seen,
            "safety_vest_detected": vest_seen,
            "restricted_area_entry_detected": restricted_entry,
            "raw_detections": detections,
            "timestamp": time.time(),
        }

    def _simulate_detection(self, zone_id: str) -> dict:
        """
        Deterministic-per-zone pseudo-random compliance signal so repeated
        calls in the same demo session look stable rather than flickering
        randomly. This is NOT a measured value — every field below exists so
        the frontend can render an unmistakable "SIMULATED" badge next to it.
        """
        seed = int(hashlib.sha256(f"{zone_id}-{int(time.time() // 20)}".encode()).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        helmet_ok = rng.random() > 0.15
        vest_ok = rng.random() > 0.2
        restricted_entry = rng.random() > 0.92

        return {
            "zone_id": zone_id,
            "source": "simulated",
            "note": "No trained PPE model loaded — see vision_agent.py module docstring for why, "
                    "and what's needed to make this real (a PPE-labelled dataset + ultralytics).",
            "helmet_detected": helmet_ok,
            "safety_vest_detected": vest_ok,
            "restricted_area_entry_detected": restricted_entry,
            "raw_detections": [],
            "timestamp": time.time(),
        }

    def status(self) -> dict:
        return {
            "ultralytics_installed": ULTRALYTICS_AVAILABLE,
            "model_path": self.model_path,
            "real_model_loaded": self.real_model_loaded,
            "mode": "real_yolov8_inference" if self.real_model_loaded else "simulated",
        }
