"""
Wildfire Intelligence & Early Warning Dashboard - Pretrained YOLO Vision Pipeline
Runs fire and smoke detection on uploaded images or field imagery presets using models/best.pt.
"""

import os
import sys
import io
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
SAMPLES_DIR = os.path.join(BASE_DIR, "assets", "samples")

# Global singleton model cache
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"YOLO weights not found at {MODEL_PATH}")
        print(f"[YOLO] Loading pretrained model from {MODEL_PATH}...")
        _yolo_model = YOLO(MODEL_PATH)
        print(f"[YOLO] Ready. Detected classes: {_yolo_model.names}")
    return _yolo_model

def decode_image(image_input):
    """
    Decodes an image input from either a file path, base64 data URL string, or PIL Image.
    Returns RGB numpy array.
    """
    if isinstance(image_input, np.ndarray):
        return image_input

    if isinstance(image_input, str):
        if image_input.startswith("data:image"):
            # Base64 string from dcc.Upload
            header, encoded = image_input.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return np.array(image)
        elif os.path.exists(image_input):
            # File path
            image = Image.open(image_input).convert("RGB")
            return np.array(image)
        else:
            # Check relative to BASE_DIR
            alt_path = os.path.join(BASE_DIR, image_input)
            if os.path.exists(alt_path):
                image = Image.open(alt_path).convert("RGB")
                return np.array(image)
            raise ValueError(f"Image path not found: {image_input}")

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    raise ValueError("Unsupported image input format.")

def run_yolo_detection(image_input, conf_threshold=0.20):
    """
    Runs YOLO fire/smoke detection, renders bounding boxes, and returns base64 image + summary.
    Includes inference latency measurement and original/annotated dual image return.
    """
    import time
    t0 = time.perf_counter()

    model = get_yolo_model()
    try:
        rgb_img = decode_image(image_input)
    except Exception as e:
        return {
            'image_uri': None,
            'orig_image_uri': None,
            'fire_count': 0,
            'smoke_count': 0,
            'total_detections': 0,
            'max_confidence': 0.0,
            'inference_time_ms': 0.0,
            'hazard_level': "DECODE ERROR",
            'hazard_msg': f"Unable to decode image: {str(e)}",
            'alert_theme': "danger",
            'detections': []
        }

    # Generate original image base64 URI
    buf_orig = io.BytesIO()
    Image.fromarray(rgb_img).save(buf_orig, format="JPEG", quality=85)
    orig_b64 = base64.b64encode(buf_orig.getvalue()).decode("utf-8")
    orig_data_uri = f"data:image/jpeg;base64,{orig_b64}"

    # Convert to BGR for OpenCV drawing
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    h, w = bgr_img.shape[:2]

    # Predict
    results = model.predict(rgb_img, conf=conf_threshold, verbose=False)[0]
    inference_time_ms = round((time.perf_counter() - t0) * 1000, 1)

    detections = []
    fire_count = 0
    smoke_count = 0
    max_conf = 0.0

    # Color mapping (BGR format)
    # Class 0: smoke (Cyan/Sky Blue), Class 1: fire (Red)
    COLORS = {
        'fire': (30, 40, 240),    # Bright Red
        'smoke': (240, 180, 50)   # Sky Blue / Cyan
    }

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, f"Class {cls_id}").lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if conf > max_conf:
            max_conf = conf

        if 'fire' in cls_name:
            fire_count += 1
            color = COLORS['fire']
            display_name = "FIRE"
        else:
            smoke_count += 1
            color = COLORS['smoke']
            display_name = "SMOKE"

        detections.append({
            'class': display_name,
            'confidence': round(conf * 100, 1),
            'bbox': [x1, y1, x2, y2]
        })

        # Draw bounding box
        thickness = max(2, int(min(h, w) / 250))
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, thickness)

        # Label badge
        label = f"{display_name} {conf*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(h, w) / 800)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

        # Draw filled background for label
        cv2.rectangle(
            bgr_img,
            (x1, max(0, y1 - text_h - 8)),
            (x1 + text_w + 6, max(text_h + 8, y1)),
            color,
            -1
        )
        # Draw text in white
        cv2.putText(
            bgr_img,
            label,
            (x1 + 3, max(text_h + 2, y1 - 4)),
            font,
            font_scale,
            (255, 255, 255),
            max(1, int(thickness / 2)),
            cv2.LINE_AA
        )

    # Convert back to RGB then to Base64 JPEG
    annotated_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_str}"

    # Determine hazard assessment
    if fire_count > 0:
        hazard_level = "CRITICAL: ACTIVE FLAME DETECTED"
        hazard_msg = f"Identified {fire_count} active flame signature{'s' if fire_count > 1 else ''}. High thermal intensity detected in field imagery."
        alert_theme = "danger"
    elif smoke_count > 0:
        hazard_level = "WARNING: SMOKE PLUME IDENTIFIED"
        hazard_msg = f"Identified {smoke_count} smoke plume signature{'s' if smoke_count > 1 else ''}. Potential ignition source or early fire evolution."
        alert_theme = "warning"
    else:
        hazard_level = "CLEAR: NO DETECTIONS"
        hazard_msg = f"Zero flame or smoke features identified above the {int(conf_threshold*100)}% confidence threshold."
        alert_theme = "info"

    return {
        'image_uri': data_uri,
        'orig_image_uri': orig_data_uri,
        'fire_count': fire_count,
        'smoke_count': smoke_count,
        'total_detections': len(detections),
        'max_confidence': round(max_conf * 100, 1),
        'inference_time_ms': inference_time_ms,
        'hazard_level': hazard_level,
        'hazard_msg': hazard_msg,
        'alert_theme': alert_theme,
        'detections': detections
    }

def get_sample_image_b64(sample_filename):
    """Returns base64 data URI of a sample image from assets/samples."""
    path = os.path.join(SAMPLES_DIR, sample_filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"
    return None

if __name__ == "__main__":
    res = run_yolo_detection("assets/samples/wildfire_aerial_sample.jpg")
    print("YOLO Pipeline Test:")
    print("Fire Count:", res['fire_count'])
    print("Smoke Count:", res['smoke_count'])
    print("Hazard:", res['hazard_level'])
