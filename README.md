# ThreatLens — AI/ML + Computer Vision Safety Prototype
**Tagline: Detect. Alert. Protect.**

## Features
- **Emergency SOS:** enter coordinates, trigger an SOS event, open the location in Google Maps, and save the event to a local CSV log.
- **CCTV image analysis:** upload an authorized image and run YOLO object detection.
- **Alert dashboard:** view and download recorded safety events.

## Run locally
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The first object-detection run downloads YOLOv8n weights and may need internet.

## Deploy a public demo
1. Push this folder to a GitHub repository.
2. Visit https://share.streamlit.io/
3. Select **New app**, choose the repository and `app.py`, then Deploy.

## Important limitations
This is an academic prototype. SOS events are stored locally and are **not** sent to police or emergency services. A secure integration with authorized emergency services/SMS provider is needed for real alerts. The default YOLOv8n COCO model does **not** reliably detect knives. Knife detection requires a custom trained and validated model saved as `best.pt` in the project root. AI detections must be verified by a human before action. Use only camera footage you are authorized to process.
