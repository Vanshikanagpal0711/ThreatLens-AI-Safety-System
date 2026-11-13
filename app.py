import csv
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
from PIL import Image

ROOT=Path(__file__).parent
LOG=ROOT/"data"/"alerts.csv"
WEIGHTS=ROOT/"best.pt"
st.set_page_config(page_title="ThreatLens",page_icon="🛡️",layout="wide")
st.markdown("""<div style="background:linear-gradient(110deg,#101c3a,#294879);padding:24px;border-radius:16px;color:white"><h1>🛡️ ThreatLens</h1><p>Detect. Alert. Protect. | AI/ML + Computer Vision Safety Prototype</p></div>""",unsafe_allow_html=True)
st.caption("Academic prototype. SOS events are logged locally; no official authority is contacted.")
def log_event(kind,detail,lat="",lon="",confidence=""):
    new=not LOG.exists()
    with open(LOG,"a",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        if new:w.writerow(["timestamp","event_type","details","latitude","longitude","confidence"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),kind,detail,lat,lon,confidence])
t1,t2,t3=st.tabs(["🚨 Emergency SOS","📷 Object Detection","📋 Alert Dashboard"])
with t1:
    st.subheader("One-button emergency assistance")
    a,b=st.columns(2)
    lat=a.number_input("Latitude",value=22.7196,format="%.6f",help="Demo default: Indore. Replace with your actual coordinates.")
    lon=b.number_input("Longitude",value=75.8577,format="%.6f")
    st.markdown(f"[View location on Google Maps](https://www.google.com/maps?q={lat},{lon})")
    note=st.text_input("Emergency note (optional)")
    ok=st.checkbox("I understand this is a prototype; this alert is stored locally.")
    if st.button("🚨 SEND EMERGENCY SOS",type="primary",use_container_width=True):
        if ok:
            log_event("EMERGENCY SOS",note or "User triggered SOS",lat,lon)
            st.error("SOS EVENT LOGGED. See Alert Dashboard.")
            st.warning("This prototype does not send alerts to police or emergency services. A secure, authorized integration is required for real deployment.")
        else: st.info("Please confirm the prototype notice.")
with t2:
    st.subheader("Analyze an authorized CCTV image")
    st.write("Upload a still frame from a camera you are authorized to access.")
    up=st.file_uploader("Upload image",type=["jpg","jpeg","png"])
    threshold=st.slider("Confidence threshold",0.1,0.95,0.35,0.05)
    st.caption("Default YOLO model detects common objects. Knife detection requires a separately trained and validated custom model (`best.pt`).")
    if up:
        img=Image.open(up).convert("RGB")
        st.image(img,use_container_width=True)
        if st.button("🔍 Analyze image",type="primary"):
            try:
                from ultralytics import YOLO
                model=YOLO(str(WEIGHTS) if WEIGHTS.exists() else "yolov8n.pt")
                with st.spinner("Analyzing image..."):
                    result=model.predict(img,conf=threshold,verbose=False)[0]
                st.image(result.plot(),channels="BGR",use_container_width=True)
                detections=[]
                for box in result.boxes:
                    label=model.names[int(box.cls[0])]
                    score=float(box.conf[0])
                    detections.append((label,score))
                if detections:
                    st.dataframe(pd.DataFrame(detections,columns=["Detected object","Confidence"]))
                else: st.info("No objects detected above the threshold.")
                flagged=[(n,s) for n,s in detections if n.lower() in {"knife","scissors","gun","weapon"}]
                if flagged:
                    detail=", ".join(f"{n} ({s:.0%})" for n,s in flagged)
                    log_event("POTENTIAL OBJECT ALERT",detail,confidence=detail)
                    st.error("Potential object alert logged: "+detail)
                else: st.success("Analysis complete. No configured suspicious class detected.")
                if not WEIGHTS.exists():
                    st.warning("Using general YOLOv8n COCO weights; they do not reliably detect knives. Add a trained custom `best.pt` model for that.")
            except ImportError: st.error("Install dependencies using: pip install -r requirements.txt")
            except Exception as e: st.error(f"Detection failed: {e}")
with t3:
    st.subheader("Safety event log")
    if LOG.exists():
        df=pd.read_csv(LOG)
        st.metric("Total events",len(df))
        st.dataframe(df.iloc[::-1],use_container_width=True)
        st.download_button("Download CSV",df.to_csv(index=False),"threatlens_alerts.csv","text/csv")
    else: st.info("No events yet. Trigger SOS or analyze an image.")
st.divider()
st.caption("Use authorized footage only. Verify AI detections manually. Protect personal data and restrict access to incident logs.")
