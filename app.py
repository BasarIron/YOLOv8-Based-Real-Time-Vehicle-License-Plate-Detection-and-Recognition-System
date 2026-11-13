import streamlit as st
import cv2
import numpy as np
import time
import re
import torch
from PIL import Image
from ultralytics import YOLO
import easyocr

st.set_page_config(page_title="Gelişmiş Araç Plaka Tanıma Sistemi", layout="wide")

st.title("Gelişmiş YOLOv8 + EasyOCR Plaka Tanıma Sistemi")
st.markdown("Bu uygulama; **Genişletilmiş Perspektif Düzeltme**, **Sol Kenar Koruması** ve **TR Plaka Kural Doğrulama** algoritmalarıyla her açıdan çekilen fotoğraflarda en doğru sonucu verir.")

@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "best.pt"
    yolo_model = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return yolo_model, reader, device

def safe_perspective_alignment(img_rgb):
    
    h, w, _ = img_rgb.shape
    padded_img = cv2.copyMakeBorder(img_rgb, 10, 10, 30, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    ph, pw, _ = padded_img.shape

    gray = cv2.cvtColor(padded_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < (ph * pw * 0.10):
        return img_rgb

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < w * 0.40 or maxHeight < h * 0.20:
        return img_rgb

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    warped = cv2.warpPerspective(padded_img, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

def parse_turkish_plate(tokens):
    
    raw = "".join(tokens)
    raw = re.sub(r'[^A-Z0-9]', '', raw.upper())

    if raw.startswith("TR") and len(raw) >= 7:
        raw = raw[2:]
    elif "TR" in raw:
        raw = raw.replace("TR", "")

    match = re.search(r'(\d{2})([A-Z]{1,3})(\d{2,5})', raw)
    if match:
        city, letters, numbers = match.groups()
        if 1 <= int(city) <= 81:
            return f"{city} {letters} {numbers}"

    c2d = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8', 'T': '7'}
    d2c = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

    for length in range(min(8, len(raw)), 5, -1):
        for start in range(len(raw) - length + 1):
            sub = list(raw[start:start + length])
            sub[0] = c2d.get(sub[0], sub[0])
            sub[1] = c2d.get(sub[1], sub[1])
            sub[2] = d2c.get(sub[2], sub[2])
            sub[-1] = c2d.get(sub[-1], sub[-1])
            sub[-2] = c2d.get(sub[-2], sub[-2])

            cand = "".join(sub)
            m = re.match(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$', cand)
            if m:
                city, letters, numbers = m.groups()
                if 1 <= int(city) <= 81:
                    return f"{city} {letters} {numbers}"

    return " ".join(tokens) if tokens else "OKUNAMADI"

def clean_and_read_plate(plate_crop, reader):
    aligned = safe_perspective_alignment(plate_crop)
    
    h, w, _ = aligned.shape
    roi = aligned[:int(h * 0.88), :]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    padded = cv2.copyMakeBorder(enhanced, 15, 15, 35, 20, cv2.BORDER_CONSTANT, value=255)

    ocr_results = reader.readtext(
        padded,
        detail=1,
        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        width_ths=0.6,
        mag_ratio=1.5
    )

    if not ocr_results:
        return "OKUNAMADI", padded

    blocks = []
    max_h = 0
    img_h = padded.shape[0]

    for (bbox, text, prob) in ocr_results:
        text = text.strip().upper()
        if not text:
            continue
        
        char_h = np.linalg.norm(np.array(bbox[3]) - np.array(bbox[0]))
        center_x = (bbox[0][0] + bbox[1][0]) / 2.0
        center_y = (bbox[0][1] + bbox[2][1]) / 2.0

        if center_y > (img_h * 0.82) and char_h < (img_h * 0.35):
            continue

        if char_h > max_h:
            max_h = char_h
        blocks.append((center_x, char_h, text, prob))

    valid_blocks = [b for b in blocks if b[1] >= (max_h * 0.40)]
    valid_blocks.sort(key=lambda x: x[0])
    tokens = [b[2] for b in valid_blocks]

    plate_str = parse_turkish_plate(tokens)
    return plate_str, padded

with st.spinner("Model ve OCR motoru yükleniyor..."):
    model, reader, device = load_models()

st.sidebar.success(f"Sistem Durumu: Aktif\n\nÇalışma Donanımı: {device.upper()}")

uploaded_files = st.file_uploader(
    "Araç fotoğraflarını yükleyin (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.subheader(f"Görüntü {idx + 1}: {uploaded_file.name}")
        
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        output_img = img_array.copy()

        start_time = time.time()
        results = model.predict(img_array, conf=0.35, device=0 if device=='cuda' else 'cpu')
        inf_time = (time.time() - start_time) * 1000

        col1, col2 = st.columns([2, 1])

        detected_plates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                plate_crop = img_array[y1:y2, x1:x2]
                plate_text, processed_crop = clean_and_read_plate(plate_crop, reader)

                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(output_img, f"{plate_text} ({conf:.2f})", (x1, max(y1 - 12, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                detected_plates.append((plate_crop, processed_crop, plate_text, conf))

        with col1:
            st.image(output_img, caption=f"Model Çıktısı (YOLO Tahmin Süresi: {inf_time:.2f} ms)", use_container_width=True)

        with col2:
            st.markdown("### Tespit Detayları")
            st.write(f"**YOLO Tahmin Süresi:** `{inf_time:.2f} ms`")
            
            if detected_plates:
                for i, (crop, proc_crop, text, conf) in enumerate(detected_plates):
                    st.image(crop, caption=f"Kırpılan Orijinal Plaka {i+1}", width=240)
                    st.image(proc_crop, caption=f"Sol Kenar Korumalı Görüntü {i+1}", width=240)
                    st.success(f"**Okunan Plaka:** `{text}`")
                    st.write(f"**Güven Skoru (Confidence):** `{conf:.2%}`")
            else:
                st.warning("Bu görüntüde plaka tespit edilemedi.")