import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
import os

app = FastAPI()

def compute_sharpness(image):
    """Laplacian variance - higher = sharper"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = min(variance / 1000.0, 1.0)
    return round(score, 4)

def compute_exposure(image):
    """Histogram analysis - 1.0 = perfect exposure"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    dark_pixels = hist[:50].sum()
    bright_pixels = hist[200:].sum()
    if dark_pixels > 0.5:
        score = 1.0 - dark_pixels
    elif bright_pixels > 0.5:
        score = 1.0 - bright_pixels
    else:
        score = 1.0 - abs(dark_pixels - bright_pixels)
    return round(float(score), 4)

def compute_face_quality(image):
    """Detect faces and check quality"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return 1.0
    avg_face_size = np.mean([w*h for (x,y,w,h) in faces])
    image_size = image.shape[0] * image.shape[1]
    face_ratio = avg_face_size / image_size
    score = min(face_ratio * 10, 1.0)
    return round(float(score), 4)

@app.post("/compute_features")
async def compute_features(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    image = cv2.imread(tmp_path)
    os.unlink(tmp_path)
    if image is None:
        return {"error": "Could not read image"}
    sharpness = compute_sharpness(image)
    exposure = compute_exposure(image)
    face_quality = compute_face_quality(image)
    return {
        "sharpness_score": sharpness,
        "exposure_score": exposure,
        "face_quality_score": face_quality
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
