from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (frontend connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.get("/")
def home():
    return {"message": "🔥 Face Swap API Enhanced"}

@app.post("/swap")
async def swap_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    strength: float = Form(1.0)
):
    img1 = cv2.imdecode(np.frombuffer(await file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(await file2.read(), np.uint8), cv2.IMREAD_COLOR)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

    if len(faces1) == 0 or len(faces2) == 0:
        return {"error": "Face not detected"}

    (x1, y1, w1, h1) = faces1[0]
    (x2, y2, w2, h2) = faces2[0]

    face1 = img1[y1:y1+h1, x1:x1+w1]
    face1 = cv2.resize(face1, (w2, h2))

    # Color match
    face1_lab = cv2.cvtColor(face1, cv2.COLOR_BGR2LAB)
    face2_lab = cv2.cvtColor(img2[y2:y2+h2, x2:x2+w2], cv2.COLOR_BGR2LAB)
    face1_lab[:,:,0] = face2_lab[:,:,0]
    face1 = cv2.cvtColor(face1_lab, cv2.COLOR_LAB2BGR)

    # Mask + blur
    mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.ellipse(mask, (w2//2, h2//2), (w2//2, h2//2), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15,15), 10)

    center = (x2 + w2//2, y2 + h2//2)

    output = cv2.seamlessClone(face1, img2, mask, center, cv2.NORMAL_CLONE)

    # Strength blend
    final = cv2.addWeighted(output, strength, img2, 1-strength, 0)

    path = "output.jpg"
    cv2.imwrite(path, final)

    return FileResponse(path, media_type="image/jpeg")
