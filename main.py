from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

@app.get("/")
def home():
    return FileResponse("templates/index.html")


def detect_face(img):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


@app.post("/swap")
async def swap(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = np.frombuffer(await file1.read(), np.uint8)
    img2 = np.frombuffer(await file2.read(), np.uint8)

    img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

    faces1 = detect_face(img1)
    faces2 = detect_face(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return {"error": "Face not detected"}

    (x1, y1, w1, h1) = faces1[0]
    (x2, y2, w2, h2) = faces2[0]

    face1 = img1[y1:y1+h1, x1:x1+w1]
    face2 = img2[y2:y2+h2, x2:x2+w2]

    face2 = cv2.resize(face2, (w1, h1))

    # mask बनाओ
    mask = np.zeros((h1, w1), dtype=np.uint8)
    cv2.ellipse(mask, (w1//2, h1//2), (w1//2, h1//2), 0, 0, 360, 255, -1)

    center = (x1 + w1//2, y1 + h1//2)

    result = cv2.seamlessClone(face2, img1, mask, center, cv2.NORMAL_CLONE)

    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
