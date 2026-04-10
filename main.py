from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import numpy as np
from io import BytesIO
import mediapipe as mp

app = FastAPI()

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1)


@app.get("/")
def home():
    return FileResponse("templates/index.html")


def detect_face(img):
    h, w, _ = img.shape
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return None

    box = results.detections[0].location_data.relative_bounding_box

    x = int(box.xmin * w)
    y = int(box.ymin * h)
    w_box = int(box.width * w)
    h_box = int(box.height * h)

    return (x, y, w_box, h_box)


@app.post("/swap")
async def swap(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = np.frombuffer(await file1.read(), np.uint8)
    img2 = np.frombuffer(await file2.read(), np.uint8)

    img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

    face1 = detect_face(img1)
    face2 = detect_face(img2)

    if face1 is None or face2 is None:
        return {"error": "Face not detected"}

    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2

    f1 = img1[y1:y1+h1, x1:x1+w1]
    f2 = img2[y2:y2+h2, x2:x2+w2]

    f2 = cv2.resize(f2, (w1, h1))

    mask = np.zeros((h1, w1), dtype=np.uint8)
    cv2.ellipse(mask, (w1//2, h1//2), (w1//2, h1//2), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31,31), 0)

    center = (x1 + w1//2, y1 + h1//2)

    result = cv2.seamlessClone(f2, img1, mask, center, cv2.NORMAL_CLONE)

    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
