from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from fastapi.responses import FileResponse

app = FastAPI()

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.get("/")
def home():
    return {"message": "Better Face Swap API 🔥"}

@app.post("/swap")
async def swap_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Read images
    img1_bytes = await file1.read()
    img2_bytes = await file2.read()

    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

    if len(faces1) == 0 or len(faces2) == 0:
        return {"error": "Face not detected ❌"}

    # Take first face
    (x1, y1, w1, h1) = faces1[0]
    (x2, y2, w2, h2) = faces2[0]

    face1 = img1[y1:y1+h1, x1:x1+w1]

    # Resize face1 to face2 size
    face1_resized = cv2.resize(face1, (w2, h2))

    # Create mask
    mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.ellipse(mask, (w2//2, h2//2), (w2//2, h2//2), 0, 0, 360, 255, -1)

    center = (x2 + w2//2, y2 + h2//2)

    # Seamless clone
    output = cv2.seamlessClone(face1_resized, img2, mask, center, cv2.NORMAL_CLONE)

    output_path = "output.jpg"
    cv2.imwrite(output_path, output)

    return FileResponse(output_path, media_type="image/jpeg", filename="result.jpg")
