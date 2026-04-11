from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import mediapipe as mp
import uuid
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Folder
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mediapipe init (safe)
try:
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1)
except:
    from mediapipe.python.solutions import face_detection
    mp_face = face_detection.FaceDetection(model_selection=1)


# 🏠 Home UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 🔥 Face Swap API
@app.post("/swap")
async def swap_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    
    # Save files
    img1_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_1.jpg")
    img2_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_2.jpg")

    with open(img1_path, "wb") as f:
        f.write(await file1.read())

    with open(img2_path, "wb") as f:
        f.write(await file2.read())

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return {"error": "Image load failed"}

    # Convert RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Detect faces
    res1 = mp_face.process(img1_rgb)
    res2 = mp_face.process(img2_rgb)

    if not res1.detections or not res2.detections:
        return {"error": "Face not detected in one image"}

    # Get bounding box
    def get_box(detection, img):
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = img.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)
        return x, y, w_box, h_box

    x1, y1, w1, h1 = get_box(res1.detections[0], img1)
    x2, y2, w2, h2 = get_box(res2.detections[0], img2)

    # Crop face
    face1 = img1[y1:y1+h1, x1:x1+w1]

    # Resize face
    face1_resized = cv2.resize(face1, (w2, h2))

    # Create mask
    mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.ellipse(mask, (w2//2, h2//2), (w2//2, h2//2), 0, 0, 360, 255, -1)

    # Center
    center = (x2 + w2//2, y2 + h2//2)

    # Seamless clone (🔥 realistic improvement)
    try:
        output = cv2.seamlessClone(face1_resized, img2, mask, center, cv2.NORMAL_CLONE)
    except:
        # fallback
        img2[y2:y2+h2, x2:x2+w2] = face1_resized
        output = img2

    # Save result
    output_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_out.jpg")
    cv2.imwrite(output_path, output)

    return FileResponse(output_path, media_type="image/jpeg")
