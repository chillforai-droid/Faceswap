from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

@app.get("/")
def home():
    return FileResponse("templates/index.html")

@app.post("/swap")
async def face_swap(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = np.frombuffer(await file1.read(), np.uint8)
    img2 = np.frombuffer(await file2.read(), np.uint8)

    img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # smoother blend
    result = cv2.seamlessClone(
        img1,
        img2,
        255 * np.ones(img1.shape[:2], np.uint8),
        (img2.shape[1]//2, img2.shape[0]//2),
        cv2.NORMAL_CLONE
    )

    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
