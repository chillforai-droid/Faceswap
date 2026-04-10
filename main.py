from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Face Swap API Running 🔥"}

@app.post("/swap")
async def face_swap(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    img1 = np.frombuffer(await file1.read(), np.uint8)
    img2 = np.frombuffer(await file2.read(), np.uint8)

    img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    result = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", result)

    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
