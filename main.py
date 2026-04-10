from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from fastapi.responses import FileResponse

app = FastAPI(
    title="Face Swap API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get("/")
def home():
    return {"message": "Face Swap API Running 🔥"}

@app.post("/swap")
async def swap_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_bytes = await file1.read()
    img2_bytes = await file2.read()

    img1_np = np.frombuffer(img1_bytes, np.uint8)
    img2_np = np.frombuffer(img2_bytes, np.uint8)

    img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)

    h, w, _ = img2.shape
    img1_resized = cv2.resize(img1, (w, h))

    output = cv2.addWeighted(img1_resized, 0.5, img2, 0.5, 0)

    output_path = "output.jpg"
    cv2.imwrite(output_path, output)

    return FileResponse(output_path, media_type="image/jpeg", filename="result.jpg")
