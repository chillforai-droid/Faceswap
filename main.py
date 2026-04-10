from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

# templates folder
templates = Jinja2Templates(directory="templates")

# ---------------- HOME (Frontend UI) ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- FACE SWAP API ----------------
@app.post("/swap")
async def face_swap(file1: UploadFile = File(...), file2: UploadFile = File(...)):

    img1_bytes = await file1.read()
    img2_bytes = await file2.read()

    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    # resize same size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # simple blend (improved)
    alpha = 0.6
    result = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

    # convert to jpg
    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")


# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {"status": "running 🚀"}
