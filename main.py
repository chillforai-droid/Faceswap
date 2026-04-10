from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

# templates setup
templates = Jinja2Templates(directory="templates")


# ---------------- HOME (UI) ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- FACE SWAP ----------------
@app.post("/swap")
async def face_swap(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # read images
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()

        img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

        # safety check
        if img1 is None or img2 is None:
            return {"error": "Invalid images"}

        # resize
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # blend (basic swap)
        result = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)

        # encode result
        _, buffer = cv2.imencode(".jpg", result)

        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}


# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "running 🚀"}
