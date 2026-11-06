import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# \# =========================================================
# \# CONFIGURATION
# \# =========================================================

BASE_DIR = Path(\_\_file\_\_).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
RESULTS_DIR = BASE_DIR / "results"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# \# =========================================================
# \# INITIALIZE MODEL
# \# =========================================================

if not MODEL_PATH.exists():
raise FileNotFoundError(
 f"Model not found: {MODEL_PATH}\n"
 "Place your trained YOLOv8 .pt file next to api.py "
 "or update MODEL_PATH."
  )

model = YOLO(str(MODEL_PATH))


# \# =========================================================
# \# INITIALIZE FASTAPI
# \# =========================================================

app = FastAPI(
  title="YOLOv8 Object Detection API",
  description=(
 "REST API for running inference using a trained YOLOv8 "
 "model. Upload an image to detect objects."
  ),
  version="1.0.0",
)

app.add_middleware(
  CORSMiddleware,
  allow_origin_regex=r"^https?://(localhost|12&#x37;**\\.**&#x30;**\\.**&#x30;**\\.**&#x31;)(:\d+)?$",
  allow_methods=["GET", "POST"],
  allow_headers=["\*"],
)

app.mount(
  "/results",
  StaticFiles(directory=str(RESULTS_DIR)),
  name="results",
)


# \# =========================================================
# \# ROOT ENDPOINT
# \# =========================================================

@app.get("/")
def home():
  return {
 "message": "YOLOv8 inference API is running.",
 "model": MODEL_PATH.name,
 "confidence_threshold": CONFIDENCE_THRESHOLD,
 "endpoints": {
     "health": "/health",
     "predict": "/predict",
     "swagger_docs": "/docs",
 },
  }


# \# =========================================================
# \# HEALTH CHECK
# \# =========================================================

@app.get("/health")
def health():
  return {
 "status": "healthy",
 "model_loaded": model is not None,
 "model_name": MODEL_PATH.name,
  }


# \# =========================================================
# \# PREDICTION ENDPOINT
# \# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  """
  Upload an image and return YOLOv8 detections.

  Response includes:
  \- Detected class names
  \- Confidence scores
  \- Bounding boxes
  \- Total detection count
  \- URL of the annotated image
  """

#   \# Validate uploaded filename
  if not file.filename:
 raise HTTPException(
     status_code=400,
     detail="No filename provided.",
 )

  extension = Path(file.filename).suffix.lower()

  if extension not in ALLOWED_EXTENSIONS:
 raise HTTPException(
     status_code=400,
     detail=(
         "Unsupported image format. "
         "Use JPG, JPEG, PNG, BMP, or WEBP."
     ),
 )

#   \# Read uploaded image
  image_bytes = await file.read()

  if not image_bytes:
 raise HTTPException(
     status_code=400,
     detail="Uploaded file is empty.",
 )

#   \# Prevent excessively large uploads
  if len(image_bytes) > 15 \* 1024 \* 1024:
 raise HTTPException(
     status_code=413,
     detail="Image size must not exceed 15 MB.",
 )

#   \# Decode image
  image_array = np.frombuffer(image_bytes, dtype=np.uint8)
  image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

  if image is None:
 raise HTTPException(
     status_code=400,
     detail="Invalid or corrupted image file.",
 )

  try:
#  \# Run YOLOv8 inference
 results = model.predict(
     source=image,
     conf=CONFIDENCE_THRESHOLD,
     imgsz=IMAGE_SIZE,
     verbose=False,
 )

 result = results[0]
 detections = []

#  \# Extract detections
 if result.boxes is not None:
     for box in result.boxes:
         class_id = int(box.cls.item())
         confidence = float(box.conf.item())

         x1, y1, x2, y2 = (
             box.xyxy[0].cpu().tolist()
         )

         class_name = result.names.get(
             class_id,
             str(class_id),
         )

         detections.append(
             {
                 "class_id": class_id,
                 "class_name": class_name,
                 "confidence": round(confidence, 4),
                 "bbox": {
                     "x1": round(float(x1), 2),
                     "y1": round(float(y1), 2),
                     "x2": round(float(x2), 2),
                     "y2": round(float(y2), 2),
                 },
             }
         )

#  \# Generate annotated image
 annotated_image = result.plot()

#  \# Save the annotated image
 image_id = uuid.uuid4().hex
 output_filename = f"{image_id}.jpg"
 output_path = RESULTS_DIR / output_filename

 success = cv2.imwrite(
     str(output_path),
     annotated_image,
 )

 if not success:
     raise RuntimeError(
         "Failed to save annotated image."
     )

#  \# Return JSON response
 return {
     "success": True,
     "filename": file.filename,
     "image_width": int(image.shape[1]),
     "image_height": int(image.shape[0]),
     "model": MODEL_PATH.name,
     "confidence_threshold": CONFIDENCE_THRESHOLD,
     "total_detections": len(detections),
     "detections": detections,
     "annotated_image_url": f"/results/{output_filename}",
 }

  except HTTPException:
 raise

  except Exception as exc:
 raise HTTPException(
     status_code=500,
     detail=f"Inference failed: {str(exc)}",
 ) from exc

  finally:
 await file.close()


# \# =========================================================
# \# ANNOTATED IMAGE DOWNLOAD ENDPOINT
# \# =========================================================

@app.get("/download/{image_name}")
def download_image(image_name: str):
  """
  Download an annotated prediction image.
  """

#   \# Only allow generated JPG filenames
  if (
 Path(image_name).name != image_name
 or not image_name.endswith(".jpg")
  ):
 raise HTTPException(
     status_code=400,
     detail="Invalid image filename.",
 )

  image_path = RESULTS_DIR / image_name

  if not image_path.is_file():
 raise HTTPException(
     status_code=404,
     detail="Annotated image not found.",
 )

  return FileResponse(
 path=str(image_path),
 media_type="image/jpeg",
 filename=image_name,
  )


# \# =========================================================
# \# RUN DIRECTLY
# \# =========================================================

if \_\_name\_\_ == "\_\_main\_\_":
  import uvicorn

  uvicorn.run(
      app,
      host="127.0.0.1",
      port=8000,
  )


