import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv

load_dotenv()
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from pwdlib import PasswordHash
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from ultralytics import YOLO

from database import Base, SessionLocal, engine, get_db
from models import Detection, MonitoringSession, User, Violation
from schemas import (
    LoginRequest, TokenResponse, UserCreate, UserOut, UserUpdate,
    SessionOut, DetectionOut, ViolationOut, DashboardStats
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "best.pt"))).expanduser()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "results"))).expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_TO_A_LONG_RANDOM_SECRET")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "60"))
password_hash = PasswordHash.recommended()

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}. Put best.pt beside main.py "
        "or set MODEL_PATH in your .env file."
    )
model = YOLO(str(MODEL_PATH))

app = FastAPI(
    title="Automated Helmet Detection System API",
    description="YOLOv8 helmet inference, authentication, dashboard, sessions, violations and user management.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "")
    if email and password:
        with SessionLocal() as db:
            existing = db.scalar(select(User).where(User.email == email))
            if existing:
                existing.full_name = os.getenv("ADMIN_FULL_NAME", "System Administrator").strip()
                existing.password_hash = password_hash.hash(password)
                existing.role = "admin"
                existing.is_active = True
                db.commit()
                return
            admins = db.scalars(select(User).where(User.role == "admin").order_by(User.id)).all()
            if len(admins) == 1:
                admin = admins[0]
                admin.full_name = os.getenv("ADMIN_FULL_NAME", "System Administrator").strip()
                admin.email = email
                admin.password_hash = password_hash.hash(password)
                admin.role = "admin"
                admin.is_active = True
            else:
                db.add(User(
                    full_name=os.getenv("ADMIN_FULL_NAME", "System Administrator"),
                    email=email,
                    password_hash=password_hash.hash(password),
                    role="admin",
                    is_active=True,
                ))
            db.commit()


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


from fastapi import Header

def current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload.get("sub", ""))
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid or expired access token.")
    user = db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User account is unavailable.")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return user


def is_without_helmet(class_name: str) -> bool:
    name = class_name.strip().lower().replace("_", " ").replace("-", " ")
    return "without helmet" in name or "no helmet" in name or name in {"without", "no helmet"}


async def read_image_upload(file: UploadFile) -> np.ndarray:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPG, JPEG, PNG, BMP, or WEBP.")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image size must not exceed 15 MB.")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")
    return image


def run_inference(image: np.ndarray):
    result = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            class_name = result.names.get(class_id, str(class_id))
            detections.append({
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                },
            })
    annotated = result.plot()
    image_name = f"{uuid.uuid4().hex}.jpg"
    image_path = RESULTS_DIR / image_name
    if not cv2.imwrite(str(image_path), annotated):
        raise HTTPException(status_code=500, detail="Failed to save annotated image.")
    return detections, image_name, image_path


@app.get("/")
def home():
    return {
        "message": "Automated Helmet Detection System API is running.",
        "model": MODEL_PATH.name,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_name": MODEL_PATH.name}


@app.post("/predict")
async def legacy_predict(file: UploadFile = File(...)):
    """Original Swagger image-inference endpoint, retained for compatibility."""
    image = await read_image_upload(file)
    detections, image_name, _ = run_inference(image)
    return {
        "success": True,
        "filename": file.filename,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "model": MODEL_PATH.name,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_detections": len(detections),
        "detections": detections,
        "annotated_image_url": f"/results/{image_name}",
    }


@app.get("/download/{image_name}")
def download_image(image_name: str):
    if Path(image_name).name != image_name or not image_name.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="Invalid image filename.")
    image_path = RESULTS_DIR / image_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Annotated image not found.")
    return FileResponse(path=str(image_path), media_type="image/jpeg", filename=image_name)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.scalar(select(User).where(User.email == email))
    if not user or not password_hash.verify(payload.password, user.password_hash) or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return TokenResponse(
        access_token=create_access_token(user),
        token_type="bearer",
        user=UserOut.model_validate(user),
    )


@app.get("/api/auth/me", response_model=UserOut)
def get_me(user: User = Depends(current_user)):
    return user


@app.get("/api/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(db: Session = Depends(get_db), user: User = Depends(current_user)):
    total_detections = db.scalar(select(func.count(Detection.id))) or 0
    total_violations = db.scalar(select(func.count(Violation.id))) or 0
    active_sessions = db.scalar(
        select(func.count(MonitoringSession.id)).where(MonitoringSession.status == "active")
    ) or 0
    users_count = db.scalar(select(func.count(User.id)).where(User.is_active.is_(True))) or 0
    return DashboardStats(
        total_detections=total_detections,
        total_violations=total_violations,
        active_sessions=active_sessions,
        active_users=users_count if user.role == "admin" else None,
    )


@app.post("/api/sessions", response_model=SessionOut, status_code=201)
def start_session(db: Session = Depends(get_db), user: User = Depends(current_user)):
    session = MonitoringSession(started_by=user.id, status="active")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.patch("/api/sessions/{session_id}/stop", response_model=SessionOut)
def stop_session(session_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    session = db.get(MonitoringSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Monitoring session not found.")
    if user.role != "admin" and session.started_by != user.id:
        raise HTTPException(status_code=403, detail="You cannot stop another user's session.")
    if session.status == "active":
        session.status = "stopped"
        session.ended_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)
    return session


@app.post("/api/detect/image", status_code=201)
async def detect_and_save(
    file: UploadFile = File(...),
    session_id: Optional[int] = Form(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    if session_id is not None:
        session = db.get(MonitoringSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Monitoring session not found.")
        if user.role != "admin" and session.started_by != user.id:
            raise HTTPException(status_code=403, detail="You cannot add detections to another user's session.")

    image = await read_image_upload(file)
    detections, image_name, image_path = run_inference(image)
    now = datetime.now(timezone.utc)
    saved_detections = []
    violation_count = 0

    for item in detections:
        bbox = item["bbox"]
        row = Detection(
            session_id=session_id,
            recorded_by=user.id,
            class_id=item["class_id"],
            class_name=item["class_name"],
            confidence=item["confidence"],
            x1=bbox["x1"], y1=bbox["y1"], x2=bbox["x2"], y2=bbox["y2"],
            image_path=f"results/{image_name}",
            detected_at=now,
        )
        db.add(row)
        db.flush()
        saved_detections.append(row)
        if is_without_helmet(item["class_name"]):
            db.add(Violation(
                detection_id=row.id,
                image_path=f"results/{image_name}",
                status="new",
                created_at=now,
            ))
            violation_count += 1

    db.commit()
    return {
        "success": True,
        "filename": file.filename,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "model": MODEL_PATH.name,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "total_detections": len(detections),
        "total_violations_created": violation_count,
        "detections": detections,
        "annotated_image_url": f"/results/{image_name}",
    }


@app.get("/api/detections", response_model=list[DetectionOut])
def list_detections(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    return db.scalars(select(Detection).order_by(Detection.detected_at.desc()).offset(offset).limit(limit)).all()


@app.get("/api/violations", response_model=list[ViolationOut])
def list_violations(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    query = select(Violation).order_by(Violation.created_at.desc())
    if status_filter:
        if status_filter not in {"new", "reviewed", "dismissed"}:
            raise HTTPException(status_code=400, detail="status must be new, reviewed, or dismissed.")
        query = query.where(Violation.status == status_filter)
    return db.scalars(query.offset(offset).limit(limit)).all()


@app.patch("/api/violations/{violation_id}")
def update_violation(
    violation_id: int,
    status_value: str = Query(..., alias="status"),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    if status_value not in {"new", "reviewed", "dismissed"}:
        raise HTTPException(status_code=400, detail="status must be new, reviewed, or dismissed.")
    violation = db.get(Violation, violation_id)
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found.")
    violation.status = status_value
    violation.reviewed_by = user.id if status_value != "new" else None
    violation.reviewed_at = datetime.now(timezone.utc) if status_value != "new" else None
    db.commit()
    db.refresh(violation)
    return violation


@app.get("/api/users", response_model=list[UserOut])
def list_users(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    return db.scalars(select(User).order_by(User.id).offset(offset).limit(limit)).all()


@app.post("/api/user", response_model=UserOut, status_code=201)
def create_user(payload: UserCreate, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    email = payload.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="A user with this email already exists.")
    if payload.role not in {"admin", "traffic_officer", "lead_inspector"}:
        raise HTTPException(status_code=400, detail="role must be admin, traffic_officer, or lead_inspector.")
    user = User(
        full_name=payload.full_name.strip(),
        email=email,
        password_hash=password_hash.hash(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.patch("/api/user/{user_id}", response_model=UserOut)
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    data = payload.model_dump(exclude_unset=True)
    if "email" in data and data["email"]:
        email = data["email"].strip().lower()
        duplicate = db.scalar(select(User).where(User.email == email, User.id != user_id))
        if duplicate:
            raise HTTPException(status_code=409, detail="A user with this email already exists.")
        data["email"] = email
    if "role" in data and data["role"] not in {"admin", "traffic_officer", "lead_inspector"}:
        raise HTTPException(status_code=400, detail="role must be admin, traffic_officer, or lead_inspector.")
    if "password" in data:
        user.password_hash = password_hash.hash(data.pop("password"))
    for key, value in data.items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user


@app.delete("/api/user/{user_id}", status_code=204)
def deactivate_user(user_id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    # Soft delete preserves references from detections and violation records.
    user.is_active = False
    db.commit()
    return None
