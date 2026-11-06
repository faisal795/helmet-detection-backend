from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    full_name: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class UserCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    role: Literal["admin", "traffic_officer", "lead_inspector"] = "traffic_officer"

class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(default=None, min_length=10, max_length=128)
    role: Optional[Literal["admin", "traffic_officer", "lead_inspector"]] = None
    is_active: Optional[bool] = None

class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    started_by: int
    started_at: datetime
    ended_at: Optional[datetime]
    status: str

class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    session_id: Optional[int]
    recorded_by: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_path: Optional[str]
    detected_at: datetime

class ViolationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    detection_id: int
    image_path: str
    status: str
    created_at: datetime
    reviewed_by: Optional[int]
    reviewed_at: Optional[datetime]

class DashboardStats(BaseModel):
    total_detections: int
    total_violations: int
    active_sessions: int
    active_users: Optional[int] = None
