# Automated Helmet Detection System — FastAPI backend starter

This starter extends the existing YOLOv8 image-inference API with:
- JWT login and `/api/auth/me`
- SQLite database using SQLAlchemy
- Admin-managed user accounts
- Monitoring session records
- Detection history and no-helmet violation records
- Dashboard summary statistics
- Existing `/`, `/health`, `/predict`, `/download/{image_name}` routes retained

## 1. Place your model

Copy your trained `best.pt` into this folder, next to `main.py`, or set `MODEL_PATH` in `.env`.

## 2. Create a virtual environment

Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows:
```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure environment

Copy `.env.example` to `.env`, then change `JWT_SECRET_KEY`, `ADMIN_EMAIL`, and `ADMIN_PASSWORD`.
The starter loads `.env` automatically only if you add `from dotenv import load_dotenv; load_dotenv()` near the top of `main.py`. Add those two lines before reading environment variables, or export the variables in your shell. (See note below.)

**Important:** For convenience in this starter, set the variables in your shell or add `load_dotenv()` at the top of `main.py` before configuration constants are read.

## 4. Run

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open Swagger UI at http://127.0.0.1:8000/docs.

## 5. First login

Use the `ADMIN_EMAIL` and `ADMIN_PASSWORD` values configured in your environment. The first admin is created automatically when the database starts for the first time and both variables are set.

Login:
`POST /api/auth/login`
```json
{
  "email": "admin@example.com",
  "password": "ChangeThisAdminPassword123!"
}
```

Copy `access_token` from the response. In Swagger, click **Authorize** and enter:
`Bearer YOUR_ACCESS_TOKEN`

## API overview

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Model health |
| POST | `/predict` | Existing image inference; does not save detection rows |
| POST | `/api/auth/login` | Login |
| GET | `/api/auth/me` | Current user |
| GET | `/api/dashboard/stats` | Dashboard totals |
| POST | `/api/sessions` | Start monitoring session record |
| PATCH | `/api/sessions/{id}/stop` | Stop session record |
| POST | `/api/detect/image` | Run inference and save detections/violations |
| GET | `/api/detections` | Detection history |
| GET | `/api/violations` | Violation history |
| PATCH | `/api/violations/{id}?status=reviewed` | Review/dismiss violation |
| GET | `/api/users` | Admin: list users |
| POST | `/api/users` | Admin: create user |
| PATCH | `/api/users/{id}` | Admin: update user |
| DELETE | `/api/users/{id}` | Admin: deactivate user |
| GET | `/results/{filename}` | View annotated image |
| GET | `/download/{filename}` | Download annotated image |

## Frontend integration

Frontend origin `http://localhost:8080` is allowed by CORS. Send the token on protected requests:
`Authorization: Bearer <access_token>`

For `/api/detect/image`, send `multipart/form-data` with:
- `file`: image file
- `session_id`: optional monitoring session ID

The live webcam pipeline is not implemented by this starter yet. `/api/sessions` records session metadata, but does not open the browser's webcam. The frontend will need to capture frames and send them periodically, or we can add a WebSocket/video-stream design later. Avoid sending every frame into the database; save selected violation events instead.

## Important prototype notes

- This is a development starter, not production-hardened deployment.
- Use HTTPS in deployment, a strong secret, strict CORS, rate limiting, backups, and secure secret management before real use.
- The current image endpoint creates a violation for each `Without Helmet` detection in an uploaded image. For continuous webcam use, add cooldown/deduplication so one rider does not create a violation on every frame.
- Only image detections are persisted by `/api/detect/image`; legacy `/predict` stays inference-only to preserve your existing Swagger test.
- Model training/update jobs are not implemented as an API operation here; training should remain a controlled separate workflow until permissions, resource limits, and deployment strategy are designed.
