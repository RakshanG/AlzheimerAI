from fastapi import FastAPI, File, UploadFile, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import base64
import io
from model import load_model, predict_auto
from gradcam import generate_gradcam
from auth import register_user, login_user, decode_token
from database import get_connection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, temperature, device = load_model()
print("Models loaded and ready")

def get_risk_level(label: str) -> str:
    if label == "Non Demented":       return "Low Risk"
    if label == "Very Mild Demented": return "Moderate Risk"
    return "High Risk"

def get_current_user(authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    return decode_token(token)

@app.post("/register")
async def register(
    name:     str = Form(...),
    email:    str = Form(...),
    password: str = Form(...),
    role:     str = Form(...)
):
    return JSONResponse(register_user(name, email, password, role))

@app.post("/login")
async def login(
    email:    str = Form(...),
    password: str = Form(...)
):
    return JSONResponse(login_user(email, password))

@app.post("/predict")
async def predict_endpoint(
    file:          UploadFile = File(...),
    patient_id:    str = Form(default=""),
    authorization: str = Header(default="")
):
    user = get_current_user(authorization)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        contents = await file.read()
        filename = file.filename or ""
        result   = predict_auto(contents, filename)

        gradcam_b64 = None
        if result["model_type"] == "2D":
            image           = Image.open(io.BytesIO(contents)).convert("RGB")
            gradcam_overlay = generate_gradcam(model, image, result["predicted_class"])
            buffer          = io.BytesIO()
            Image.fromarray(gradcam_overlay).save(buffer, format="PNG")
            gradcam_b64     = base64.b64encode(buffer.getvalue()).decode("utf-8")

        risk_level = get_risk_level(result["predicted_label"])
        pid = int(patient_id) if patient_id else user["id"]
        did = user["id"] if user["role"] == "doctor" else None

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO scans
                    (patient_id, doctor_id, filename, predicted_label, risk_level,
                     confidence, uncertainty_flag, prob_non_demented,
                     prob_very_mild, prob_mild, model_type, gradcam_image)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    pid, did, filename,
                    result["predicted_label"], risk_level,
                    result["confidence"], result["uncertainty_flag"],
                    result["probabilities"]["Non Demented"],
                    result["probabilities"]["Very Mild Demented"],
                    result["probabilities"]["Mild Demented"],
                    result["model_type"],
                    gradcam_b64
                ))
                conn.commit()
        finally:
            conn.close()

        return JSONResponse({
            "predicted_label":  result["predicted_label"],
            "risk_level":       risk_level,
            "confidence":       result["confidence"],
            "probabilities":    result["probabilities"],
            "uncertainty_flag": result["uncertainty_flag"],
            "model_type":       result["model_type"],
            "model_name":       result["model_name"],
            "gradcam_image":    gradcam_b64,
            "slices_3d":        result.get("slices_3d")
        })

    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/scans")
async def get_scans(authorization: str = Header(default="")):
    user = get_current_user(authorization)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if user["role"] == "doctor":
                cur.execute("""
                    SELECT s.*, u.name as patient_name, u.email as patient_email
                    FROM scans s
                    JOIN users u ON s.patient_id = u.id
                    ORDER BY s.timestamp DESC
                """)
            else:
                cur.execute("""
                    SELECT * FROM scans
                    WHERE patient_id = %s
                    ORDER BY timestamp DESC
                """, (user["id"],))
            scans = cur.fetchall()
            for s in scans:
                s["gradcam_image"] = None
                if s.get("timestamp"):
                    s["timestamp"] = str(s["timestamp"])
            return JSONResponse(scans)
    finally:
        conn.close()

@app.get("/dashboard")
async def get_dashboard(authorization: str = Header(default="")):
    user = get_current_user(authorization)
    if not user or user["role"] != "doctor":
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM scans")
            total = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(*) as c FROM scans WHERE risk_level = 'Low Risk'")
            low = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) as c FROM scans WHERE risk_level = 'Moderate Risk'")
            moderate = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) as c FROM scans WHERE risk_level = 'High Risk'")
            high = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) as c FROM scans WHERE uncertainty_flag = TRUE")
            flagged = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(DISTINCT patient_id) as c FROM scans")
            unique_patients = cur.fetchone()["c"]

            cur.execute("SELECT model_type, COUNT(*) as count FROM scans GROUP BY model_type")
            model_usage = {row["model_type"]: row["count"] for row in cur.fetchall() if row["model_type"]}

            cur.execute("""
                SELECT s.*, u.name as patient_name
                FROM scans s JOIN users u ON s.patient_id = u.id
                WHERE s.uncertainty_flag = TRUE
                ORDER BY s.timestamp DESC
            """)
            flagged_scans = cur.fetchall()
            for s in flagged_scans:
                s["gradcam_image"] = None
                s["timestamp"]     = str(s["timestamp"])

            return JSONResponse({
                "total_scans":     total,
                "low_risk":        low,
                "moderate_risk":   moderate,
                "high_risk":       high,
                "flagged":         flagged,
                "unique_patients": unique_patients,
                "model_usage":     model_usage,
                "flagged_scans":   flagged_scans
            })
    finally:
        conn.close()

@app.get("/patients")
async def get_patients(authorization: str = Header(default="")):
    user = get_current_user(authorization)
    if not user or user["role"] != "doctor":
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT u.id, u.name, u.email, u.created_at,
                       COUNT(s.id) as total_scans,
                       MAX(s.risk_level) as latest_risk
                FROM users u
                LEFT JOIN scans s ON u.id = s.patient_id
                WHERE u.role = 'patient'
                GROUP BY u.id
                ORDER BY u.created_at DESC
            """)
            patients = cur.fetchall()
            for p in patients:
                p["created_at"] = str(p["created_at"])
            return JSONResponse(patients)
    finally:
        conn.close()

@app.get("/")
def root():
    return {"status": "NeuroScan AI API running — 2D + 3D active"}