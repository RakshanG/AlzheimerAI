from jose import JWTError, jwt
from datetime import datetime, timedelta
from database import get_connection
import hashlib

SECRET_KEY         = "neuroscan-secret-key-2024"
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hashlib.sha256(plain.encode()).hexdigest() == hashed

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire    = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

def register_user(name: str, email: str, password: str, role: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                return {"error": "Email already registered"}
            hashed = hash_password(password)
            cur.execute(
                "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, %s)",
                (name, email, hashed, role)
            )
            conn.commit()
            return {"success": True, "message": "Account created successfully"}
    finally:
        conn.close()

def login_user(email: str, password: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
            if not user or not verify_password(password, user["password"]):
                return {"error": "Invalid email or password"}
            token = create_token({
                "id":    user["id"],
                "email": user["email"],
                "role":  user["role"],
                "name":  user["name"]
            })
            return {
                "success": True,
                "token":   token,
                "user": {
                    "id":    user["id"],
                    "name":  user["name"],
                    "role":  user["role"],
                    "email": user["email"]
                }
            }
    finally:
        conn.close()