import os
import re
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from database import SessionLocal
from db_models import User

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(user_id: str):
    expire = datetime.utcnow() + timedelta(days=7)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def validate_phone(phone: str, country_code: str = "NG") -> bool:
    if not phone.startswith("+"):
        return False
    digits = phone[1:]
    if digits.startswith("0"):
        return False
    lengths = {"NG": 13, "GH": 12, "US": 11, "IN": 12, "GB": 12}
    expected = lengths.get(country_code)
    if expected and len(digits) != expected:
        return False
    return True

def compute_user_cell(full_name: str, phone: str):
    """Compute start row and column for user's message box."""
    clean_name = re.sub(r'[^a-zA-Z]', '', full_name)
    L = len(clean_name)
    S = sum(int(d) for d in phone if d.isdigit())
    c = ord(clean_name[0].lower()) - 97 if clean_name else 0
    start_row = ((L + S - 1) % 64) + 1
    start_col = c % 26
    return start_row, start_col

def get_current_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(401)
    except JWTError:
        raise HTTPException(401, "Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user