import os
import secrets
from fastapi import HTTPException, Header, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from db_models import APIKey, User

ENV_API_KEY = os.getenv("PUBLIC_API_KEY", "")

def generate_api_key(length=32):
    return secrets.token_urlsafe(length)

def create_api_key(user_id=None, db: Session = None):
    if db is None:
        db = SessionLocal()
        close_db = True
    else:
        close_db = False
    try:
        key = generate_api_key()
        api_key = APIKey(key=key, user_id=user_id)
        db.add(api_key)
        db.commit()
        return key
    finally:
        if close_db:
            db.close()

def verify_api_key(x_api_key: str = Header(None), db: Session = Depends(get_db)) -> bool:
    # Check environment key first for testing
    if ENV_API_KEY and x_api_key == ENV_API_KEY:
        return True
    if not x_api_key:
        raise HTTPException(401, "API key required")
    key_obj = db.query(APIKey).filter(APIKey.key == x_api_key, APIKey.is_active == True).first()
    if not key_obj:
        raise HTTPException(401, "Invalid API key")
    return True