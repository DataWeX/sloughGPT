"""
Auth Router - JWT token management + login/register/me/logout endpoints
"""
import uuid, json, os, hashlib, secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Header, Request, Depends
from pydantic import BaseModel
from typing import Optional

from schemas.common import success_response

router = APIRouter(prefix="/auth", tags=["auth"])
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
USERS_FILE = os.path.join(_REPO_ROOT, "data", "users.json")


# ---------- in‑memory user store (persisted to JSON) ----------

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    os.makedirs(os.path.dirname(USERS_FILE) or ".", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2, default=str)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ---------- request / response models ----------

class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class UserInfo(BaseModel):
    id: str
    username: str
    email: str


class AuthResponse(BaseModel):
    token: str
    user: UserInfo


# ---------- helpers ----------

def _get_auth_deps():
    from infrastructure.auth import get_jwt_auth, get_audit_logger
    from settings import get_security_settings
    sec = get_security_settings()
    return sec.valid_api_keys, sec.jwt_expiration_hours, get_jwt_auth(), get_audit_logger


def _get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Extract the authenticated user from the JWT token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    _, _, jwt_auth, _ = _get_auth_deps()
    token = authorization[7:]
    payload = jwt_auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = payload.get("sub", "")
    users = _load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": user_id, "username": user["username"], "email": user["email"]}


# ---------- endpoints ----------

@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Authenticate with username+password and receive a JWT token."""
    users = _load_users()
    for uid, u in users.items():
        if u["username"] == req.username:
            if u.get("password_hash") != _hash_password(req.password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            _, exp_hours, jwt_auth, _ = _get_auth_deps()
            token = jwt_auth.create_token(subject=uid)
            return AuthResponse(
                token=token,
                user=UserInfo(id=uid, username=u["username"], email=u["email"]),
            )
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Create a new user account and return a JWT token."""
    users = _load_users()
    if any(u["username"] == req.username for u in users.values()):
        raise HTTPException(status_code=409, detail="Username already exists")
    uid = str(uuid.uuid4())
    users[uid] = {
        "username": req.username,
        "email": req.email,
        "password_hash": _hash_password(req.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)
    _, exp_hours, jwt_auth, _ = _get_auth_deps()
    token = jwt_auth.create_token(subject=uid)
    return AuthResponse(
        token=token,
        user=UserInfo(id=uid, username=req.username, email=req.email),
    )


@router.get("/me", response_model=UserInfo)
async def get_me(current_user: dict = Depends(_get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserInfo(**current_user)


@router.post("/token", response_model=TokenResponse)
async def create_token(token_request: TokenRequest, request: Request):
    """Create a JWT access token using API key."""
    from fastapi import HTTPException

    valid_keys, exp_hours, jwt_auth, audit_logger = _get_auth_deps()
    client_ip = request.client.host if request.client else "unknown"

    if token_request.api_key not in valid_keys:
        audit_logger.log("auth_failed", client_ip, resource="/auth/token", action="token_create", status="failure")
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = jwt_auth.create_token(subject=token_request.api_key[:8])
    audit_logger.log("auth_success", client_ip, resource="/auth/token", action="token_create", status="success")

    return TokenResponse(access_token=token, token_type="bearer", expires_in=exp_hours * 3600)


@router.post("/verify")
async def verify_token(authorization: Optional[str] = Header(None)):
    """Verify a JWT token."""
    from fastapi import HTTPException
    _, _, jwt_auth, _ = _get_auth_deps()

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization[7:]
    payload = jwt_auth.verify_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return success_response(data={"valid": True, "subject": payload.get("sub"), "expires": payload.get("exp")})


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(authorization: Optional[str] = Header(None)):
    """Refresh a JWT token."""
    from fastapi import HTTPException
    _, exp_hours, jwt_auth, _ = _get_auth_deps()

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization[7:]
    new_token = jwt_auth.refresh_token(token)

    if not new_token:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return TokenResponse(access_token=new_token, token_type="bearer", expires_in=exp_hours * 3600)
