"""
Auth Router - JWT token management + login/register/me/logout endpoints
"""
import uuid, json, os, hashlib, secrets
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Header, Request, Depends
from pydantic import BaseModel
from typing import Optional

from schemas.common import success_response

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
USERS_FILE = os.path.join(_REPO_ROOT, "data", "users.json")


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


# ---------- router class ----------

class AuthRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/auth", tags=["auth"])
        self._register_routes()

    # ---------- in-memory user store (persisted to JSON) ----------

    @staticmethod
    def _load_users() -> dict:
        if not os.path.exists(USERS_FILE):
            return {}
        try:
            with open(USERS_FILE) as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _save_users(users: dict) -> None:
        os.makedirs(os.path.dirname(USERS_FILE) or ".", exist_ok=True)
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2, default=str)

    @staticmethod
    def _hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return f"v1:{salt}:{dk.hex()}"

    @staticmethod
    def _verify_password(password: str, stored: str) -> bool:
        if stored.startswith("v1:"):
            parts = stored.split(":")
            if len(parts) == 3:
                _, salt, expected_hex = parts
                dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
                return secrets.compare_digest(dk.hex(), expected_hex)
        legacy = hashlib.sha256(password.encode()).hexdigest()
        return secrets.compare_digest(legacy, stored)

    # ---------- helpers ----------

    @staticmethod
    def _get_auth_deps():
        from infrastructure.auth import get_jwt_auth, get_audit_logger
        from settings import get_security_settings
        sec = get_security_settings()
        return sec.valid_api_keys, sec.jwt_expiration_hours, get_jwt_auth(), get_audit_logger

    def _get_current_user(self, authorization: Optional[str] = Header(None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        _, _, jwt_auth, _ = self._get_auth_deps()
        token = authorization[7:]
        payload = jwt_auth.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        user_id = payload.get("sub", "")
        users = self._load_users()
        user = users.get(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return {"id": user_id, "username": user["username"], "email": user["email"]}

    # ---------- route registration ----------

    def _register_routes(self):
        current_user_dep = Depends(self._get_current_user)

        async def login(req: LoginRequest):
            users = self._load_users()
            for uid, u in users.items():
                if u["username"] == req.username:
                    if not self._verify_password(req.password, u.get("password_hash", "")):
                        raise HTTPException(status_code=401, detail="Invalid credentials")
                    if not u.get("password_hash", "").startswith("v1:"):
                        u["password_hash"] = self._hash_password(req.password)
                        self._save_users(users)
                    _, exp_hours, jwt_auth, _ = self._get_auth_deps()
                    token = jwt_auth.create_token(subject=uid)
                    return AuthResponse(
                        token=token,
                        user=UserInfo(id=uid, username=u["username"], email=u["email"]),
                    )
            raise HTTPException(status_code=401, detail="Invalid credentials")

        async def register(req: RegisterRequest):
            users = self._load_users()
            if any(u["username"] == req.username for u in users.values()):
                raise HTTPException(status_code=409, detail="Username already exists")
            uid = str(uuid.uuid4())
            users[uid] = {
                "username": req.username,
                "email": req.email,
                "password_hash": self._hash_password(req.password),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_users(users)
            _, exp_hours, jwt_auth, _ = self._get_auth_deps()
            token = jwt_auth.create_token(subject=uid)
            return AuthResponse(
                token=token,
                user=UserInfo(id=uid, username=req.username, email=req.email),
            )

        async def get_me(current_user: dict = current_user_dep):
            return UserInfo(**current_user)

        async def create_token(token_request: TokenRequest, request: Request):
            valid_keys, exp_hours, jwt_auth, audit_logger = self._get_auth_deps()
            client_ip = request.client.host if request.client else "unknown"
            if token_request.api_key not in valid_keys:
                audit_logger.log("auth_failed", client_ip, resource="/auth/token", action="token_create", status="failure")
                raise HTTPException(status_code=401, detail="Invalid API key")
            token = jwt_auth.create_token(subject=token_request.api_key[:8])
            audit_logger.log("auth_success", client_ip, resource="/auth/token", action="token_create", status="success")
            return TokenResponse(access_token=token, token_type="bearer", expires_in=exp_hours * 3600)

        async def verify_token(authorization: Optional[str] = Header(None)):
            _, _, jwt_auth, _ = self._get_auth_deps()
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
            token = authorization[7:]
            payload = jwt_auth.verify_token(token)
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            return success_response(data={"valid": True, "subject": payload.get("sub"), "expires": payload.get("exp")})

        async def refresh_token(authorization: Optional[str] = Header(None)):
            _, exp_hours, jwt_auth, _ = self._get_auth_deps()
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
            token = authorization[7:]
            new_token = jwt_auth.refresh_token(token)
            if not new_token:
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            return TokenResponse(access_token=new_token, token_type="bearer", expires_in=exp_hours * 3600)

        self.router.add_api_route("/login", login, methods=["POST"], response_model=AuthResponse)
        self.router.add_api_route("/register", register, methods=["POST"], response_model=AuthResponse)
        self.router.add_api_route("/me", get_me, methods=["GET"], response_model=UserInfo)
        self.router.add_api_route("/token", create_token, methods=["POST"], response_model=TokenResponse)
        self.router.add_api_route("/verify", verify_token, methods=["POST"])
        self.router.add_api_route("/refresh", refresh_token, methods=["POST"], response_model=TokenResponse)


# ---------- module-level backward-compat shims ----------

_auth_instance = AuthRouter()
router = _auth_instance.router


def _load_users() -> dict:
    return _auth_instance._load_users()


def _save_users(users: dict) -> None:
    return _auth_instance._save_users(users)


def _hash_password(password: str) -> str:
    return _auth_instance._hash_password(password)


def _get_auth_deps():
    return _auth_instance._get_auth_deps()
