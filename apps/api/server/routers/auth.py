"""
Auth Router - JWT token management + login/register/me/logout endpoints

Storage backed by MogDB (the project's embedded document DB). User
records are stored in a ``users`` collection instead of a raw JSON file.
"""
import uuid, os, hashlib, secrets, logging
from datetime import datetime, timezone
from fastapi import APIRouter, Header, Request, Depends
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

from schemas.common import success_response, raise_error, safe_audit_log


# ---------- request / response models ----------

class TokenRequest(BaseModel):
    api_key: str = Field(max_length=500)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=500)


class RegisterRequest(BaseModel):
    username: str = Field(max_length=100)
    email: str = Field(max_length=254)
    password: str = Field(max_length=500)


class UserInfo(BaseModel):
    id: str
    username: str
    email: str


class AuthResponse(BaseModel):
    token: str
    user: UserInfo


# ---------- router class ----------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _get_mogdb(db_path: str):
    """Create a MogDB instance at the given path."""
    from mogdb import MogDB
    return MogDB(db_path)


class AuthRouter:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(_REPO_ROOT, "data", "auth_mogdb")
        self._db = _get_mogdb(db_path)
        self._users = self._db.collection("users")
        self._users.create_index("username")
        self.router = APIRouter(prefix="/auth", tags=["auth"])
        self._register_routes()

    @property
    def users_collection(self):
        """Expose the users collection for testing."""
        return self._users

    # ---------- user store backed by MogDB ----------

    def _load_users(self) -> dict:
        """Load all users from MogDB, returned as a dict keyed by user ID.

        Returns:
            Dict mapping user IDs to user record dicts.
        """
        results = {}
        for doc in self._users.find():
            uid = doc["_id"]
            results[uid] = {k: v for k, v in doc.items() if k != "_id"}
        return results

    def _save_user(self, uid: str, user_data: dict) -> None:
        """Insert or update a single user record in MogDB.

        Args:
            uid: The user ID (document _id).
            user_data: Dict of user fields (username, email, etc.).
        """
        doc = {"_id": uid, **user_data}
        existing = self._users.find_one({"_id": uid})
        if existing:
            self._users.update_one({"_id": uid}, {"$set": user_data})
        else:
            self._users.insert_one(doc)

    def _delete_user(self, uid: str) -> None:
        """Delete a single user record from MogDB.

        Args:
            uid: The user ID to remove.
        """
        self._users.delete_one({"_id": uid})

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
        return sec.valid_api_keys, sec.jwt_expiration_hours, get_jwt_auth(), get_audit_logger()

    def _get_current_user(self, authorization: Optional[str] = Header(None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise_error("Missing or invalid authorization header", "E_AUTH_MISSING", status_code=401)
        _, _, jwt_auth, _ = self._get_auth_deps()
        token = authorization[7:]
        payload = jwt_auth.verify_token(token)
        if not payload:
            raise_error("Invalid or expired token", "E_AUTH_MISSING", status_code=401)
        user_id = payload.get("sub", "")
        user = self._users.find_one({"_id": user_id})
        if not user:
            raise_error("User not found", "E_AUTH_MISSING", status_code=401)
        return success_response(data={"id": user_id, "username": user["username"], "email": user["email"]})

    # ---------- route registration ----------

    def _register_routes(self):
        current_user_dep = Depends(self._get_current_user)

        async def login(req: LoginRequest) -> dict:
            """Authenticate a user with username and password.

            Args:
                req: LoginRequest containing username and password.

            Returns:
                AuthResponse with JWT token and UserInfo (id, username, email).

            Side effects:
                Verifies password against stored hash.
                Upgrades legacy SHA-256 hashes to PBKDF2 on first login.
                Raises 401 if credentials are invalid.
            """
            user = self._users.find_one({"username": req.username})
            if not user:
                safe_audit_log("auth.login_failed", resource=req.username, detail="user_not_found")
                raise_error("Invalid credentials", "E_AUTH_MISSING", status_code=401)
            uid = user["_id"]
            if not self._verify_password(req.password, user.get("password_hash", "")):
                safe_audit_log("auth.login_failed", resource=req.username, detail="invalid_password")
                raise_error("Invalid credentials", "E_AUTH_MISSING", status_code=401)
            if not user.get("password_hash", "").startswith("v1:"):
                user["password_hash"] = self._hash_password(req.password)
                self._save_user(uid, user)
            _, exp_hours, jwt_auth, _ = self._get_auth_deps()
            token = jwt_auth.create_token(user_id=uid)
            safe_audit_log("auth.login_success", resource=uid, detail=f"username={req.username}")
            return AuthResponse(
                token=token,
                user=UserInfo(id=uid, username=user["username"], email=user["email"]),
            )

        async def register(req: RegisterRequest) -> dict:
            """Register a new user account with username, email, and password.

            Args:
                req: RegisterRequest with username (max 100 chars), email
                    (max 254 chars), and password (max 500 chars).

            Returns:
                AuthResponse with JWT token and UserInfo.

            Side effects:
                Creates a new user record in the MogDB user store.
                Hashes the password with PBKDF2 + random salt.
                Raises 409 if the username already exists.
            """
            existing = self._users.find_one({"username": req.username})
            if existing:
                raise_error("Username already exists", "E_INFRA_BUSY", status_code=409)
            uid = str(uuid.uuid4())
            user_data = {
                "username": req.username,
                "email": req.email,
                "password_hash": self._hash_password(req.password),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_user(uid, user_data)
            _, exp_hours, jwt_auth, _ = self._get_auth_deps()
            token = jwt_auth.create_token(user_id=uid)
            safe_audit_log("auth.register", resource=uid, detail=f"username={req.username}")
            return AuthResponse(
                token=token,
                user=UserInfo(id=uid, username=req.username, email=req.email),
            )

        async def get_me(current_user: dict = current_user_dep) -> dict:
            """Return the current authenticated user's profile.

            Args:
                current_user: Injected by the Depends(get_current_user)
                    dependency; contains id, username, email.

            Returns:
                UserInfo with id, username, and email fields.

            Side effects:
                Raises 401 if the authorization header is missing or the
                token is invalid or the user no longer exists.
            """
            return UserInfo(**current_user)

        async def create_token(token_request: TokenRequest, request: Request) -> dict:
            """Create a JWT access token from a valid API key.

            Args:
                token_request: TokenRequest containing the api_key string.
                request: FastAPI Request used to extract the client IP for
                    audit logging.

            Returns:
                TokenResponse with access_token, token_type "bearer", and
                expires_in (seconds).

            Side effects:
                Validates the api_key against the configured key list.
                Logs an audit entry for auth_success or auth_failed.
                Raises 401 if the API key is not in the valid key list.
            """
            valid_keys, exp_hours, jwt_auth, audit_logger = self._get_auth_deps()
            client_ip = request.client.host if request.client else "unknown"
            if token_request.api_key not in valid_keys:
                audit_logger.log("auth_failed", client_ip, resource="/auth/token", extra={"action": "token_create", "status": "failure"})
                raise_error("Invalid API key", "E_AUTH_MISSING", status_code=401)
            token = jwt_auth.create_token(user_id=token_request.api_key[:8])
            audit_logger.log("auth_success", client_ip, resource="/auth/token", extra={"action": "token_create", "status": "success"})
            return TokenResponse(access_token=token, token_type="bearer", expires_in=exp_hours * 3600)

        async def verify_token(authorization: Optional[str] = Header(None)) -> dict:
            """Verify whether a JWT bearer token is valid and not expired.

            Args:
                authorization: The Authorization header value, expected
                    format "Bearer <token>". Injected by FastAPI.

            Returns:
                Success envelope with valid=True, subject (user ID), and
                expires (epoch timestamp).

            Side effects:
                Raises 401 if the header is missing, malformed, or the
                token is invalid or expired.
            """
            _, _, jwt_auth, _ = self._get_auth_deps()
            if not authorization or not authorization.startswith("Bearer "):
                raise_error("Missing or invalid authorization header", "E_AUTH_MISSING", status_code=401)
            token = authorization[7:]
            payload = jwt_auth.verify_token(token)
            if not payload:
                raise_error("Invalid or expired token", "E_AUTH_MISSING", status_code=401)
            return success_response(data={"valid": True, "subject": payload.get("sub"), "expires": payload.get("exp")})

        async def refresh_token(authorization: Optional[str] = Header(None)) -> dict:
            """Issue a new JWT token from an existing valid token.

            Args:
                authorization: The Authorization header value, expected
                    format "Bearer <token>". Injected by FastAPI.

            Returns:
                TokenResponse with the new access_token, token_type
                "bearer", and expires_in (seconds).

            Side effects:
                Raises 401 if the header is missing, malformed, or the
                token is invalid or expired.
            """
            _, exp_hours, jwt_auth, _ = self._get_auth_deps()
            if not authorization or not authorization.startswith("Bearer "):
                raise_error("Missing or invalid authorization header", "E_AUTH_MISSING", status_code=401)
            token = authorization[7:]
            new_token = jwt_auth.refresh_token(token)
            if not new_token:
                raise_error("Invalid or expired token", "E_AUTH_MISSING", status_code=401)
            safe_audit_log("auth.token_refresh")
            return TokenResponse(access_token=new_token, token_type="bearer", expires_in=exp_hours * 3600)

        self.router.add_api_route("/login", login, methods=["POST"], response_model=AuthResponse)
        self.router.add_api_route("/register", register, methods=["POST"], response_model=AuthResponse)
        self.router.add_api_route("/me", get_me, methods=["GET"], response_model=UserInfo)
        self.router.add_api_route("/token", create_token, methods=["POST"], response_model=TokenResponse)
        self.router.add_api_route("/verify", verify_token, methods=["POST"])
        self.router.add_api_route("/refresh", refresh_token, methods=["POST"], response_model=TokenResponse)


# ---------- module-level backward-compat shims ----------

_auth_instance: Optional[AuthRouter] = None
router = None


def _get_auth_router() -> AuthRouter:
    """Get or create the global AuthRouter singleton."""
    global _auth_instance, router
    if _auth_instance is None:
        db_path = os.environ.get("MOGDB_AUTH_PATH")
        _auth_instance = AuthRouter(db_path=db_path)
        router = _auth_instance.router
    return _auth_instance


def _load_users() -> dict:
    return _get_auth_router()._load_users()


def _save_user(uid: str, user_data: dict) -> None:
    return _get_auth_router()._save_user(uid, user_data)


def _hash_password(password: str) -> str:
    return AuthRouter._hash_password(password)


def _get_auth_deps():
    return AuthRouter._get_auth_deps()


def set_auth_router(auth_router: AuthRouter) -> None:
    """Replace the global AuthRouter singleton (for testing)."""
    global _auth_instance, router
    _auth_instance = auth_router
    router = auth_router.router


def reset_auth_router() -> None:
    """Reset the global singleton so the next call creates a fresh instance."""
    global _auth_instance, router
    _auth_instance = None
    router = None


# Initialize on import
_get_auth_router()
