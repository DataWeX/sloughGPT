"""Auth domain models — User, Tenant, Workspace, RBAC."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


# ─── Roles & Permissions ──────────────────────────────────────


class Permission(str, enum.Enum):
    """Granular permissions."""

    # Model
    MODEL_READ = "model:read"
    MODEL_LOAD = "model:load"
    MODEL_UNLOAD = "model:unload"

    # Training
    TRAIN_READ = "train:read"
    TRAIN_START = "train:start"
    TRAIN_CANCEL = "train:cancel"

    # Chat
    CHAT_READ = "chat:read"
    CHAT_SEND = "chat:send"

    # Datasets
    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    DATASET_DELETE = "dataset:delete"

    # Knowledge
    KNOWLEDGE_READ = "knowledge:read"
    KNOWLEDGE_WRITE = "knowledge:write"
    KNOWLEDGE_DELETE = "knowledge:delete"

    # Users
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"

    # Tenants
    TENANT_READ = "tenant:read"
    TENANT_WRITE = "tenant:write"

    # Workspaces
    WORKSPACE_READ = "workspace:read"
    WORKSPACE_WRITE = "workspace:write"
    WORKSPACE_DELETE = "workspace:delete"

    # System
    SYSTEM_READ = "system:read"
    SYSTEM_ADMIN = "system:admin"


class Role(str, enum.Enum):
    """Built-in roles with predefined permission sets."""

    VIEWER = "viewer"
    USER = "user"
    ADMIN = "admin"
    OWNER = "owner"


# Permission sets for each role
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.MODEL_READ,
        Permission.CHAT_READ,
        Permission.DATASET_READ,
        Permission.KNOWLEDGE_READ,
        Permission.TRAIN_READ,
    },
    Role.USER: {
        Permission.MODEL_READ,
        Permission.MODEL_LOAD,
        Permission.CHAT_READ,
        Permission.CHAT_SEND,
        Permission.DATASET_READ,
        Permission.DATASET_WRITE,
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_WRITE,
        Permission.TRAIN_READ,
        Permission.TRAIN_START,
        Permission.TRAIN_CANCEL,
    },
    Role.ADMIN: {
        Permission.MODEL_READ,
        Permission.MODEL_LOAD,
        Permission.MODEL_UNLOAD,
        Permission.CHAT_READ,
        Permission.CHAT_SEND,
        Permission.DATASET_READ,
        Permission.DATASET_WRITE,
        Permission.DATASET_DELETE,
        Permission.KNOWLEDGE_READ,
        Permission.KNOWLEDGE_WRITE,
        Permission.KNOWLEDGE_DELETE,
        Permission.TRAIN_READ,
        Permission.TRAIN_START,
        Permission.TRAIN_CANCEL,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.TENANT_READ,
        Permission.TENANT_WRITE,
        Permission.WORKSPACE_READ,
        Permission.WORKSPACE_WRITE,
        Permission.WORKSPACE_DELETE,
        Permission.SYSTEM_READ,
    },
    Role.OWNER: {p for p in Permission},  # all permissions
}


# ─── User ──────────────────────────────────────────────────────


class UserRole(str, enum.Enum):
    """User status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


@dataclass
class User:
    """User domain entity."""

    id: str = field(default_factory=_new_id)
    username: str = ""
    email: str = ""
    password_hash: str = ""
    display_name: str = ""
    role: Role = Role.USER
    status: UserRole = UserRole.ACTIVE
    tenant_id: str = ""
    avatar_url: str = ""
    last_login_at: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "_id": self.id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "display_name": self.display_name,
            "role": self.role.value,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "avatar_url": self.avatar_url,
            "last_login_at": self.last_login_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> User:
        return cls(
            id=data.get("_id", data.get("id", "")),
            username=data.get("username", ""),
            email=data.get("email", ""),
            password_hash=data.get("password_hash", ""),
            display_name=data.get("display_name", ""),
            role=Role(data.get("role", "user")),
            status=UserRole(data.get("status", "active")),
            tenant_id=data.get("tenant_id", ""),
            avatar_url=data.get("avatar_url", ""),
            last_login_at=data.get("last_login_at", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @property
    def is_active(self) -> bool:
        return self.status == UserRole.ACTIVE

    @property
    def is_admin(self) -> bool:
        return self.role in (Role.ADMIN, Role.OWNER)


# ─── Tenant ────────────────────────────────────────────────────


@dataclass
class Tenant:
    """Tenant (organization) domain entity."""

    id: str = field(default_factory=_new_id)
    name: str = ""
    slug: str = ""
    plan: str = "free"
    max_users: int = 5
    max_workspaces: int = 3
    settings: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "_id": self.id,
            "name": self.name,
            "slug": self.slug,
            "plan": self.plan,
            "max_users": self.max_users,
            "max_workspaces": self.max_workspaces,
            "settings": self.settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Tenant:
        return cls(
            id=data.get("_id", data.get("id", "")),
            name=data.get("name", ""),
            slug=data.get("slug", ""),
            plan=data.get("plan", "free"),
            max_users=data.get("max_users", 5),
            max_workspaces=data.get("max_workspaces", 3),
            settings=data.get("settings", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


# ─── Workspace ─────────────────────────────────────────────────


@dataclass
class Workspace:
    """Workspace domain entity — a collaborative space within a tenant."""

    id: str = field(default_factory=_new_id)
    name: str = ""
    tenant_id: str = ""
    description: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "_id": self.id,
            "name": self.name,
            "tenant_id": self.tenant_id,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Workspace:
        return cls(
            id=data.get("_id", data.get("id", "")),
            name=data.get("name", ""),
            tenant_id=data.get("tenant_id", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class WorkspaceMember:
    """Workspace membership — links a user to a workspace with a role."""

    id: str = field(default_factory=_new_id)
    workspace_id: str = ""
    user_id: str = ""
    role: Role = Role.USER
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "_id": self.id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "role": self.role.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkspaceMember:
        return cls(
            id=data.get("_id", data.get("id", "")),
            workspace_id=data.get("workspace_id", ""),
            user_id=data.get("user_id", ""),
            role=Role(data.get("role", "user")),
            created_at=data.get("created_at", ""),
        )
