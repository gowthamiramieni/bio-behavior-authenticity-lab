"""
Authentication and user management utilities using bcrypt and SQLAlchemy.
"""
from typing import Optional, List
import bcrypt
from sqlalchemy.orm import Session
from core.db import User


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def get_user_by_username(session: Session, username: str) -> Optional[User]:
    """Retrieve a user by username."""
    return session.query(User).filter(User.username == username.strip()).first()


def get_user_by_id(session: Session, user_id: int) -> Optional[User]:
    """Retrieve a user by primary key ID."""
    return session.query(User).filter(User.id == user_id).first()


def list_users(session: Session) -> List[User]:
    """List all registered users ordered by username."""
    return session.query(User).order_by(User.username.asc()).all()


def create_user(session: Session, username: str, password: str) -> User:
    """Create a new user with bcrypt-hashed password."""
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("Username cannot be empty.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    existing = get_user_by_username(session, clean_username)
    if existing:
        raise ValueError(f"User '{clean_username}' already exists.")

    hashed = hash_password(password)
    new_user = User(username=clean_username, password_hash=hashed)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user


def authenticate_user(session: Session, username: str, password: str) -> Optional[User]:
    """Verify username and password, returning User if valid, else None."""
    user = get_user_by_username(session, username)
    if not user:
        return None
    if verify_password(password, user.password_hash):
        return user
    return None
