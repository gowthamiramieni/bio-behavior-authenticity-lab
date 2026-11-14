"""
Database models and connection management for BioAuth Lab using SQLAlchemy.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    create_engine,
    Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

Base = declarative_base()

# Default SQLite database path
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bioauth.db"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    samples = relationship("Sample", back_populates="user", cascade="all, delete-orphan")
    models = relationship("ModelRecord", back_populates="user", cascade="all, delete-orphan")
    results = relationship("EvaluationResult", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Sample(Base):
    __tablename__ = "samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    modality = Column(String(32), nullable=False, index=True)  # 'keystroke', 'mouse', 'touch'
    raw_data = Column(Text, nullable=False)  # JSON-encoded raw events and coordinates
    features = Column(Text, nullable=True)   # JSON-encoded extracted feature dictionary
    label = Column(String(16), nullable=False, default="genuine", index=True)  # 'genuine' or 'impostor'
    session_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="samples")

    __table_args__ = (
        Index("idx_user_modality_label", "user_id", "modality", "label"),
    )

    def get_raw_data(self):
        return json.loads(self.raw_data) if self.raw_data else {}

    def get_features(self):
        return json.loads(self.features) if self.features else {}

    def __repr__(self):
        return f"<Sample(id={self.id}, user_id={self.user_id}, modality='{self.modality}', label='{self.label}')>"


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    modality = Column(String(32), nullable=False, index=True)  # 'keystroke', 'mouse', 'touch', 'fusion'
    model_type = Column(String(64), nullable=False)  # 'RandomForest', 'OneClassSVM', etc.
    filepath = Column(String(256), nullable=False)
    params = Column(Text, nullable=True)    # JSON-encoded hyperparameters
    metrics = Column(Text, nullable=True)   # JSON-encoded validation metrics (EER, AUC, etc.)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="models")

    def get_params(self):
        return json.loads(self.params) if self.params else {}

    def get_metrics(self):
        return json.loads(self.metrics) if self.metrics else {}

    def __repr__(self):
        return f"<ModelRecord(id={self.id}, model_type='{self.model_type}', modality='{self.modality}')>"


class EvaluationResult(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    stage = Column(String(32), nullable=False, index=True)  # 'before_improvement', 'after_improvement'
    modality = Column(String(32), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)
    metrics = Column(Text, nullable=False)  # JSON-encoded dictionary of all metric values
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="results")

    def get_metrics(self):
        return json.loads(self.metrics) if self.metrics else {}

    def __repr__(self):
        return f"<EvaluationResult(id={self.id}, model='{self.model_name}', stage='{self.stage}')>"


def get_engine(db_path: Optional[str or Path] = None):
    """Create or return SQLAlchemy engine for SQLite."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{target_path.resolve()}", echo=False)


def init_db(db_path: Optional[str or Path] = None):
    """Initialize database tables."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_path: Optional[str or Path] = None) -> Session:
    """Return a new SQLAlchemy session."""
    engine = get_engine(db_path)
    session_factory = sessionmaker(bind=engine)
    return session_factory()
