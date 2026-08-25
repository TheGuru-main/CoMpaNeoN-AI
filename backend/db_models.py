from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    Boolean,
    JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base
import uuid
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    phone = Column(String(15), unique=True, nullable=False, index=True)

    full_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    language = Column(String(10), default="en")

    country = Column(String(100), default="Nigeria")
    temperament = Column(String(20), default="sanguine")

    settings = Column(JSON, default=dict)

    start_row = Column(Integer, nullable=False)
    start_col = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    workspaces = relationship(
        "Workspace",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # Human-readable project identity
    project_name = Column(String(255), nullable=False)

    # Primary project domain
    project_domain = Column(String(100), default="general")

    # Keywords extracted from project/room conversations
    project_keywords = Column(JSON, default=list)

    # Project's deterministic GridCV mapping
    #
    # Example:
    # {
    #     "version": 1,
    #     "keywords": {
    #         "python": {"column": 3, "row": 17},
    #         "backend": {"column": 8, "row": 42}
    #     }
    # }
    project_grid_cv = Column(JSON, default=dict)

    # Compiled understanding of the room/project

    context_summary = Column(Text, default="")

    # Current temporal context for the project
    #
    # Example:
    # {
    #     "timezone": "Africa/Lagos",
    #     "date": "2026-08-09",
    #     "day": "Sunday",
    #     "hour": 22,
    #     "period": "night"
    # }

    temporal_context = Column(JSON, default=dict)

    # Version allows context modelling to evolve without
    # corrupting older workspace records.
    context_version = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    user = relationship(
        "User",
        back_populates="workspaces"
    )

    messages = relationship(
        "Message",
        back_populates="workspace",
        cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    role = Column(String(20), nullable=False)

    content = Column(Text, nullable=False)

    # AI interpretation of this particular message
    detected_domain = Column(String(100), default="general")

    # Important concepts extracted from the message
    keywords = Column(JSON, default=list)

    # Message-level deterministic GridCV placement
    grid_cv = Column(JSON, default=dict)

    # Date/time/hour interpretation associated with the message
    temporal_context = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)

    workspace = relationship(
        "Workspace",
        back_populates="messages"
    )


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    key = Column(String(64), unique=True, nullable=False)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True
    )

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)