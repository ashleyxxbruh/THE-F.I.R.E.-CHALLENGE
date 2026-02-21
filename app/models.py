from __future__ import annotations

from datetime import date
from typing import Any, Optional

from sqlalchemy import Date, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class BusinessUnit(Base):
    __tablename__ = "business_units"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    address: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    managers: Mapped[list["Manager"]] = relationship(back_populates="business_unit")


class Manager(Base):
    __tablename__ = "managers"
    __table_args__ = (
        UniqueConstraint("full_name", "business_unit_id", name="uq_manager_full_name_business_unit"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    skills: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    business_unit_id: Mapped[int] = mapped_column(ForeignKey("business_units.id"), nullable=False, index=True)
    current_load: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    business_unit: Mapped["BusinessUnit"] = relationship(back_populates="managers")


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_guid: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    gender: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    segment: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    attachment_ref: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    region: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    street: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    house_raw: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    ai_analysis: Mapped[Optional["AiAnalysis"]] = relationship(back_populates="ticket", uselist=False)


class AiAnalysis(Base):
    __tablename__ = "ai_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[int] = mapped_column(ForeignKey("tickets.id"), nullable=False, unique=True, index=True)
    ticket_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="RU")
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    ticket: Mapped["Ticket"] = relationship(back_populates="ai_analysis")
