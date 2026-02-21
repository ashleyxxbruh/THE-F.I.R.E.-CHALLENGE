from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import Date, ForeignKey, Integer, String, UniqueConstraint
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
