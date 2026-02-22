from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OfficeOut(BaseModel):
    id: int
    name: str
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class ManagerOut(BaseModel):
    id: int
    full_name: str
    position: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    business_unit_id: int
    business_unit_name: str
    current_load: int
