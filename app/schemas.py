from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

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


class TicketAiAnalysisOut(BaseModel):
    ticket_type: Optional[str] = None
    sentiment: Optional[str] = None
    priority: int
    language: str
    summary: Optional[str] = None
    recommendation: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class TicketAssignmentOut(BaseModel):
    status: str
    business_unit_id: int
    business_unit_name: Optional[str] = None
    manager_id: Optional[int] = None
    manager_name: Optional[str] = None
    assigned_at: Optional[datetime] = None


class TicketAssignmentDetailOut(TicketAssignmentOut):
    reason: Optional[dict[str, Any]] = None


class TicketListItemOut(BaseModel):
    id: int
    client_guid: str
    segment: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    street: Optional[str] = None
    house_raw: Optional[str] = None
    description: Optional[str] = None
    ai_analysis: Optional[TicketAiAnalysisOut] = None
    assignment: Optional[TicketAssignmentOut] = None


class TicketDetailOut(BaseModel):
    id: int
    client_guid: str
    gender: Optional[str] = None
    birth_date: Optional[date] = None
    segment: Optional[str] = None
    description: Optional[str] = None
    attachment_ref: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    street: Optional[str] = None
    house_raw: Optional[str] = None
    ai_analysis: Optional[TicketAiAnalysisOut] = None
    assignment: Optional[TicketAssignmentDetailOut] = None


class StatsTotalsOut(BaseModel):
    tickets_total: int
    assigned: int
    unassigned: int
    dropped_spam: int


class TicketTypeCountOut(BaseModel):
    ticket_type: str
    count: int


class SentimentCountOut(BaseModel):
    sentiment: str
    count: int


class LanguageCountOut(BaseModel):
    language: str
    count: int


class OfficeAssignedCountOut(BaseModel):
    office_name: str
    count: int


class ManagerLoadOut(BaseModel):
    manager_id: int
    manager_name: str
    office_name: str
    current_load: int


class StatsOut(BaseModel):
    totals: StatsTotalsOut
    by_ticket_type: list[TicketTypeCountOut] = Field(default_factory=list)
    by_sentiment: list[SentimentCountOut] = Field(default_factory=list)
    by_language: list[LanguageCountOut] = Field(default_factory=list)
    by_office_assigned: list[OfficeAssignedCountOut] = Field(default_factory=list)
    manager_loads: list[ManagerLoadOut] = Field(default_factory=list)
