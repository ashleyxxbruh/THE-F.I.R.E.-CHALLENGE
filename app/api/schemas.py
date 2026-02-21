from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel


class TicketAIShort(BaseModel):
    ticket_type: Optional[str] = None
    sentiment: Optional[str] = None
    priority: Optional[int] = None
    language: Optional[str] = None


class TicketAssignmentShort(BaseModel):
    status: Optional[str] = None
    office_name: Optional[str] = None
    manager_name: Optional[str] = None


class TicketListItem(BaseModel):
    ticket_id: int
    client_guid: str
    segment: Optional[str] = None
    city: Optional[str] = None
    description: Optional[str] = None
    ai: TicketAIShort
    assignment: TicketAssignmentShort
    assigned_at: Optional[datetime] = None


class TicketData(BaseModel):
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


class AiAnalysisData(BaseModel):
    id: int
    ticket_id: int
    ticket_type: Optional[str] = None
    sentiment: Optional[str] = None
    priority: Optional[int] = None
    language: Optional[str] = None
    summary: Optional[str] = None
    recommendation: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    raw_json: Optional[dict[str, Any]] = None


class AssignmentData(BaseModel):
    id: int
    ticket_id: int
    business_unit_id: int
    manager_id: Optional[int] = None
    status: str
    assigned_at: Optional[datetime] = None
    office_name: Optional[str] = None
    manager_name: Optional[str] = None
    reason: Optional[dict[str, Any]] = None


class TicketDetailResponse(BaseModel):
    ticket: TicketData
    ai_analysis: Optional[AiAnalysisData] = None
    assignment: Optional[AssignmentData] = None


class ManagerOut(BaseModel):
    manager_id: int
    full_name: str
    position: Optional[str] = None
    skills: list[str]
    office_name: str
    current_load: int
    assigned_tickets_count: int


class OfficeOut(BaseModel):
    office_id: int
    name: str
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    managers_count: int
    assigned_tickets_count: int


class KeyCount(BaseModel):
    key: str
    count: int


class StatsResponse(BaseModel):
    total_tickets: int
    total_assigned: int
    total_unassigned: int
    avg_priority: Optional[float] = None
    counts_by_office: list[KeyCount]
    counts_by_ticket_type: list[KeyCount]
    counts_by_sentiment: list[KeyCount]
    counts_by_language: list[KeyCount]
    counts_by_city: list[KeyCount]
