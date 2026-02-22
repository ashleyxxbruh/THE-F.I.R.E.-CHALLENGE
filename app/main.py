from __future__ import annotations

from typing import Any, Generator

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import AiAnalysis, Assignment, BusinessUnit, Manager, Ticket
from app.schemas import (
    ManagerOut,
    OfficeOut,
    TicketAiAnalysisOut,
    TicketAssignmentDetailOut,
    TicketAssignmentOut,
    TicketDetailOut,
    TicketListItemOut,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def normalize_skills(raw_skills: Any) -> list[str]:
    if isinstance(raw_skills, list):
        return [str(item).strip() for item in raw_skills if str(item).strip()]
    if isinstance(raw_skills, str):
        return [item.strip() for item in raw_skills.split(",") if item.strip()]
    return []


def clean_control_chars(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = "".join(ch if (ord(ch) >= 32 or ch in ("\n", "\t")) else " " for ch in str(text))
    stripped = cleaned.strip()
    return stripped if stripped else None


def build_ai_analysis_payload(ai_analysis: AiAnalysis | None) -> TicketAiAnalysisOut | None:
    if ai_analysis is None:
        return None
    return TicketAiAnalysisOut(
        ticket_type=ai_analysis.ticket_type,
        sentiment=ai_analysis.sentiment,
        priority=int(ai_analysis.priority if ai_analysis.priority is not None else 5),
        language=ai_analysis.language or "RU",
        summary=ai_analysis.summary,
        recommendation=ai_analysis.recommendation,
        lat=ai_analysis.lat,
        lon=ai_analysis.lon,
    )


def build_assignment_payload(
    assignment: Assignment | None,
    business_unit_name: str | None,
    manager_name: str | None,
) -> TicketAssignmentOut | None:
    if assignment is None:
        return None
    return TicketAssignmentOut(
        status=assignment.status,
        business_unit_id=assignment.business_unit_id,
        business_unit_name=business_unit_name,
        manager_id=assignment.manager_id,
        manager_name=manager_name,
        assigned_at=assignment.assigned_at,
    )


def build_assignment_detail_payload(
    assignment: Assignment | None,
    business_unit_name: str | None,
    manager_name: str | None,
) -> TicketAssignmentDetailOut | None:
    if assignment is None:
        return None
    return TicketAssignmentDetailOut(
        status=assignment.status,
        business_unit_id=assignment.business_unit_id,
        business_unit_name=business_unit_name,
        manager_id=assignment.manager_id,
        manager_name=manager_name,
        assigned_at=assignment.assigned_at,
        reason=assignment.reason,
    )


@app.get("/offices", response_model=list[OfficeOut])
def get_offices(db: Session = Depends(get_db)) -> list[OfficeOut]:
    offices = db.execute(select(BusinessUnit).order_by(BusinessUnit.name.asc())).scalars().all()
    return [
        OfficeOut(
            id=office.id,
            name=office.name,
            address=clean_control_chars(office.address),
            lat=office.lat,
            lon=office.lon,
        )
        for office in offices
    ]


@app.get("/managers", response_model=list[ManagerOut])
def get_managers(db: Session = Depends(get_db)) -> list[ManagerOut]:
    rows = db.execute(
        select(Manager, BusinessUnit.name)
        .join(BusinessUnit, BusinessUnit.id == Manager.business_unit_id)
        .order_by(BusinessUnit.name.asc(), Manager.current_load.asc(), Manager.id.asc())
    ).all()
    return [
        ManagerOut(
            id=manager.id,
            full_name=manager.full_name,
            position=manager.position,
            skills=normalize_skills(manager.skills),
            business_unit_id=manager.business_unit_id,
            business_unit_name=business_unit_name,
            current_load=manager.current_load,
        )
        for manager, business_unit_name in rows
    ]


@app.get("/tickets", response_model=list[TicketListItemOut])
def get_tickets(
    segment: str | None = Query(default=None),
    ticket_type: str | None = Query(default=None),
    sentiment: str | None = Query(default=None),
    language: str | None = Query(default=None),
    priority_min: int | None = Query(default=None),
    priority_max: int | None = Query(default=None),
    status: str | None = Query(default=None),
    office_id: int | None = Query(default=None),
    manager_id: int | None = Query(default=None),
    q: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[TicketListItemOut]:
    if priority_min is not None and priority_max is not None and priority_min > priority_max:
        raise HTTPException(status_code=400, detail="priority_min cannot be greater than priority_max")

    stmt = (
        select(
            Ticket,
            AiAnalysis,
            Assignment,
            BusinessUnit.name.label("business_unit_name"),
            Manager.full_name.label("manager_name"),
        )
        .outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id)
        .outerjoin(Assignment, Assignment.ticket_id == Ticket.id)
        .outerjoin(BusinessUnit, BusinessUnit.id == Assignment.business_unit_id)
        .outerjoin(Manager, Manager.id == Assignment.manager_id)
    )

    if segment:
        stmt = stmt.where(Ticket.segment == segment)
    if ticket_type:
        stmt = stmt.where(AiAnalysis.ticket_type == ticket_type)
    if sentiment:
        stmt = stmt.where(AiAnalysis.sentiment == sentiment)
    if language:
        stmt = stmt.where(AiAnalysis.language == language)
    if priority_min is not None:
        stmt = stmt.where(AiAnalysis.priority >= priority_min)
    if priority_max is not None:
        stmt = stmt.where(AiAnalysis.priority <= priority_max)

    normalized_status = status.strip().upper() if status and status.strip() else None
    if normalized_status == "UNASSIGNED":
        stmt = stmt.where(or_(Assignment.id.is_(None), Assignment.status == "UNASSIGNED"))
    elif normalized_status:
        stmt = stmt.where(Assignment.status == normalized_status)

    if office_id is not None:
        stmt = stmt.where(Assignment.business_unit_id == office_id)
    if manager_id is not None:
        stmt = stmt.where(Assignment.manager_id == manager_id)
    if q and q.strip():
        stmt = stmt.where(Ticket.description.ilike(f"%{q.strip()}%"))

    rows = db.execute(stmt.order_by(Ticket.id.desc()).limit(limit).offset(offset)).all()
    return [
        TicketListItemOut(
            id=ticket.id,
            client_guid=ticket.client_guid,
            segment=ticket.segment,
            country=ticket.country,
            region=ticket.region,
            city=ticket.city,
            street=clean_control_chars(ticket.street),
            house_raw=ticket.house_raw,
            description=clean_control_chars(ticket.description),
            ai_analysis=build_ai_analysis_payload(ai_analysis),
            assignment=build_assignment_payload(
                assignment=assignment,
                business_unit_name=business_unit_name,
                manager_name=manager_name,
            ),
        )
        for ticket, ai_analysis, assignment, business_unit_name, manager_name in rows
    ]


@app.get("/tickets/{ticket_id}", response_model=TicketDetailOut)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)) -> TicketDetailOut:
    stmt = (
        select(
            Ticket,
            AiAnalysis,
            Assignment,
            BusinessUnit.name.label("business_unit_name"),
            Manager.full_name.label("manager_name"),
        )
        .outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id)
        .outerjoin(Assignment, Assignment.ticket_id == Ticket.id)
        .outerjoin(BusinessUnit, BusinessUnit.id == Assignment.business_unit_id)
        .outerjoin(Manager, Manager.id == Assignment.manager_id)
        .where(Ticket.id == ticket_id)
        .limit(1)
    )
    row = db.execute(stmt).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket, ai_analysis, assignment, business_unit_name, manager_name = row
    return TicketDetailOut(
        id=ticket.id,
        client_guid=ticket.client_guid,
        gender=ticket.gender,
        birth_date=ticket.birth_date,
        segment=ticket.segment,
        description=clean_control_chars(ticket.description),
        attachment_ref=ticket.attachment_ref,
        country=ticket.country,
        region=ticket.region,
        city=ticket.city,
        street=clean_control_chars(ticket.street),
        house_raw=ticket.house_raw,
        ai_analysis=build_ai_analysis_payload(ai_analysis),
        assignment=build_assignment_detail_payload(
            assignment=assignment,
            business_unit_name=business_unit_name,
            manager_name=manager_name,
        ),
    )
