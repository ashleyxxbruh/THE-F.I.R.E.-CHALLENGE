from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, aliased

from app.api.deps import get_db
from app.api.schemas import (
    AiAnalysisData,
    AssignmentData,
    KeyCount,
    ManagerOut,
    OfficeOut,
    StatsResponse,
    TicketAIShort,
    TicketAssignmentShort,
    TicketData,
    TicketDetailResponse,
    TicketListItem,
)
from app.models import AiAnalysis, Assignment, BusinessUnit, Manager, Ticket

router = APIRouter()


def _snippet(text: Optional[str], size: int = 180) -> Optional[str]:
    if not text:
        return None
    cleaned = str(text).strip()
    if len(cleaned) <= size:
        return cleaned
    return f"{cleaned[:size].rstrip()}..."


def _key_count_rows(rows: list[tuple[Optional[str], int]]) -> list[KeyCount]:
    return [KeyCount(key=(key if key else "—"), count=int(count or 0)) for key, count in rows]


@router.get("/tickets", response_model=list[TicketListItem])
def get_tickets(
    office: Optional[str] = None,
    status: Optional[str] = None,
    segment: Optional[str] = None,
    ticket_type: Optional[str] = None,
    sentiment: Optional[str] = None,
    language: Optional[str] = None,
    priority_min: Optional[int] = None,
    priority_max: Optional[int] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[TicketListItem]:
    office_alias = aliased(BusinessUnit)

    stmt = (
        select(
            Ticket.id.label("ticket_id"),
            Ticket.client_guid,
            Ticket.segment,
            Ticket.city,
            Ticket.description,
            AiAnalysis.ticket_type,
            AiAnalysis.sentiment,
            AiAnalysis.priority,
            AiAnalysis.language,
            Assignment.status,
            Assignment.assigned_at,
            office_alias.name.label("office_name"),
            Manager.full_name.label("manager_name"),
        )
        .outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id)
        .outerjoin(Assignment, Assignment.ticket_id == Ticket.id)
        .outerjoin(Manager, Manager.id == Assignment.manager_id)
        .outerjoin(office_alias, office_alias.id == Assignment.business_unit_id)
        .order_by(Ticket.id.desc())
    )

    if office:
        stmt = stmt.where(func.lower(office_alias.name) == office.strip().lower())
    if status:
        stmt = stmt.where(func.upper(Assignment.status) == status.strip().upper())
    if segment:
        stmt = stmt.where(func.lower(Ticket.segment) == segment.strip().lower())
    if ticket_type:
        stmt = stmt.where(func.lower(AiAnalysis.ticket_type) == ticket_type.strip().lower())
    if sentiment:
        stmt = stmt.where(func.lower(AiAnalysis.sentiment) == sentiment.strip().lower())
    if language:
        stmt = stmt.where(func.upper(AiAnalysis.language) == language.strip().upper())
    if priority_min is not None:
        stmt = stmt.where(AiAnalysis.priority >= priority_min)
    if priority_max is not None:
        stmt = stmt.where(AiAnalysis.priority <= priority_max)

    stmt = stmt.limit(limit).offset(offset)
    rows = db.execute(stmt).all()

    result: list[TicketListItem] = []
    for row in rows:
        result.append(
            TicketListItem(
                ticket_id=row.ticket_id,
                client_guid=row.client_guid,
                segment=row.segment,
                city=row.city,
                description=_snippet(row.description),
                ai=TicketAIShort(
                    ticket_type=row.ticket_type,
                    sentiment=row.sentiment,
                    priority=row.priority,
                    language=row.language,
                ),
                assignment=TicketAssignmentShort(
                    status=row.status,
                    office_name=row.office_name,
                    manager_name=row.manager_name,
                ),
                assigned_at=row.assigned_at,
            )
        )
    return result


@router.get("/tickets/{ticket_id}", response_model=TicketDetailResponse)
def get_ticket_detail(ticket_id: int, db: Session = Depends(get_db)) -> TicketDetailResponse:
    office_alias = aliased(BusinessUnit)
    stmt = (
        select(
            Ticket,
            AiAnalysis,
            Assignment,
            office_alias.name.label("office_name"),
            Manager.full_name.label("manager_name"),
        )
        .outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id)
        .outerjoin(Assignment, Assignment.ticket_id == Ticket.id)
        .outerjoin(office_alias, office_alias.id == Assignment.business_unit_id)
        .outerjoin(Manager, Manager.id == Assignment.manager_id)
        .where(Ticket.id == ticket_id)
    )
    row = db.execute(stmt).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket, ai_analysis, assignment, office_name, manager_name = row

    detail = TicketDetailResponse(
        ticket=TicketData(
            id=ticket.id,
            client_guid=ticket.client_guid,
            gender=ticket.gender,
            birth_date=ticket.birth_date,
            segment=ticket.segment,
            description=ticket.description,
            attachment_ref=ticket.attachment_ref,
            country=ticket.country,
            region=ticket.region,
            city=ticket.city,
            street=ticket.street,
            house_raw=ticket.house_raw,
        ),
        ai_analysis=(
            AiAnalysisData(
                id=ai_analysis.id,
                ticket_id=ai_analysis.ticket_id,
                ticket_type=ai_analysis.ticket_type,
                sentiment=ai_analysis.sentiment,
                priority=ai_analysis.priority,
                language=ai_analysis.language,
                summary=ai_analysis.summary,
                recommendation=ai_analysis.recommendation,
                lat=ai_analysis.lat,
                lon=ai_analysis.lon,
                raw_json=ai_analysis.raw_json,
            )
            if ai_analysis
            else None
        ),
        assignment=(
            AssignmentData(
                id=assignment.id,
                ticket_id=assignment.ticket_id,
                business_unit_id=assignment.business_unit_id,
                manager_id=assignment.manager_id,
                status=assignment.status,
                assigned_at=assignment.assigned_at,
                office_name=office_name,
                manager_name=manager_name,
                reason=assignment.reason,
            )
            if assignment
            else None
        ),
    )
    return detail


@router.get("/managers", response_model=list[ManagerOut])
def get_managers(db: Session = Depends(get_db)) -> list[ManagerOut]:
    stmt = (
        select(
            Manager.id,
            Manager.full_name,
            Manager.position,
            Manager.skills,
            BusinessUnit.name.label("office_name"),
            Manager.current_load,
            func.count(Assignment.id).label("assigned_tickets_count"),
        )
        .join(BusinessUnit, BusinessUnit.id == Manager.business_unit_id)
        .outerjoin(
            Assignment,
            (Assignment.manager_id == Manager.id) & (Assignment.status == "ASSIGNED"),
        )
        .group_by(
            Manager.id,
            Manager.full_name,
            Manager.position,
            Manager.skills,
            BusinessUnit.name,
            Manager.current_load,
        )
        .order_by(Manager.current_load.desc(), Manager.id.asc())
    )
    rows = db.execute(stmt).all()
    result: list[ManagerOut] = []
    for row in rows:
        skills_value = row.skills if isinstance(row.skills, list) else []
        result.append(
            ManagerOut(
                manager_id=row.id,
                full_name=row.full_name,
                position=row.position,
                skills=[str(skill) for skill in skills_value],
                office_name=row.office_name,
                current_load=int(row.current_load or 0),
                assigned_tickets_count=int(row.assigned_tickets_count or 0),
            )
        )
    return result


@router.get("/offices", response_model=list[OfficeOut])
def get_offices(db: Session = Depends(get_db)) -> list[OfficeOut]:
    manager_counts = (
        select(
            Manager.business_unit_id.label("business_unit_id"),
            func.count(Manager.id).label("managers_count"),
        )
        .group_by(Manager.business_unit_id)
        .subquery()
    )
    assigned_counts = (
        select(
            Assignment.business_unit_id.label("business_unit_id"),
            func.count(Assignment.id).label("assigned_tickets_count"),
        )
        .where(Assignment.status == "ASSIGNED")
        .group_by(Assignment.business_unit_id)
        .subquery()
    )

    stmt = (
        select(
            BusinessUnit.id,
            BusinessUnit.name,
            BusinessUnit.address,
            BusinessUnit.lat,
            BusinessUnit.lon,
            func.coalesce(manager_counts.c.managers_count, 0),
            func.coalesce(assigned_counts.c.assigned_tickets_count, 0),
        )
        .outerjoin(manager_counts, manager_counts.c.business_unit_id == BusinessUnit.id)
        .outerjoin(assigned_counts, assigned_counts.c.business_unit_id == BusinessUnit.id)
        .order_by(BusinessUnit.name.asc())
    )
    rows = db.execute(stmt).all()
    return [
        OfficeOut(
            office_id=row.id,
            name=row.name,
            address=row.address,
            lat=row.lat,
            lon=row.lon,
            managers_count=int(row[5] or 0),
            assigned_tickets_count=int(row[6] or 0),
        )
        for row in rows
    ]


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)) -> StatsResponse:
    totals_row = db.execute(
        select(
            func.count(Ticket.id).label("total_tickets"),
            func.sum(case((Assignment.status == "ASSIGNED", 1), else_=0)).label("assigned_count"),
            func.sum(case((Assignment.status == "UNASSIGNED", 1), else_=0)).label("unassigned_count"),
            func.sum(case((Assignment.id.is_(None), 1), else_=0)).label("missing_assignment_count"),
        ).outerjoin(Assignment, Assignment.ticket_id == Ticket.id)
    ).one()

    avg_priority = db.scalar(select(func.avg(AiAnalysis.priority)))

    counts_by_office_rows = db.execute(
        select(BusinessUnit.name, func.count(Assignment.id))
        .join(Assignment, Assignment.business_unit_id == BusinessUnit.id)
        .where(Assignment.status == "ASSIGNED")
        .group_by(BusinessUnit.name)
        .order_by(func.count(Assignment.id).desc(), BusinessUnit.name.asc())
    ).all()

    counts_by_ticket_type_rows = db.execute(
        select(AiAnalysis.ticket_type, func.count(AiAnalysis.id))
        .where(AiAnalysis.ticket_type.is_not(None))
        .group_by(AiAnalysis.ticket_type)
        .order_by(func.count(AiAnalysis.id).desc(), AiAnalysis.ticket_type.asc())
    ).all()

    counts_by_sentiment_rows = db.execute(
        select(AiAnalysis.sentiment, func.count(AiAnalysis.id))
        .where(AiAnalysis.sentiment.is_not(None))
        .group_by(AiAnalysis.sentiment)
        .order_by(func.count(AiAnalysis.id).desc(), AiAnalysis.sentiment.asc())
    ).all()

    counts_by_language_rows = db.execute(
        select(AiAnalysis.language, func.count(AiAnalysis.id))
        .where(AiAnalysis.language.is_not(None))
        .group_by(AiAnalysis.language)
        .order_by(func.count(AiAnalysis.id).desc(), AiAnalysis.language.asc())
    ).all()

    counts_by_city_rows = db.execute(
        select(Ticket.city, func.count(Ticket.id))
        .where(Ticket.city.is_not(None))
        .group_by(Ticket.city)
        .order_by(func.count(Ticket.id).desc(), Ticket.city.asc())
        .limit(10)
    ).all()

    total_tickets = int(totals_row.total_tickets or 0)
    total_assigned = int(totals_row.assigned_count or 0)
    total_unassigned = int(totals_row.unassigned_count or 0) + int(totals_row.missing_assignment_count or 0)

    return StatsResponse(
        total_tickets=total_tickets,
        total_assigned=total_assigned,
        total_unassigned=total_unassigned,
        avg_priority=float(avg_priority) if avg_priority is not None else None,
        counts_by_office=_key_count_rows(counts_by_office_rows),
        counts_by_ticket_type=_key_count_rows(counts_by_ticket_type_rows),
        counts_by_sentiment=_key_count_rows(counts_by_sentiment_rows),
        counts_by_language=_key_count_rows(counts_by_language_rows),
        counts_by_city=_key_count_rows(counts_by_city_rows),
    )
