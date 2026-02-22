from __future__ import annotations

from typing import Any, Generator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import BusinessUnit, Manager
from app.schemas import ManagerOut, OfficeOut

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


@app.get("/offices", response_model=list[OfficeOut])
def get_offices(db: Session = Depends(get_db)) -> list[OfficeOut]:
    offices = db.execute(select(BusinessUnit).order_by(BusinessUnit.name.asc())).scalars().all()
    return [
        OfficeOut(
            id=office.id,
            name=office.name,
            address=office.address,
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
