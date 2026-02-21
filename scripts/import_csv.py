"""
Run:
  export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/db
  python scripts/import_csv.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db import SessionLocal, engine
from app.models import Base, BusinessUnit, Manager, Ticket

logger = logging.getLogger("import_csv")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1251"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [str(col).strip() for col in df.columns]
            logger.info("Loaded %s using %s", path, encoding)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not decode CSV file {path}: {last_error}")


def clean_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def clean_skills(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in text.split(","):
        skill = raw.strip().upper()
        if not skill:
            continue
        if skill in seen:
            continue
        seen.add(skill)
        normalized.append(skill)
    return normalized


def clean_house_raw(value: Any) -> str | None:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return str(value).strip()

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = Decimal(text.replace(",", "."))
            if numeric == numeric.to_integral_value():
                return str(int(numeric))
        except InvalidOperation:
            pass
        return text

    text = str(value).strip()
    return text if text else None


def clean_birth_date(value: Any):
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def clean_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def ensure_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns {missing}")


def import_business_units(session, csv_path: Path) -> int:
    df = read_csv_with_fallback(csv_path)
    ensure_columns(df, ["Офис", "Адрес"], "business_units.csv")

    deduped: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        name = clean_text(row.get("Офис"))
        if not name:
            continue
        deduped[name] = {
            "name": name,
            "address": clean_text(row.get("Адрес")),
        }

    records = list(deduped.values())
    if not records:
        return 0

    stmt = insert(BusinessUnit).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=[BusinessUnit.name],
        set_={"address": stmt.excluded.address},
    )
    session.execute(stmt)
    return len(records)


def business_unit_name_to_id(session) -> dict[str, int]:
    rows = session.execute(select(BusinessUnit.name, BusinessUnit.id)).all()
    return {name: unit_id for name, unit_id in rows}


def import_managers(session, csv_path: Path) -> tuple[int, int]:
    df = read_csv_with_fallback(csv_path)
    ensure_columns(
        df,
        ["ФИО", "Должность", "Офис", "Навыки", "Количество обращений в работе"],
        "managers.csv",
    )

    office_to_id = business_unit_name_to_id(session)
    deduped: dict[tuple[str, int], dict[str, Any]] = {}
    skipped_unknown_office = 0

    for _, row in df.iterrows():
        full_name = clean_text(row.get("ФИО"))
        office = clean_text(row.get("Офис"))
        if not full_name:
            continue

        if not office or office not in office_to_id:
            skipped_unknown_office += 1
            logger.warning("Skipping manager '%s': unknown office '%s'", full_name, office)
            continue

        business_unit_id = office_to_id[office]
        key = (full_name, business_unit_id)
        deduped[key] = {
            "full_name": full_name,
            "position": clean_text(row.get("Должность")),
            "skills": clean_skills(row.get("Навыки")),
            "business_unit_id": business_unit_id,
            "current_load": clean_int(row.get("Количество обращений в работе"), default=0),
        }

    records = list(deduped.values())
    if records:
        stmt = insert(Manager).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=[Manager.full_name, Manager.business_unit_id],
            set_={
                "position": stmt.excluded.position,
                "skills": stmt.excluded.skills,
                "current_load": stmt.excluded.current_load,
            },
        )
        session.execute(stmt)

    return len(records), skipped_unknown_office


def import_tickets(session, csv_path: Path) -> int:
    df = read_csv_with_fallback(csv_path)
    ensure_columns(
        df,
        [
            "GUID клиента",
            "Пол клиента",
            "Дата рождения",
            "Описание",
            "Вложения",
            "Сегмент клиента",
            "Страна",
            "Область",
            "Населённый пункт",
            "Улица",
            "Дом",
        ],
        "tickets.csv",
    )

    deduped: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        client_guid = clean_text(row.get("GUID клиента"))
        if not client_guid:
            continue

        deduped[client_guid] = {
            "client_guid": client_guid,
            "gender": clean_text(row.get("Пол клиента")),
            "birth_date": clean_birth_date(row.get("Дата рождения")),
            "segment": clean_text(row.get("Сегмент клиента")),
            "description": clean_text(row.get("Описание")),
            "attachment_ref": clean_text(row.get("Вложения")),
            "country": clean_text(row.get("Страна")),
            "region": clean_text(row.get("Область")),
            "city": clean_text(row.get("Населённый пункт")),
            "street": clean_text(row.get("Улица")),
            "house_raw": clean_house_raw(row.get("Дом")),
        }

    records = list(deduped.values())
    if not records:
        return 0

    stmt = insert(Ticket).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Ticket.client_guid],
        set_={
            "gender": stmt.excluded.gender,
            "birth_date": stmt.excluded.birth_date,
            "segment": stmt.excluded.segment,
            "description": stmt.excluded.description,
            "attachment_ref": stmt.excluded.attachment_ref,
            "country": stmt.excluded.country,
            "region": stmt.excluded.region,
            "city": stmt.excluded.city,
            "street": stmt.excluded.street,
            "house_raw": stmt.excluded.house_raw,
        },
    )
    session.execute(stmt)
    return len(records)


def table_counts(session) -> dict[str, int]:
    return {
        "business_units": session.scalar(select(func.count(BusinessUnit.id))) or 0,
        "managers": session.scalar(select(func.count(Manager.id))) or 0,
        "tickets": session.scalar(select(func.count(Ticket.id))) or 0,
    }


def office_grouping(session) -> list[tuple[str, int]]:
    rows = session.execute(
        select(BusinessUnit.name, func.count(Manager.id))
        .outerjoin(Manager, Manager.business_unit_id == BusinessUnit.id)
        .group_by(BusinessUnit.name)
        .order_by(BusinessUnit.name)
    ).all()
    return [(name, count) for name, count in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import FIRE CSV datasets into PostgreSQL.")
    parser.add_argument("--business-units", default="business_units.csv")
    parser.add_argument("--managers", default="managers.csv")
    parser.add_argument("--tickets", default="tickets.csv")
    args = parser.parse_args()

    configure_logging()

    business_units_path = Path(args.business_units)
    managers_path = Path(args.managers)
    tickets_path = Path(args.tickets)

    for path in [business_units_path, managers_path, tickets_path]:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        before = table_counts(session)

    with SessionLocal() as session:
        with session.begin():
            business_units_upserted = import_business_units(session, business_units_path)
            managers_upserted, skipped_unknown_office = import_managers(session, managers_path)
            tickets_upserted = import_tickets(session, tickets_path)

    with SessionLocal() as session:
        after = table_counts(session)
        by_office = office_grouping(session)

    print("\nPost-import sanity report")
    print(f"BusinessUnits upserted from CSV: {business_units_upserted}")
    print(f"Managers upserted from CSV: {managers_upserted}")
    print(f"Tickets upserted from CSV: {tickets_upserted}")
    print(f"Managers skipped due to unknown office: {skipped_unknown_office}")
    print(f"BusinessUnits inserted (new rows): {after['business_units'] - before['business_units']}")
    print(f"Managers inserted (new rows): {after['managers'] - before['managers']}")
    print(f"Tickets inserted (new rows): {after['tickets'] - before['tickets']}")
    print("Managers by office:")
    for office_name, count in by_office:
        print(f"  {office_name}: {count}")


if __name__ == "__main__":
    main()
