"""
Run:
  export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/db
  python scripts/route_tickets.py
  python scripts/route_tickets.py --force
  python scripts/route_tickets.py --limit 10
  python scripts/route_tickets.py --tickets <GUID>
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    import requests
    from requests import RequestException
except ImportError:  # pragma: no cover
    requests = None

    class RequestException(Exception):
        pass

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db import SessionLocal, engine
from app.models import AiAnalysis, Assignment, Base, BusinessUnit, Manager, Ticket

logger = logging.getLogger("route_tickets")

DEFAULT_TICKET_TYPE = "Консультация"
DEFAULT_LANGUAGE = "RU"
STATUS_ASSIGNED = "ASSIGNED"
STATUS_UNASSIGNED = "UNASSIGNED"

KAZAKHSTAN_ALIASES = {"казахстан", "kazakhstan", "kz", "рк", "rk"}

ALMATY_COORDS = (43.238949, 76.889709)
ASTANA_COORDS = (51.169392, 71.449074)


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_spaces_lower(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return " ".join(text.split()).lower()


def clean_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


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
    raise RuntimeError(f"Could not decode CSV file {path}: {last_error}")


def manager_office_key(full_name: Any, office_name: Any) -> tuple[str, str]:
    return (normalize_spaces_lower(full_name), normalize_spaces_lower(office_name))


def normalize_language(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return DEFAULT_LANGUAGE
    upper = text.upper()
    return upper if upper in {"KZ", "ENG", "RU"} else DEFAULT_LANGUAGE


def normalize_country_token(value: Any) -> str:
    text = normalize_spaces_lower(value)
    if not text:
        return ""
    for token in (".", ",", "-", "_"):
        text = text.replace(token, "")
    return text.replace(" ", "")


def is_abroad_or_unknown(country: Any) -> bool:
    token = normalize_country_token(country)
    if not token:
        return True
    return token not in KAZAKHSTAN_ALIASES


def is_vip_or_priority(segment: Any) -> bool:
    segment_norm = normalize_spaces_lower(segment)
    return segment_norm in {"vip", "priority"}


def is_data_change(ticket_type: Any) -> bool:
    return normalize_spaces_lower(ticket_type) == normalize_spaces_lower("Смена данных")


def is_main_specialist(position: Any) -> bool:
    return normalize_spaces_lower(position) == normalize_spaces_lower("Глав спец")


def manager_skills(manager: Manager) -> set[str]:
    skills: set[str] = set()
    raw = manager.skills
    if isinstance(raw, list):
        for item in raw:
            text = clean_text(item)
            if text:
                skills.add(text.upper())
    elif isinstance(raw, str):
        for item in raw.split(","):
            text = clean_text(item)
            if text:
                skills.add(text.upper())
    return skills


def normalize_coord(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def geocode_address(address: str, cache: dict[str, Optional[tuple[float, float]]]) -> Optional[tuple[float, float]]:
    if not address:
        return None

    if address in cache:
        return cache[address]

    if requests is None:
        cache[address] = None
        return None

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "fire-hackathon-routing-engine/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            first = payload[0]
            lat = normalize_coord(first.get("lat"))
            lon = normalize_coord(first.get("lon"))
            if lat is not None and lon is not None:
                cache[address] = (lat, lon)
                return cache[address]
    except (RequestException, ValueError, TypeError, KeyError):
        pass

    cache[address] = None
    return None


def office_city_centroid(office_name: Any) -> Optional[tuple[float, float]]:
    name = normalize_spaces_lower(office_name)
    if not name:
        return None
    if "алматы" in name:
        return ALMATY_COORDS
    if "астана" in name or "нур-султан" in name or "нур султан" in name:
        return ASTANA_COORDS
    return None


def ensure_office_coordinates(skip_geocode: bool) -> None:
    address_cache: dict[str, Optional[tuple[float, float]]] = {}

    with SessionLocal() as session:
        with session.begin():
            offices = session.execute(select(BusinessUnit).order_by(BusinessUnit.id)).scalars().all()
            for office in offices:
                if office.lat is not None and office.lon is not None:
                    continue

                coords: Optional[tuple[float, float]] = None
                source: Optional[str] = None

                if not skip_geocode:
                    address = clean_text(office.address)
                    if address:
                        coords = geocode_address(address, address_cache)
                        if coords is not None:
                            source = "nominatim"

                if coords is None:
                    centroid = office_city_centroid(office.name)
                    if centroid is not None:
                        coords = centroid
                        source = "city_centroid"

                if coords is not None:
                    office.lat = coords[0]
                    office.lon = coords[1]
                    logger.info("Office '%s' coordinates set from %s", office.name, source)
                else:
                    logger.warning("Office '%s' coordinates unavailable", office.name)


def load_manager_base_loads_from_csv(
    managers_csv_path: Path,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], tuple[str, str]], int]:
    if not managers_csv_path.exists():
        raise FileNotFoundError(f"Managers CSV file not found: {managers_csv_path}")

    df = read_csv_with_fallback(managers_csv_path)
    required_columns = ["ФИО", "Офис", "Количество обращений в работе"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in managers CSV {managers_csv_path}: {missing_columns}"
        )

    base_loads: dict[tuple[str, str], int] = {}
    source_labels: dict[tuple[str, str], tuple[str, str]] = {}
    invalid_rows = 0

    for index, row in df.iterrows():
        full_name = clean_text(row.get("ФИО"))
        office_name = clean_text(row.get("Офис"))
        base_load = clean_int(row.get("Количество обращений в работе"), default=0)

        if not full_name or not office_name:
            invalid_rows += 1
            logger.warning(
                "CSV row %s has missing ФИО/Офис and cannot be matched to DB manager",
                index + 2,
            )
            continue

        key = manager_office_key(full_name, office_name)
        base_loads[key] = base_load
        source_labels[key] = (full_name, office_name)

    return base_loads, source_labels, invalid_rows


def reset_routing_state_for_force(managers_csv_path: Path) -> None:
    logger.info("FORCE mode: deleting all assignments and resetting manager loads from managers.csv")
    csv_load_map, csv_labels, invalid_csv_rows = load_manager_base_loads_from_csv(managers_csv_path)

    assignments_deleted = 0
    managers_updated = 0
    managers_missing_in_csv = 0
    csv_rows_with_no_db_match = 0

    with SessionLocal() as session:
        with session.begin():
            assignments_deleted = session.scalar(select(func.count(Assignment.id))) or 0
            session.execute(delete(Assignment))

            db_rows = session.execute(
                select(Manager, BusinessUnit.name).join(BusinessUnit, BusinessUnit.id == Manager.business_unit_id)
            ).all()

            db_manager_by_key: dict[tuple[str, str], Manager] = {}
            for manager, office_name in db_rows:
                key = manager_office_key(manager.full_name, office_name)
                db_manager_by_key[key] = manager

            for key, manager in db_manager_by_key.items():
                if key in csv_load_map:
                    manager.current_load = csv_load_map[key]
                    managers_updated += 1
                else:
                    manager.current_load = 0
                    managers_missing_in_csv += 1

            for key, (full_name, office_name) in csv_labels.items():
                if key not in db_manager_by_key:
                    csv_rows_with_no_db_match += 1
                    logger.warning(
                        "No DB manager match for CSV entry: full_name='%s', office='%s'",
                        full_name,
                        office_name,
                    )

            csv_rows_with_no_db_match += invalid_csv_rows

    logger.info("assignments_deleted=%s", assignments_deleted)
    logger.info("managers_updated=%s", managers_updated)
    logger.info("managers_missing_in_csv=%s", managers_missing_in_csv)
    logger.info("csv_rows_with_no_db_match=%s", csv_rows_with_no_db_match)


def load_offices() -> list[BusinessUnit]:
    with SessionLocal() as session:
        return session.execute(select(BusinessUnit).order_by(BusinessUnit.id)).scalars().all()


def parse_tickets_arg(value: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    if not value:
        return None, None
    raw = value.strip()
    if not raw:
        return None, None
    try:
        numeric = int(raw)
        return None, max(numeric, 0)
    except ValueError:
        return raw, None


def effective_limit(limit_a: Optional[int], limit_b: Optional[int]) -> Optional[int]:
    values = [value for value in (limit_a, limit_b) if value is not None and value > 0]
    if not values:
        return None
    return min(values)


def select_tickets(force: bool, tickets_arg: Optional[str], limit: Optional[int]) -> list[tuple[Ticket, Optional[AiAnalysis]]]:
    guid_filter, count_filter = parse_tickets_arg(tickets_arg)
    limit_value = effective_limit(limit, count_filter)

    stmt = (
        select(Ticket, AiAnalysis)
        .outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id)
        .order_by(Ticket.id)
    )

    if not force:
        stmt = stmt.outerjoin(Assignment, Assignment.ticket_id == Ticket.id).where(Assignment.id.is_(None))

    if guid_filter:
        stmt = stmt.where(Ticket.client_guid == guid_filter)

    if limit_value is not None:
        stmt = stmt.limit(limit_value)

    with SessionLocal() as session:
        rows = session.execute(stmt).all()
    return [(row[0], row[1]) for row in rows]


def find_almaty_office(offices: list[BusinessUnit]) -> Optional[BusinessUnit]:
    for office in offices:
        if "алматы" in normalize_spaces_lower(office.name):
            return office
    return None


def find_astana_office(offices: list[BusinessUnit]) -> Optional[BusinessUnit]:
    for office in offices:
        name = normalize_spaces_lower(office.name)
        if "астана" in name or "нур-султан" in name or "нур султан" in name:
            return office
    return None


def choose_fallback_office(
    offices: list[BusinessUnit],
    fallback_state: dict[str, int],
) -> tuple[Optional[BusinessUnit], str]:
    astana = find_astana_office(offices)
    almaty = find_almaty_office(offices)

    if astana is not None and almaty is not None:
        idx = fallback_state.get("astana_almaty_rr", 0) % 2
        chosen = [astana, almaty][idx]
        fallback_state["astana_almaty_rr"] = (idx + 1) % 2
        return chosen, "astana_almaty_rr"

    if astana is not None:
        return astana, "astana_only"

    if almaty is not None:
        return almaty, "almaty_only"

    if not offices:
        return None, "no_offices_available"

    if len(offices) == 1:
        return offices[0], "single_office_fallback"

    idx = fallback_state.get("generic_rr", 0) % 2
    chosen = [offices[0], offices[1]][idx]
    fallback_state["generic_rr"] = (idx + 1) % 2
    return chosen, "generic_rr"


def choose_nearest_office(ticket_lat: float, ticket_lon: float, offices: list[BusinessUnit]) -> Optional[BusinessUnit]:
    candidates = [office for office in offices if office.lat is not None and office.lon is not None]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda office: haversine_km(ticket_lat, ticket_lon, float(office.lat), float(office.lon)),
    )


def build_filters(ticket: Ticket, ai_analysis: Optional[AiAnalysis]) -> tuple[dict[str, Any], str]:
    vip_required = is_vip_or_priority(ticket.segment)
    ticket_type = clean_text(ai_analysis.ticket_type if ai_analysis else None) or DEFAULT_TICKET_TYPE
    data_change_required = is_data_change(ticket_type)
    language = normalize_language(ai_analysis.language if ai_analysis else None)
    language_required = language if language in {"KZ", "ENG"} else None

    segment_bucket = "VIP_OR_PRIORITY" if vip_required else "NORMAL"
    bucket = f"{segment_bucket}|{normalize_spaces_lower(ticket_type)}|{language}"

    filters = {
        "vip_required": vip_required,
        "data_change_required": data_change_required,
        "language_required": language_required,
    }
    return filters, bucket


def eligible_managers_for_office(
    session,
    office_id: int,
    filters: dict[str, Any],
) -> list[Manager]:
    managers = session.execute(
        select(Manager)
        .where(Manager.business_unit_id == office_id)
        .order_by(Manager.current_load.asc(), Manager.id.asc())
    ).scalars().all()

    vip_required = bool(filters["vip_required"])
    data_change_required = bool(filters["data_change_required"])
    language_required = filters["language_required"]

    eligible: list[Manager] = []
    for manager in managers:
        skills = manager_skills(manager)

        if vip_required and "VIP" not in skills:
            continue

        if data_change_required and not is_main_specialist(manager.position):
            continue

        if language_required and language_required not in skills:
            continue

        eligible.append(manager)

    return eligible


def pick_manager_with_rr(
    eligible: list[Manager],
    business_unit_id: int,
    bucket: str,
    rr_state: dict[tuple[int, str], int],
) -> tuple[Optional[Manager], str]:
    rr_key = f"{business_unit_id}:{bucket}"
    if not eligible:
        return None, rr_key

    top_two = eligible[:2]
    if len(top_two) == 1:
        return top_two[0], rr_key

    state_key = (business_unit_id, bucket)
    toggle = rr_state.get(state_key, 0) % 2
    chosen = top_two[toggle]
    rr_state[state_key] = (toggle + 1) % 2
    return chosen, rr_key


def upsert_assignment(
    session,
    values: dict[str, Any],
) -> None:
    stmt = insert(Assignment).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Assignment.ticket_id],
        set_={
            "business_unit_id": stmt.excluded.business_unit_id,
            "manager_id": stmt.excluded.manager_id,
            "assigned_at": func.now(),
            "status": stmt.excluded.status,
            "reason": stmt.excluded.reason,
        },
    )
    session.execute(stmt)


def process_ticket(
    ticket: Ticket,
    ai_analysis: Optional[AiAnalysis],
    offices: list[BusinessUnit],
    rr_state: dict[tuple[int, str], int],
    fallback_state: dict[str, int],
    debug: bool,
) -> dict[str, Any]:
    result = {
        "processed": 1,
        "assigned": 0,
        "unassigned": 0,
        "fallback_50_50_count": 0,
        "nearest_office_count": 0,
        "office_name": None,
        "manager_id": None,
    }

    ticket_lat = normalize_coord(ai_analysis.lat if ai_analysis else None)
    ticket_lon = normalize_coord(ai_analysis.lon if ai_analysis else None)
    abroad_or_unknown = is_abroad_or_unknown(ticket.country)

    selected_office: Optional[BusinessUnit] = None
    office_choice = "nearest"
    office_fallback_mode = None

    if abroad_or_unknown or ticket_lat is None or ticket_lon is None:
        selected_office, office_fallback_mode = choose_fallback_office(offices, fallback_state)
        office_choice = "fallback_50_50"
        result["fallback_50_50_count"] = 1
    else:
        selected_office = choose_nearest_office(ticket_lat, ticket_lon, offices)
        if selected_office is None:
            selected_office, office_fallback_mode = choose_fallback_office(offices, fallback_state)
            office_choice = "fallback_50_50"
            result["fallback_50_50_count"] = 1
        else:
            result["nearest_office_count"] = 1

    if selected_office is None:
        logger.error("No office available for ticket %s; skipping.", ticket.client_guid)
        return result

    filters, bucket = build_filters(ticket, ai_analysis)

    try:
        with SessionLocal() as session:
            with session.begin():
                office = session.get(BusinessUnit, selected_office.id)
                if office is None:
                    raise RuntimeError(f"Office id {selected_office.id} not found during assignment")

                eligible = eligible_managers_for_office(
                    session=session,
                    office_id=office.id,
                    filters=filters,
                )
                selected_manager, rr_key = pick_manager_with_rr(
                    eligible=eligible,
                    business_unit_id=office.id,
                    bucket=bucket,
                    rr_state=rr_state,
                )

                status = STATUS_ASSIGNED if selected_manager is not None else STATUS_UNASSIGNED
                unassigned_reason = None
                if selected_manager is None:
                    unassigned_reason = "no_eligible_managers"

                reason = {
                    "office_choice": office_choice,
                    "office_name": office.name,
                    "office_fallback_mode": office_fallback_mode,
                    "filters": filters,
                    "eligible_managers_count": len(eligible),
                    "rr_key": rr_key,
                    "selected_manager_id": selected_manager.id if selected_manager else None,
                    "unassigned_reason": unassigned_reason,
                }

                values = {
                    "ticket_id": ticket.id,
                    "business_unit_id": office.id,
                    "manager_id": selected_manager.id if selected_manager else None,
                    "status": status,
                    "reason": reason,
                }
                upsert_assignment(session, values)

                if selected_manager is not None:
                    session.execute(
                        update(Manager)
                        .where(Manager.id == selected_manager.id)
                        .values(current_load=Manager.current_load + 1)
                    )

                result["office_name"] = office.name
                result["manager_id"] = selected_manager.id if selected_manager else None
                if selected_manager is not None:
                    result["assigned"] = 1
                else:
                    result["unassigned"] = 1
    except Exception as exc:  # pragma: no cover
        if debug:
            logger.exception("Routing failed for ticket %s", ticket.client_guid)
        else:
            logger.error("Routing failed for ticket %s: %s", ticket.client_guid, exc)
        return result

    return result


def manager_loads_report() -> list[tuple[str, str, int]]:
    with SessionLocal() as session:
        rows = session.execute(
            select(Manager.full_name, BusinessUnit.name, Manager.current_load)
            .join(BusinessUnit, BusinessUnit.id == Manager.business_unit_id)
            .order_by(Manager.current_load.desc(), Manager.id.asc())
        ).all()
    return [(full_name, office_name, current_load) for full_name, office_name, current_load in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Route tickets to managers with hard filters and balancing.")
    parser.add_argument("--force", action="store_true", help="Re-route even if assignment already exists.")
    parser.add_argument("--limit", type=int, help="Maximum number of tickets to process.")
    parser.add_argument("--tickets", help="Either integer count or a specific ticket client GUID.")
    parser.add_argument(
        "--managers-csv",
        default="managers.csv",
        help="Path to managers.csv used to reset base manager loads in --force mode.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and stack traces.")
    parser.add_argument(
        "--no-office-geocode",
        action="store_true",
        help="Skip Nominatim office geocoding; use existing coordinates and city-centroid fallback only.",
    )
    args = parser.parse_args()

    configure_logging(debug=args.debug)
    Base.metadata.create_all(bind=engine)

    if args.force:
        managers_csv_path = Path(args.managers_csv)
        if not managers_csv_path.exists():
            raise FileNotFoundError(f"Managers CSV file not found: {managers_csv_path}")
        reset_routing_state_for_force(managers_csv_path=managers_csv_path)

    ensure_office_coordinates(skip_geocode=args.no_office_geocode)
    offices = load_offices()
    if not offices:
        print("\nRouting Report")
        print("total selected tickets: 0")
        print("processed: 0")
        print("assigned: 0")
        print("unassigned: 0")
        print("fallback_50_50_count: 0")
        print("nearest_office_count: 0")
        print("per-office assigned counts:")
        print("per-manager final loads:")
        return

    ticket_rows = select_tickets(force=args.force, tickets_arg=args.tickets, limit=args.limit)

    rr_state: dict[tuple[int, str], int] = {}
    fallback_state: dict[str, int] = {}
    per_office_assigned = defaultdict(int)

    total_selected = len(ticket_rows)
    processed = 0
    assigned = 0
    unassigned = 0
    fallback_50_50_count = 0
    nearest_office_count = 0

    for ticket, ai_analysis in ticket_rows:
        processed += 1
        try:
            item = process_ticket(
                ticket=ticket,
                ai_analysis=ai_analysis,
                offices=offices,
                rr_state=rr_state,
                fallback_state=fallback_state,
                debug=args.debug,
            )
            assigned += int(item["assigned"])
            unassigned += int(item["unassigned"])
            fallback_50_50_count += int(item["fallback_50_50_count"])
            nearest_office_count += int(item["nearest_office_count"])
            if item["assigned"] and item["office_name"]:
                per_office_assigned[item["office_name"]] += 1
        except Exception as exc:  # pragma: no cover
            if args.debug:
                logger.exception("Unexpected failure for ticket %s", ticket.client_guid)
            else:
                logger.error("Unexpected failure for ticket %s: %s", ticket.client_guid, exc)

    loads = manager_loads_report()

    print("\nRouting Report")
    print(f"total selected tickets: {total_selected}")
    print(f"processed: {processed}")
    print(f"assigned: {assigned}")
    print(f"unassigned: {unassigned}")
    print(f"fallback_50_50_count: {fallback_50_50_count}")
    print(f"nearest_office_count: {nearest_office_count}")
    print("per-office assigned counts:")
    for office_name, count in sorted(per_office_assigned.items(), key=lambda x: x[0]):
        print(f"  {office_name}: {count}")
    print("per-manager final loads:")
    for full_name, office_name, current_load in loads:
        print(f"  {full_name} ({office_name}): {current_load}")


if __name__ == "__main__":
    main()
