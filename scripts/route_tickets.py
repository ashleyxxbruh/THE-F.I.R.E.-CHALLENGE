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
import re
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
STATUS_DROPPED_SPAM = "DROPPED_SPAM"
SPAM_TICKET_CANONICAL = "\u0441\u043f\u0430\u043c"
SPAM_TICKET_LABEL = "\u0421\u043f\u0430\u043c"
SPAM_URL_MARKERS = (
    "http://",
    "https://",
    "www.",
    "safelinks.protection.outlook.com",
    "t.me",
    "bit.ly",
    "wa.me",
)
SPAM_INVITE_EVENT_KEYWORDS = (
    "приглаша",
    "приглашение",
    "мероприят",
    "вебинар",
    "семинар",
    "конференц",
    "форум",
    "день инвестора",
    "презентац",
    "roadshow",
    "инвестор",
    "инвестора",
)
SPAM_CTA_KEYWORDS = (
    "регистрац",
    "зарегистр",
    "ссылка",
    "перейдите",
    "подробнее",
    "участие",
)
SPAM_MARKETING_KEYWORDS = (
    "топ-",
    "рейтинг",
    "победител",
    "лидер",
    "эмитент",
    "облигац",
    "esg",
    "raex",
    "эксперт ра",
)
SPAM_AD_KEYWORDS = (
    "в наличии",
    "отгрузка",
    "минимальный заказ",
    "оптом",
    "цена",
    "скидка",
    "акция",
    "предложение",
    "купить",
    "заказать",
    "доставка",
    "welding",
    "aggregat",
    "агрегат",
    "свароч",
    "тюльпан",
    "розы",
    "продам",
    "продажа",
)
SPAM_DATE_MONTH_KEYWORDS = (
    "феврал",
    "марта",
    "апрел",
    "мая",
    "июн",
    "июл",
    "август",
    "сентябр",
    "октябр",
    "ноябр",
    "декабр",
)
SUPPORT_CONTEXT_KEYWORDS = (
    "не работает",
    "ошибк",
    "вход",
    "логин",
    "парол",
    "перевод",
    "карта",
    "счет",
    "счёт",
    "заблок",
    "комисси",
    "мошенн",
    "верификац",
    "смс",
    "приложени",
    "платеж",
    "платёж",
    "транзакц",
    "возврат",
)
CITY_ALIASES: dict[str, str] = {
    "aktau": "Актау",
    "актау": "Актау",
    "aktobe": "Актобе",
    "актобе": "Актобе",
    "atyrau": "Атырау",
    "атырау": "Атырау",
    "karaganda": "Караганда",
    "караганда": "Караганда",
    "kokshetau": "Кокшетау",
    "кокшетау": "Кокшетау",
    "kostanay": "Костанай",
    "костанай": "Костанай",
    "kyzylorda": "Кызылорда",
    "кызылорда": "Кызылорда",
    "pavlodar": "Павлодар",
    "павлодар": "Павлодар",
    "petropavlovsk": "Петропавловск",
    "петропавловск": "Петропавловск",
    "taraz": "Тараз",
    "тараз": "Тараз",
    "uralsk": "Уральск",
    "oral": "Уральск",
    "уральск": "Уральск",
    "ораль": "Уральск",
    "ustkamenogorsk": "Усть-Каменогорск",
    "ust kamenogorsk": "Усть-Каменогорск",
    "ust-kamenogorsk": "Усть-Каменогорск",
    "устькаменогорск": "Усть-Каменогорск",
    "усть каменогорск": "Усть-Каменогорск",
    "uсть каменогорск": "Усть-Каменогорск",
    "shymkent": "Шымкент",
    "шымкент": "Шымкент",
    "astana": "Астана",
    "нурсултан": "Астана",
    "нур султан": "Астана",
    "нур-султан": "Астана",
    "almaty": "Алматы",
    "алматы": "Алматы",
}
REGION_TO_OFFICES: dict[str, list[str]] = {
    "акмолинская": ["Кокшетау", "Астана"],
    "вко": ["Усть-Каменогорск"],
    "восточно": ["Усть-Каменогорск"],
    "семипалатинская": ["Усть-Каменогорск"],
    "абайская": ["Усть-Каменогорск"],
    "юко": ["Шымкент"],
    "туркестанская": ["Шымкент"],
    "шымкент": ["Шымкент"],
    "алматинская": ["Алматы"],
    "павлодарская": ["Павлодар"],
    "карагандинская": ["Караганда"],
    "костанайская": ["Костанай"],
    "атырауская": ["Атырау"],
    "мангиста": ["Актау"],
    "западно": ["Уральск"],
    "урал": ["Уральск"],
}

KAZAKHSTAN_ALIASES = {"казахстан", "kazakhstan", "kz", "рк", "rk"}

ALMATY_COORDS = (43.238949, 76.889709)
ASTANA_COORDS = (51.169392, 71.449074)
OFFICE_CENTROIDS: dict[str, tuple[float, float]] = {
    "актау": (43.6511, 51.1978),
    "актобе": (50.2839, 57.1669),
    "алматы": ALMATY_COORDS,
    "астана": ASTANA_COORDS,
    "атырау": (47.1126, 51.9235),
    "караганда": (49.8047, 73.1094),
    "кокшетау": (53.2833, 69.3833),
    "костанай": (53.2144, 63.6246),
    "кызылорда": (44.8488, 65.4823),
    "павлодар": (52.2873, 76.9674),
    "петропавловск": (54.8753, 69.1628),
    "тараз": (42.9006, 71.3658),
    "уральск": (51.2278, 51.3865),
    "усть каменогорск": (49.9483, 82.6289),
    "шымкент": (42.3417, 69.5901),
    # aliases
    "нур султан": ASTANA_COORDS,
    "орал": (51.2278, 51.3865),
    "оскемен": (49.9483, 82.6289),
}
OFFICE_CITY_LABELS: dict[str, str] = {
    "актау": "Актау",
    "актобе": "Актобе",
    "алматы": "Алматы",
    "астана": "Астана",
    "атырау": "Атырау",
    "караганда": "Караганда",
    "кокшетау": "Кокшетау",
    "костанай": "Костанай",
    "кызылорда": "Кызылорда",
    "павлодар": "Павлодар",
    "петропавловск": "Петропавловск",
    "тараз": "Тараз",
    "уральск": "Уральск",
    "усть каменогорск": "Усть-Каменогорск",
    "шымкент": "Шымкент",
    "нур султан": "Астана",
    "орал": "Уральск",
    "оскемен": "Усть-Каменогорск",
}


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


def normalize_place(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"[.,/_\\\-()\"'«»]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def tokenize_place(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    raw_tokens = re.split(r"[\/|;]", text)
    tokens: list[str] = []
    for token in raw_tokens:
        normalized = normalize_place(token)
        if normalized and normalized not in tokens:
            tokens.append(normalized)
    return tokens


def normalize_geo_lookup_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    lowered = printable.lower().replace("ё", "е")
    lowered = lowered.replace("нур-султан", "нур султан")
    lowered = re.sub(r"[«»\"'`]", " ", lowered)
    lowered = re.sub(r"[-_/]", " ", lowered)
    lowered = re.sub(r"[.,;:(){}\[\]]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if "нур султан" in lowered:
        lowered = lowered.replace("нур султан", "астана")
    return lowered


def match_office_city(office_name: Any) -> tuple[Optional[str], Optional[tuple[float, float]]]:
    normalized = normalize_geo_lookup_text(office_name)
    if not normalized:
        return None, None

    for city_key, coords in OFFICE_CENTROIDS.items():
        if city_key in normalized:
            return OFFICE_CITY_LABELS.get(city_key, city_key.title()), coords
    return None, None


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


def is_spam(ticket_type: Any) -> bool:
    return normalize_spaces_lower(ticket_type) == SPAM_TICKET_CANONICAL


def is_spam_heuristic(ticket: Ticket, ai_analysis: Optional[AiAnalysis]) -> tuple[bool, dict[str, Any]]:
    description = clean_text(ticket.description) or ""
    summary = clean_text(ai_analysis.summary if ai_analysis else None) or ""
    combined = " ".join(part for part in (description, summary) if part).strip()
    normalized = normalize_spaces_lower(combined)

    url_hits = [marker for marker in SPAM_URL_MARKERS if marker in normalized]
    invite_hits = [keyword for keyword in SPAM_INVITE_EVENT_KEYWORDS if keyword in normalized]
    cta_hits = [keyword for keyword in SPAM_CTA_KEYWORDS if keyword in normalized]
    marketing_hits = [keyword for keyword in SPAM_MARKETING_KEYWORDS if keyword in normalized]
    ad_hits = [keyword for keyword in SPAM_AD_KEYWORDS if keyword in normalized]
    date_hits = [keyword for keyword in SPAM_DATE_MONTH_KEYWORDS if keyword in normalized]
    has_time_pattern = bool(re.search(r"\b\d{1,2}:\d{2}\b", normalized))
    support_hits = [keyword for keyword in SUPPORT_CONTEXT_KEYWORDS if keyword in normalized]

    link_detected = len(url_hits) > 0
    invite_detected = len(invite_hits) > 0
    cta_detected = len(cta_hits) > 0
    marketing_detected = len(marketing_hits) > 0 or len(ad_hits) > 0
    date_detected = len(date_hits) > 0 or has_time_pattern
    support_detected = len(support_hits) > 0

    score = 0
    if link_detected:
        score += 3
    if invite_detected:
        score += 2
    if cta_detected:
        score += 2
    if marketing_detected:
        score += 1
    if date_detected:
        score += 1

    trigger_rule = None
    is_spam_like = False
    if score >= 5:
        is_spam_like = True
        trigger_rule = "score_threshold"
    if link_detected and invite_detected:
        is_spam_like = True
        trigger_rule = "link_and_invite"

    suppression_triggered = False
    if support_detected and score < 7:
        suppression_triggered = True
        is_spam_like = False
        trigger_rule = "support_suppression"

    matched_categories = []
    if link_detected:
        matched_categories.append("link")
    if invite_detected:
        matched_categories.append("invite_event")
    if cta_detected:
        matched_categories.append("registration_cta")
    if marketing_detected:
        matched_categories.append("marketing")
    if date_detected:
        matched_categories.append("date_time")
    if support_detected:
        matched_categories.append("support_context")

    matched_keywords = {
        "link": url_hits,
        "invite_event": invite_hits,
        "registration_cta": cta_hits,
        "marketing": marketing_hits + ad_hits,
        "date_time": date_hits + (["time_pattern"] if has_time_pattern else []),
        "support_context": support_hits,
    }

    debug_info: dict[str, Any] = {
        "score": score,
        "matched_keywords": matched_keywords,
        "matched_categories": matched_categories,
        "link_detected": link_detected,
        "suppression_triggered": suppression_triggered,
        "trigger_rule": trigger_rule,
        "text_length": len(normalized),
    }
    return is_spam_like, debug_info


def is_main_specialist(position: Any) -> bool:
    return normalize_spaces_lower(position) == normalize_spaces_lower("Главный специалист")


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


def clean_office_address_for_nominatim(address: Optional[str], office_name: str) -> Optional[str]:
    city_label, _ = match_office_city(office_name)
    city = city_label or clean_text(office_name)
    if not city:
        return None

    raw_address = clean_text(address)
    if not raw_address:
        return f"{city}, Kazakhstan"

    cleaned = "".join(ch if ch.isprintable() else " " for ch in raw_address)
    cleaned = cleaned.replace("«", " ").replace("»", " ").replace('"', " ")
    cleaned = cleaned.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\bг\.\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:район|р-н)\b[^,;]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:бизнес[\s-]*центр|бц)\b[^,;]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+\s*[-–—]?\s*(?:й|ый|ий)?\s*этаж\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bэтаж\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bофис\b\s*№?\s*[\w/-]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bмикрорайон\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bмкр\b\.?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned).strip(" ,.-")

    segments = [seg.strip(" .,-") for seg in re.split(r"[;,]", cleaned) if clean_text(seg)]
    primary = ""
    if segments:
        primary = segments[0]
        if not re.search(r"\d", primary) and len(segments) > 1:
            primary = f"{primary} {segments[1]}".strip()

    primary = re.sub(r"\b(?:обл|область)\b\.?", " ", primary, flags=re.IGNORECASE)
    primary = re.sub(r"\s+", " ", primary).strip(" ,.-")
    primary = primary.replace("Қ", "К").replace("қ", "к").replace("Ә", "А").replace("ә", "а")

    if not primary:
        return f"{city}, Kazakhstan"
    return f"{primary}, {city}, Kazakhstan"


def office_city_centroid(office_name: Any) -> Optional[tuple[float, float]]:
    _, coords = match_office_city(office_name)
    return coords


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
                    query = clean_office_address_for_nominatim(office.address, office.name)
                    if query:
                        coords = geocode_address(query, address_cache)
                        if coords is not None:
                            source = "nominatim_cleaned"

                if coords is None:
                    centroid = office_city_centroid(office.name)
                    if centroid is not None:
                        coords = centroid
                        source = "centroid_map"

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


def choose_office_for_ticket(
    ticket: Ticket,
    ai_analysis: Optional[AiAnalysis],
    offices: list[BusinessUnit],
    fallback_state: dict[str, int],
) -> tuple[Optional[BusinessUnit], str, Optional[str], Optional[list[str]]]:
    office_by_normalized_name: dict[str, BusinessUnit] = {
        normalize_place(office.name): office for office in offices
    }

    city_tokens = tokenize_place(ticket.city)
    for token in city_tokens:
        canonical_name = CITY_ALIASES.get(token)
        if canonical_name:
            office = office_by_normalized_name.get(normalize_place(canonical_name))
            if office is not None:
                return office, "exact_city_or_alias", None, None
        direct_match_office = office_by_normalized_name.get(token)
        if direct_match_office is not None:
            return direct_match_office, "exact_city_or_alias", None, None

    region_norm = normalize_place(ticket.region)
    region_candidates_names: list[str] = []
    if region_norm:
        for region_key, office_names in REGION_TO_OFFICES.items():
            if region_key in region_norm:
                for office_name in office_names:
                    if office_name not in region_candidates_names:
                        region_candidates_names.append(office_name)

    region_candidate_offices = [
        office_by_normalized_name.get(normalize_place(office_name))
        for office_name in region_candidates_names
    ]
    region_candidate_offices = [office for office in region_candidate_offices if office is not None]
    region_candidates_debug = [office.name for office in region_candidate_offices] if region_candidate_offices else None

    ticket_lat = normalize_coord(ai_analysis.lat if ai_analysis else None)
    ticket_lon = normalize_coord(ai_analysis.lon if ai_analysis else None)

    if region_candidate_offices:
        if ticket_lat is not None and ticket_lon is not None:
            candidates_with_coords = [
                office for office in region_candidate_offices if office.lat is not None and office.lon is not None
            ]
            if candidates_with_coords:
                chosen = min(
                    candidates_with_coords,
                    key=lambda office: haversine_km(ticket_lat, ticket_lon, float(office.lat), float(office.lon)),
                )
                return chosen, "region_candidate_nearest", None, region_candidates_debug
            return region_candidate_offices[0], "region_candidate_no_coords", None, region_candidates_debug

        if len(region_candidate_offices) == 1:
            return region_candidate_offices[0], "region_candidate_no_coords", None, region_candidates_debug

        rr_key = f"region_no_coords_rr:{region_norm}"
        toggle = fallback_state.get(rr_key, 0) % 2
        fallback_state[rr_key] = (toggle + 1) % 2
        return region_candidate_offices[toggle], "region_candidate_no_coords", rr_key, region_candidates_debug

    abroad_or_unknown = is_abroad_or_unknown(ticket.country)
    if not abroad_or_unknown and ticket_lat is not None and ticket_lon is not None:
        nearest_global = choose_nearest_office(ticket_lat, ticket_lon, offices)
        if nearest_global is not None:
            return nearest_global, "nearest_global", None, None

    fallback_office, fallback_mode = choose_fallback_office(offices, fallback_state)
    return fallback_office, "fallback_50_50", fallback_mode, None


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


def choose_best_hub_office(
    session,
    offices: list[BusinessUnit],
    filters: dict[str, Any],
    bucket: str,
    rr_state: dict[tuple[int, str], int],
) -> tuple[Optional[BusinessUnit], list[Manager], dict[str, Any]]:
    astana_office = find_astana_office(offices)
    almaty_office = find_almaty_office(offices)

    hub_offices: list[BusinessUnit] = []
    for office in (astana_office, almaty_office):
        if office is None:
            continue
        if all(existing.id != office.id for existing in hub_offices):
            hub_offices.append(office)

    hub_candidates_debug: list[dict[str, Any]] = []
    eligible_hubs: list[tuple[BusinessUnit, list[Manager], int]] = []

    for hub in hub_offices:
        eligible = eligible_managers_for_office(session=session, office_id=hub.id, filters=filters)
        top_two = eligible[:2]
        top_loads = [int(manager.current_load) for manager in top_two]
        score = sum(top_loads) if top_two else None
        hub_candidates_debug.append(
            {
                "office": hub.name,
                "eligible_count": len(eligible),
                "top_loads": top_loads,
                "score": score,
            }
        )
        if score is not None:
            eligible_hubs.append((hub, eligible, score))

    debug_info: dict[str, Any] = {
        "hub_candidates": hub_candidates_debug,
        "hub_choice_rule": "min_sum_top2_loads",
    }

    if not eligible_hubs:
        return None, [], debug_info

    min_score = min(item[2] for item in eligible_hubs)
    best_hubs = [item for item in eligible_hubs if item[2] == min_score]

    if len(best_hubs) == 1:
        best_office, best_eligible, _ = best_hubs[0]
        return best_office, best_eligible, debug_info

    tie_state_key = (-1, f"hub_rr:{bucket}")
    toggle = rr_state.get(tie_state_key, 0) % len(best_hubs)
    rr_state[tie_state_key] = (toggle + 1) % len(best_hubs)
    best_office, best_eligible, _ = best_hubs[toggle]
    return best_office, best_eligible, debug_info


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
        "hub_fallback_count": 0,
        "dropped_spam_count": 0,
        "spam_source": None,
        "spam_score": None,
        "office_name": None,
        "manager_id": None,
    }

    ticket_type_value = clean_text(ai_analysis.ticket_type if ai_analysis else None) or DEFAULT_TICKET_TYPE
    selected_office, office_choice, office_fallback_mode, region_candidates = choose_office_for_ticket(
        ticket=ticket,
        ai_analysis=ai_analysis,
        offices=offices,
        fallback_state=fallback_state,
    )
    primary_office_choice = office_choice
    if office_choice == "fallback_50_50":
        result["fallback_50_50_count"] = 1
    if office_choice == "nearest_global":
        result["nearest_office_count"] = 1

    if selected_office is None:
        logger.error("No office available for ticket %s; skipping.", ticket.client_guid)
        return result

    spam_by_label = is_spam(ticket_type_value)
    spam_by_heuristic, spam_debug_info = is_spam_heuristic(ticket, ai_analysis)
    if spam_by_label or spam_by_heuristic:
        spam_source = "ai_label" if spam_by_label else "heuristic"
        try:
            with SessionLocal() as session:
                with session.begin():
                    office = session.get(BusinessUnit, selected_office.id)
                    if office is None:
                        raise RuntimeError(f"Office id {selected_office.id} not found during spam drop")

                    reason = {
                        "dropped_reason": "spam",
                        "spam_source": spam_source,
                        "spam_debug": spam_debug_info if spam_by_heuristic else None,
                        "ticket_type": SPAM_TICKET_LABEL,
                        "detected_ticket_type": ticket_type_value,
                        "office_choice": office_choice,
                        "office_name": office.name,
                        "region_candidates": region_candidates,
                    }
                    values = {
                        "ticket_id": ticket.id,
                        "business_unit_id": office.id,
                        "manager_id": None,
                        "status": STATUS_DROPPED_SPAM,
                        "reason": reason,
                    }
                    upsert_assignment(session, values)

                    result["office_name"] = office.name
                    result["manager_id"] = None
                    result["dropped_spam_count"] = 1
                    result["spam_source"] = spam_source
                    result["spam_score"] = spam_debug_info.get("score") if spam_debug_info else None
        except Exception as exc:  # pragma: no cover
            if debug:
                logger.exception("Spam drop failed for ticket %s", ticket.client_guid)
            else:
                logger.error("Spam drop failed for ticket %s: %s", ticket.client_guid, exc)
            return result

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
                primary_eligible_count = len(eligible)
                primary_office_info = {
                    "id": office.id,
                    "name": office.name,
                    "choice": primary_office_choice,
                    "eligible_count": primary_eligible_count,
                }
                hub_debug: Optional[dict[str, Any]] = None
                hub_fallback_used = False

                if not eligible:
                    hub_candidate_offices = [item for item in offices if item.id != office.id]
                    hub_office, hub_eligible, hub_debug = choose_best_hub_office(
                        session=session,
                        offices=hub_candidate_offices,
                        filters=filters,
                        bucket=bucket,
                        rr_state=rr_state,
                    )
                    if hub_office is not None and hub_eligible:
                        rerouted_office = session.get(BusinessUnit, hub_office.id)
                        if rerouted_office is None:
                            raise RuntimeError(f"Hub office id {hub_office.id} not found during assignment")
                        office = rerouted_office
                        eligible = hub_eligible
                        office_choice = "hub_fallback"
                        office_fallback_mode = "no_eligible_in_primary_office"
                        hub_fallback_used = True
                        result["hub_fallback_count"] = 1

                selected_manager, rr_key = pick_manager_with_rr(
                    eligible=eligible,
                    business_unit_id=office.id,
                    bucket=bucket,
                    rr_state=rr_state,
                )

                status = STATUS_ASSIGNED if selected_manager is not None else STATUS_UNASSIGNED
                unassigned_reason = None
                if selected_manager is None:
                    if primary_eligible_count == 0:
                        unassigned_reason = "no_eligible_managers_even_in_hubs"
                    else:
                        unassigned_reason = "no_eligible_managers"

                reason = {
                    "office_choice": office_choice,
                    "office_name": office.name,
                    "office_fallback_mode": office_fallback_mode,
                    "region_candidates": region_candidates,
                    "primary_office": primary_office_info,
                    "hub_fallback": {
                        "used": hub_fallback_used,
                        "hub_debug": hub_debug if hub_debug else None,
                    },
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


def run_geo_self_check(offices: list[BusinessUnit]) -> None:
    target_ids = [3, 10, 13, 19, 20, 28]
    with SessionLocal() as session:
        rows = session.execute(
            select(Ticket, AiAnalysis)
            .outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id)
            .where(Ticket.id.in_(target_ids))
            .order_by(Ticket.id.asc())
        ).all()

    if not rows:
        print("\nGeo Self Check")
        print("No target tickets found for ids: 3, 10, 13, 19, 20, 28")
        return

    local_fallback_state: dict[str, int] = {}
    print("\nGeo Self Check")
    for ticket, ai_analysis in rows:
        office, office_choice, _, region_candidates = choose_office_for_ticket(
            ticket=ticket,
            ai_analysis=ai_analysis,
            offices=offices,
            fallback_state=local_fallback_state,
        )
        print(
            f"ticket_id={ticket.id} city={clean_text(ticket.city)} region={clean_text(ticket.region)} "
            f"choice={office_choice} office={(office.name if office else None)} "
            f"region_candidates={region_candidates}"
        )


def run_spam_sanity_checks(ticket_ids: Optional[list[int]] = None) -> dict[str, int]:
    with SessionLocal() as session:
        stmt = (
            select(Ticket, AiAnalysis, Assignment)
            .outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id)
            .outerjoin(Assignment, Assignment.ticket_id == Ticket.id)
        )
        if ticket_ids:
            stmt = stmt.where(Ticket.id.in_(ticket_ids))
        rows = session.execute(stmt).all()

    checked = 0
    label_spam_cases = 0
    heuristic_spam_cases = 0
    for ticket, ai_analysis, assignment in rows:
        checked += 1

        ticket_type_value = clean_text(ai_analysis.ticket_type if ai_analysis else None)
        label_spam = is_spam(ticket_type_value)
        heuristic_spam, _ = is_spam_heuristic(ticket, ai_analysis)
        detected_spam = label_spam or heuristic_spam

        if detected_spam and assignment is not None and assignment.status == STATUS_ASSIGNED:
            raise AssertionError(
                f"Detected spam ticket assigned to manager: ticket_id={ticket.id}, client_guid={ticket.client_guid}"
            )

        if label_spam:
            label_spam_cases += 1

        if heuristic_spam:
            heuristic_spam_cases += 1
            if assignment is None or assignment.status != STATUS_DROPPED_SPAM:
                raise AssertionError(
                    "Heuristic spam ticket is not DROPPED_SPAM: "
                    f"ticket_id={ticket.id}, client_guid={ticket.client_guid}, "
                    f"status={(assignment.status if assignment else None)}"
                )
            if assignment.manager_id is not None:
                raise AssertionError(
                    f"Heuristic spam ticket has manager assigned: ticket_id={ticket.id}, manager_id={assignment.manager_id}"
                )

    return {
        "checked_tickets": checked,
        "label_spam_cases": label_spam_cases,
        "heuristic_spam_cases": heuristic_spam_cases,
    }


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
    parser.add_argument(
        "--self-check-spam",
        action="store_true",
        help="Run lightweight post-routing assertions for spam drop invariants.",
    )
    parser.add_argument(
        "--self-check-geo",
        action="store_true",
        help="Dry-run office choice printout for known problematic ticket ids.",
    )
    parser.add_argument(
        "--print-spam",
        action="store_true",
        help="Print dropped spam tickets with source and heuristic score.",
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
        print("dropped_spam: 0")
        print("fallback_50_50_count: 0")
        print("nearest_office_count: 0")
        print("hub_fallback_count: 0")
        print("per-office assigned counts:")
        print("per-manager final loads:")
        return

    if args.self_check_geo:
        run_geo_self_check(offices)

    ticket_rows = select_tickets(force=args.force, tickets_arg=args.tickets, limit=args.limit)

    rr_state: dict[tuple[int, str], int] = {}
    fallback_state: dict[str, int] = {}
    per_office_assigned = defaultdict(int)
    processed_ticket_ids: list[int] = []
    dropped_spam_items: list[dict[str, Any]] = []

    total_selected = len(ticket_rows)
    processed = 0
    assigned = 0
    unassigned = 0
    dropped_spam_count = 0
    fallback_50_50_count = 0
    nearest_office_count = 0
    hub_fallback_count = 0

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
            processed_ticket_ids.append(ticket.id)
            assigned += int(item["assigned"])
            unassigned += int(item["unassigned"])
            dropped_spam_count += int(item["dropped_spam_count"])
            fallback_50_50_count += int(item["fallback_50_50_count"])
            nearest_office_count += int(item["nearest_office_count"])
            hub_fallback_count += int(item["hub_fallback_count"])
            if item["dropped_spam_count"]:
                dropped_spam_items.append(
                    {
                        "ticket_id": ticket.id,
                        "client_guid": ticket.client_guid,
                        "spam_source": item["spam_source"],
                        "score": item["spam_score"],
                    }
                )
            if item["assigned"] and item["office_name"]:
                per_office_assigned[item["office_name"]] += 1
        except Exception as exc:  # pragma: no cover
            if args.debug:
                logger.exception("Unexpected failure for ticket %s", ticket.client_guid)
            else:
                logger.error("Unexpected failure for ticket %s: %s", ticket.client_guid, exc)

    if args.self_check_spam:
        spam_check_stats = run_spam_sanity_checks(ticket_ids=processed_ticket_ids)
        logger.info("Spam sanity checks passed: %s", spam_check_stats)

    if args.print_spam and dropped_spam_items:
        print("\nDropped Spam Tickets")
        for spam_item in dropped_spam_items:
            print(
                f"ticket_id={spam_item['ticket_id']} "
                f"client_guid={spam_item['client_guid']} "
                f"source={spam_item['spam_source']} "
                f"score={spam_item['score']}"
            )

    loads = manager_loads_report()

    print("\nRouting Report")
    print(f"total selected tickets: {total_selected}")
    print(f"processed: {processed}")
    print(f"assigned: {assigned}")
    print(f"unassigned: {unassigned}")
    print(f"dropped_spam: {dropped_spam_count}")
    print(f"fallback_50_50_count: {fallback_50_50_count}")
    print(f"nearest_office_count: {nearest_office_count}")
    print(f"hub_fallback_count: {hub_fallback_count}")
    print("per-office assigned counts:")
    for office_name, count in sorted(per_office_assigned.items(), key=lambda x: x[0]):
        print(f"  {office_name}: {count}")
    print("per-manager final loads:")
    for full_name, office_name, current_load in loads:
        print(f"  {full_name} ({office_name}): {current_load}")


if __name__ == "__main__":
    main()
