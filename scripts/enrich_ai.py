"""
Run (PowerShell):
  $env:DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/db"
  $env:GROQ_API_KEY="your_groq_api_key"
  $env:GROQ_MODEL="llama-3.3-70b-versatile"
  python scripts/enrich_ai.py
  python scripts/enrich_ai.py --force
  python scripts/enrich_ai.py --limit 10

Run (bash):
  export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/db
  export GROQ_API_KEY=your_groq_api_key
  export GROQ_MODEL=llama-3.3-70b-versatile
  python scripts/enrich_ai.py
  python scripts/enrich_ai.py --force
  python scripts/enrich_ai.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

try:
    import requests
    from requests import RequestException
except ImportError:  # pragma: no cover
    requests = None

    class RequestException(Exception):
        pass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db import SessionLocal, engine
from app.models import AiAnalysis, Base, Ticket

logger = logging.getLogger("enrich_ai")

ALLOWED_TICKET_TYPES = {
    "Жалоба",
    "Смена данных",
    "Консультация",
    "Претензия",
    "Неработоспособность приложения",
    "Мошеннические действия",
    "Спам",
}
ALLOWED_SENTIMENTS = {"Позитивный", "Нейтральный", "Негативный"}
ALLOWED_LANGUAGES = {"KZ", "ENG", "RU"}

DEFAULT_TICKET_TYPE = "Консультация"
DEFAULT_SENTIMENT = "Нейтральный"
DEFAULT_PRIORITY = 5
DEFAULT_LANGUAGE = "RU"
DEFAULT_RECOMMENDATION = "Проверьте детали обращения и уточните информацию у клиента."
OVERRIDE_RULE_A_BLOCK_KEYWORDS = ("заблок", "блокиров", "счет", "счёт", "карта", "доступ")
OVERRIDE_RULE_A_COMPENSATION_KEYWORDS = ("верните", "возмест", "компенс", "ущерб", "требую", "суд", "претенз")
OVERRIDE_RULE_B_KEYWORDS = (
    "смена номера",
    "изменить номер",
    "номер телефона",
    "смс не приходит",
    "код не приходит",
    "верификац",
    "паспорт",
    "иин",
    "документ",
    "обновить данные",
)
OVERRIDE_RULE_C_KEYWORDS = ("мошенн", "обман", "развод", "жертва", "украли", "списали", "фишинг", "злоумышлен")
OVERRIDE_RULE_D_KEYWORDS = (
    "ордер",
    "order",
    "не исполнил",
    "ошибка исполнения",
    "не работает",
    "bug",
    "ошибка",
    "краш",
    "зависло",
)
OVERRIDE_RULE_E_TRANSFER_KEYWORDS = ("перевод не приш", "не поступил", "завис перевод")
OVERRIDE_RULE_E_CURRENCY_KEYWORDS = ("тг", "₸", "руб", "₽", "usd", "$", "евро", "€")

ALMATY_COORDS = (43.238949, 76.889709)
ASTANA_COORDS = (51.169392, 71.449074)

CITY_CENTROIDS: dict[str, tuple[float, float]] = {
    "алматы": ALMATY_COORDS,
    "алматин": ALMATY_COORDS,
    "астана": ASTANA_COORDS,
    "нур-султан": ASTANA_COORDS,
    "нур султан": ASTANA_COORDS,
    "шымкент": (42.3417, 69.5901),
    "караганд": (49.8047, 73.1094),
    "павлодар": (52.2873, 76.9674),
    "усть-каменогорск": (49.9483, 82.6289),
    "оскемен": (49.9483, 82.6289),
    "кокшетау": (53.2833, 69.3833),
    "костанай": (53.2144, 63.6246),
    "атырау": (47.1126, 51.9235),
    "актобе": (50.2839, 57.1669),
    "актау": (43.6511, 51.1978),
    "тараз": (42.9006, 71.3658),
    "уральск": (51.2278, 51.3865),
    "орал": (51.2278, 51.3865),
    "петропавловск": (54.8753, 69.1628),
    "кызылорда": (44.8488, 65.4823),
    "туркестан": (43.2973, 68.2518),
}


class LLMAdapterError(Exception):
    pass


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def default_payload(description: Optional[str]) -> dict[str, Any]:
    fallback_summary = None
    text = clean_text(description)
    if text:
        fallback_summary = text[:200]
    return {
        "ticket_type": DEFAULT_TICKET_TYPE,
        "sentiment": DEFAULT_SENTIMENT,
        "priority": DEFAULT_PRIORITY,
        "language": DEFAULT_LANGUAGE,
        "summary": fallback_summary,
        "recommendation": DEFAULT_RECOMMENDATION,
    }


def normalize_priority(value: Any) -> int:
    try:
        priority = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return DEFAULT_PRIORITY
    if 1 <= priority <= 10:
        return priority
    return DEFAULT_PRIORITY


def normalize_ticket_type(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return DEFAULT_TICKET_TYPE
    canonical = {item.lower(): item for item in ALLOWED_TICKET_TYPES}
    return canonical.get(text.lower(), DEFAULT_TICKET_TYPE)


def normalize_sentiment(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return DEFAULT_SENTIMENT
    canonical = {item.lower(): item for item in ALLOWED_SENTIMENTS}
    return canonical.get(text.lower(), DEFAULT_SENTIMENT)


def normalize_language(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return DEFAULT_LANGUAGE
    language = text.upper()
    return language if language in ALLOWED_LANGUAGES else DEFAULT_LANGUAGE


def validate_ai_payload(payload: Any, description: Optional[str]) -> tuple[dict[str, Any], Optional[dict[str, Any]], bool]:
    defaults = default_payload(description)
    if not isinstance(payload, dict):
        return defaults, None, True

    result = defaults.copy()
    result["ticket_type"] = normalize_ticket_type(payload.get("ticket_type"))
    result["sentiment"] = normalize_sentiment(payload.get("sentiment"))
    result["priority"] = normalize_priority(payload.get("priority"))
    result["language"] = normalize_language(payload.get("language"))

    summary = clean_text(payload.get("summary"))
    recommendation = clean_text(payload.get("recommendation"))
    result["summary"] = summary if summary else defaults["summary"]
    result["recommendation"] = recommendation if recommendation else defaults["recommendation"]

    return result, payload, False


def _normalize_text_for_override(*parts: Optional[str]) -> str:
    values = [clean_text(part) for part in parts]
    values = [value for value in values if value]
    if not values:
        return ""
    normalized = " ".join(values).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", normalized).strip()


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def apply_type_overrides(description: str, ai: dict[str, Any]) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    updated = dict(ai)
    text = _normalize_text_for_override(description, updated.get("summary"))
    original_type = normalize_ticket_type(updated.get("ticket_type"))
    original_priority = normalize_priority(updated.get("priority"))

    updated["ticket_type"] = original_type
    updated["priority"] = original_priority

    if not text:
        return updated, None

    block_hits = _keyword_hits(text, OVERRIDE_RULE_A_BLOCK_KEYWORDS)
    compensation_hits = _keyword_hits(text, OVERRIDE_RULE_A_COMPENSATION_KEYWORDS)
    rule_b_hits = _keyword_hits(text, OVERRIDE_RULE_B_KEYWORDS)
    fraud_hits = _keyword_hits(text, OVERRIDE_RULE_C_KEYWORDS)
    rule_d_hits = _keyword_hits(text, OVERRIDE_RULE_D_KEYWORDS)
    transfer_hits = _keyword_hits(text, OVERRIDE_RULE_E_TRANSFER_KEYWORDS)
    currency_hits = _keyword_hits(text, OVERRIDE_RULE_E_CURRENCY_KEYWORDS)
    has_large_number = bool(re.search(r"\b\d{3,}\b", text))

    new_type = original_type
    new_priority = original_priority
    rule_id: Optional[str] = None
    matched_keywords: list[str] = []

    if fraud_hits:
        rule_id = "RULE_C_FRAUD_VICTIM"
        new_type = "Мошеннические действия"
        new_priority = max(original_priority, 9)
        matched_keywords = fraud_hits
    elif rule_b_hits:
        rule_id = "RULE_B_CHANGE_DATA"
        new_type = "Смена данных"
        new_priority = min(max(original_priority, 5), 10)
        matched_keywords = rule_b_hits
    elif transfer_hits and (currency_hits or has_large_number):
        rule_id = "RULE_E_TRANSFER_CLAIM"
        new_type = "Претензия"
        new_priority = max(original_priority, 7)
        matched_keywords = transfer_hits + currency_hits + (["amount_3plus"] if has_large_number else [])
    elif rule_d_hits:
        rule_id = "RULE_D_APP_NOT_WORKING"
        new_type = "Неработоспособность приложения"
        new_priority = max(original_priority, 7)
        matched_keywords = rule_d_hits
    elif block_hits and not compensation_hits:
        rule_id = "RULE_A_BLOCKED_ACCOUNT_COMPLAINT"
        new_type = "Жалоба"
        new_priority = original_priority
        matched_keywords = block_hits

    new_type = normalize_ticket_type(new_type)
    if new_type not in ALLOWED_TICKET_TYPES:
        new_type = DEFAULT_TICKET_TYPE
    new_priority = normalize_priority(new_priority)

    if rule_id is None:
        return updated, None
    if new_type == original_type and new_priority == original_priority:
        return updated, None

    updated["ticket_type"] = new_type
    updated["priority"] = new_priority

    override_reason = {
        "applied": True,
        "original_type": original_type,
        "new_type": new_type,
        "original_priority": original_priority,
        "new_priority": new_priority,
        "rule_id": rule_id,
        "matched_keywords": matched_keywords,
    }
    return updated, override_reason


def _build_llm_prompt(description: str) -> str:
    prompt_description = description.strip() if description else ""
    ticket_types = ", ".join(sorted(ALLOWED_TICKET_TYPES))
    sentiments = ", ".join(sorted(ALLOWED_SENTIMENTS))
    return (
        "Classify this client support ticket and return ONLY a JSON object with keys: "
        "ticket_type, sentiment, priority, language, summary, recommendation.\n\n"
        f"Ticket description:\n{prompt_description}\n\n"
        f"Allowed ticket_type values: {ticket_types}\n"
        f"Allowed sentiment values: {sentiments}\n"
        "priority: integer from 1 to 10\n"
        "language: one of KZ, ENG, RU\n"
        "summary: 1-2 sentences\n"
        "recommendation: short next steps for manager"
    )


def _parse_json_response_text(text: str) -> dict[str, Any]:
    if text is None:
        raise LLMAdapterError("LLM response content is empty")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMAdapterError(f"LLM JSON decode error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMAdapterError("LLM response is not a JSON object")
    return parsed


def _call_groq_llm(description: str, api_key: str) -> dict[str, Any]:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    prompt = _build_llm_prompt(description)
    base_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only a valid JSON object.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
    }
    payload_candidates = [
        {
            **base_payload,
            "response_format": {"type": "json_object"},
        },
        base_payload,
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    errors: list[str] = []
    for payload in payload_candidates:
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if not response.ok:
                body_preview = (response.text or "").strip().replace("\n", " ")
                errors.append(f"status={response.status_code} body={body_preview[:300]}")
                continue
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return _parse_json_response_text(content)
        except RequestException as exc:
            errors.append(f"request_error={exc}")
            continue
        except ValueError as exc:
            errors.append(f"non_json_response={exc}")
            continue
        except (KeyError, IndexError, TypeError, LLMAdapterError) as exc:
            errors.append(f"parse_error={exc}")
            continue

    joined = "; ".join(errors[-2:]) if errors else "unknown error"
    raise LLMAdapterError(f"Groq request failed: {joined}")


def call_llm(description: str) -> dict[str, Any]:
    if requests is None:
        raise LLMAdapterError("requests library is not installed")

    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        return _call_groq_llm(description, groq_api_key)

    raise LLMAdapterError("GROQ_API_KEY is not set")


def _join_geo_parts(*parts: Optional[str]) -> Optional[str]:
    cleaned_parts = [clean_text(part) for part in parts]
    values = [part for part in cleaned_parts if part]
    if not values:
        return None
    return ", ".join(values)


def _append_unique(queries: list[str], value: Optional[str]) -> None:
    query = clean_text(value)
    if not query:
        return
    if query not in queries:
        queries.append(query)


def normalize_region_for_geocoding(region: Optional[str]) -> Optional[str]:
    text = clean_text(region)
    if not text:
        return None

    normalized = text.replace("\\", "/")
    if "/" in normalized:
        parts = [part.strip() for part in normalized.split("/") if part.strip()]
        if parts:
            normalized = parts[-1]

    normalized = re.sub(r"\b(обл\.?|область|облысы|район|р-н)\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[(){}\[\];:]", " ", normalized)
    normalized = re.sub(r"[,.]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" -")
    return normalized or None


def city_variants_for_geocoding(city: Optional[str]) -> list[str]:
    city_text = clean_text(city)
    if not city_text:
        return []

    variants: list[str] = []
    _append_unique(variants, city_text)

    if "/" in city_text:
        slash_parts = [clean_text(part) for part in city_text.split("/")]
        slash_parts = [part for part in slash_parts if part]
        if slash_parts:
            _append_unique(variants, ", ".join(slash_parts))
            for part in slash_parts:
                _append_unique(variants, part)

    if "(" in city_text and ")" in city_text:
        main_city = re.sub(r"\(.*?\)", "", city_text).strip(" ,")
        _append_unique(variants, main_city)
        inside_parts = re.findall(r"\(([^)]+)\)", city_text)
        for inside in inside_parts:
            _append_unique(variants, inside)

    return variants


def build_geocode_queries(ticket: Ticket) -> list[str]:
    country = clean_text(ticket.country)
    region_raw = clean_text(ticket.region)
    region_clean = normalize_region_for_geocoding(region_raw)
    street = clean_text(ticket.street)
    house = clean_text(ticket.house_raw)
    city_variants = city_variants_for_geocoding(ticket.city)

    queries: list[str] = []
    if not city_variants:
        city_variants = [None]

    def add_for_city(city_value: Optional[str]) -> None:
        # a) full
        _append_unique(queries, _join_geo_parts(country, region_raw, city_value, street, house))
        if region_clean and region_clean != region_raw:
            _append_unique(queries, _join_geo_parts(country, region_clean, city_value, street, house))

        # b) street-level
        _append_unique(queries, _join_geo_parts(country, region_raw, city_value, street))
        if region_clean and region_clean != region_raw:
            _append_unique(queries, _join_geo_parts(country, region_clean, city_value, street))

        # c) city-level
        _append_unique(queries, _join_geo_parts(country, region_raw, city_value))
        if region_clean and region_clean != region_raw:
            _append_unique(queries, _join_geo_parts(country, region_clean, city_value))

        # d) city + Kazakhstan
        _append_unique(queries, _join_geo_parts(city_value, "Казахстан"))

    add_for_city(city_variants[0])
    for city_variant in city_variants[1:]:
        add_for_city(city_variant)

    return queries


def geocode_nominatim(
    queries: list[str],
    cache: dict[str, Optional[tuple[float, float]]],
) -> tuple[Optional[tuple[float, float]], Optional[str]]:
    if not queries:
        return None, None

    for raw_query in queries:
        query = clean_text(raw_query)
        if not query:
            continue

        if query in cache:
            cached_value = cache[query]
            if cached_value is not None:
                return cached_value, query
            continue

        if requests is None:
            cache[query] = None
            continue

        did_request = False
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 1},
                headers={"User-Agent": "fire-hackathon-ai-enricher/1.0"},
                timeout=15,
            )
            did_request = True
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list) and payload:
                first = payload[0]
                lat = float(first["lat"])
                lon = float(first["lon"])
                cache[query] = (lat, lon)
                return cache[query], query
            cache[query] = None
        except (RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError):
            cache[query] = None
        finally:
            if did_request:
                time.sleep(1.0)

    return None, None


def fallback_city_centroid(city: Optional[str]) -> Optional[tuple[float, float]]:
    text = clean_text(city)
    if not text:
        return None

    city_lc = text.lower()
    for city_pattern, coords in CITY_CENTROIDS.items():
        if city_pattern in city_lc:
            return coords
    return None


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


def select_tickets(force: bool, tickets_arg: Optional[str], limit: Optional[int]) -> list[Ticket]:
    ticket_guid_filter, ticket_count_filter = parse_tickets_arg(tickets_arg)
    limit_value = effective_limit(limit, ticket_count_filter)

    stmt = select(Ticket).order_by(Ticket.id)

    if not force:
        stmt = stmt.outerjoin(AiAnalysis, AiAnalysis.ticket_id == Ticket.id).where(AiAnalysis.id.is_(None))

    if ticket_guid_filter:
        stmt = stmt.where(Ticket.client_guid == ticket_guid_filter)

    if limit_value is not None:
        stmt = stmt.limit(limit_value)

    with SessionLocal() as session:
        return session.execute(stmt).scalars().all()


def upsert_ai_analysis(session, values: dict[str, Any]) -> None:
    stmt = insert(AiAnalysis).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[AiAnalysis.ticket_id],
        set_={
            "ticket_type": stmt.excluded.ticket_type,
            "sentiment": stmt.excluded.sentiment,
            "priority": stmt.excluded.priority,
            "language": stmt.excluded.language,
            "summary": stmt.excluded.summary,
            "recommendation": stmt.excluded.recommendation,
            "lat": stmt.excluded.lat,
            "lon": stmt.excluded.lon,
            "raw_json": stmt.excluded.raw_json,
        },
    )
    session.execute(stmt)


def run_self_check_types(tickets: list[Ticket]) -> None:
    override_counts: dict[str, int] = {}
    processed = 0
    ai_call_failures = 0

    for ticket in tickets:
        processed += 1
        raw_payload: Any = None
        try:
            raw_payload = call_llm(ticket.description or "")
        except LLMAdapterError as exc:
            ai_call_failures += 1
            logger.warning("AI call failed for ticket %s during type self-check: %s", ticket.client_guid, exc)

        normalized, _, _ = validate_ai_payload(raw_payload, ticket.description)
        _, override_reason = apply_type_overrides(ticket.description or "", normalized)
        if override_reason is not None:
            rule_id = str(override_reason["rule_id"])
            override_counts[rule_id] = override_counts.get(rule_id, 0) + 1

    print("\nType Override Self-Check")
    print(f"processed: {processed}")
    print(f"ai_call_failures: {ai_call_failures}")
    print("overrides by rule:")
    if not override_counts:
        print("  none")
        return
    for rule_id, count in sorted(override_counts.items(), key=lambda x: x[0]):
        print(f"  {rule_id}: {count}")


def process_ticket(
    ticket: Ticket,
    geocode_cache: dict[str, Optional[tuple[float, float]]],
    debug: bool,
) -> tuple[bool, int, int, int, Optional[str]]:
    failed_ai_parses = 0
    failed_geocodes = 0
    used_fallback_city_centroid = 0

    raw_payload: Any = None
    ai_call_failed = False
    try:
        raw_payload = call_llm(ticket.description or "")
    except LLMAdapterError as exc:
        ai_call_failed = True
        failed_ai_parses += 1
        logger.warning("AI call failed for ticket %s: %s", ticket.client_guid, exc)

    normalized, raw_json, invalid_json = validate_ai_payload(raw_payload, ticket.description)
    if invalid_json and not ai_call_failed:
        failed_ai_parses += 1
        logger.warning("Invalid AI JSON for ticket %s. Applied fallback defaults.", ticket.client_guid)

    normalized, override_reason = apply_type_overrides(ticket.description or "", normalized)
    if override_reason is not None:
        logger.info(
            "Type override for ticket %s: %s -> %s (rule=%s)",
            ticket.client_guid,
            override_reason["original_type"],
            override_reason["new_type"],
            override_reason["rule_id"],
        )

    raw_json_to_store: Any = raw_json if raw_json is not None else (raw_payload if isinstance(raw_payload, dict) else None)
    if override_reason is not None:
        if isinstance(raw_json, dict):
            raw_json_to_store = dict(raw_json)
            raw_json_to_store["override"] = override_reason
        else:
            raw_json_to_store = {
                "llm": raw_json if raw_json is not None else (raw_payload if isinstance(raw_payload, dict) else None),
                "override": override_reason,
            }

    geocode_queries = build_geocode_queries(ticket)
    coords, geocode_query = geocode_nominatim(geocode_queries, geocode_cache)
    if coords is not None:
        if debug:
            logger.debug(
                "Geocode ok for ticket %s: query='%s' source=nominatim",
                ticket.client_guid,
                geocode_query,
            )
    else:
        city_coords = fallback_city_centroid(ticket.city)
        if city_coords is not None:
            coords = city_coords
            used_fallback_city_centroid = 1
            if debug:
                logger.debug(
                    "Geocode ok for ticket %s: city='%s' source=centroid",
                    ticket.client_guid,
                    clean_text(ticket.city),
                )
        else:
            failed_geocodes = 1
            if debug:
                logger.debug(
                    "Geocode failed for ticket %s. Tried queries: %s",
                    ticket.client_guid,
                    geocode_queries,
                )

    lat = coords[0] if coords else None
    lon = coords[1] if coords else None

    values = {
        "ticket_id": ticket.id,
        "ticket_type": normalized["ticket_type"],
        "sentiment": normalized["sentiment"],
        "priority": normalized["priority"],
        "language": normalized["language"],
        "summary": normalized["summary"],
        "recommendation": normalized["recommendation"],
        "lat": lat,
        "lon": lon,
        "raw_json": raw_json_to_store,
    }

    try:
        with SessionLocal() as session:
            with session.begin():
                upsert_ai_analysis(session, values)
    except Exception as exc:  # pragma: no cover
        if debug:
            logger.exception("Database upsert failed for ticket %s", ticket.client_guid)
        else:
            logger.error("Database upsert failed for ticket %s: %s", ticket.client_guid, exc)
        return False, failed_ai_parses, failed_geocodes, used_fallback_city_centroid, (
            override_reason["rule_id"] if override_reason else None
        )

    return True, failed_ai_parses, failed_geocodes, used_fallback_city_centroid, (
        override_reason["rule_id"] if override_reason else None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI enrichment + geocoding for FIRE tickets.")
    parser.add_argument("--tickets", help="Either integer count (e.g. 50) or a specific client GUID.")
    parser.add_argument("--force", action="store_true", help="Re-enrich tickets even when ai_analysis exists.")
    parser.add_argument("--limit", type=int, help="Maximum number of tickets to process.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and exception traces.")
    parser.add_argument(
        "--self-check-types",
        action="store_true",
        help="Dry-run type override checks and print override counts by rule (no DB writes).",
    )
    args = parser.parse_args()

    configure_logging(debug=args.debug)
    Base.metadata.create_all(bind=engine)

    tickets = select_tickets(force=args.force, tickets_arg=args.tickets, limit=args.limit)
    total_selected = len(tickets)

    if args.self_check_types:
        run_self_check_types(tickets)
        return

    processed = 0
    successes = 0
    failed_ai_parses = 0
    failed_geocodes = 0
    used_fallback_city_centroids = 0

    geocode_cache: dict[str, Optional[tuple[float, float]]] = {}

    for ticket in tickets:
        processed += 1
        try:
            ok, ai_fail_cnt, geocode_fail_cnt, fallback_cnt, _override_rule_id = process_ticket(
                ticket=ticket,
                geocode_cache=geocode_cache,
                debug=args.debug,
            )
            failed_ai_parses += ai_fail_cnt
            failed_geocodes += geocode_fail_cnt
            used_fallback_city_centroids += fallback_cnt
            if ok:
                successes += 1
        except Exception as exc:  # pragma: no cover
            if args.debug:
                logger.exception("Unexpected failure for ticket %s", ticket.client_guid)
            else:
                logger.error("Unexpected failure for ticket %s: %s", ticket.client_guid, exc)

    print("\nAI Enrichment Report")
    print(f"total selected tickets: {total_selected}")
    print(f"processed: {processed}")
    print(f"successes: {successes}")
    print(f"failed_ai_parses: {failed_ai_parses}")
    print(f"failed_geocodes: {failed_geocodes}")
    print(f"used_fallback_city_centroids: {used_fallback_city_centroids}")


if __name__ == "__main__":
    main()
