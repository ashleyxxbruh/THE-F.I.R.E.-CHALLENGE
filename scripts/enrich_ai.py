"""
Run (PowerShell):
  $env:DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/db"
  $env:GOOGLE_API_KEY="your_google_ai_studio_key"
  python scripts/enrich_ai.py
  python scripts/enrich_ai.py --force
  python scripts/enrich_ai.py --limit 10

Run (bash):
  export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/db
  export GOOGLE_API_KEY=your_google_ai_studio_key
  python scripts/enrich_ai.py
  python scripts/enrich_ai.py --force
  python scripts/enrich_ai.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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

ALMATY_COORDS = (43.238949, 76.889709)
ASTANA_COORDS = (51.169392, 71.449074)


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


def _call_openai_llm(description: str, api_key: str) -> dict[str, Any]:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = _build_llm_prompt(description)

    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
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

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_json_response_text(content)
    except RequestException as exc:
        raise LLMAdapterError(f"OpenAI network error: {exc}") from exc
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMAdapterError(f"OpenAI response parse error: {exc}") from exc


def _call_gemini_llm(description: str, api_key: str) -> dict[str, Any]:
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = _build_llm_prompt(description)

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_json_response_text(content)
    except RequestException as exc:
        raise LLMAdapterError(f"Gemini network error: {exc}") from exc
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMAdapterError(f"Gemini response parse error: {exc}") from exc


def call_llm(description: str) -> dict[str, Any]:
    if requests is None:
        raise LLMAdapterError("requests library is not installed")

    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if google_api_key:
        return _call_gemini_llm(description, google_api_key)

    if openai_api_key:
        return _call_openai_llm(description, openai_api_key)

    raise LLMAdapterError("No LLM API key set. Use GOOGLE_API_KEY (or GEMINI_API_KEY) / OPENAI_API_KEY.")


def build_address(ticket: Ticket) -> str:
    parts = [
        clean_text(ticket.country),
        clean_text(ticket.region),
        clean_text(ticket.city),
        clean_text(ticket.street),
        clean_text(ticket.house_raw),
    ]
    return ", ".join(part for part in parts if part)


def geocode_nominatim(address: str, cache: dict[str, Optional[tuple[float, float]]]) -> Optional[tuple[float, float]]:
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
            headers={"User-Agent": "fire-hackathon-ai-enricher/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            first = payload[0]
            lat = float(first["lat"])
            lon = float(first["lon"])
            cache[address] = (lat, lon)
            return cache[address]
    except (RequestException, ValueError, KeyError, TypeError):
        pass

    cache[address] = None
    return None


def fallback_city_centroid(city: Optional[str]) -> Optional[tuple[float, float]]:
    text = clean_text(city)
    if not text:
        return None

    city_lc = text.lower()
    if "алматы" in city_lc:
        return ALMATY_COORDS
    if "астана" in city_lc or "нур-султан" in city_lc:
        return ASTANA_COORDS
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


def process_ticket(
    ticket: Ticket,
    geocode_cache: dict[str, Optional[tuple[float, float]]],
    debug: bool,
) -> tuple[bool, int, int, int]:
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

    address = build_address(ticket)
    coords = geocode_nominatim(address, geocode_cache) if address else None
    if coords is None:
        city_coords = fallback_city_centroid(ticket.city)
        if city_coords is not None:
            coords = city_coords
            used_fallback_city_centroid = 1
        else:
            failed_geocodes = 1

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
        "raw_json": raw_json if raw_json is not None else (raw_payload if isinstance(raw_payload, dict) else None),
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
        return False, failed_ai_parses, failed_geocodes, used_fallback_city_centroid

    return True, failed_ai_parses, failed_geocodes, used_fallback_city_centroid


def main() -> None:
    parser = argparse.ArgumentParser(description="AI enrichment + geocoding for FIRE tickets.")
    parser.add_argument("--tickets", help="Either integer count (e.g. 50) or a specific client GUID.")
    parser.add_argument("--force", action="store_true", help="Re-enrich tickets even when ai_analysis exists.")
    parser.add_argument("--limit", type=int, help="Maximum number of tickets to process.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and exception traces.")
    args = parser.parse_args()

    configure_logging(debug=args.debug)
    Base.metadata.create_all(bind=engine)

    tickets = select_tickets(force=args.force, tickets_arg=args.tickets, limit=args.limit)
    total_selected = len(tickets)

    processed = 0
    successes = 0
    failed_ai_parses = 0
    failed_geocodes = 0
    used_fallback_city_centroids = 0

    geocode_cache: dict[str, Optional[tuple[float, float]]] = {}

    for ticket in tickets:
        processed += 1
        try:
            ok, ai_fail_cnt, geocode_fail_cnt, fallback_cnt = process_ticket(
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
