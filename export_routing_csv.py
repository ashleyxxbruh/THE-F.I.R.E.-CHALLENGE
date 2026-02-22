from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.db import engine

OUT_PATH = Path("routing_results.csv")

QUERY = """
SELECT
  t.id AS ticket_id,
  t.client_guid,
  t.segment,
  a.ticket_type,
  a.language,
  a.priority,
  bu.name AS assigned_office,
  m.full_name AS assigned_manager,
  m.position AS manager_position,
  asg.status,
  asg.reason::text AS reason_json
FROM tickets t
LEFT JOIN ai_analysis a ON a.ticket_id = t.id
LEFT JOIN assignments asg ON asg.ticket_id = t.id
LEFT JOIN business_units bu ON bu.id = asg.business_unit_id
LEFT JOIN managers m ON m.id = asg.manager_id
ORDER BY t.id
"""

def main() -> None:
    df = pd.read_sql(text(QUERY), engine)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()