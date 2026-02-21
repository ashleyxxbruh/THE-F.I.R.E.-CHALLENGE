import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def resolve_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    if user and password and database:
        quoted_user = quote_plus(user)
        quoted_password = quote_plus(password)
        return f"postgresql+psycopg2://{quoted_user}:{quoted_password}@{host}:{port}/{database}"

    raise RuntimeError(
        "Database config is missing. Set DATABASE_URL or POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB "
        "(optional: POSTGRES_HOST, POSTGRES_PORT). Example PowerShell: "
        "$env:DATABASE_URL='postgresql+psycopg2://postgres:postgres@localhost:5432/fire_db'"
    )


DATABASE_URL = resolve_database_url()

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
