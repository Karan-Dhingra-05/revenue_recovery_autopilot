import redis as redis_lib
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Returns the liveness status of the API and its dependencies.

    Always returns HTTP 200. Callers should inspect the 'status' field:
      - "ok"       — all dependencies healthy
      - "degraded" — at least one dependency is unhealthy

    This endpoint is intentionally lightweight (no ML, no business logic).
    """
    # ── Database ─────────────────────────────────────────────────────────────
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001
        db_status = f"error: {exc}"

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        redis_status = "ok"
    except Exception as exc:  # noqa: BLE001
        redis_status = f"error: {exc}"

    overall = "ok" if (db_status == "ok" and redis_status == "ok") else "degraded"

    return {
        "status": overall,
        "db": db_status,
        "redis": redis_status,
    }
