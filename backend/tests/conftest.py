"""
Shared pytest fixtures for the backend test suite.

Fixtures defined here are available to all test modules automatically.

Two fixture categories:
  - Mocked (no infrastructure): used by test_health.py — no changes needed.
  - Live DB (requires docker compose): db_engine and db_session used by
    test_db_schema.py. These are function-scoped and roll back all writes
    after each test so the database stays clean.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings


@pytest.fixture(scope="session")
def db_engine():
    """
    Session-scoped SQLAlchemy engine.

    Created once per pytest session. Requires PostgreSQL to be running
    (docker compose up). Only instantiated if a test actually requests
    this fixture — health tests that mock everything are unaffected.
    """
    engine = create_engine(settings.database_url, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """
    Function-scoped transactional session.

    Each test gets its own connection with an open outer transaction.
    The session uses SAVEPOINTs (join_transaction_mode='create_savepoint')
    so that tests which deliberately trigger IntegrityErrors can call
    session.rollback() to unwind only the failing operation — the outer
    transaction stays alive and is rolled back by this fixture at the end.

    All writes are rolled back after each test: no truncation needed.
    """
    connection = db_engine.connect()
    outer_tx = connection.begin()

    session = Session(bind=connection, join_transaction_mode="create_savepoint")

    yield session

    session.close()
    if outer_tx.is_active:
        outer_tx.rollback()
    connection.close()
