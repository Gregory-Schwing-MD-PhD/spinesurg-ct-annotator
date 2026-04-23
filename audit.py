"""
Audit logging — SQLite with WAL mode for concurrent-safe writes.

Two tables:
  • annotation_events  — one row per mask save
  • session_events     — login, logout, study_opened, unauthorized_attempt, etc.

The DB file is pushed to the target HF dataset alongside every mask save,
so the provenance record lives with the data it describes.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotation_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    username         TEXT    NOT NULL,
    case_id          TEXT    NOT NULL,
    source_filename  TEXT    NOT NULL,
    mask_filename    TEXT    NOT NULL,
    mask_sha256      TEXT    NOT NULL,
    mask_bytes       INTEGER NOT NULL,
    session_id       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS session_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    username         TEXT    NOT NULL,
    event_type       TEXT    NOT NULL,
    case_id          TEXT,
    session_id       TEXT    NOT NULL,
    details          TEXT
);

CREATE INDEX IF NOT EXISTS idx_ann_case ON annotation_events(case_id);
CREATE INDEX IF NOT EXISTS idx_ann_user ON annotation_events(username);
CREATE INDEX IF NOT EXISTS idx_ann_ts   ON annotation_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_ses_user ON session_events(username);
CREATE INDEX IF NOT EXISTS idx_ses_type ON session_events(event_type);
"""


class AuditLogger:
    """Thread-safe audit logger. SQLite with WAL handles concurrent writers."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=30)
            try:
                yield conn
            finally:
                conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- Annotation events ------------------------------------------------ #

    def log_annotation(
        self,
        username: str,
        case_id: str,
        source_filename: str,
        mask_filename: str,
        mask_bytes: bytes,
        session_id: str,
    ) -> str:
        """Returns the sha256 hex digest of the mask."""
        sha = hashlib.sha256(mask_bytes).hexdigest()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO annotation_events
                    (timestamp, username, case_id, source_filename,
                     mask_filename, mask_sha256, mask_bytes, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._now(), username, case_id, source_filename,
                    mask_filename, sha, len(mask_bytes), session_id,
                ),
            )
        return sha

    # --- Session events --------------------------------------------------- #

    def log_session(
        self,
        username: str,
        event_type: str,
        session_id: str,
        case_id: str | None = None,
        details: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO session_events
                    (timestamp, username, event_type, case_id, session_id, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self._now(), username, event_type, case_id, session_id, details),
            )

    # --- Read helpers (for a future dashboard) ---------------------------- #

    def case_annotators(self, case_id: str) -> list[tuple[str, str]]:
        """List (username, timestamp) pairs for everyone who annotated a case."""
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT username, timestamp FROM annotation_events "
                "WHERE case_id = ? ORDER BY timestamp ASC",
                (case_id,),
            )
            return cur.fetchall()

    def user_annotation_count(self, username: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM annotation_events WHERE username = ?",
                (username,),
            )
            return cur.fetchone()[0]
