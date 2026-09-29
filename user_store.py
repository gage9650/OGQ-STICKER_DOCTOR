"""간단한 사용자 계정 + 사용자별 진단 기록 저장소.

프로토타입 단계에서는 SQLite를 사용한다. 비밀번호는 평문으로 저장하지 않고
PBKDF2-HMAC-SHA256으로 해시한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime
from typing import Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sticker_doctor_users.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnosis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                score REAL,
                pass_count INTEGER NOT NULL DEFAULT 0,
                warn_count INTEGER NOT NULL DEFAULT 0,
                fail_count INTEGER NOT NULL DEFAULT 0,
                checklist_done INTEGER NOT NULL DEFAULT 0,
                checklist_total INTEGER NOT NULL DEFAULT 0,
                file_count INTEGER NOT NULL DEFAULT 0,
                market_keywords TEXT,
                diagnosis_json TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310_000,
    )
    return salt.hex(), digest.hex()


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 2 or len(username) > 30:
        return False, "아이디는 2~30자로 입력해주세요."
    if len(password) < 6:
        return False, "비밀번호는 6자 이상으로 입력해주세요."

    salt, password_hash = _hash_password(password)
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, password_salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        return True, "회원가입이 완료됐습니다."
    except sqlite3.IntegrityError:
        return False, "이미 사용 중인 아이디입니다."
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    username = username.strip()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT password_hash, password_salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return False

    try:
        _, candidate = _hash_password(password, bytes.fromhex(row["password_salt"]))
    except Exception:
        return False
    return secrets.compare_digest(candidate, row["password_hash"])


def get_user_id(username: str) -> int | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        return int(row["id"]) if row else None
    finally:
        conn.close()


def save_diagnosis_record(
    username: str,
    *,
    score: float,
    pass_count: int,
    warn_count: int,
    fail_count: int,
    checklist_done: int,
    checklist_total: int,
    file_count: int,
    market_keywords: list[str] | None = None,
    diagnosis_payload: dict[str, Any] | None = None,
) -> int | None:
    user_id = get_user_id(username)
    if user_id is None:
        return None

    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO diagnosis_records (
                user_id, created_at, score, pass_count, warn_count, fail_count,
                checklist_done, checklist_total, file_count, market_keywords, diagnosis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                datetime.now().isoformat(timespec="seconds"),
                round(float(score), 1),
                int(pass_count),
                int(warn_count),
                int(fail_count),
                int(checklist_done),
                int(checklist_total),
                int(file_count),
                json.dumps(market_keywords or [], ensure_ascii=False),
                json.dumps(diagnosis_payload or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def load_user_diagnosis_history(username: str, limit: int = 30) -> list[dict[str, Any]]:
    user_id = get_user_id(username)
    if user_id is None:
        return []

    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, score, pass_count, warn_count, fail_count,
                   checklist_done, checklist_total, file_count, market_keywords, diagnosis_json
            FROM diagnosis_records
            WHERE user_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (user_id, max(1, min(int(limit), 100))),
        ).fetchall()
    finally:
        conn.close()

    history: list[dict[str, Any]] = []
    for row in rows:
        try:
            keywords = json.loads(row["market_keywords"] or "[]")
        except json.JSONDecodeError:
            keywords = []
        try:
            diagnosis_payload = json.loads(row["diagnosis_json"] or "{}")
        except json.JSONDecodeError:
            diagnosis_payload = {}
        history.append(
            {
                "id": row["id"],
                "timestamp": row["created_at"],
                "score": row["score"],
                "pass": row["pass_count"],
                "warn": row["warn_count"],
                "fail": row["fail_count"],
                "checklist": f"{row['checklist_done']}/{row['checklist_total']}",
                "file_count": row["file_count"],
                "market_keywords": keywords,
                "diagnosis": diagnosis_payload,
            }
        )
    return history


init_db()
