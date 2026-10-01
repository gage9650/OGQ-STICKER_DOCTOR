"""간단한 사용자 계정 + 사용자별 진단 기록 + 리뷰 저장소.

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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diagnosis_record_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                overall_rating INTEGER NOT NULL,
                usefulness_rating INTEGER NOT NULL,
                accuracy_rating INTEGER NOT NULL,
                issue_tags TEXT,
                comment TEXT NOT NULL DEFAULT '',
                visibility TEXT NOT NULL DEFAULT 'public',
                FOREIGN KEY(diagnosis_record_id) REFERENCES diagnosis_records(id) ON DELETE CASCADE,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(diagnosis_record_id, user_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_user ON diagnosis_records(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_record ON reviews(diagnosis_record_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_visibility ON reviews(visibility, id DESC)")
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


def load_user_diagnosis_history(username: str, limit: int = 50) -> list[dict[str, Any]]:
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


def save_review(
    username: str,
    diagnosis_record_id: int,
    *,
    overall_rating: int,
    usefulness_rating: int,
    accuracy_rating: int,
    issue_tags: list[str] | None = None,
    comment: str = "",
    visibility: str = "public",
) -> int | None:
    user_id = get_user_id(username)
    if user_id is None:
        return None

    overall_rating = max(1, min(5, int(overall_rating)))
    usefulness_rating = max(1, min(5, int(usefulness_rating)))
    accuracy_rating = max(1, min(5, int(accuracy_rating)))
    visibility = visibility if visibility in {"public", "developer"} else "public"
    comment = str(comment or "").strip()[:3000]
    tags_json = json.dumps(issue_tags or [], ensure_ascii=False)

    conn = _connect()
    try:
        # 한 진단에 한 리뷰만 남기고 다시 작성하면 업데이트한다.
        existing = conn.execute(
            "SELECT id FROM reviews WHERE diagnosis_record_id = ? AND user_id = ?",
            (int(diagnosis_record_id), user_id),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE reviews
                SET created_at = ?, overall_rating = ?, usefulness_rating = ?, accuracy_rating = ?,
                    issue_tags = ?, comment = ?, visibility = ?
                WHERE id = ?
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    overall_rating,
                    usefulness_rating,
                    accuracy_rating,
                    tags_json,
                    comment,
                    visibility,
                    int(existing["id"]),
                ),
            )
            conn.commit()
            return int(existing["id"])

        # 본인의 진단 기록에만 리뷰할 수 있도록 검증
        owned = conn.execute(
            "SELECT id FROM diagnosis_records WHERE id = ? AND user_id = ?",
            (int(diagnosis_record_id), user_id),
        ).fetchone()
        if not owned:
            return None

        cur = conn.execute(
            """
            INSERT INTO reviews (
                diagnosis_record_id, user_id, created_at, overall_rating,
                usefulness_rating, accuracy_rating, issue_tags, comment, visibility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(diagnosis_record_id),
                user_id,
                datetime.now().isoformat(timespec="seconds"),
                overall_rating,
                usefulness_rating,
                accuracy_rating,
                tags_json,
                comment,
                visibility,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _review_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    try:
        issue_tags = json.loads(row["issue_tags"] or "[]")
    except json.JSONDecodeError:
        issue_tags = []
    return {
        "id": row["id"],
        "record_id": row["diagnosis_record_id"],
        "username": row["username"],
        "timestamp": row["created_at"],
        "overall_rating": row["overall_rating"],
        "usefulness_rating": row["usefulness_rating"],
        "accuracy_rating": row["accuracy_rating"],
        "issue_tags": issue_tags,
        "comment": row["comment"],
        "visibility": row["visibility"],
    }


def load_reviews_for_record(diagnosis_record_id: int, include_developer_only: bool = True) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        if include_developer_only:
            rows = conn.execute(
                """
                SELECT r.*, u.username
                FROM reviews r JOIN users u ON u.id = r.user_id
                WHERE r.diagnosis_record_id = ?
                ORDER BY r.id DESC
                """,
                (int(diagnosis_record_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT r.*, u.username
                FROM reviews r JOIN users u ON u.id = r.user_id
                WHERE r.diagnosis_record_id = ? AND r.visibility = 'public'
                ORDER BY r.id DESC
                """,
                (int(diagnosis_record_id),),
            ).fetchall()
    finally:
        conn.close()
    return [_review_row_to_dict(row) for row in rows]


def load_public_reviews(limit: int = 30) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT r.*, u.username
            FROM reviews r JOIN users u ON u.id = r.user_id
            WHERE r.visibility = 'public'
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 100)),),
        ).fetchall()
    finally:
        conn.close()
    return [_review_row_to_dict(row) for row in rows]


def load_all_reviews(limit: int = 100) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT r.*, u.username
            FROM reviews r JOIN users u ON u.id = r.user_id
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    finally:
        conn.close()
    return [_review_row_to_dict(row) for row in rows]


init_db()
