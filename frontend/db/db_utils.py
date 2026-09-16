"""
QuantaMED — Database Utilities
=============================
SQLite helper functions for patient, visit, and prediction persistence.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "quantamed.db"
SCHEMA_PATH = DB_DIR / "schema.sql"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database with Row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Initialize the database schema if tables do not exist."""
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()


def generate_patient_id() -> str:
    """Generate the next sequential patient ID in format QMED-000123."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT patient_id FROM patients 
            WHERE patient_id LIKE 'QMED-%' 
            ORDER BY patient_id DESC LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row and row["patient_id"]:
            last_id = row["patient_id"]
            try:
                num_part = int(last_id.split("-")[1])
                next_num = num_part + 1
            except (IndexError, ValueError):
                next_num = 1
        else:
            # Fallback: total count of patients + 1
            cursor2 = conn.execute("SELECT COUNT(*) AS cnt FROM patients")
            next_num = cursor2.fetchone()["cnt"] + 1
            
        return f"QMED-{next_num:06d}"


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve patient record by ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT patient_id, age, gender, created_at FROM patients WHERE patient_id = ?",
            (patient_id.strip(),),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def insert_patient(
    patient_id: str, age: Optional[int] = None, gender: Optional[str] = None
) -> str:
    """Insert a new patient if not already present. Returns patient_id."""
    init_db()
    clean_id = patient_id.strip()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO patients (patient_id, age, gender)
            VALUES (?, ?, ?)
            """,
            (clean_id, age, gender),
        )
        conn.commit()
    return clean_id


def insert_visit(
    patient_id: str,
    mode: str,
    readings: Dict[str, Any],
    notes: Optional[str] = None,
) -> int:
    """Insert a clinical visit record. Returns the new visit_id."""
    init_db()
    insert_patient(patient_id)
    
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO visits (
                patient_id, mode, flat_k_D, flat_k_axis_deg,
                steep_k_D, steep_k_axis_deg, cylinder_D,
                pachy_central_um, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id.strip(),
                mode,
                float(readings["flat_k_D"]),
                float(readings["flat_k_axis_deg"]),
                float(readings["steep_k_D"]),
                float(readings["steep_k_axis_deg"]),
                float(readings["cylinder_D"]),
                float(readings["pachy_central_um"]) if readings.get("pachy_central_um") is not None else None,
                notes,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def insert_prediction(
    visit_id: int,
    model_version: str,
    risk_score: float,
    calculation_trail: List[Dict[str, Any]],
) -> int:
    """Insert a prediction record linked to visit_id with JSON calculation trail."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                visit_id, model_version, risk_score, calculation_trail
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                visit_id,
                model_version,
                float(risk_score),
                json.dumps(calculation_trail),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_visits(patient_id: str) -> List[Dict[str, Any]]:
    """Get all visits for a patient (newest first) joined with prediction details."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT 
                v.visit_id, v.patient_id, v.visit_date, v.mode,
                v.flat_k_D, v.flat_k_axis_deg, v.steep_k_D, v.steep_k_axis_deg,
                v.cylinder_D, v.pachy_central_um, v.notes,
                p.model_version, p.risk_score, p.calculation_trail, p.created_at AS prediction_created_at
            FROM visits v
            LEFT JOIN predictions p ON v.visit_id = p.visit_id
            WHERE v.patient_id = ?
            ORDER BY v.visit_date DESC, v.visit_id DESC
            """,
            (patient_id.strip(),),
        )
        rows = cursor.fetchall()
        visits = []
        for r in rows:
            v_dict = dict(r)
            if v_dict.get("calculation_trail"):
                try:
                    v_dict["calculation_trail"] = json.loads(v_dict["calculation_trail"])
                except Exception:
                    v_dict["calculation_trail"] = []
            else:
                v_dict["calculation_trail"] = []
            visits.append(v_dict)
        return visits


def update_visit_notes(visit_id: int, notes: str) -> None:
    """Update notes for a specific visit."""
    init_db()
    with get_connection() as conn:
        conn.execute(
            "UPDATE visits SET notes = ? WHERE visit_id = ?",
            (notes, visit_id),
        )
        conn.commit()


def get_db_stats() -> Dict[str, Any]:
    """Return database health metrics: counts and last write timestamp."""
    init_db()
    with get_connection() as conn:
        pat_cnt = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        vis_cnt = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        pred_cnt = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        
        # Latest write timestamp across visits or predictions
        last_visit = conn.execute(
            "SELECT visit_date FROM visits ORDER BY visit_date DESC, visit_id DESC LIMIT 1"
        ).fetchone()
        last_write = last_visit[0] if last_visit and last_visit[0] else "No writes yet"
        
        return {
            "patient_count": pat_cnt,
            "visit_count": vis_cnt,
            "prediction_count": pred_cnt,
            "last_write": last_write,
        }
