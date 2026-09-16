-- QuantaMED Clinical Dashboard — Database Schema
-- SQLite, single file (db/quantamed.db)

CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,        -- e.g. 'QMED-000123'
    age          INTEGER,
    gender       TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id        TEXT NOT NULL REFERENCES patients(patient_id),
    visit_date        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mode              TEXT NOT NULL CHECK(mode IN ('ark_only','ark_pachy')),
    flat_k_D          REAL NOT NULL,
    flat_k_axis_deg   REAL NOT NULL,
    steep_k_D         REAL NOT NULL,
    steep_k_axis_deg  REAL NOT NULL,
    cylinder_D        REAL NOT NULL,
    pachy_central_um  REAL,              -- nullable, only in ark_pachy mode
    notes             TEXT               -- doctor's comments / preventive measures
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id           INTEGER NOT NULL REFERENCES visits(visit_id),
    model_version      TEXT NOT NULL,     -- e.g. 'heuristic_v0', later 'tabular_v1'
    risk_score         REAL NOT NULL,     -- 0-100
    calculation_trail  TEXT,              -- JSON: list of factor contribution entries
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
