"""Extract finished distillation runs from mlruns.db into results/ablation_results.json.

Ensures that any progress made before interruption is permanently recorded.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def sync_progress() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    db_path = base_dir / "mlruns.db"
    results_dir = base_dir / "results"
    results_path = results_dir / "ablation_results.json"

    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("""
        SELECT run_uuid, name, status, start_time, end_time 
        FROM runs 
        WHERE status = 'FINISHED' AND name LIKE 'student_%'
        ORDER BY start_time ASC
    """)
    finished_runs = c.fetchall()

    ablation = {"classical": [], "quantum": [], "logistic_baseline": []}

    # Load existing if available
    if results_path.exists():
        try:
            with open(results_path) as f:
                existing = json.load(f)
                for k in ablation:
                    if k in existing and isinstance(existing[k], list):
                        ablation[k] = existing[k]
        except Exception:
            pass

    for run_uuid, name, status, start_t, end_t in finished_runs:
        c.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_uuid,))
        params = dict(c.fetchall())

        c.execute("SELECT key, value FROM metrics WHERE run_uuid = ?", (run_uuid,))
        metrics_raw = c.fetchall()

        metrics = {}
        for k, v in metrics_raw:
            if k.startswith("best_"):
                metrics[k[5:]] = v
            elif k not in metrics:
                metrics[k] = v

        if not metrics:
            continue

        for pk in ["fold", "head_type", "alpha", "beta", "seed"]:
            if pk in params:
                try:
                    metrics[pk] = int(params[pk]) if pk in ["fold", "seed"] else float(params[pk])
                except ValueError:
                    metrics[pk] = params[pk]

        ht = metrics.get("head_type", "")
        if ht in ablation:
            dup = any(
                r.get("fold") == metrics.get("fold")
                and r.get("seed") == metrics.get("seed")
                and abs(float(r.get("alpha", -1.0)) - float(metrics.get("alpha", -1.0))) < 1e-4
                and abs(float(r.get("beta", -1.0)) - float(metrics.get("beta", -1.0))) < 1e-4
                for r in ablation[ht]
            )
            if not dup:
                ablation[ht].append(metrics)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(ablation, f, indent=2, default=str)

    print("Progress synchronized successfully!")
    print(f"Stored runs: Classical={len(ablation['classical'])}, Quantum={len(ablation['quantum'])}, Logistic={len(ablation['logistic_baseline'])}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    sync_progress()
