#!/usr/bin/env bash
# run_a100_session.sh — Ordered, time-boxed A100 GPU session orchestrator.
#
# Runs ALL phases in order inside the activated .venv.
# Save results after each phase. Stop and report if any phase fails.
#
# Usage (inside tmux — ALWAYS run inside tmux):
#   tmux new-session -s a100
#   source .venv/bin/activate
#   bash run_a100_session.sh 2>&1 | tee a100_session.log
#
# Phases:
#   Phase 0a: Environment verification (already done by setup_a100.sh)
#   Phase 0b+0c: Provenance check + Teacher vs Hybrid comparison (~5 min)
#   Phase 1: Teacher 5-fold CV, 2 conditions (~45 min)
#   Phase 2: Cache embeddings (~5 min) → GPU no longer needed after this
#   Phase 3: QSVM gate check (conditional)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-50}"
PHASE1_SEED="${PHASE1_SEED:-42}"
RESULTS_DIR="$REPO_ROOT/results"
LOG_DIR="$REPO_ROOT/results/a100_logs"

mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%dT%H:%M:%S"; }

phase_header() {
    echo ""
    echo "################################################################"
    echo "##  $1"
    echo "##  Started: $(timestamp)"
    echo "################################################################"
}

phase_done() {
    echo ""
    echo ">> $1 COMPLETE — $(timestamp)"
    echo ">> Results saved to: $2"
}

# ── Sanity check: GPU available ──────────────────────────────────
phase_header "Phase 0a — Environment Sanity Check"
$PYTHON -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available! Check setup_a100.sh output.'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'torch.cuda.is_available() = True ✓')
from quantum_kc.config import config, get_device
device = get_device()
print(f'get_device() = {device}')
assert str(device) == 'cuda', f'Expected cuda, got {device}'
print('All environment checks passed ✓')
"
phase_done "Phase 0a" "N/A (in-process check)"

# ── Phase 0b + 0c: Provenance + comparison ───────────────────────
phase_header "Phase 0b+0c — Provenance Check + Teacher vs Hybrid"
$PYTHON scripts/a100_phase0_provenance.py 2>&1 | tee "$LOG_DIR/phase0_provenance.log"
phase_done "Phase 0b+0c" "$RESULTS_DIR/a100_phase0_results.json"

# ── Phase 1: Teacher 5-fold CV ────────────────────────────────────
phase_header "Phase 1 — Teacher 5-fold CV (epochs=$PHASE1_EPOCHS, seed=$PHASE1_SEED)"
echo "WARNING: This phase runs 10 training jobs (5 folds × 2 conditions)."
echo "         Estimated time: ~45 min on A100."

# First, verify Stage-1 pretrained encoder exists
echo ""
echo "Checking Stage-1 pretrained_encoder.pt..."
$PYTHON scripts/a100_phase1_teacher_cv.py --check-pretrain-only

PRETRAIN_CKPT="$REPO_ROOT/checkpoints/pretrained_encoder.pt"
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
    echo ""
    echo "[INFO] Stage-1 pretrained_encoder.pt not found."
    echo "[INFO] Running Stage-1 pretraining on 2,633 unlabeled images..."
    echo "[INFO] (This runs ONCE — not per fold)"
    $PYTHON scripts/run_pretrain.py --epochs 100 2>&1 | tee "$LOG_DIR/phase1_pretrain.log"
    echo "[INFO] Stage-1 pretraining complete."
fi

echo ""
echo "Running 5-fold Teacher CV (both conditions)..."
$PYTHON scripts/a100_phase1_teacher_cv.py \
    --epochs "$PHASE1_EPOCHS" \
    --seed "$PHASE1_SEED" \
    2>&1 | tee "$LOG_DIR/phase1_teacher_cv.log"
phase_done "Phase 1" "$RESULTS_DIR/a100_phase1_teacher_cv.json"

# ── Phase 2: Cache embeddings ─────────────────────────────────────
phase_header "Phase 2 — Cache Teacher Embeddings (all 1,454 eyes)"
echo "Using canonical Teacher from Phase 1 output..."

# Read canonical checkpoint from Phase 1 results
CANONICAL_CKPT=$(python -c "
import json
with open('$RESULTS_DIR/a100_phase1_teacher_cv.json') as f:
    d = json.load(f)
ckpt = d.get('canonical_teacher', {}).get('canonical_checkpoint', '')
print(ckpt)
" 2>/dev/null || echo "")

if [[ -z "$CANONICAL_CKPT" ]] || [[ ! -f "$CANONICAL_CKPT" ]]; then
    echo "[WARN] Could not read canonical checkpoint from Phase 1 results."
    echo "[WARN] Falling back to: checkpoints/teacher_finetuned_with_pretrain_s42_fold1.pt"
    CANONICAL_CKPT="$REPO_ROOT/checkpoints/teacher_finetuned_with_pretrain_s42_fold1.pt"
fi
echo "Canonical Teacher: $CANONICAL_CKPT"

$PYTHON scripts/a100_phase2_cache_embeddings.py \
    --teacher-ckpt "$CANONICAL_CKPT" \
    --batch-size 64 \
    --num-workers 4 \
    2>&1 | tee "$LOG_DIR/phase2_cache.log"
phase_done "Phase 2" "$REPO_ROOT/data/teacher_embeddings.npz"

echo ""
echo "################################################################"
echo "##  >>> GPU IS NO LONGER NEEDED FOR STUDENT TRAINING <<<       "
echo "##  >>> Student (tiny MLP + 8-qubit VQC) runs on CPU  <<<      "
echo "##  If Phase 3 gate fails, terminate the A100 now.             "
echo "################################################################"

# ── Phase 3: QSVM gate check (conditional) ────────────────────────
phase_header "Phase 3 — QSVM Gate Check (conditional)"
echo "Running gate checks before any kernel matrix computation..."
$PYTHON scripts/a100_phase3_qsvm_gate.py \
    --n-values "50,100,200" \
    2>&1 | tee "$LOG_DIR/phase3_gate.log"
phase_done "Phase 3" "$RESULTS_DIR/a100_phase3_gate_check.json"

# Check if both gates passed
GATES_PASS=$(python -c "
import json, sys
try:
    with open('$RESULTS_DIR/a100_phase3_gate_check.json') as f:
        d = json.load(f)
    g1 = d.get('gate_i', {}).get('gate_pass', False)
    g2 = d.get('gate_ii', {}).get('gate_pass', False)
    print('YES' if (g1 and g2) else 'NO')
except:
    print('NO')
" 2>/dev/null || echo "NO")

if [[ "$GATES_PASS" == "YES" ]]; then
    echo ""
    echo "Both gates PASSED — computing kernel matrices..."
    echo "(Kernel matrix computation script goes here — not yet gated in)"
    echo "See: results/a100_phase3_gate_check.json for gate details."
else
    echo ""
    echo "One or more gates FAILED. NOT proceeding to kernel matrix."
    echo "See: $RESULTS_DIR/a100_phase3_gate_check.json for details."
    echo ""
    echo "################################################################"
    echo "##  ALL PHASES COMPLETE. GPU INSTANCE CAN BE TERMINATED NOW.  "
    echo "################################################################"
fi

echo ""
echo "================================================================"
echo "  A100 session complete — $(timestamp)"
echo "  All results in: $RESULTS_DIR/"
echo "  Logs in:        $LOG_DIR/"
echo "================================================================"
