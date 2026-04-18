#!/usr/bin/env bash
#
# Compare DSPy compiled vs uncompiled agent accuracy (v2)
# =======================================================
# Runs eval_dspy.py twice (with and without compiled demos),
# then prints a side-by-side comparison with statistical rigor.
#
# Usage:
#   ./evals/compare_dspy.sh              # full eval, 1 run
#   ./evals/compare_dspy.sh 5            # quick smoke test (5 questions)
#   ./evals/compare_dspy.sh --runs 3     # 3 runs for significance
#   ./evals/compare_dspy.sh 5 --runs 3   # both
#
set -euo pipefail

cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

LIMIT_FLAG=""
RUNS_FLAG=""

for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        LIMIT_FLAG="--limit $arg"
    elif [[ "$arg" == "--runs" ]]; then
        : # next arg is the count
    elif [[ -n "${PREV_WAS_RUNS:-}" ]]; then
        RUNS_FLAG="--runs $arg"
        unset PREV_WAS_RUNS
    fi
    [[ "$arg" == "--runs" ]] && PREV_WAS_RUNS=1
done
# Handle "--runs N" as two args
if [[ -z "$RUNS_FLAG" ]]; then
    for i in $(seq 1 $(($# - 1))); do
        arg="${!i}"
        next_i=$((i + 1))
        next_arg="${!next_i}"
        if [[ "$arg" == "--runs" ]]; then
            RUNS_FLAG="--runs $next_arg"
        fi
    done
fi

echo "============================================="
echo "  DSPy Compiled vs Uncompiled Comparison (v2)"
echo "============================================="
echo ""

# --- Run 1: Uncompiled ---
echo ">>> [1/2] Running UNCOMPILED eval..."
.venv/bin/python -m evals.eval_dspy \
    --out evals/results_dspy_uncompiled.json \
    $LIMIT_FLAG $RUNS_FLAG

echo ""
echo ">>> [2/2] Running COMPILED eval..."
.venv/bin/python -m evals.eval_dspy \
    --agent-dir my_profile.agent \
    --out evals/results_dspy_compiled.json \
    $LIMIT_FLAG $RUNS_FLAG

# --- Compare ---
echo ""
echo "============================================="
echo "  COMPARISON"
echo "============================================="

.venv/bin/python3 -c "
import json

u = json.load(open('evals/results_dspy_uncompiled.json'))
c = json.load(open('evals/results_dspy_compiled.json'))

u_acc = u['accuracy_mean']
c_acc = c['accuracy_mean']
u_std = u.get('accuracy_std', 0)
c_std = c.get('accuracy_std', 0)
diff = c_acc - u_acc

n_runs = u.get('runs', 1)

print()
if n_runs > 1:
    print(f'  Uncompiled: {u_acc:.1%} ± {u_std:.1%}')
    print(f'  Compiled:   {c_acc:.1%} ± {c_std:.1%}')
else:
    print(f'  Uncompiled: {u_acc:.1%}')
    print(f'  Compiled:   {c_acc:.1%}')
print(f'  Delta:      {diff:+.1%}')
print()

# Per-kind
for label, data in [('Uncompiled', u), ('Compiled', c)]:
    by_kind = {}
    for qid, info in data['per_question'].items():
        k = info['kind']
        by_kind.setdefault(k, [])
        by_kind[k].append(info['correct_rate'])
    print(f'  {label}:')
    for k in sorted(by_kind):
        rates = by_kind[k]
        avg = sum(rates) / len(rates)
        print(f'    {k:11s} {avg:.0%} ({len(rates)} qs)')
    print()

# Latency
print(f'  Latency (warmup excluded):')
print(f'    Uncompiled: {u[\"mean_latency_s\"]:.1f}s')
print(f'    Compiled:   {c[\"mean_latency_s\"]:.1f}s')
lat_diff = (c['mean_latency_s'] - u['mean_latency_s']) / max(u['mean_latency_s'], 0.01)
print(f'    Delta:      {lat_diff:+.0%}')
print()

# Retries
print(f'  Structured-output retries:')
print(f'    Uncompiled: {u[\"retry_rate\"]:.0%} of questions')
print(f'    Compiled:   {c[\"retry_rate\"]:.0%} of questions')
print()

# Per-question regressions/fixes
print(f'  Per-question changes:')
changed = False
for qid in u['per_question']:
    if qid in c['per_question']:
        ur = u['per_question'][qid]['correct_rate']
        cr = c['per_question'][qid]['correct_rate']
        if abs(ur - cr) > 0.01:
            changed = True
            if cr > ur:
                print(f'    {qid}: IMPROVED ({ur:.0%} → {cr:.0%})')
            else:
                print(f'    {qid}: REGRESSED ({ur:.0%} → {cr:.0%})')
if not changed:
    print(f'    (no changes)')

# Verdict
print()
if diff > 0.05:
    print(f'  VERDICT: Compiled demos improved accuracy by {diff:.1%}')
elif diff < -0.05:
    print(f'  VERDICT: Compiled demos HURT accuracy by {abs(diff):.1%} — review your metric')
else:
    print(f'  VERDICT: No significant accuracy difference ({diff:+.1%})')

if c['retry_rate'] < u['retry_rate']:
    print(f'  VERDICT: Compiled demos reduced retries ({u[\"retry_rate\"]:.0%} → {c[\"retry_rate\"]:.0%}) — likely explains latency improvement')
"
