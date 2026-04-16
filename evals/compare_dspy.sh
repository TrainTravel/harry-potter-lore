#!/usr/bin/env bash
#
# Compare DSPy compiled vs uncompiled agent accuracy
# ===================================================
# Runs eval_dspy.py twice (with and without compiled demos),
# then prints a side-by-side comparison.
#
# Usage:
#   ./evals/compare_dspy.sh              # full eval (12 questions)
#   ./evals/compare_dspy.sh --limit 5    # quick smoke test
#
set -euo pipefail

cd "$(dirname "$0")/.."
source .env 2>/dev/null || true

LIMIT="${1:-}"
LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit ${LIMIT#--limit }"
    # Handle both "./compare_dspy.sh --limit 5" and "./compare_dspy.sh 5"
    if [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
        LIMIT_FLAG="--limit $LIMIT"
    fi
fi

echo "============================================="
echo "  DSPy Compiled vs Uncompiled Comparison"
echo "============================================="
echo ""

# --- Run 1: Uncompiled (zero-shot) ---
echo ">>> [1/2] Running UNCOMPILED eval (zero-shot)..."
.venv/bin/python -m evals.eval_dspy \
    --out evals/results_dspy_uncompiled.json \
    $LIMIT_FLAG

echo ""
echo ">>> [2/2] Running COMPILED eval (with bootstrapped demos)..."
.venv/bin/python -m evals.eval_dspy \
    --agent-dir my_profile.agent \
    --out evals/results_dspy_compiled.json \
    $LIMIT_FLAG

# --- Compare ---
echo ""
echo "============================================="
echo "  COMPARISON"
echo "============================================="

.venv/bin/python3 -c "
import json

u = json.load(open('evals/results_dspy_uncompiled.json'))
c = json.load(open('evals/results_dspy_compiled.json'))

u_acc = u['accuracy']
c_acc = c['accuracy']
diff = c_acc - u_acc

print(f'')
print(f'  Uncompiled accuracy: {u_acc:.1%}')
print(f'  Compiled accuracy:   {c_acc:.1%}')
print(f'  Delta:               {diff:+.1%}')
print(f'')

# Per-kind breakdown
for label, data in [('Uncompiled', u), ('Compiled', c)]:
    by_kind = {}
    for r in data['rows']:
        k = r['kind']
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += int(r['correct'])
        by_kind[k][1] += 1
    print(f'  {label}:')
    for k, (correct, total) in sorted(by_kind.items()):
        print(f'    {k:11s} {correct}/{total}')
    print()

# Per-question diff
print('  Per-question changes:')
u_by_id = {r['id']: r for r in u['rows']}
c_by_id = {r['id']: r for r in c['rows']}
changed = False
for qid in u_by_id:
    if qid in c_by_id:
        uc = u_by_id[qid]['correct']
        cc = c_by_id[qid]['correct']
        if uc != cc:
            changed = True
            arrow = 'FIXED' if cc else 'REGRESSED'
            print(f'    {qid}: {arrow}')
if not changed:
    print(f'    (no changes)')

# Latency
u_lat = sum(r['latency_s'] for r in u['rows']) / len(u['rows'])
c_lat = sum(r['latency_s'] for r in c['rows']) / len(c['rows'])
print(f'')
print(f'  Mean latency:')
print(f'    Uncompiled: {u_lat:.1f}s')
print(f'    Compiled:   {c_lat:.1f}s')

if diff > 0:
    print(f'')
    print(f'  VERDICT: Compiled demos improved accuracy by {diff:.1%}')
elif diff < 0:
    print(f'')
    print(f'  VERDICT: Compiled demos HURT accuracy by {diff:.1%} — investigate metrics')
else:
    print(f'')
    print(f'  VERDICT: No accuracy difference — DSPy demos had no measurable effect')
"
