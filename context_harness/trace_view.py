"""
CLI trace viewer — Skill 6: Evaluation and Observability
=========================================================
Pretty-prints a turn's event sequence from SQLite.
Follows the IBM video's recommendation: "replay the trace, not the user."

Usage:
    python -m context_harness.trace_view <turn_id>
    python -m context_harness.trace_view <turn_id> --db data/traces.db
    python -m context_harness.trace_view --list          # show recent turn IDs
    python -m context_harness.trace_view --list --limit 20
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Colour palette for each event kind
_KIND_STYLE: dict[str, str] = {
    "turn_start":    "bold green",
    "turn_end":      "bold green",
    "llm_call":      "cyan",
    "tool_call":     "blue",
    "tool_result":   "blue",
    "retrieval":     "magenta",
    "context_built": "yellow",
    "escalation":    "bold yellow",
    "error":         "bold red",
    "security":      "bold red",
}


def _format_payload(payload: dict, max_len: int = 120) -> str:
    s = json.dumps(payload, default=str)
    return s[:max_len] + "…" if len(s) > max_len else s


def print_trace(turn_id: str, db_path: str, console: Console) -> int:
    """
    Query SQLite and render the trace as a Rich table.
    Returns 0 on success, 1 if no events found.
    """
    if not Path(db_path).exists():
        console.print(f"[red]Database not found:[/red] {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT seq, kind, payload, ts, latency_ms "
        "FROM trace_events WHERE turn_id=? ORDER BY seq",
        (turn_id,),
    ).fetchall()
    conn.close()

    if not rows:
        console.print(f"[red]No events found for turn_id=[/red]{turn_id!r}")
        return 1

    table = Table(
        title=f"Trace  turn_id=[bold]{turn_id}[/bold]",
        box=box.ROUNDED,
        show_lines=False,
        expand=False,
    )
    table.add_column("#",        style="dim",    width=4,  justify="right")
    table.add_column("Kind",     style="white",  width=18)
    table.add_column("Latency",  style="yellow", width=10, justify="right")
    table.add_column("Payload",  style="white",  no_wrap=False)

    total_latency = 0.0
    first_ts = rows[0][3] if rows else 0.0
    errors = 0

    for seq, kind, payload_raw, ts, latency_ms in rows:
        style = _KIND_STYLE.get(kind, "white")
        lat_str = f"{latency_ms:.1f}ms" if latency_ms is not None else ""
        if latency_ms:
            total_latency += latency_ms
        if kind == "error":
            errors += 1
        try:
            payload = json.loads(payload_raw)
        except (json.JSONDecodeError, TypeError):
            payload = {"raw": str(payload_raw)}

        table.add_row(
            str(seq),
            Text(kind, style=style),
            lat_str,
            _format_payload(payload),
        )

    console.print(table)

    # Summary footer
    elapsed = (rows[-1][3] - first_ts) * 1000 if len(rows) > 1 else 0.0
    summary = (
        f"  [dim]{len(rows)} events[/dim]"
        f"  [yellow]wall {elapsed:.0f}ms[/yellow]"
        f"  [yellow]sum latency {total_latency:.0f}ms[/yellow]"
    )
    if errors:
        summary += f"  [bold red]{errors} error(s)[/bold red]"
    console.print(summary)
    return 0


def list_turns(db_path: str, limit: int, console: Console) -> int:
    """Print the most recent turn IDs from the trace store."""
    if not Path(db_path).exists():
        console.print(f"[red]Database not found:[/red] {db_path}")
        return 1

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT turn_id, COUNT(*) AS events, MIN(ts) AS first_ts "
        "FROM trace_events GROUP BY turn_id ORDER BY first_ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No traces in database.[/yellow]")
        return 0

    table = Table(title="Recent turns", box=box.SIMPLE_HEAD)
    table.add_column("turn_id",  style="cyan")
    table.add_column("events",   style="dim", justify="right")
    table.add_column("first_ts", style="dim")

    import datetime
    for turn_id, event_count, ts in rows:
        ts_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(turn_id, str(event_count), ts_str)

    console.print(table)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m context_harness.trace_view",
        description="Pretty-print a turn trace or list recent turns.",
    )
    ap.add_argument("turn_id", nargs="?", help="Turn ID to display")
    ap.add_argument("--db",    default="data/traces.db",
                    help="Path to traces SQLite database (default: data/traces.db)")
    ap.add_argument("--list",  action="store_true",
                    help="List recent turn IDs instead of printing a trace")
    ap.add_argument("--limit", type=int, default=10,
                    help="Number of turns to list (default: 10)")
    args = ap.parse_args()

    console = Console()

    if args.list:
        return list_turns(args.db, args.limit, console)

    if not args.turn_id:
        ap.error("turn_id is required unless --list is given")

    return print_trace(args.turn_id, args.db, console)


if __name__ == "__main__":
    raise SystemExit(main())
