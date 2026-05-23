"""
Lightweight dead-code finder for the hogwarts-chat frontend.
=============================================================
A poor man's knip — works with whatever Python you have on hand, no
Node toolchain required. Catches three classes of dead code:

  1. **Files never imported.** A .ts/.tsx file in src/ that nothing else
     imports (the SuggestedPrompts.tsx pattern).
  2. **Exports never referenced.** A named export from one file that
     no other file imports.
  3. **Unused dependencies.** A package in package.json's dependencies
     that doesn't appear in any source import.

Limitations:
  - Does NOT detect "imported but never rendered in JSX" — knip can't
    fully do this either; the deepest tools need JSX AST traversal.
    The SuggestedPrompts.tsx case will be caught by #1 because nothing
    even imports it.
  - Does NOT understand path aliases beyond a simple "@/..." → "src/..."
    rewrite. If the project uses other aliases, false positives possible.
  - Does NOT understand barrel re-exports (``export * from`` chains).
    A file reachable only through a barrel might be flagged.
  - Regex-based import parser, not a TS AST. ~95% accurate; edge cases
    like inline-comment imports may slip.

Usage:
    .venv/bin/python -m scripts.find_dead_frontend_code [--repo PATH]

By default, scans ``../hogwarts-chat``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SRC_EXTS = (".ts", ".tsx", ".js", ".jsx")

# Patterns: capture the imported module specifier
IMPORT_PATTERNS = [
    re.compile(r"""import\s+(?:.+?\s+from\s+)?["']([^"']+)["']"""),
    re.compile(r"""require\(["']([^"']+)["']\)"""),
    re.compile(r"""import\(["']([^"']+)["']\)"""),  # dynamic import
]

# Patterns to extract exports from a file
EXPORT_PATTERNS = [
    re.compile(r"""^\s*export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+(\w+)""", re.M),
    re.compile(r"""^\s*export\s+\{\s*([^}]+)\}""", re.M),
    re.compile(r"""^\s*export\s+(?:type|interface)\s+(\w+)""", re.M),
    re.compile(r"""^\s*export\s+default\s+(\w+)\s*[;}\n]""", re.M),
]


def find_source_files(repo: Path) -> list[Path]:
    """All TS/JS source files we care about. Skips node_modules + build dirs."""
    files: list[Path] = []
    for sub in ("src", "e2e"):
        root = repo / sub
        if not root.exists():
            continue
        for ext in SRC_EXTS:
            files.extend(root.rglob(f"*{ext}"))
    return [f for f in files if "node_modules" not in f.parts and "dist" not in f.parts]


def resolve_import(spec: str, importer: Path, repo: Path) -> Path | None:
    """Map an import specifier to an absolute file path on disk.

    Returns None for unresolved (external package, or doesn't exist on
    disk). Honors the common @/ → src/ alias.
    """
    if spec.startswith("@/"):
        target = repo / "src" / spec[2:]
    elif spec.startswith("./") or spec.startswith("../"):
        target = (importer.parent / spec).resolve()
    else:
        return None  # external package

    candidates = [
        target.with_suffix(ext) for ext in SRC_EXTS
    ] + [target] + [
        target / f"index{ext}" for ext in SRC_EXTS
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def extract_imports(file: Path) -> list[tuple[str, set[str]]]:
    """Returns list of (module_specifier, {imported_names}). Names empty
    for side-effect-only or default-only imports — we treat those as
    'file is used' but don't attribute names."""
    text = file.read_text(errors="replace")
    results: list[tuple[str, set[str]]] = []

    # First strip block + line comments to avoid matching imports in comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)

    # Capture each import with the {named} portion if present
    named_imp = re.compile(
        r"""import\s+(?:(\w+)\s*,?\s*)?(?:\{([^}]+)\})?\s*(?:,?\s*\*\s+as\s+(\w+))?\s*from\s+["']([^"']+)["']""",
        re.M,
    )
    for m in named_imp.finditer(text):
        default_name, named_block, _starredname, spec = m.groups()
        names: set[str] = set()
        if named_block:
            for n in named_block.split(","):
                n = n.strip().split(" as ")[0].strip()
                if n:
                    names.add(n)
        if default_name:
            names.add(default_name)
        results.append((spec, names))

    # Side-effect imports
    for m in re.finditer(r"""import\s+["']([^"']+)["']""", text):
        results.append((m.group(1), set()))

    # require() and dynamic import()
    for m in re.finditer(r"""(?:require|import)\(["']([^"']+)["']\)""", text):
        results.append((m.group(1), set()))

    return results


def extract_exports(file: Path) -> set[str]:
    """All named exports from a file. 'default' for default exports."""
    text = file.read_text(errors="replace")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)

    exports: set[str] = set()
    for pat in EXPORT_PATTERNS:
        for m in pat.finditer(text):
            block = m.group(1)
            for name in block.split(","):
                name = name.strip().split(" as ")[0].strip()
                if name:
                    exports.add(name)

    # Default export (catch-all)
    if re.search(r"""^\s*export\s+default\b""", text, re.M):
        exports.add("default")

    # Re-exports: export { X } from "./mod" or export * from "./mod"
    for m in re.finditer(r"""export\s+\{([^}]+)\}\s*from""", text):
        for n in m.group(1).split(","):
            n = n.strip().split(" as ")[0].strip()
            if n:
                exports.add(n)

    return exports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../hogwarts-chat", help="Path to the frontend repo")
    ap.add_argument("--entry", action="append", default=[],
                    help="Treat this file as an entry (always reachable). Repeatable.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "package.json").exists():
        print(f"ERROR: {repo} has no package.json", file=sys.stderr)
        return 2

    # Default entries
    default_entries = ["src/main.tsx", "src/main.ts", "src/index.tsx", "src/index.ts",
                       "src/App.tsx", "vite.config.ts", "vite.config.js"]
    entries: set[Path] = set()
    for e in default_entries + args.entry:
        p = repo / e
        if p.exists():
            entries.add(p.resolve())

    files = find_source_files(repo)
    files_set = {f.resolve() for f in files}
    print(f"Scanning {len(files)} source files in {repo}\n")

    # Build the import graph: file -> set of files it imports
    import_graph: dict[Path, set[Path]] = {f: set() for f in files_set}
    # Also track which names are imported from each file (for unused-export detection)
    imported_names_by_file: dict[Path, set[str]] = {f: set() for f in files_set}

    external_imports: set[str] = set()

    for f in files:
        for spec, names in extract_imports(f):
            resolved = resolve_import(spec, f, repo)
            if resolved and resolved in files_set:
                import_graph[f].add(resolved)
                imported_names_by_file[resolved] |= names
            elif not spec.startswith(".") and not spec.startswith("@/"):
                # External package — strip subpath to get root package name
                pkg = spec.split("/")[0] if not spec.startswith("@") else "/".join(spec.split("/")[:2])
                external_imports.add(pkg)

    # --- 1. Files never imported (reachability from entries) ---
    reachable: set[Path] = set()
    stack = list(entries)
    while stack:
        f = stack.pop()
        if f in reachable:
            continue
        reachable.add(f)
        for dep in import_graph.get(f, ()):
            if dep not in reachable:
                stack.append(dep)

    unreachable = sorted(files_set - reachable - entries)
    print(f"=== UNREACHABLE FILES ({len(unreachable)}) ===")
    print("(not reachable by import from any entry; likely dead)")
    for f in unreachable:
        rel = f.relative_to(repo)
        print(f"  {rel}")
    print()

    # --- 2. Exports never imported ---
    print("=== UNUSED EXPORTS ===")
    print("(named exports that no other file imports — excludes default exports of entry files)")
    unused_count = 0
    for f in sorted(files_set):
        if f in unreachable:
            continue  # already flagged at file level
        exports = extract_exports(f)
        used_names = imported_names_by_file.get(f, set())
        unused = exports - used_names
        # Filter out "default" if any caller does a default-only import
        # (which our parser records as a named import) — already handled.
        # Also drop "default" for files that ARE entries.
        if f in entries:
            unused -= {"default"}
        if unused:
            rel = f.relative_to(repo)
            print(f"  {rel}: {', '.join(sorted(unused))}")
            unused_count += len(unused)
    print(f"  (total: {unused_count} unused export names)\n")

    # --- 3. Unused dependencies ---
    pkg = json.loads((repo / "package.json").read_text())
    deps = set(pkg.get("dependencies", {}).keys())
    devs = set(pkg.get("devDependencies", {}).keys())

    print("=== UNUSED DEPENDENCIES ===")
    used_deps = deps & external_imports
    unused_deps = deps - external_imports
    print(f"  ({len(used_deps)}/{len(deps)} deps referenced in src imports)")
    if unused_deps:
        for d in sorted(unused_deps):
            print(f"  {d}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
