from __future__ import annotations

import argparse
import json
from pathlib import Path


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} under {root}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous prefix: {matches}")
    return matches[0]


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def render_view(storage_dir: str | Path, doc_prefix: str, page: int) -> str:
    root = Path(storage_dir)
    page_path = _find_one(root, f"{doc_prefix}*.pages.jsonl")
    stem = page_path.name[: -len(".pages.jsonl")]
    table_candidates = list(root.rglob(f"{stem}.tables.jsonl"))
    table_path = table_candidates[0] if table_candidates else None

    pages = _load_jsonl(page_path)
    page_row = next((p for p in pages if int(p.get("pdf_page_index", 0)) == page), None)
    if page_row is None:
        raise ValueError(f"page {page} not found in {page_path}")

    lines: list[str] = []
    lines.append(f"=== PAGE {page} ({stem}) ===")
    lines.append(page_row.get("page_text", ""))
    lines.append("")

    tables: list[dict] = []
    if table_path is not None and table_path.is_file():
        tables = _load_jsonl(table_path)
    page_tables = [t for t in tables
                   if int(t.get("page_start", -1)) <= page <= int(t.get("page_end", -1))]

    if not page_tables:
        lines.append("(no tables on this page)")
        return "\n".join(lines) + "\n"

    for ti, t in enumerate(page_tables, start=1):
        lines.append(f"--- TABLE {ti} ---")
        cols = t.get("columns", [])
        header = [str(c.get("header_raw", "")) for c in cols]
        rows = t.get("rows", [])
        cells_matrix = []
        for r in rows:
            cells = r.get("cells", [])
            cells_matrix.append([str(c.get("raw", "")) for c in cells])
        widths = [max([len(h)] + [len(cm[i]) if i < len(cm) else 0
                  for cm in cells_matrix]) for i, h in enumerate(header)]
        lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for cm in cells_matrix:
            padded = []
            for i, w in enumerate(widths):
                val = cm[i] if i < len(cm) else ""
                padded.append(val.ljust(w))
            lines.append("  ".join(padded))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only view of persisted page/table JSONL")
    parser.add_argument("--storage-dir", required=True)
    parser.add_argument("--doc", required=True)
    parser.add_argument("--page", type=int, required=True)
    args = parser.parse_args()
    print(render_view(args.storage_dir, args.doc, args.page), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
