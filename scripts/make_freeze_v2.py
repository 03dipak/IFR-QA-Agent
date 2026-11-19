"""Build corpus freeze v2 from the current ledger (same definition as freeze v1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

LEDGER = Path("data/corpus/ledger.jsonl")
RAW = Path("data/corpus/raw")
PAGES = Path("data/corpus/extract/page_extracts")
FREEZE_V1 = Path("data/corpus/freeze_v1.json")
FREEZE_V2 = Path("data/corpus/freeze_v2.json")
GATE_PASS = Path("data/corpus/gate_pass_v2.json")
CORRECTIONS = [Path("data/corpus/ledger_corrections_v2.json"),
                Path("data/corpus/ledger_corrections_v3.jsonl")]

DOC_KEYS = ["company_id", "doc_type", "doc_type_confidence", "document_id",
            "extraction_config_hash", "extraction_status", "fiscal_year",
            "period_type", "sha256", "source", "variant", "version"]


def replay_ledger(path: Path) -> tuple[dict[str, dict], set[str], set[str]]:
    """Last-wins document rows; superseded ids; withdrawn ids."""
    docs: dict[str, dict] = {}
    superseded: set[str] = set()
    withdrawn: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind, data = record.get("kind"), record.get("data") or {}
            if kind == "document" and data.get("checksum"):
                docs[data["checksum"]] = data
            elif kind == "superseded":
                if data.get("document_id"):
                    superseded.add(data["document_id"])
            elif kind == "revision":
                if data.get("effective_status") == "superseded" and data.get("from_document_id"):
                    superseded.add(data["from_document_id"])
            elif kind == "withdrawn" and data.get("document_id"):
                withdrawn.add(data["document_id"])
    return docs, superseded, withdrawn


def verify_doc(checksum: str) -> tuple[bool, str]:
    raw = RAW / f"{checksum}.pdf"
    if not raw.exists():
        return False, "missing_raw"
    if hashlib.sha256(raw.read_bytes()).hexdigest() != checksum:
        return False, "checksum_mismatch"
    if not (PAGES / f"{checksum}.pages.jsonl").exists():
        return False, "missing_extract"
    return True, ""


def load_gate_pass() -> dict[str, dict]:
    if not GATE_PASS.exists():
        return {}
    rows = json.loads(GATE_PASS.read_text(encoding="utf-8"))
    return {row["checksum"]: row for row in rows}


def load_corrections() -> tuple[dict[str, str], set[str]]:
    """Variant overrides and revoked supersessions from the corrections files."""
    overrides: dict[str, str] = {}
    revoked: set[str] = set()
    for path in CORRECTIONS:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = [data]
        for entry in entries:
            kind = entry.get("kind")
            if kind == "variant_override":
                overrides[entry["document_id"]] = entry["to"]
            elif kind == "supersession_revocation":
                revoked.add(entry["from_document_id"])
            elif kind is None:
                overrides.update(entry.get("variant_overrides") or {})
                revoked.update(entry.get("revoked_supersessions") or [])
    return overrides, revoked


def build_freeze(gate_pass: dict[str, dict], corrections: tuple[dict[str, str], set[str]]
                 ) -> tuple[list[dict], Counter]:
    docs, superseded, withdrawn = replay_ledger(LEDGER)
    overrides, revoked = corrections
    superseded -= revoked
    reasons: Counter = Counter()
    kept: list[dict] = []
    for doc in docs.values():
        checksum = doc.get("checksum")
        if checksum in overrides:
            doc = {**doc, "variant": overrides[checksum]}
        gate = gate_pass.get(checksum, {}).get("decision", doc.get("gate_decision"))
        if gate != "keep":
            reasons[f"gate:{gate}"] += 1
            continue
        if checksum in superseded:
            reasons["superseded"] += 1
            continue
        if checksum in withdrawn:
            reasons["withdrawn"] += 1
            continue
        ok, why = verify_doc(checksum)
        if not ok:
            reasons[why] += 1
            continue
        if doc.get("extraction_status") != "extracted":
            reasons["not_extracted"] += 1
            continue
        kept.append(doc)
    return kept, reasons


def manifest(kept: list[dict]) -> dict:
    documents = [{k: doc.get(k) for k in DOC_KEYS} for doc in kept]
    documents.sort(key=lambda d: d["document_id"])
    counts_by_company = Counter(d["company_id"] for d in documents)
    counts_by_doc_type = Counter(d["doc_type"] for d in documents)
    manifest = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts_by_company": dict(sorted(counts_by_company.items())),
        "counts_by_doc_type": dict(sorted(counts_by_doc_type.items())),
        "definition": ("freeze v1 definition re-run on current ledger: last-wins doc rows with "
                       "gate_decision=='keep', not superseded/withdrawn, extraction_status=="
                       "'extracted', raw sha256 verified, page extracts present"),
        "documents": documents,
        "freeze_version": "v2",
        "source_ledger": str(LEDGER),
    }
    body = json.dumps(manifest["documents"], sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    manifest["source_ledger_sha256"] = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    if CORRECTIONS:
        blob = b"".join(p.read_bytes() for p in CORRECTIONS if p.exists())
        manifest["corrections_sha256"] = hashlib.sha256(blob).hexdigest()
        manifest["corrections_files"] = [str(p) for p in CORRECTIONS if p.exists()]
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build freeze v2 manifest")
    parser.add_argument("--apply", action="store_true", help="write freeze_v2.json")
    args = parser.parse_args()
    started = time.time()
    kept, reasons = build_freeze(load_gate_pass(), load_corrections())
    doc = manifest(kept)
    v1 = json.loads(FREEZE_V1.read_text(encoding="utf-8"))
    v1_ids = {d["document_id"] for d in v1["documents"]}
    v2_ids = {d["document_id"] for d in doc["documents"]}
    print(f"freeze v2 docs: {len(kept)} | companies: {len(doc['counts_by_company'])}")
    print("by doc_type:", doc["counts_by_doc_type"])
    print("excluded:", dict(reasons))
    print(f"vs v1: shared={len(v1_ids & v2_ids)} v1-only={len(v1_ids - v2_ids)} "
          f"v2-new={len(v2_ids - v1_ids)}")
    if args.apply:
        FREEZE_V2.write_text(json.dumps(doc, indent=1, sort_keys=False), encoding="utf-8")
        print(f"wrote {FREEZE_V2} manifest_sha256={doc['manifest_sha256'][:16]}")
    print(f"elapsed_s={int(time.time() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
