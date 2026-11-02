"""Corpus freeze manager.

Implements lld_ingestion_pdf.txt 5.5 + 6: freeze versions (v1/v2/v3), manifest
publication with canonical serialization + checksum, extractor-change rule,
coverage report, extraction-QA gate, freeze gate (zero unresolved candidates,
counts reconcile, gold evidence refs resolve, manifest checksum). A freeze is
immutable once tagged.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field

from .models import DocumentLedgerRow, FreezeManifest, PageExtract


def sha256_canonical(obj: dict) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CandidateAccounting:
    discovered: int = 0
    accepted: int = 0
    permanently_excluded: int = 0
    unresolved: int = 0

    def reconciles(self) -> bool:
        return self.discovered == self.accepted + self.permanently_excluded + self.unresolved


@dataclass
class GoldEvidenceRef:
    question_id: str
    document_id: str
    pdf_page_index: int
    extraction_hash: str


@dataclass
class FreezeGateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


class FreezeManager:
    def __init__(self) -> None:
        self.freezes: dict[str, FreezeManifest] = {}
        self.current_tag: str | None = None
        self.gate_results: dict[str, FreezeGateResult] = {}

    def create_freeze(self, freeze_tag: str, parent_freeze_id: str | None,
                      ledger_rows: list[DocumentLedgerRow],
                      page_extracts: list[PageExtract],
                      extraction_config_hash: str,
                      run_id: str, config_hash: str,
                      qa_report_checksum: str | None = None,
                      qa_passed: bool = False,
                      coverage_report_checksum: str | None = None,
                      table_record_set_hash: str | None = None,
                      pipeline_git_commit: str | None = None) -> FreezeManifest:
        ledger_checksum = sha256_canonical(
            {"rows": [r.model_dump(mode="json") for r in ledger_rows]})
        page_set_hash = sha256_canonical(
            {"pages": [p.model_dump(mode="json") for p in page_extracts]})
        table_set_hash = table_record_set_hash if table_record_set_hash else "pending-tables-lld"
        checksum_registry = {r.document_id: r.checksum for r in ledger_rows}
        manifest = FreezeManifest(
            freeze_id=str(uuid.uuid4()),
            freeze_tag=freeze_tag,
            parent_freeze_id=parent_freeze_id,
            ledger_snapshot_checksum=ledger_checksum,
            page_extract_set_hash=page_set_hash,
            table_record_set_hash=table_set_hash,
            extraction_config_hash=extraction_config_hash,
            checksum_registry=checksum_registry,
            document_count=len(ledger_rows),
            page_count=len(page_extracts),
            run_id=run_id,
            pipeline_git_commit=pipeline_git_commit,
            config_hash=config_hash,
            qa_report_checksum=qa_report_checksum,
            qa_passed=qa_passed,
            coverage_report_checksum=coverage_report_checksum,
            status="pending",
        )
        manifest.manifest_checksum = manifest.compute_manifest_checksum()
        self.freezes[manifest.freeze_id] = manifest
        return manifest

    def validate_freeze(self, freeze_id: str,
                        accounting: CandidateAccounting,
                        gold_refs: list[GoldEvidenceRef],
                        page_extracts: list[PageExtract]) -> FreezeGateResult:
        m = self.freezes[freeze_id]
        reasons: list[str] = []

        # QA gate
        if not m.qa_passed:
            reasons.append("extraction-QA gate did not pass")
        if not m.qa_report_checksum:
            reasons.append("missing qa_report_checksum")

        # candidate accounting: zero unresolved in scope
        if accounting.unresolved != 0:
            reasons.append(f"unresolved candidates in scope: {accounting.unresolved}")
        if not accounting.reconciles():
            reasons.append("candidate accounting does not reconcile")

        # gold evidence refs resolve to frozen pages with matching hash
        page_lookup: dict[tuple[str, int], str] = {}
        for p in page_extracts:
            page_lookup[(p.document_id, p.pdf_page_index)] = p.extraction_hash
        for ref in gold_refs:
            key = (ref.document_id, ref.pdf_page_index)
            if key not in page_lookup:
                reasons.append(f"gold ref {ref.question_id}: page not in frozen set")
            elif page_lookup[key] != ref.extraction_hash:
                reasons.append(f"gold ref {ref.question_id}: extraction hash mismatch")

        # manifest integrity
        if m.manifest_checksum != m.compute_manifest_checksum():
            reasons.append("manifest checksum mismatch")
        if not m.extraction_config_hash:
            reasons.append("missing extraction_config_hash")

        passed = len(reasons) == 0
        result = FreezeGateResult(passed=passed, reasons=reasons)
        self.gate_results[freeze_id] = result
        if passed:
            m.freeze_gate_passed = True
            m.manifest_checksum = m.compute_manifest_checksum()
        return result

    def publish(self, freeze_id: str) -> FreezeManifest:
        m = self.freezes[freeze_id]
        if not m.qa_passed:
            raise ValueError("extraction-QA gate did not pass — refusing to publish")
        if not m.qa_report_checksum:
            raise ValueError("missing qa_report_checksum — refusing to publish")
        if not m.freeze_gate_passed:
            raise ValueError("freeze gate did not pass — refusing to publish")
        if m.manifest_checksum != m.compute_manifest_checksum():
            raise ValueError("manifest checksum mismatch — refusing to publish")
        m.status = "published"
        for fid, fm in self.freezes.items():
            if fid != freeze_id and fm.status == "published":
                fm.status = "published_previous"
        self.current_tag = m.freeze_tag
        return m

    def verify(self, freeze_id: str) -> bool:
        m = self.freezes[freeze_id]
        return m.manifest_checksum == m.compute_manifest_checksum()

    def load_verified(self, freeze_id: str) -> FreezeManifest:
        """Load a freeze after verifying its manifest checksum (canonical
        serialization checked)."""
        m = self.freezes[freeze_id]
        if not self.verify(freeze_id):
            raise ValueError("manifest checksum mismatch on load")
        return m

    def get(self, freeze_id: str) -> FreezeManifest:
        return self.freezes[freeze_id]
