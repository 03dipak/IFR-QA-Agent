"""Tables as first-class records — data models.

Implements lld_tables.txt section 3: TableRecord, TableColumn, TableRow,
TableCell, FootnoteAssertion, TableContinuation, RowUnitSignal, and the
closed enumerations. Canonical owner of these types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Closed enumerations (lld_tables.txt section 3)
# ---------------------------------------------------------------------------


class StatementKind(StrEnum):
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    EQUITY = "equity"
    SEGMENT = "segment"
    NOTES = "notes"
    OTHER = "other"
    UNCLEAR = "unclear"


class ReportingBasis(StrEnum):
    CONSOLIDATED = "consolidated"
    STANDALONE = "standalone"
    PARENT_ONLY = "parent-only"
    UNCLEAR = "unclear"


class AuditStatus(StrEnum):
    AUDITED = "audited"
    UNAUDITED = "unaudited"
    REVIEWED = "reviewed"
    NOT_APPLICABLE = "not_applicable"
    UNCLEAR = "unclear"


class ColumnKind(StrEnum):
    REPORTING_PERIOD = "reporting_period"
    NOTE_REFERENCE = "note_reference"
    PERCENTAGE = "percentage"
    VARIANCE = "variance"
    OTHER = "other"
    UNCLEAR = "unclear"


class RowKind(StrEnum):
    ITEM = "item"
    SUBTOTAL = "subtotal"
    TOTAL = "total"
    OTHER = "other"


class CellKind(StrEnum):
    VALUE = "value"
    NIL = "nil"
    NOT_APPLICABLE = "not_applicable"
    BELOW_ROUNDING = "below_rounding"
    MISSING = "missing"
    UNCLEAR = "unclear"


class ScopeType(StrEnum):
    TABLE = "table"
    ROW = "row"
    COLUMN = "column"
    CELL = "cell"
    UNRESOLVED = "unresolved"


class RelationshipType(StrEnum):
    CONFIRMED_CONTINUATION = "confirmed_continuation"
    POSSIBLE_CONTINUATION = "possible_continuation"
    UNRELATED = "unrelated"


class ReviewStatus(StrEnum):
    AUTO = "auto"
    QA_PASSED = "qa_passed"
    QA_FAILED = "qa_failed"
    MANUAL = "manual"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    QA_FAILED = "qa_failed"


class PeriodResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class SignalType(StrEnum):
    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    FACE_VALUE = "face_value"
    EXPLICIT_UNIT = "explicit_unit"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class RowUnitSignal:
    signal_type: SignalType
    printed_text: str
    source: str  # row_label | inline_suffix | footnote
    confidence: float = 0.0


@dataclass
class TableCell:
    cell_id: str
    row_id: str
    column_id: str
    raw: str
    is_bracket_negative: bool = False
    cell_kind: CellKind = CellKind.VALUE
    footnote_refs: list[str] = field(default_factory=list)
    source_page_index: int = 0
    source_cell_coordinates: str = ""
    normalised_value: Any = None  # NormalisedValue (normalisation LLD)
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class TableRow:
    row_id: str
    row_ordinal: int
    row_label_raw: str
    row_label: str = ""
    row_label_normalized: str = ""
    row_kind: RowKind = RowKind.ITEM
    row_unit_signal: RowUnitSignal | None = None
    source_page_index: int = 0
    source_row_coordinates: str = ""
    cells: list[TableCell] = field(default_factory=list)


@dataclass
class TableColumn:
    column_id: str
    column_ordinal: int
    header_raw: str
    header_path: str = ""
    column_kind: ColumnKind = ColumnKind.REPORTING_PERIOD
    period_label: str | None = None
    period_start_date: str | None = None
    period_end_date: str | None = None
    duration_label: str | None = None  # 3M | 6M | 9M | 12M
    period_resolution_status: PeriodResolutionStatus = PeriodResolutionStatus.UNRESOLVED
    column_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class FootnoteAssertion:
    footnote_id: str
    text_raw: str
    scope_type: ScopeType = ScopeType.UNRESOLVED
    scope_target_ids: list[str] = field(default_factory=list)
    source_page_index: int = 0
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class TableContinuation:
    continuation_group_id: str
    previous_table_id: str
    next_table_id: str
    relationship_type: RelationshipType = RelationshipType.POSSIBLE_CONTINUATION
    confidence: float = 0.0
    review_status: ReviewStatus = ReviewStatus.AUTO


@dataclass
class TableRecord:
    table_logical_key: str
    table_extraction_id: str
    document_id: str
    extraction_run_id: str
    extraction_config_hash: str
    freeze_version: str
    page_start: int
    page_end: int
    printed_labels: list[str | None] = field(default_factory=list)
    caption: str = ""
    section_context: str = ""
    statement_kind: StatementKind = StatementKind.UNCLEAR
    reporting_basis: ReportingBasis = ReportingBasis.UNCLEAR
    audit_status: AuditStatus = AuditStatus.UNCLEAR
    unit_header: str = ""
    columns: list[TableColumn] = field(default_factory=list)
    rows: list[TableRow] = field(default_factory=list)
    footnote_assertions: list[FootnoteAssertion] = field(default_factory=list)
    extraction_confidence: float = 0.0
    review_status: ReviewStatus = ReviewStatus.AUTO
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    bbox: tuple[float, float, float, float] | None = None

    def validate_invariants(self) -> list[str]:
        problems: list[str] = []
        if self.page_end < self.page_start:
            problems.append("page_end < page_start")
        if self.statement_kind is None:
            problems.append("statement_kind unset")
        if self.reporting_basis is None:
            problems.append("reporting_basis unset")
        # every cell on exactly one page in [page_start, page_end]
        for row in self.rows:
            for cell in row.cells:
                outside = (cell.source_page_index < self.page_start
                           or cell.source_page_index > self.page_end)
                if outside:
                    problems.append(
                        f"cell {cell.cell_id} page {cell.source_page_index} outside span"
                    )
        # every cell resolves to exactly one column_id
        col_ids = {c.column_id for c in self.columns}
        for row in self.rows:
            for cell in row.cells:
                if cell.column_id not in col_ids:
                    problems.append(f"cell {cell.cell_id} references unknown column")
        return problems

    @staticmethod
    def make_logical_key(document_id: str, statement_kind: str,
                        section_context: str, header_signature: str) -> str:
        payload = f"{document_id}|{statement_kind}|{section_context}|{header_signature}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def make_extraction_id(logical_key: str, extraction_config_hash: str,
                           content_hash: str) -> str:
        payload = f"{logical_key}|{extraction_config_hash}|{content_hash}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class TableQAReference:
    reference_id: str
    document_id: str
    pdf_page_index: int
    reference_table_id: str
    reference_rows: list[str] = field(default_factory=list)
    reference_columns: list[str] = field(default_factory=list)
    reference_cells: list[str] = field(default_factory=list)
    reference_units: list[str] = field(default_factory=list)
    reference_footnotes: list[str] = field(default_factory=list)
    reviewer_id: str = ""
    review_status: str = "auto"
    annotation_version: str = "v1"


@dataclass
class RestatementRecord:
    restatement_id: str
    source_document_id: str
    source_table_id: str
    source_row_id: str
    source_cell_id: str
    restating_document_id: str
    restating_table_id: str
    restating_row_id: str
    restating_cell_id: str
    original_value_as_printed: str
    restated_value_as_printed: str
    reporting_period: str
    reason_as_printed: str = ""
    relationship_confidence: float = 0.0
    verification_status: str = "unverified"
