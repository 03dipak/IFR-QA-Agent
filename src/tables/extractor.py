"""Structured table extraction from frozen page text.

No OCR: tables are recovered from native text only. The implementation is
deliberately conservative and records uncertainty rather than guessing.
"""

from __future__ import annotations

import re

from .models import (
    ReportingBasis,
    RowKind,
    StatementKind,
    TableCell,
    TableColumn,
    TableRecord,
    TableRow,
)

_NUMERIC_RE = re.compile(r"\d")


def _statement_kind(text: str) -> StatementKind:
    low = text.lower()
    if "balance sheet" in low:
        return StatementKind.BALANCE_SHEET
    if "profit and loss" in low or "profit & loss" in low:
        return StatementKind.INCOME_STATEMENT
    if "cash flow" in low:
        return StatementKind.CASH_FLOW
    if "segment" in low:
        return StatementKind.SEGMENT
    return StatementKind.UNCLEAR


def _reporting_basis(text: str) -> ReportingBasis:
    low = text.lower()
    if "consolidated" in low:
        return ReportingBasis.CONSOLIDATED
    if "standalone" in low:
        return ReportingBasis.STANDALONE
    return ReportingBasis.UNCLEAR


def _unit_header(text: str) -> str:
    for line in text.splitlines():
        low = line.lower()
        if "rs in" in low or "rs. in" in low or "in crore" in low or "in lakh" in low:
            return line.strip()[:120]
    return ""


def _is_table_line(line: str) -> bool:
    tokens = line.split()
    return len(tokens) >= 3 and any(_NUMERIC_RE.search(tok) for tok in tokens)


def extract_tables_from_page(
    page_text: str,
    page_index: int,
    document_id: str,
    extraction_run_id: str,
    extraction_config_hash: str,
    freeze_version: str = "v1",
    printed_label=None,
) -> list[TableRecord]:
    """Extract conservative table regions from one page's text."""
    lines = [ln for ln in page_text.splitlines()]
    regions: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _is_table_line(line):
            current.append(line)
        else:
            if len(current) >= 3:
                regions.append(current)
            current = []
    if len(current) >= 3:
        regions.append(current)

    out: list[TableRecord] = []
    for region in regions:
        header_tokens = region[0].split()
        columns = [
            TableColumn(column_id=f"col-{i}", column_ordinal=i, header_raw=tok)
            for i, tok in enumerate(header_tokens)
        ]
        rows: list[TableRow] = []
        for ordinal, line in enumerate(region[1:], start=1):
            tokens = line.split()
            cells: list[TableCell] = []
            for i, tok in enumerate(tokens):
                if i >= len(columns):
                    break
                cells.append(TableCell(
                    cell_id=f"cell-{ordinal}-{i}",
                    row_id=f"row-{ordinal}",
                    column_id=columns[i].column_id,
                    raw=tok,
                    source_page_index=page_index,
                ))
            rows.append(TableRow(
                row_id=f"row-{ordinal}",
                row_ordinal=ordinal,
                row_label_raw=line,
                row_kind=RowKind.ITEM,
                source_page_index=page_index,
                cells=cells,
            ))
        logical_key = TableRecord.make_logical_key(
            document_id, _statement_kind(page_text).value, "", "|".join(header_tokens)
        )
        extraction_id = TableRecord.make_extraction_id(
            logical_key, extraction_config_hash, hashlib_sha256(region)
        )
        out.append(TableRecord(
            table_logical_key=logical_key,
            table_extraction_id=extraction_id,
            document_id=document_id,
            extraction_run_id=extraction_run_id,
            extraction_config_hash=extraction_config_hash,
            freeze_version=freeze_version,
            page_start=page_index,
            page_end=page_index,
            printed_labels=[printed_label],
            caption="",
            section_context="",
            statement_kind=_statement_kind(page_text),
            reporting_basis=_reporting_basis(page_text),
            unit_header=_unit_header(page_text),
            columns=columns,
            rows=rows,
            extraction_confidence=0.5,
        ))
    return out


def extract_tables(
    pages,
    document_id: str,
    extraction_run_id: str,
    extraction_config_hash: str,
    freeze_version: str = "v1",
) -> list[TableRecord]:
    out: list[TableRecord] = []
    for page in pages:
        out.extend(extract_tables_from_page(
            page.page_text,
            page.pdf_page_index,
            document_id,
            extraction_run_id,
            extraction_config_hash,
            freeze_version=freeze_version,
            printed_label=getattr(page, "printed_page_label", None),
        ))
    return out


def hashlib_sha256(value) -> str:
    import hashlib
    if isinstance(value, str):
        data = value.encode("utf-8")
    elif isinstance(value, list):
        data = "\n".join(value).encode("utf-8")
    else:
        data = str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
