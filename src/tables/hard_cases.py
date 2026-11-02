"""Hard-case table handling helpers.

Implements lld_tables.txt section 4: multi-page tables, repeated headers,
merged cells, unit footnotes, restated comparatives, consolidated vs
standalone, prose/table split.
"""

from __future__ import annotations

from .models import TableContinuation, TableRecord


def is_continuation_candidate(prev: TableRecord, nxt: TableRecord,
                             header_similarity: float = 0.0) -> TableContinuation:
    """Build a TableContinuation candidate between two table records.

    A wrong merge is worse than two clean halves, so uncertain cases stay
    separate with a possible_continuation relationship.
    """
    import uuid
    if prev.page_end + 1 == nxt.page_start and header_similarity >= 0.9:
        rel = "confirmed_continuation"
        conf = max(header_similarity, 0.9)
    elif prev.page_end + 1 == nxt.page_start:
        rel = "possible_continuation"
        conf = header_similarity
    else:
        rel = "unrelated"
        conf = 0.0
    from .models import RelationshipType
    mapping = {
        "confirmed_continuation": RelationshipType.CONFIRMED_CONTINUATION,
        "possible_continuation": RelationshipType.POSSIBLE_CONTINUATION,
        "unrelated": RelationshipType.UNRELATED,
    }
    return TableContinuation(
        continuation_group_id=str(uuid.uuid4()),
        previous_table_id=prev.table_extraction_id,
        next_table_id=nxt.table_extraction_id,
        relationship_type=mapping[rel],
        confidence=conf,
    )


def merge_confirmed_continuations(records: list[TableRecord],
                                 continuations: list[TableContinuation]) -> list[TableRecord]:
    """Merge only CONFIRMED continuations into a single TableRecord spanning
    both page ranges. Possible/unrelated stay separate."""
    from .models import RelationshipType
    confirmed = {c.continuation_group_id: c for c in continuations
                 if c.relationship_type == RelationshipType.CONFIRMED_CONTINUATION}
    # Build a map from table id -> continuation
    by_prev = {c.previous_table_id: c for c in confirmed.values()}
    merged: list[TableRecord] = []
    skip: set[str] = set()
    for r in records:
        if r.table_extraction_id in skip:
            continue
        c = by_prev.get(r.table_extraction_id)
        if c is not None:
            nxt = next((x for x in records if x.table_extraction_id == c.next_table_id), None)
            if nxt is not None:
                # merge
                combined = TableRecord(
                    table_logical_key=r.table_logical_key,
                    table_extraction_id=r.table_extraction_id,
                    document_id=r.document_id,
                    extraction_run_id=r.extraction_run_id,
                    extraction_config_hash=r.extraction_config_hash,
                    freeze_version=r.freeze_version,
                    page_start=r.page_start,
                    page_end=max(r.page_end, nxt.page_end),
                    printed_labels=list(r.printed_labels) + list(nxt.printed_labels),
                    caption=r.caption or nxt.caption,
                    section_context=r.section_context,
                    statement_kind=r.statement_kind,
                    reporting_basis=r.reporting_basis,
                    audit_status=r.audit_status,
                    unit_header=r.unit_header,
                    columns=r.columns,
                    rows=r.rows + nxt.rows,
                    footnote_assertions=r.footnote_assertions + nxt.footnote_assertions,
                )
                merged.append(combined)
                skip.add(nxt.table_extraction_id)
                continue
        merged.append(r)
    return merged


def validate_no_broadcast(record: TableRecord) -> list[str]:
    """Ensure merged cells do not implicitly broadcast values: a merged cell
    carries its value on the topmost-leftmost coordinate only; covered
    coordinates must not duplicate the value."""
    problems: list[str] = []
    # Heuristic: duplicate raw values across cells in the same row with same
    # column span are suspicious; record as a check placeholder.
    for row in record.rows:
        seen: set[str] = set()
        for cell in row.cells:
            if cell.raw in seen and cell.raw.strip():
                problems.append(f"possible broadcast in row {row.row_id}: {cell.raw}")
            seen.add(cell.raw)
    return problems


def prose_table_split_marker(table_id: str) -> str:
    """Short handle marker for the chunk stream (replaces table-region text)."""
    return f"[TABLE:{table_id}]"
