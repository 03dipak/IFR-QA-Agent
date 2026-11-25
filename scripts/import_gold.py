import json
import re
from pathlib import Path

TEAM_GOLD = Path("data/eval/datasets/gold_v2.jsonl")
EXISTING_GOLD = Path("eval/gold.jsonl")


def _answer_values(text: object) -> list[str]:
    return re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(text or ""))


def _map_row(row: dict) -> dict:
    company_ids = [str(row["company_id"])] if row.get("company_id") else []
    expected_values = [str(row.get("expected_value") or "")]
    expected_values.extend(_answer_values(row.get("expected_answer") or ""))
    expected_values.extend(_answer_values(row.get("answer_text") or ""))
    page = None
    for evidence in row.get("evidence", []):
        page = evidence.get("pdf_page")
        if page is not None:
            break
    unit = row.get("expected_unit") or row.get("expected_unit_raw") or "unit not printed"
    return {
        "id": str(row.get("id") or row.get("row_id") or ""),
        "question": str(row.get("question_text") or row.get("question") or ""),
        "company_ids": company_ids,
        "period": str(row.get("fiscal_year") or row.get("period") or ""),
        "basis": str(row.get("variant") or ""),
        "unit": str(unit),
        "expected_values": sorted({v for v in expected_values if v}),
        "expected_winner": str(row.get("company_id") or ""),
        "page": page,
        "abstain": str(row.get("expected_behaviour") or "").lower() in {
            "abstain", "insufficient", "abstention"
        },
    }


def main() -> int:
    team_rows = [
        json.loads(line)
        for line in TEAM_GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapped = [row for row in (_map_row(r) for r in team_rows) if row["id"]]
    seen = {row["id"] for row in mapped}
    existing = [
        json.loads(line)
        for line in EXISTING_GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [row for row in existing if row.get("id") not in seen]
    final = mapped + kept
    EXISTING_GOLD.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
        encoding="utf-8",
    )
    missing_page = sum(1 for row in final if not row.get("page"))
    missing_unit = sum(1 for row in final if not row.get("unit"))
    print(f"imported {len(mapped)} team rows; kept {len(kept)} existing rows; total {len(final)}")
    print(f"missing_page={missing_page} missing_unit={missing_unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
