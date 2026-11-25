"""Calculator tool: safe arithmetic wrapper around src.tables.arithmetic (no eval())."""

from __future__ import annotations

import ast
import operator
from decimal import Decimal
from typing import Any

from src.tables import arithmetic

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _walk(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _walk(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_walk(node.left), _walk(node.right))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate a numeric expression with Decimal via AST (never Python eval)."""
    tree = ast.parse(str(expression), mode="eval")
    value = _walk(tree.body)
    return {"expression": str(expression), "value": str(value), "status": "ok"}


def check_statement(text: str, *, company: str = "", checksum: str = "",
                    basis: str = "", statement: str = "BS",
                    start_pdf_page: int = 0, end_pdf_page: int = 0) -> dict[str, Any]:
    """Run the existing statement checker; returns serialisable result."""
    result = arithmetic.evaluate_statement(
        company, checksum, basis, statement, start_pdf_page, end_pdf_page, text,
    )
    return {
        "status": result.status,
        "statement": result.statement,
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail}
                   for c in result.checks],
    }


TOOLS = {"calculate": calculate, "check_statement": check_statement}
