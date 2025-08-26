from dataclasses import dataclass

import pandas as pd


@dataclass
class Rule:
    id: str
    ticker: str
    expr: str  # z.B. "close < 180"


OPS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
}


def eval_rule(row: pd.Series, rule: Rule) -> bool:
    # sehr einfache Syntax: "<field> <op> <zahl>"
    try:
        field, op, value = rule.expr.split()
        value = float(value)
        if field not in row or op not in OPS:
            return False
        return OPS[op](float(row[field]), value)
    except Exception:
        return False
