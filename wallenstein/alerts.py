from dataclasses import dataclass

import duckdb

from .config import settings


@dataclass
class Alert:
    id: int
    ticker: str
    op: str
    price: float
    active: bool


def _ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            ticker VARCHAR,
            op VARCHAR,
            price DOUBLE,
            active BOOLEAN
        )
        """
    )


def add_alert(ticker: str, op: str, price: float, db_path: str | None = None) -> int:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        alert_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM alerts").fetchone()[0]
        con.execute(
            "INSERT INTO alerts (id, ticker, op, price, active) VALUES (?, ?, ?, ?, TRUE)",
            [alert_id, ticker.upper(), op, float(price)],
        )
        return int(alert_id)
    finally:
        con.close()


def list_alerts(db_path: str | None = None) -> list[Alert]:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        rows = con.execute(
            "SELECT id, ticker, op, price, active FROM alerts ORDER BY id"
        ).fetchall()
        return [Alert(int(r[0]), r[1], r[2], float(r[3]), bool(r[4])) for r in rows]
    finally:
        con.close()


def delete_alert(alert_id: int, db_path: str | None = None) -> None:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        con.execute("DELETE FROM alerts WHERE id = ?", [alert_id])
    finally:
        con.close()


def _set_active(alert_id: int, active: bool, db_path: str | None = None) -> None:
    db_path = db_path or settings.WALLENSTEIN_DB_PATH
    con = duckdb.connect(db_path)
    try:
        _ensure_table(con)
        con.execute("UPDATE alerts SET active = ? WHERE id = ?", [active, alert_id])
    finally:
        con.close()


def activate_alert(alert_id: int, db_path: str | None = None) -> None:
    _set_active(alert_id, True, db_path)


def deactivate_alert(alert_id: int, db_path: str | None = None) -> None:
    _set_active(alert_id, False, db_path)
