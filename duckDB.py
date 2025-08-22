# quick_check.py
import duckdb; con=duckdb.connect("data/wallenstein.duckdb")
print(con.execute("SELECT COUNT(*) FROM stocks").fetchone())
print(con.execute("SELECT * FROM stocks ORDER BY date DESC LIMIT 5").fetchdf())
