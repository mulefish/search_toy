"""
Utility to print all rows from the search_item table.
"""

import argparse
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "db.sqlite3"


def fetch_rows(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM search_item ORDER BY name;")
        return cur.fetchall()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump search_item table contents.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Path to DB (default: {DEFAULT_DB})")
    args = parser.parse_args()

    rows = fetch_rows(args.db)
    if not rows:
        print(f"No rows found in search_item (DB: {args.db})")
        return

    print(f"search_item contents from {args.db}:")
    for row in rows:
        print(f"[{row['id']:02d}] {row['name']} ({row['category']}) -> {row['description']}")


if __name__ == "__main__":
    main()

