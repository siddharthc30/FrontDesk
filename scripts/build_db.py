"""
Build hotels.db from hotels_synthetic.csv.

Run:  .venv/bin/python -m scripts.build_db
"""

from core.load import DEFAULT_CSV, DEFAULT_DB, load_csv


def main() -> None:
    n = load_csv(DEFAULT_CSV, DEFAULT_DB)
    print(f"loaded {n} rows from {DEFAULT_CSV.name} → {DEFAULT_DB.name}")


if __name__ == "__main__":
    main()
