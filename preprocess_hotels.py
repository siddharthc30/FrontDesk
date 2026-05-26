"""
Preprocessing script for Hotel_Reviews.csv.

Reads the raw Booking.com review dataset, deduplicates to unique hotels,
randomly samples 1000, parses city + country from the address field, and
writes the resulting DataFrame to hotels_sample.csv.

Run with:
    .venv/bin/python preprocess_hotels.py
"""

import re
import pandas as pd


RAW_CSV = "Hotel_Reviews.csv"
OUT_CSV = "hotels_sample.csv"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def parse_city_country(address: str) -> tuple[str, str]:
    """
    Parse city and country from Hotel_Address strings like:
        's Gravesandestraat 55 Oost 1092 AA Amsterdam Netherlands'
        '57 59 Welbeck Street Westminster Borough London W1G 9BL United Kingdom'
        '3 Rue Christine 6th arr 75006 Paris France'
        'Schottengasse 1 1010 Vienna Austria'

    Known countries (multi-word handled explicitly):
        Netherlands, United Kingdom, France, Austria, Spain, Italy

    Strategy:
        1. Strip the country suffix (handling "United Kingdom" as two words).
        2. Find the last postcode-like token(s) in what remains.
           - UK:          two-part alphanumeric  e.g. 'W1G 9BL'
           - France/Italy/Spain: 5-digit numeric e.g. '75006'
           - Netherlands: 4-digit + 2-letter     e.g. '1092 AA'
           - Austria:     4-digit numeric         e.g. '1010'
        3. Everything after the postcode block (before the country) is the city.
        4. Normalize French districts: strip leading '75NNN '/ '1300N ' etc.
           to keep just the final city word (e.g. 'Paris', 'Marseille').
    """
    address = address.strip()
    parts = address.split()

    # --- 1. Identify country ---
    if len(parts) >= 2 and " ".join(parts[-2:]) == "United Kingdom":
        country = "United Kingdom"
        body = parts[:-2]
    else:
        country = parts[-1]
        body = parts[:-1]

    # --- 2. Find last postcode-like token(s) in body (right-to-left) ---
    # Patterns (checked in order of specificity):
    #   a) Two-token UK postcode:    alpha(2-4) + alpha-digit(3)  e.g. 'W1G 9BL'
    #   b) Two-token NL postcode:    4-digit + 2-alpha             e.g. '1092 AA'
    #   c) Single 5-digit postcode:  France / Spain / Italy        e.g. '75006'
    #   d) Single 4-digit postcode:  Austria                       e.g. '1010'

    uk_part1 = re.compile(r'^[A-Z]{1,2}\d[A-Z0-9]?$', re.IGNORECASE)
    uk_part2 = re.compile(r'^\d[A-Z]{2}$', re.IGNORECASE)
    nl_part2 = re.compile(r'^[A-Z]{2}$', re.IGNORECASE)
    digit4 = re.compile(r'^\d{4}$')
    digit5 = re.compile(r'^\d{5}$')

    postcode_start = None  # index of FIRST token of the postcode block
    postcode_end = None    # index of LAST  token of the postcode block
    i = len(body) - 1
    while i >= 0:
        tok = body[i]
        # UK two-part (city is BEFORE the postcode)
        if i >= 1 and uk_part1.match(body[i - 1]) and uk_part2.match(tok):
            postcode_start = i - 1
            postcode_end = i
            break
        # NL two-part: 4-digit + 2-alpha (city is AFTER the postcode)
        if i >= 1 and digit4.match(body[i - 1]) and nl_part2.match(tok):
            postcode_start = i - 1
            postcode_end = i
            break
        # 5-digit (France/Spain/Italy) — city AFTER
        if digit5.match(tok):
            postcode_start = postcode_end = i
            break
        # 4-digit (Austria) — city AFTER
        if digit4.match(tok):
            postcode_start = postcode_end = i
            break
        i -= 1

    if postcode_end is not None:
        if country == "United Kingdom":
            # UK: city is the last alphabetic word immediately before the postcode
            pre = body[:postcode_start]
            city_tokens = [pre[-1]] if pre else []
        else:
            city_tokens = body[postcode_end + 1:]
    else:
        # Fallback: last token of body
        city_tokens = [body[-1]] if body else []

    city = " ".join(city_tokens)

    # --- 3. Normalize French arrondissement labels ---
    # e.g. "75006 Paris" → body may produce city_tokens = ['Paris'] already;
    # but sometimes "75008 Paris Ile De France" → strip the numeric prefix part
    city = re.sub(r'^\d{4,5}\s+', '', city).strip()

    return city, country


def load_unique_hotels(path: str) -> pd.DataFrame:
    """
    Read the raw CSV and collapse multiple reviews per hotel into one row,
    keeping the hotel-level fields: name, address, lat, lng, avg_score,
    total_reviews.
    """
    df = pd.read_csv(path, usecols=[
        "Hotel_Name",
        "Hotel_Address",
        "lat",
        "lng",
        "Average_Score",
        "Total_Number_of_Reviews",
    ])

    # Strip whitespace from text columns
    df["Hotel_Name"] = df["Hotel_Name"].str.strip()
    df["Hotel_Address"] = df["Hotel_Address"].str.strip()

    # Each hotel appears many times (once per review). Keep one row per
    # unique (name, address) pair; all hotel-level fields are constant across
    # reviews so first() is safe.
    hotels = (
        df.groupby(["Hotel_Name", "Hotel_Address"], as_index=False)
        .agg(
            latitude=("lat", "first"),
            longitude=("lng", "first"),
            avg_score=("Average_Score", "first"),
            total_reviews=("Total_Number_of_Reviews", "first"),
        )
        .rename(columns={"Hotel_Name": "name", "Hotel_Address": "address"})
    )

    return hotels


def main() -> None:
    print(f"Loading {RAW_CSV} ...")
    hotels = load_unique_hotels(RAW_CSV)
    print(f"  Unique hotels found: {len(hotels)}")

    # Drop rows with missing coordinates or name
    before = len(hotels)
    hotels = hotels.dropna(subset=["name", "latitude", "longitude"])
    dropped = before - len(hotels)
    if dropped:
        print(f"  Dropped {dropped} rows with missing name/coords")

    # Sample
    n = min(SAMPLE_SIZE, len(hotels))
    sample = hotels.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Sampled {n} hotels (seed={RANDOM_SEED})")

    # Parse city and country from address
    parsed = sample["address"].apply(parse_city_country)
    sample["city"] = parsed.apply(lambda t: t[0])
    sample["country"] = parsed.apply(lambda t: t[1])

    # Reorder columns to match the target schema
    sample = sample[[
        "name",
        "address",
        "city",
        "country",
        "latitude",
        "longitude",
        "avg_score",
        "total_reviews",
    ]]

    # Preview
    print("\nSample (first 5 rows):")
    print(sample.head().to_string(index=False))
    print(f"\nCity value counts (top 10):\n{sample['city'].value_counts().head(10)}")
    print(f"\nCountry value counts:\n{sample['country'].value_counts()}")

    # Diagnostic: show one address per country to verify parsing
    print("\nParsing check (one per country):")
    for country, grp in sample.groupby("country"):
        row = grp.iloc[0]
        print(f"  [{country}]  city='{row['city']}'  | {row['address']}")

    # Save
    sample.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(sample)} hotels → {OUT_CSV}")


if __name__ == "__main__":
    main()
