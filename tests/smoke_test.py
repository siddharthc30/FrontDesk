"""Phase 0 smoke test — verifies DB connection, haversine, read-only, and data dictionary."""
from core.db import get_connection, get_data_dictionary, get_schema_info

EXPECTED_COLUMNS = {
    "id", "name", "address", "city", "country",
    "latitude", "longitude", "avg_score", "total_reviews",
    "price_per_night",
    "has_wifi", "has_pool", "has_gym", "has_sauna",
    "has_restaurant", "has_room_service", "has_lounge", "has_event_space",
}

conn = get_connection("data/hotels.db")
schema = get_schema_info(conn)

print("Tables:", list(schema.keys()))
col_names = {col["name"] for col in schema["hotels"]["columns"]}
print("Columns:", sorted(col_names))
print("Row count:", schema["hotels"]["row_count"])
sample = schema["hotels"]["sample_rows"][0]
print(f"Sample row: name={sample['name']!r}, price={sample['price_per_night']}, has_pool={sample['has_pool']}")

missing = EXPECTED_COLUMNS - col_names
assert not missing, f"Missing columns: {missing}"
print("All expected columns present: OK")

result = conn.execute("SELECT haversine(51.5074, -0.1278, 48.8566, 2.3522)").fetchone()
print(f"\nLondon -> Paris: {result[0]:.1f} km  (expect ~343)")

result = conn.execute("SELECT haversine(48.8566, 2.3522, 48.8566, 2.3522)").fetchone()
print(f"Paris -> Paris: {result[0]}  (expect 0.0)")

try:
    conn.execute("INSERT INTO hotels (name, latitude, longitude) VALUES ('test', 0, 0)")
    print("\nERROR: write succeeded — connection is NOT read-only!")
except Exception as e:
    print(f"\nRead-only confirmed: {e}")

print("\n--- Data Dictionary ---")
print(get_data_dictionary())

conn.close()
