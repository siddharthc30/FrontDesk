"""
Interactive test harness for the hotel NL search pipeline.
Run: python test_interactive.py
Optionally set USER_LAT and USER_LNG env vars for geo queries.

Example geo coordinates:
  London centre:    USER_LAT=51.5074  USER_LNG=-0.1278
  Paris centre:     USER_LAT=48.8566  USER_LNG=2.3522
  Barcelona centre: USER_LAT=41.3851  USER_LNG=2.1734
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from core.pipeline import ask  # noqa: E402  (must be after load_dotenv)


async def main() -> None:
    raw_lat = os.environ.get("USER_LAT", "0")
    raw_lng = os.environ.get("USER_LNG", "0")

    try:
        user_lat: float | None = float(raw_lat) or None
        user_lng: float | None = float(raw_lng) or None
    except ValueError:
        user_lat = user_lng = None

    print("=" * 60)
    print("Hotel NL Search — Interactive Test")
    if user_lat and user_lng:
        print(f"Location: {user_lat}, {user_lng}")
    else:
        print("No location set (set USER_LAT/USER_LNG for geo queries)")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            result = await ask(question, user_lat=user_lat, user_lng=user_lng)

            print(f"\n--- Path: {result.path} ---")
            if result.declined:
                print(f"Declined: {result.decline_reason}")
            else:
                print(f"Answer: {result.answer}")
                print(f"Insights: {result.insights}")
                if result.hotels:
                    print(f"Hotels returned: {len(result.hotels)}")
                    for h in result.hotels[:5]:
                        print(f"  • {h.name} ({h.city}) — score: {h.avg_score}")
                    if len(result.hotels) > 5:
                        print(f"  ... and {len(result.hotels) - 5} more")
            print(f"Query ran: {result.query_ran}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")


asyncio.run(main())
