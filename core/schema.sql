DROP TABLE IF EXISTS hotels;

CREATE TABLE hotels (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    address          TEXT,
    city             TEXT,
    country          TEXT,
    latitude         REAL NOT NULL,
    longitude        REAL NOT NULL,
    avg_score        REAL,
    rating           REAL GENERATED ALWAYS AS (avg_score / 2.0) STORED,
    total_reviews    INTEGER,
    price_per_night  INTEGER,
    has_wifi         INTEGER NOT NULL DEFAULT 0,
    has_pool         INTEGER NOT NULL DEFAULT 0,
    has_gym          INTEGER NOT NULL DEFAULT 0,
    has_sauna        INTEGER NOT NULL DEFAULT 0,
    has_restaurant   INTEGER NOT NULL DEFAULT 0,
    has_room_service INTEGER NOT NULL DEFAULT 0,
    has_lounge       INTEGER NOT NULL DEFAULT 0,
    has_event_space  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_hotels_city      ON hotels(city);
CREATE INDEX idx_hotels_country   ON hotels(country);
CREATE INDEX idx_hotels_avg_score ON hotels(avg_score);
