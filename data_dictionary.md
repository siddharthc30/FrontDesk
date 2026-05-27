Table: hotels (1000 rows)
Columns:
- id: INTEGER, auto-incrementing primary key
- name: TEXT, the hotel's name
- address: TEXT, full street address
- city: TEXT, city name (values: London, Paris, Barcelona, Milan, Vienna, Amsterdam)
- country: TEXT, country name (values: United Kingdom, France, Spain, Italy, Austria, Netherlands)
- latitude: REAL, geographic latitude
- longitude: REAL, geographic longitude
- avg_score: REAL, average guest review score on a 1.0-10.0 scale
- total_reviews: INTEGER, total number of guest reviews
- price_per_night: INTEGER, representative nightly price in USD (range: 80-465)
- has_wifi: INTEGER (0 or 1), whether the hotel has Wi-Fi
- has_pool: INTEGER (0 or 1), whether the hotel has a pool
- has_gym: INTEGER (0 or 1), whether the hotel has a gym
- has_sauna: INTEGER (0 or 1), whether the hotel has a sauna
- has_restaurant: INTEGER (0 or 1), whether the hotel has a restaurant
- has_room_service: INTEGER (0 or 1), whether the hotel offers room service
- has_lounge: INTEGER (0 or 1), whether the hotel has a lounge
- has_event_space: INTEGER (0 or 1), whether the hotel has event space

Available SQL function:
- haversine(lat1, lon1, lat2, lon2): returns distance in km between two coordinates

NOTE: Amenity columns are 0/1 integers (1 = has the amenity, 0 = does not).
Price is a representative nightly rate in USD, not a live booking price.
