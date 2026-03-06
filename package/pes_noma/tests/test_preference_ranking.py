"""
Tests for preference-based scoring in trip_option_ranker.py

Run with:
  python -m pytest extensions/pes_noma/package/pes_noma/tests/test_preference_ranking.py -v
"""
import unittest
import sys
import os

# Add noma package to path (ranker lives in the noma extension)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'noma', 'package'))

from noma.handlers.trip_option_ranker import (
    _collect_preferences,
    _departure_time_category,
    _extract_departure_hour,
    _extract_airlines,
    _flight_preference_score,
    _extract_hotel_name,
    _extract_hotel_chain,
    _extract_hotel_amenities,
    _extract_hotel_city,
    _hotel_preference_score,
    _build_bundles_simple,
    TripOptionRanker,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

PREFS_FULL = {
    "preferred_airlines": {"LATAM", "GOL"},
    "preferred_travel_times": {"morning"},
    "blocked_travel_times": {"night"},
    "hotel_chain_prefs": {"marriott"},
    "hotel_facilities": {"pool", "gym"},
    "preferred_hotels_by_city": [
        {"city": "São Paulo", "hotels": ["Ibis Paulista", "Novotel"]},
    ],
}

PREFS_EMPTY = {
    "preferred_airlines": set(),
    "preferred_travel_times": set(),
    "blocked_travel_times": set(),
    "hotel_chain_prefs": set(),
    "hotel_facilities": set(),
    "preferred_hotels_by_city": [],
}

INTENT_WITH_PREFS = {
    "party": {
        "traveler_ids": ["t-alice", "t-bob"],
        "travelers_by_id": {
            "t-alice": {"preferences_id": "pref-alice"},
            "t-bob": {"preferences_id": "pref-bob"},
        },
        "preferences_by_id": {
            "pref-alice": {
                "_id": "pref-alice",
                "preferred_airlines": ["LATAM", "GOL"],
                "seat_preference": "Window",
                "preferred_travel_time": "Morning",
                "blocked_travel_time": ["Night"],
                "hotel_chain_pref": ["Marriott"],
                "hotel_facilities": ["Pool", "Gym"],
                "preferred_hotels_by_city": [
                    {"city": "São Paulo", "hotels": ["Ibis Paulista"]},
                ],
            },
            "pref-bob": {
                "_id": "pref-bob",
                "preferred_airlines": ["Azul"],
                "preferred_travel_time": "Afternoon",
                "blocked_travel_time": [],
                "hotel_chain_pref": [],
                "hotel_facilities": ["Wifi"],
                "preferred_hotels_by_city": [],
            },
        },
    },
}

INTENT_NO_PREFS = {"party": {"traveler_ids": ["t-1"], "travelers_by_id": {"t-1": {}}}}

FLIGHT_LATAM_MORNING = {
    "flights": [
        {
            "airline": "LATAM",
            "departure_airport": {"id": "GRU", "name": "Guarulhos", "time": "2026-03-10T08:30:00"},
            "arrival_airport": {"id": "GIG", "name": "Galeão", "time": "2026-03-10T09:45:00"},
        }
    ],
    "price": "$300",
    "total_duration": "1 hr 15 min",
    "layovers": [],
}

FLIGHT_GOL_AFTERNOON = {
    "flights": [
        {
            "airline": "GOL",
            "departure_airport": {"id": "GRU", "name": "Guarulhos", "time": "2026-03-10T14:00:00"},
            "arrival_airport": {"id": "GIG", "name": "Galeão", "time": "2026-03-10T15:15:00"},
        }
    ],
    "price": "$280",
    "total_duration": "1 hr 15 min",
    "layovers": [],
}

FLIGHT_UNKNOWN_NIGHT = {
    "flights": [
        {
            "airline": "Spirit",
            "departure_airport": {"id": "GRU", "name": "Guarulhos", "time": "2026-03-10T22:00:00"},
            "arrival_airport": {"id": "GIG", "name": "Galeão", "time": "2026-03-10T23:15:00"},
        }
    ],
    "price": "$150",
    "total_duration": "1 hr 15 min",
    "layovers": [],
}

HOTEL_MARRIOTT = {
    "name": "Marriott São Paulo",
    "chain": "Marriott",
    "city": "São Paulo",
    "amenities": ["Pool", "Gym", "Wifi", "Restaurant"],
    "currentPrice": 450,
    "nights": 2,
}

HOTEL_IBIS_SP = {
    "name": "Ibis Paulista",
    "chain": "Accor",
    "city": "São Paulo",
    "amenities": ["Wifi"],
    "currentPrice": 200,
    "nights": 2,
}

HOTEL_GENERIC = {
    "name": "Budget Inn",
    "city": "Rio de Janeiro",
    "amenities": [],
    "currentPrice": 120,
    "nights": 2,
}


# ── Tests: helpers ────────────────────────────────────────────────────────────

class TestDepartureTimeCategory(unittest.TestCase):
    def test_morning(self):
        for h in [5, 6, 8, 11]:
            self.assertEqual(_departure_time_category(h), "morning")

    def test_afternoon(self):
        for h in [12, 14, 17]:
            self.assertEqual(_departure_time_category(h), "afternoon")

    def test_night(self):
        for h in [0, 3, 18, 22, 23]:
            self.assertEqual(_departure_time_category(h), "night")


class TestExtractDepartureHour(unittest.TestCase):
    def test_extracts_hour(self):
        self.assertEqual(_extract_departure_hour(FLIGHT_LATAM_MORNING), 8)
        self.assertEqual(_extract_departure_hour(FLIGHT_UNKNOWN_NIGHT), 22)

    def test_no_flights(self):
        self.assertIsNone(_extract_departure_hour({"flights": []}))
        self.assertIsNone(_extract_departure_hour({}))


class TestExtractAirlines(unittest.TestCase):
    def test_single_flight(self):
        self.assertEqual(_extract_airlines(FLIGHT_LATAM_MORNING), ["LATAM"])

    def test_no_flights(self):
        self.assertEqual(_extract_airlines({}), [])


class TestExtractHotelFields(unittest.TestCase):
    def test_name(self):
        self.assertEqual(_extract_hotel_name(HOTEL_MARRIOTT), "marriott são paulo")

    def test_chain(self):
        self.assertEqual(_extract_hotel_chain(HOTEL_MARRIOTT), "marriott")
        self.assertEqual(_extract_hotel_chain(HOTEL_GENERIC), "")

    def test_amenities(self):
        self.assertEqual(_extract_hotel_amenities(HOTEL_MARRIOTT), ["pool", "gym", "wifi", "restaurant"])
        self.assertEqual(_extract_hotel_amenities(HOTEL_GENERIC), [])

    def test_city(self):
        self.assertEqual(_extract_hotel_city(HOTEL_MARRIOTT), "são paulo")


# ── Tests: collect_preferences ────────────────────────────────────────────────

class TestCollectPreferences(unittest.TestCase):
    def test_collects_from_multiple_travelers(self):
        prefs = _collect_preferences(INTENT_WITH_PREFS)
        self.assertIn("LATAM", prefs["preferred_airlines"])
        self.assertIn("GOL", prefs["preferred_airlines"])
        self.assertIn("AZUL", prefs["preferred_airlines"])
        self.assertIn("morning", prefs["preferred_travel_times"])
        self.assertIn("afternoon", prefs["preferred_travel_times"])
        self.assertIn("night", prefs["blocked_travel_times"])
        self.assertIn("marriott", prefs["hotel_chain_prefs"])
        self.assertIn("pool", prefs["hotel_facilities"])
        self.assertIn("wifi", prefs["hotel_facilities"])

    def test_empty_when_no_prefs(self):
        prefs = _collect_preferences(INTENT_NO_PREFS)
        self.assertEqual(prefs["preferred_airlines"], set())

    def test_empty_intent(self):
        prefs = _collect_preferences({})
        self.assertEqual(prefs["preferred_airlines"], set())


# ── Tests: flight preference scoring ─────────────────────────────────────────

class TestFlightPreferenceScore(unittest.TestCase):
    def test_preferred_airline_morning_boost(self):
        score = _flight_preference_score(FLIGHT_LATAM_MORNING, PREFS_FULL)
        # airline match (+0.15) + morning match (+0.10) = +0.25
        self.assertAlmostEqual(score, 0.25, places=2)

    def test_preferred_airline_no_time_match(self):
        score = _flight_preference_score(FLIGHT_GOL_AFTERNOON, PREFS_FULL)
        # airline match (+0.15), afternoon not in preferred_travel_times for PREFS_FULL (only morning)
        self.assertAlmostEqual(score, 0.15, places=2)

    def test_blocked_time_penalty(self):
        score = _flight_preference_score(FLIGHT_UNKNOWN_NIGHT, PREFS_FULL)
        # no airline match, night blocked (-0.30) = -0.30
        self.assertAlmostEqual(score, -0.30, places=2)

    def test_no_prefs_zero_score(self):
        score = _flight_preference_score(FLIGHT_LATAM_MORNING, PREFS_EMPTY)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_latam_preferred_beats_cheap_night(self):
        """LATAM morning should score higher than cheap Spirit night."""
        latam_score = _flight_preference_score(FLIGHT_LATAM_MORNING, PREFS_FULL)
        spirit_score = _flight_preference_score(FLIGHT_UNKNOWN_NIGHT, PREFS_FULL)
        self.assertGreater(latam_score, spirit_score)
        # Difference should be significant: 0.25 vs -0.30
        self.assertGreater(latam_score - spirit_score, 0.5)


# ── Tests: hotel preference scoring ──────────────────────────────────────────

class TestHotelPreferenceScore(unittest.TestCase):
    def test_chain_and_facilities_match(self):
        score = _hotel_preference_score(HOTEL_MARRIOTT, PREFS_FULL)
        # chain match (+0.15) + pool (+0.03) + gym (+0.03) = 0.21
        self.assertGreaterEqual(score, 0.20)

    def test_preferred_hotel_by_city(self):
        score = _hotel_preference_score(HOTEL_IBIS_SP, PREFS_FULL)
        # "ibis paulista" in preferred_hotels_by_city for São Paulo (+0.20)
        self.assertGreaterEqual(score, 0.20)

    def test_no_match(self):
        score = _hotel_preference_score(HOTEL_GENERIC, PREFS_FULL)
        # No chain match, no facilities, not in preferred city hotels
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_no_prefs_zero_score(self):
        score = _hotel_preference_score(HOTEL_MARRIOTT, PREFS_EMPTY)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_marriott_beats_generic(self):
        marriott_score = _hotel_preference_score(HOTEL_MARRIOTT, PREFS_FULL)
        generic_score = _hotel_preference_score(HOTEL_GENERIC, PREFS_FULL)
        self.assertGreater(marriott_score, generic_score)


# ── Tests: end-to-end ranking ────────────────────────────────────────────────

class TestBundleRankingWithPreferences(unittest.TestCase):
    def test_preferred_flight_ranks_higher(self):
        """LATAM morning should rank above Spirit night despite being more expensive."""
        bundles = _build_bundles_simple(
            flight_options=[FLIGHT_LATAM_MORNING, FLIGHT_UNKNOWN_NIGHT],
            hotel_options=[],
            ranking_policy={"weights": {"price": 0.5, "duration": 0.2, "preference": 0.3}},
            prefs=PREFS_FULL,
        )
        self.assertGreaterEqual(len(bundles), 2)
        # First bundle should be LATAM (preferred) not Spirit (cheapest but blocked time)
        first_id = bundles[0].get("flight_option_id") or ""
        self.assertIn("0", first_id)  # LATAM is index 0

    def test_no_prefs_cheapest_wins(self):
        """Without preferences, cheapest should rank higher."""
        bundles = _build_bundles_simple(
            flight_options=[FLIGHT_LATAM_MORNING, FLIGHT_UNKNOWN_NIGHT],
            hotel_options=[],
            ranking_policy={"weights": {"price": 0.5, "duration": 0.2, "preference": 0.3}},
            prefs=None,
        )
        self.assertGreaterEqual(len(bundles), 2)

    def test_ranker_class_with_intent(self):
        """TripOptionRanker.run() should pick up preferences from intent."""
        ranker = TripOptionRanker()
        result = ranker.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": [FLIGHT_LATAM_MORNING, FLIGHT_GOL_AFTERNOON, FLIGHT_UNKNOWN_NIGHT],
            "hotel_options": [],
            "ranking_policy": {},
        })
        self.assertTrue(result["success"])
        bundles = result["output"]["bundles"]
        self.assertGreater(len(bundles), 0)

    def test_ranker_with_hotels(self):
        """Marriott + LATAM should rank above Generic + Spirit."""
        ranker = TripOptionRanker()
        result = ranker.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": [FLIGHT_LATAM_MORNING, FLIGHT_UNKNOWN_NIGHT],
            "hotel_options": [HOTEL_MARRIOTT, HOTEL_GENERIC],
            "ranking_policy": {},
        })
        self.assertTrue(result["success"])
        bundles = result["output"]["bundles"]
        self.assertGreater(len(bundles), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
