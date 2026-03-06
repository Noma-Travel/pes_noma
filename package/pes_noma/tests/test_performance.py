"""
Performance tests: ranking pipeline should not add significant latency.

Run with:
  python -m pytest pes_noma/tests/test_performance.py -v
"""
import unittest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from noma.handlers.policy_filter import PolicyFilter
from noma.handlers.trip_option_ranker import TripOptionRanker


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_flight(index):
    hours = [6, 8, 10, 12, 14, 16, 18, 20, 22]
    airlines = ["LATAM", "GOL", "Azul", "Spirit", "Emirates", "TAP", "Delta", "United"]
    return {
        "price_amount": 200 + (index * 17) % 800,
        "cabin_class": "economy",
        "total_duration": f"{1 + index % 3} hr {10 + index % 50} min",
        "layovers": [] if index % 3 == 0 else [{"duration": "1 hr"}],
        "flights": [{
            "airline": airlines[index % len(airlines)],
            "departure_airport": {"id": "GRU", "time": f"2026-03-10T{hours[index % len(hours)]:02d}:00:00"},
            "arrival_airport": {"id": "GIG", "time": f"2026-03-10T{(hours[index % len(hours)] + 2) % 24:02d}:00:00"},
        }],
        "refundable": index % 5 == 0,
    }


def make_hotel(index):
    chains = ["Marriott", "Accor", "Hilton", "IHG", "Hyatt", "", "", ""]
    names = ["Grand Hotel", "Budget Inn", "Plaza Resort", "City Lodge", "Beach View", "Airport Inn"]
    return {
        "name": f"{names[index % len(names)]} {index}",
        "chain": chains[index % len(chains)],
        "city": "São Paulo" if index % 2 == 0 else "Rio de Janeiro",
        "amenities": ["Pool", "Gym", "Wifi"][:index % 4],
        "currentPrice": 100 + (index * 23) % 500,
        "stars": 2 + index % 4,
        "nights": 2,
    }


INTENT_WITH_PREFS = {
    "preferences": {
        "flight": {"max_budget": 900.0, "max_class": "business"},
        "hotel": {"max_daily_rate": 500.0, "max_stars": 5},
    },
    "constraints": {"enabled_services": ["flights", "hotels"]},
    "party": {
        "travelers": {"adults": 2},
        "traveler_ids": ["t-1", "t-2"],
        "travelers_by_id": {
            "t-1": {"preferences_id": "pref-1"},
            "t-2": {"preferences_id": "pref-2"},
        },
        "preferences_by_id": {
            "pref-1": {
                "preferred_airlines": ["LATAM", "GOL"],
                "preferred_travel_time": "Morning",
                "blocked_travel_time": ["Night"],
                "hotel_chain_pref": ["Marriott"],
                "hotel_facilities": ["Pool", "Gym"],
                "preferred_hotels_by_city": [
                    {"city": "São Paulo", "hotels": ["Grand Hotel"]},
                ],
            },
            "pref-2": {
                "preferred_airlines": ["Azul"],
                "preferred_travel_time": "Afternoon",
                "blocked_travel_time": [],
                "hotel_chain_pref": ["Hilton"],
                "hotel_facilities": ["Wifi"],
                "preferred_hotels_by_city": [],
            },
        },
    },
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPolicyFilterPerformance(unittest.TestCase):
    def test_filter_50_flights_50_hotels_under_50ms(self):
        flights = [make_flight(i) for i in range(50)]
        hotels = [make_hotel(i) for i in range(50)]

        pf = PolicyFilter()
        start = time.perf_counter()
        pf.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": flights,
            "hotel_options": hotels,
        })
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertLess(elapsed_ms, 50, f"PolicyFilter took {elapsed_ms:.1f}ms (limit: 50ms)")

    def test_filter_200_flights_under_100ms(self):
        flights = [make_flight(i) for i in range(200)]

        pf = PolicyFilter()
        start = time.perf_counter()
        pf.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": flights,
            "hotel_options": [],
        })
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertLess(elapsed_ms, 100, f"PolicyFilter took {elapsed_ms:.1f}ms (limit: 100ms)")


class TestRankerPerformance(unittest.TestCase):
    def test_rank_20_flights_20_hotels_under_200ms(self):
        """20×20 = 400 combos — typical real-world scenario."""
        flights = [make_flight(i) for i in range(20)]
        hotels = [make_hotel(i) for i in range(20)]

        ranker = TripOptionRanker()
        start = time.perf_counter()
        result = ranker.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": flights,
            "hotel_options": hotels,
            "ranking_policy": {},
        })
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertTrue(result["success"])
        self.assertGreater(len(result["output"]["bundles"]), 0)
        self.assertLess(elapsed_ms, 200, f"Ranker took {elapsed_ms:.1f}ms (limit: 200ms)")

    def test_rank_50_flights_only_under_100ms(self):
        flights = [make_flight(i) for i in range(50)]

        ranker = TripOptionRanker()
        start = time.perf_counter()
        result = ranker.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": flights,
            "hotel_options": [],
            "ranking_policy": {},
        })
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertTrue(result["success"])
        self.assertLess(elapsed_ms, 100, f"Ranker took {elapsed_ms:.1f}ms (limit: 100ms)")

    def test_rank_50_hotels_only_under_100ms(self):
        hotels = [make_hotel(i) for i in range(50)]

        ranker = TripOptionRanker()
        start = time.perf_counter()
        result = ranker.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": [],
            "hotel_options": hotels,
            "ranking_policy": {},
        })
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertTrue(result["success"])
        self.assertLess(elapsed_ms, 100, f"Ranker took {elapsed_ms:.1f}ms (limit: 100ms)")


class TestFullPipelinePerformance(unittest.TestCase):
    def test_full_pipeline_20x20_under_300ms(self):
        """End-to-end: PolicyFilter + TripOptionRanker with 20 flights × 20 hotels."""
        flights = [make_flight(i) for i in range(20)]
        hotels = [make_hotel(i) for i in range(20)]

        pf = PolicyFilter()
        ranker = TripOptionRanker()

        start = time.perf_counter()

        filtered = pf.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": flights,
            "hotel_options": hotels,
        })
        ranked = ranker.run({
            "intent": INTENT_WITH_PREFS,
            "flight_options": filtered["output"]["flight_options"],
            "hotel_options": filtered["output"]["hotel_options"],
            "ranking_policy": {},
        })

        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertTrue(ranked["success"])
        self.assertGreater(len(ranked["output"]["bundles"]), 0)
        self.assertLess(elapsed_ms, 300, f"Full pipeline took {elapsed_ms:.1f}ms (limit: 300ms)")

    def test_no_prefs_same_speed(self):
        """Without preferences, pipeline should be equally fast or faster."""
        flights = [make_flight(i) for i in range(20)]
        hotels = [make_hotel(i) for i in range(20)]
        intent_no_prefs = {
            "preferences": INTENT_WITH_PREFS["preferences"],
            "constraints": {},
            "party": {"travelers": {"adults": 1}},
        }

        pf = PolicyFilter()
        ranker = TripOptionRanker()

        start = time.perf_counter()
        filtered = pf.run({
            "intent": intent_no_prefs,
            "flight_options": flights,
            "hotel_options": hotels,
        })
        ranker.run({
            "intent": intent_no_prefs,
            "flight_options": filtered["output"]["flight_options"],
            "hotel_options": filtered["output"]["hotel_options"],
            "ranking_policy": {},
        })
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertLess(elapsed_ms, 300, f"No-prefs pipeline took {elapsed_ms:.1f}ms (limit: 300ms)")


if __name__ == '__main__':
    unittest.main(verbosity=2)
