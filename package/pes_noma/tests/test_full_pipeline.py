"""
Tests for the full ranking pipeline: PolicyFilter (hard) → TripOptionRanker (soft + score)

Verifies that policy filtering and preference scoring work together correctly
when chained as they are in the real flow.

Run with:
  python -m pytest extensions/pes_noma/package/pes_noma/tests/test_full_pipeline.py -v
"""
import unittest
import sys
import os

# Add noma package to path (policy_filter and ranker live in the noma extension)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'noma', 'package'))

from noma.handlers.policy_filter import PolicyFilter
from noma.handlers.trip_option_ranker import (
    TripOptionRanker,
    _collect_preferences,
    _flight_preference_score,
    _hotel_preference_score,
    _build_bundles_simple,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

INTENT_FULL = {
    "preferences": {
        "flight": {"max_budget": 800.0, "max_class": "premium_economy"},
        "hotel": {"max_daily_rate": 400.0, "max_stars": 5},
    },
    "constraints": {
        "enabled_services": ["flights", "hotels"],
    },
    "party": {
        "travelers": {"adults": 1},
        "traveler_ids": ["t-alice"],
        "travelers_by_id": {
            "t-alice": {"preferences_id": "pref-alice"},
        },
        "preferences_by_id": {
            "pref-alice": {
                "_id": "pref-alice",
                "preferred_airlines": ["LATAM", "GOL"],
                "preferred_travel_time": "Morning",
                "blocked_travel_time": ["Night"],
                "hotel_chain_pref": ["Marriott"],
                "hotel_facilities": ["Pool", "Gym"],
                "preferred_hotels_by_city": [
                    {"city": "São Paulo", "hotels": ["Ibis Paulista"]},
                ],
            },
        },
    },
}

INTENT_NO_PREFS = {
    "preferences": {
        "flight": {"max_budget": 800.0},
        "hotel": {"max_daily_rate": 400.0},
    },
    "constraints": {},
    "party": {
        "travelers": {"adults": 1},
        "traveler_ids": ["t-bob"],
        "travelers_by_id": {"t-bob": {}},
    },
}

INTENT_VERY_RESTRICTIVE = {
    "preferences": {
        "flight": {"max_budget": 200.0, "max_class": "economy"},
        "hotel": {"max_daily_rate": 100.0, "max_stars": 3},
    },
    "constraints": {
        "enabled_services": ["flights", "hotels"],
    },
    "party": {
        "travelers": {"adults": 1},
        "traveler_ids": ["t-alice"],
        "travelers_by_id": {
            "t-alice": {"preferences_id": "pref-alice"},
        },
        "preferences_by_id": {
            "pref-alice": {
                "_id": "pref-alice",
                "preferred_airlines": ["LATAM"],
                "preferred_travel_time": "Morning",
                "blocked_travel_time": ["Night"],
                "hotel_chain_pref": ["Marriott"],
                "hotel_facilities": ["Pool"],
                "preferred_hotels_by_city": [],
            },
        },
    },
}

# ── Flight options ────────────────────────────────────────────────────────────

FLIGHT_LATAM_MORNING_CHEAP = {
    "price_amount": 350,
    "cabin_class": "economy",
    "total_duration": "1 hr 15 min",
    "layovers": [],
    "flights": [{
        "airline": "LATAM",
        "departure_airport": {"id": "GRU", "time": "2026-03-10T08:30:00"},
        "arrival_airport": {"id": "GIG", "time": "2026-03-10T09:45:00"},
    }],
}

FLIGHT_GOL_AFTERNOON = {
    "price_amount": 280,
    "cabin_class": "economy",
    "total_duration": "1 hr 15 min",
    "layovers": [],
    "flights": [{
        "airline": "GOL",
        "departure_airport": {"id": "GRU", "time": "2026-03-10T14:00:00"},
        "arrival_airport": {"id": "GIG", "time": "2026-03-10T15:15:00"},
    }],
}

FLIGHT_SPIRIT_NIGHT_CHEAPEST = {
    "price_amount": 150,
    "cabin_class": "economy",
    "total_duration": "1 hr 30 min",
    "layovers": [],
    "flights": [{
        "airline": "Spirit",
        "departure_airport": {"id": "GRU", "time": "2026-03-10T22:00:00"},
        "arrival_airport": {"id": "GIG", "time": "2026-03-10T23:30:00"},
    }],
}

FLIGHT_EMIRATES_FIRST_EXPENSIVE = {
    "price_amount": 3000,
    "cabin_class": "first",
    "total_duration": "1 hr 10 min",
    "layovers": [],
    "flights": [{
        "airline": "Emirates",
        "departure_airport": {"id": "GRU", "time": "2026-03-10T10:00:00"},
        "arrival_airport": {"id": "GIG", "time": "2026-03-10T11:10:00"},
    }],
}

FLIGHT_AZUL_BUSINESS = {
    "price_amount": 900,
    "cabin_class": "business",
    "total_duration": "1 hr 15 min",
    "layovers": [],
    "flights": [{
        "airline": "Azul",
        "departure_airport": {"id": "GRU", "time": "2026-03-10T09:00:00"},
        "arrival_airport": {"id": "GIG", "time": "2026-03-10T10:15:00"},
    }],
}

# ── Hotel options ─────────────────────────────────────────────────────────────

HOTEL_MARRIOTT_SP = {
    "name": "Marriott São Paulo",
    "chain": "Marriott",
    "city": "São Paulo",
    "amenities": ["Pool", "Gym", "Wifi", "Restaurant"],
    "currentPrice": 380,
    "stars": 5,
    "nights": 2,
}

HOTEL_IBIS_SP = {
    "name": "Ibis Paulista",
    "chain": "Accor",
    "city": "São Paulo",
    "amenities": ["Wifi"],
    "currentPrice": 180,
    "stars": 3,
    "nights": 2,
}

HOTEL_LUXURY_OVERPRICED = {
    "name": "Grand Luxury Palace",
    "chain": "Four Seasons",
    "city": "São Paulo",
    "amenities": ["Pool", "Spa", "Gym", "Restaurant", "Bar"],
    "currentPrice": 900,
    "stars": 5,
    "nights": 2,
}

HOTEL_BUDGET_RIO = {
    "name": "Budget Inn",
    "city": "Rio de Janeiro",
    "amenities": [],
    "currentPrice": 90,
    "stars": 2,
    "nights": 2,
}

ALL_FLIGHTS = [
    FLIGHT_LATAM_MORNING_CHEAP,
    FLIGHT_GOL_AFTERNOON,
    FLIGHT_SPIRIT_NIGHT_CHEAPEST,
    FLIGHT_EMIRATES_FIRST_EXPENSIVE,
    FLIGHT_AZUL_BUSINESS,
]

ALL_HOTELS = [
    HOTEL_MARRIOTT_SP,
    HOTEL_IBIS_SP,
    HOTEL_LUXURY_OVERPRICED,
    HOTEL_BUDGET_RIO,
]


# ── Helper: run full pipeline ─────────────────────────────────────────────────

def run_pipeline(intent, flights, hotels):
    """PolicyFilter → TripOptionRanker, returns (filter_output, ranker_output)."""
    pf = PolicyFilter()
    filtered = pf.run({
        "intent": intent,
        "flight_options": list(flights),
        "hotel_options": list(hotels),
    })

    ranker = TripOptionRanker()
    ranked = ranker.run({
        "intent": intent,
        "flight_options": filtered["output"]["flight_options"],
        "hotel_options": filtered["output"]["hotel_options"],
        "ranking_policy": {"weights": {"price": 0.4, "duration": 0.1, "preference": 0.3, "refundable": 0.1, "convenience": 0.1}},
    })

    return filtered["output"], ranked["output"]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPolicyThenRanking(unittest.TestCase):
    """Policy filters first, then ranker scores what remains."""

    def test_policy_removes_violations_before_ranking(self):
        filter_out, rank_out = run_pipeline(INTENT_FULL, ALL_FLIGHTS, ALL_HOTELS)

        self.assertEqual(filter_out["violations_count"], 3)
        remaining_flights = filter_out["flight_options"]
        remaining_hotels = filter_out["hotel_options"]
        self.assertEqual(len(remaining_flights), 3)
        self.assertEqual(len(remaining_hotels), 3)

        bundles = rank_out["bundles"]
        self.assertGreater(len(bundles), 0)

    def test_preferred_flight_ranks_high_after_filter(self):
        filter_out, rank_out = run_pipeline(INTENT_FULL, ALL_FLIGHTS, [])

        bundles = rank_out["bundles"]
        self.assertGreater(len(bundles), 0)

        first_bundle = bundles[0]
        self.assertIn("flt_seg0_0", first_bundle["flight_option_id"])

    def test_preferred_hotel_ranks_high_after_filter(self):
        filter_out, rank_out = run_pipeline(INTENT_FULL, [], ALL_HOTELS)

        bundles = rank_out["bundles"]
        self.assertGreater(len(bundles), 0)

        scores = {}
        prefs = _collect_preferences(INTENT_FULL)
        for h in filter_out["hotel_options"]:
            scores[h["name"]] = _hotel_preference_score(h, prefs)
        self.assertGreater(scores["Marriott São Paulo"], scores["Budget Inn"])

    def test_no_prefs_ranking_uses_price(self):
        filter_out, rank_out = run_pipeline(INTENT_NO_PREFS, ALL_FLIGHTS, [])

        bundles = rank_out["bundles"]
        self.assertGreater(len(bundles), 0)
        first_price = bundles[0]["estimated_total"]["amount"]
        self.assertEqual(first_price, 150.0)


class TestVeryRestrictivePolicy(unittest.TestCase):
    """When policy is very restrictive, most options are filtered out."""

    def test_most_options_filtered(self):
        filter_out, rank_out = run_pipeline(INTENT_VERY_RESTRICTIVE, ALL_FLIGHTS, ALL_HOTELS)

        self.assertEqual(len(filter_out["flight_options"]), 1)
        self.assertEqual(filter_out["flight_options"][0]["flights"][0]["airline"], "Spirit")

        self.assertEqual(len(filter_out["hotel_options"]), 1)
        self.assertEqual(filter_out["hotel_options"][0]["name"], "Budget Inn")

        bundles = rank_out["bundles"]
        self.assertEqual(len(bundles), 1)

    def test_preference_still_scored_on_surviving_options(self):
        """Even with strict policy, pref scores apply to what's left."""
        prefs = _collect_preferences(INTENT_VERY_RESTRICTIVE)

        spirit_score = _flight_preference_score(FLIGHT_SPIRIT_NIGHT_CHEAPEST, prefs)
        self.assertLess(spirit_score, 0)

        budget_score = _hotel_preference_score(HOTEL_BUDGET_RIO, prefs)
        self.assertAlmostEqual(budget_score, 0.0)


class TestPolicyNoFilterPlusPrefs(unittest.TestCase):
    """No policy rules → all options pass, preferences determine ranking."""

    def test_all_options_survive(self):
        intent_open = {
            "preferences": {},
            "constraints": {},
            "party": INTENT_FULL["party"],
        }
        filter_out, rank_out = run_pipeline(intent_open, ALL_FLIGHTS, ALL_HOTELS)

        self.assertEqual(filter_out["violations_count"], 0)
        self.assertEqual(len(filter_out["flight_options"]), 5)
        self.assertEqual(len(filter_out["hotel_options"]), 4)

        bundles = rank_out["bundles"]
        self.assertGreater(len(bundles), 0)


class TestEnabledServicesPlusPrefs(unittest.TestCase):
    """Disabled services are removed before ranking even considers them."""

    def test_flights_disabled_hotel_only_ranking(self):
        intent = {
            "preferences": {},
            "constraints": {"enabled_services": ["hotels"]},
            "party": INTENT_FULL["party"],
        }
        filter_out, rank_out = run_pipeline(intent, ALL_FLIGHTS, ALL_HOTELS)

        self.assertEqual(len(filter_out["flight_options"]), 0)
        self.assertEqual(len(filter_out["hotel_options"]), 4)

        bundles = rank_out["bundles"]
        self.assertGreater(len(bundles), 0)
        for b in bundles:
            self.assertIsNone(b.get("flight_option_id"))

    def test_hotels_disabled_flight_only_ranking(self):
        intent = {
            "preferences": {},
            "constraints": {"enabled_services": ["flights"]},
            "party": INTENT_FULL["party"],
        }
        filter_out, rank_out = run_pipeline(intent, ALL_FLIGHTS, ALL_HOTELS)

        self.assertEqual(len(filter_out["flight_options"]), 5)
        self.assertEqual(len(filter_out["hotel_options"]), 0)

        bundles = rank_out["bundles"]
        self.assertGreater(len(bundles), 0)
        for b in bundles:
            self.assertIsNone(b.get("hotel_option_id"))


class TestPreferenceVsPriceTradeoff(unittest.TestCase):
    """Preference weight should override pure price sorting."""

    def test_preferred_beats_cheapest(self):
        """LATAM morning ($350) should beat Spirit night ($150) with prefs enabled."""
        prefs = _collect_preferences(INTENT_FULL)
        bundles = _build_bundles_simple(
            flight_options=[FLIGHT_LATAM_MORNING_CHEAP, FLIGHT_SPIRIT_NIGHT_CHEAPEST],
            hotel_options=[],
            ranking_policy={"weights": {"price": 0.4, "duration": 0.1, "preference": 0.3, "refundable": 0.1, "convenience": 0.1}},
            prefs=prefs,
        )
        self.assertGreaterEqual(len(bundles), 2)
        self.assertIn("0", bundles[0]["flight_option_id"])

    def test_without_prefs_cheapest_wins(self):
        """Without preferences, $150 Spirit should beat $350 LATAM."""
        bundles = _build_bundles_simple(
            flight_options=[FLIGHT_LATAM_MORNING_CHEAP, FLIGHT_SPIRIT_NIGHT_CHEAPEST],
            hotel_options=[],
            ranking_policy={"weights": {"price": 0.5, "duration": 0.2, "preference": 0.3}},
            prefs=None,
        )
        self.assertGreaterEqual(len(bundles), 2)
        first_price = bundles[0]["estimated_total"]["amount"]
        self.assertEqual(first_price, 150.0)


class TestBundleCombinationScoring(unittest.TestCase):
    """Flight+hotel bundles combine both preference scores."""

    def test_best_combo_is_preferred_flight_plus_preferred_hotel(self):
        prefs = _collect_preferences(INTENT_FULL)

        latam_pref = _flight_preference_score(FLIGHT_LATAM_MORNING_CHEAP, prefs)
        spirit_pref = _flight_preference_score(FLIGHT_SPIRIT_NIGHT_CHEAPEST, prefs)
        marriott_pref = _hotel_preference_score(HOTEL_MARRIOTT_SP, prefs)
        budget_pref = _hotel_preference_score(HOTEL_BUDGET_RIO, prefs)

        best_combo = latam_pref + marriott_pref
        worst_combo = spirit_pref + budget_pref
        self.assertGreater(best_combo, worst_combo)
        self.assertGreater(best_combo - worst_combo, 0.4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
