"""
Tests for preference_utils.py

Run with:
  python -m pytest extensions/pes_noma/package/pes_noma/tests/test_preference_utils.py -v
"""
import unittest
import sys
import os

# Add pes_noma package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pes_noma.utils.preference_utils import (
    get_traveler_preference,
    get_all_traveler_preferences,
    get_preferred_airlines,
    get_seat_preference,
    get_preferred_travel_time,
    get_blocked_travel_times,
    get_hotel_chain_preferences,
    get_hotel_floor_preference,
    get_hotel_facilities,
    get_preferred_hotels_by_city,
    get_preferred_hotels_for_city,
    is_smoker,
    get_special_needs,
    get_dietary_requirements,
    get_car_type_preference,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

PREF_ALICE = {
    '_id': 'pref-alice',
    'user_id': 'traveler-alice',
    'preferred_airlines': ['LATAM', 'GOL'],
    'seat_preference': 'Window',
    'preferred_travel_time': 'Morning',
    'blocked_travel_time': ['Night'],
    'hotel_chain_pref': ['Marriott'],
    'hotel_floor_pref': 'High',
    'hotel_facilities': ['Pool', 'Gym'],
    'is_smoker': False,
    'special_needs': 'Wheelchair access',
    'dietary_requirements': 'Vegan',
    'car_type_pref': 'SUV',
    'preferred_hotels_by_city': [
        {'city': 'São Paulo', 'hotels': ['Ibis SP', 'Novotel']},
        {'city': 'Rio de Janeiro', 'hotels': ['Windsor']},
    ],
}

PREF_BOB = {
    '_id': 'pref-bob',
    'user_id': 'traveler-bob',
    'preferred_airlines': ['Azul'],
    'seat_preference': 'Aisle',
    'preferred_travel_time': 'Afternoon',
    'blocked_travel_time': [],
    'hotel_chain_pref': [],
    'hotel_floor_pref': '',
    'hotel_facilities': [],
    'is_smoker': True,
    'special_needs': '',
    'dietary_requirements': 'Gluten-free',
    'car_type_pref': '',
    'preferred_hotels_by_city': [],
}

INTENT_TWO_TRAVELERS = {
    'party': {
        'traveler_ids': ['traveler-alice', 'traveler-bob'],
        'travelers_by_id': {
            'traveler-alice': {'policy_id': 'pol-1', 'preferences_id': 'pref-alice'},
            'traveler-bob':   {'policy_id': 'pol-2', 'preferences_id': 'pref-bob'},
        },
        'preferences_by_id': {
            'pref-alice': PREF_ALICE,
            'pref-bob':   PREF_BOB,
        },
    }
}

INTENT_NO_PREFS = {
    'party': {
        'traveler_ids': ['traveler-charlie'],
        'travelers_by_id': {
            'traveler-charlie': {'policy_id': 'pol-3'},
        },
        'preferences_by_id': {},
    }
}

INTENT_EMPTY = {}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGetTravelerPreference(unittest.TestCase):
    def test_returns_pref_for_known_traveler(self):
        pref = get_traveler_preference(INTENT_TWO_TRAVELERS, 'traveler-alice')
        self.assertEqual(pref['_id'], 'pref-alice')

    def test_returns_none_for_traveler_without_pref(self):
        pref = get_traveler_preference(INTENT_NO_PREFS, 'traveler-charlie')
        self.assertIsNone(pref)

    def test_returns_none_for_unknown_traveler(self):
        pref = get_traveler_preference(INTENT_TWO_TRAVELERS, 'traveler-unknown')
        self.assertIsNone(pref)

    def test_returns_none_for_empty_intent(self):
        pref = get_traveler_preference(INTENT_EMPTY, 'traveler-alice')
        self.assertIsNone(pref)


class TestGetAllTravelerPreferences(unittest.TestCase):
    def test_returns_all_travelers_with_prefs(self):
        result = get_all_traveler_preferences(INTENT_TWO_TRAVELERS)
        self.assertIn('traveler-alice', result)
        self.assertIn('traveler-bob', result)
        self.assertEqual(result['traveler-alice']['_id'], 'pref-alice')

    def test_excludes_travelers_without_pref(self):
        result = get_all_traveler_preferences(INTENT_NO_PREFS)
        self.assertEqual(result, {})

    def test_empty_intent(self):
        result = get_all_traveler_preferences(INTENT_EMPTY)
        self.assertEqual(result, {})


class TestFlightPreferences(unittest.TestCase):
    def test_preferred_airlines(self):
        self.assertEqual(get_preferred_airlines(INTENT_TWO_TRAVELERS, 'traveler-alice'), ['LATAM', 'GOL'])
        self.assertEqual(get_preferred_airlines(INTENT_TWO_TRAVELERS, 'traveler-bob'), ['Azul'])

    def test_preferred_airlines_missing(self):
        self.assertEqual(get_preferred_airlines(INTENT_NO_PREFS, 'traveler-charlie'), [])

    def test_seat_preference(self):
        self.assertEqual(get_seat_preference(INTENT_TWO_TRAVELERS, 'traveler-alice'), 'Window')
        self.assertEqual(get_seat_preference(INTENT_TWO_TRAVELERS, 'traveler-bob'), 'Aisle')

    def test_seat_preference_missing(self):
        self.assertIsNone(get_seat_preference(INTENT_NO_PREFS, 'traveler-charlie'))

    def test_preferred_travel_time(self):
        self.assertEqual(get_preferred_travel_time(INTENT_TWO_TRAVELERS, 'traveler-alice'), 'Morning')

    def test_blocked_travel_times(self):
        self.assertEqual(get_blocked_travel_times(INTENT_TWO_TRAVELERS, 'traveler-alice'), ['Night'])
        self.assertEqual(get_blocked_travel_times(INTENT_TWO_TRAVELERS, 'traveler-bob'), [])


class TestHotelPreferences(unittest.TestCase):
    def test_hotel_chain_preferences(self):
        self.assertEqual(get_hotel_chain_preferences(INTENT_TWO_TRAVELERS, 'traveler-alice'), ['Marriott'])
        self.assertEqual(get_hotel_chain_preferences(INTENT_TWO_TRAVELERS, 'traveler-bob'), [])

    def test_hotel_floor_preference(self):
        self.assertEqual(get_hotel_floor_preference(INTENT_TWO_TRAVELERS, 'traveler-alice'), 'High')
        self.assertIsNone(get_hotel_floor_preference(INTENT_TWO_TRAVELERS, 'traveler-bob'))

    def test_hotel_facilities(self):
        self.assertEqual(get_hotel_facilities(INTENT_TWO_TRAVELERS, 'traveler-alice'), ['Pool', 'Gym'])

    def test_preferred_hotels_by_city(self):
        entries = get_preferred_hotels_by_city(INTENT_TWO_TRAVELERS, 'traveler-alice')
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]['city'], 'São Paulo')

    def test_preferred_hotels_for_city_match(self):
        hotels = get_preferred_hotels_for_city(INTENT_TWO_TRAVELERS, 'traveler-alice', 'São Paulo')
        self.assertEqual(hotels, ['Ibis SP', 'Novotel'])

    def test_preferred_hotels_for_city_case_insensitive(self):
        hotels = get_preferred_hotels_for_city(INTENT_TWO_TRAVELERS, 'traveler-alice', 'são paulo')
        self.assertEqual(hotels, ['Ibis SP', 'Novotel'])

    def test_preferred_hotels_for_city_no_match(self):
        hotels = get_preferred_hotels_for_city(INTENT_TWO_TRAVELERS, 'traveler-alice', 'Curitiba')
        self.assertEqual(hotels, [])


class TestPersonalPreferences(unittest.TestCase):
    def test_is_smoker_false(self):
        self.assertFalse(is_smoker(INTENT_TWO_TRAVELERS, 'traveler-alice'))

    def test_is_smoker_true(self):
        self.assertTrue(is_smoker(INTENT_TWO_TRAVELERS, 'traveler-bob'))

    def test_is_smoker_missing(self):
        self.assertFalse(is_smoker(INTENT_NO_PREFS, 'traveler-charlie'))

    def test_special_needs(self):
        self.assertEqual(get_special_needs(INTENT_TWO_TRAVELERS, 'traveler-alice'), 'Wheelchair access')
        self.assertIsNone(get_special_needs(INTENT_TWO_TRAVELERS, 'traveler-bob'))

    def test_dietary_requirements(self):
        self.assertEqual(get_dietary_requirements(INTENT_TWO_TRAVELERS, 'traveler-alice'), 'Vegan')
        self.assertEqual(get_dietary_requirements(INTENT_TWO_TRAVELERS, 'traveler-bob'), 'Gluten-free')

    def test_car_type_preference(self):
        self.assertEqual(get_car_type_preference(INTENT_TWO_TRAVELERS, 'traveler-alice'), 'SUV')
        self.assertIsNone(get_car_type_preference(INTENT_TWO_TRAVELERS, 'traveler-bob'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
