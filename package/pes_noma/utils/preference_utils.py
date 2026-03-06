"""
Utility functions for reading travel preferences from a resolved intent.

Preferences are stored in intent.party after generate_plan runs:
  intent['party']['preferences_by_id'][pref_id]  -> preference object
  intent['party']['travelers_by_id'][tid]['preferences_id'] -> pref_id

All functions are pure (no I/O) and return safe defaults when data is absent.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


def get_traveler_preference(intent: Dict[str, Any], traveler_id: str) -> Optional[Dict[str, Any]]:
    """Return the full preference object for a traveler, or None if not found."""
    party = intent.get('party') or {}
    traveler = (party.get('travelers_by_id') or {}).get(traveler_id) or {}
    pref_id = traveler.get('preferences_id')
    if not pref_id:
        return None
    return (party.get('preferences_by_id') or {}).get(pref_id)


def get_all_traveler_preferences(intent: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return {traveler_id: preference_obj} for all travelers that have a preference."""
    party = intent.get('party') or {}
    travelers_by_id = party.get('travelers_by_id') or {}
    preferences_by_id = party.get('preferences_by_id') or {}
    result = {}
    for tid, traveler in travelers_by_id.items():
        pref_id = (traveler or {}).get('preferences_id')
        if pref_id and pref_id in preferences_by_id:
            result[tid] = preferences_by_id[pref_id]
    return result


# ── Flight ────────────────────────────────────────────────────────────────────

def get_preferred_airlines(intent: Dict[str, Any], traveler_id: str) -> List[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return list((pref or {}).get('preferred_airlines') or [])


def get_seat_preference(intent: Dict[str, Any], traveler_id: str) -> Optional[str]:
    """Window / Aisle / Middle / No Preference — or None if unset."""
    pref = get_traveler_preference(intent, traveler_id)
    return (pref or {}).get('seat_preference') or None


def get_preferred_travel_time(intent: Dict[str, Any], traveler_id: str) -> Optional[str]:
    """Morning / Afternoon / Night — or None if unset."""
    pref = get_traveler_preference(intent, traveler_id)
    return (pref or {}).get('preferred_travel_time') or None


def get_blocked_travel_times(intent: Dict[str, Any], traveler_id: str) -> List[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return list((pref or {}).get('blocked_travel_time') or [])


# ── Hotel ─────────────────────────────────────────────────────────────────────

def get_hotel_chain_preferences(intent: Dict[str, Any], traveler_id: str) -> List[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return list((pref or {}).get('hotel_chain_pref') or [])


def get_hotel_floor_preference(intent: Dict[str, Any], traveler_id: str) -> Optional[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return (pref or {}).get('hotel_floor_pref') or None


def get_hotel_facilities(intent: Dict[str, Any], traveler_id: str) -> List[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return list((pref or {}).get('hotel_facilities') or [])


def get_preferred_hotels_by_city(intent: Dict[str, Any], traveler_id: str) -> List[Dict[str, Any]]:
    """Return [{city: str, hotels: [str]}] for this traveler."""
    pref = get_traveler_preference(intent, traveler_id)
    return list((pref or {}).get('preferred_hotels_by_city') or [])


def get_preferred_hotels_for_city(intent: Dict[str, Any], traveler_id: str, city: str) -> List[str]:
    """Return the list of preferred hotel names for a specific city."""
    entries = get_preferred_hotels_by_city(intent, traveler_id)
    for entry in entries:
        if (entry.get('city') or '').lower() == city.lower():
            return list(entry.get('hotels') or [])
    return []


# ── Personal / Special ────────────────────────────────────────────────────────

def is_smoker(intent: Dict[str, Any], traveler_id: str) -> bool:
    pref = get_traveler_preference(intent, traveler_id)
    return bool((pref or {}).get('is_smoker'))


def get_special_needs(intent: Dict[str, Any], traveler_id: str) -> Optional[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return (pref or {}).get('special_needs') or None


def get_dietary_requirements(intent: Dict[str, Any], traveler_id: str) -> Optional[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return (pref or {}).get('dietary_requirements') or None


# ── Car ───────────────────────────────────────────────────────────────────────

def get_car_type_preference(intent: Dict[str, Any], traveler_id: str) -> Optional[str]:
    pref = get_traveler_preference(intent, traveler_id)
    return (pref or {}).get('car_type_pref') or None
