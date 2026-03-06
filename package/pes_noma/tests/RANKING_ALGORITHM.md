# Ranking Algorithm Documentation

## Overview

The ranking system has two stages:

1. **PolicyFilter** (hard constraints) — removes options that violate travel policies
2. **TripOptionRanker** (soft scoring) — scores and ranks surviving options into bundles

## 1. PolicyFilter (Hard Constraints)

**File:** `noma/handlers/policy_filter.py`

Reads merged policy from `intent.preferences` and `intent.constraints`. Options that violate rules are **removed** (not just downranked).

### Rules

| Rule | Source | Effect |
|---|---|---|
| `max_flight_budget` | `intent.preferences.flight.max_budget` | Removes flights above price |
| `max_flight_class` | `intent.preferences.flight.max_class` | Removes flights above class (economy < premium_economy < business < first) |
| `max_hotel_daily_rate` | `intent.preferences.hotel.max_daily_rate` | Removes hotels above nightly rate |
| `max_hotel_stars` | `intent.preferences.hotel.max_stars` | Removes hotels above star rating |
| `enabled_services` | `intent.constraints.enabled_services` | Removes entire category (flights/hotels) if not in list |

### Output

```python
{
    "flight_options": [...],           # compliant options
    "hotel_options": [...],            # compliant options
    "violations": [...],               # removed options with _policy_violation reason
    "policy_applied": ["max_flight_budget=500"],  # rules that were enforced
    "violations_count": 3,
}
```

## 2. TripOptionRanker (Soft Scoring)

**File:** `noma/handlers/trip_option_ranker.py`

Scores every flight×hotel combination and returns the top 10 bundles.

### Score Formula

```
score = w_price      * price_score
      + w_duration   * duration_score
      + w_refundable * refundable_score
      + w_convenience* convenience_score
      + w_preference * preference_score
```

### Default Weights

| Weight | Default | Description |
|---|---|---|
| `price` | 0.5 | Lower price = higher score |
| `duration` | 0.2 | Shorter flight = higher score |
| `refundable` | 0.2 | Refundable = 1.0, non-refundable = 0.0 |
| `convenience` | 0.1 | Shorter travel time, direct flights |
| `preference` | 0.3 | Traveler preference match |

Weights are passed via `ranking_policy.weights` in the payload. If omitted, defaults apply.

### Sub-scores

**Price score:** `1.0 / (1.0 + total_price / 500.0)` — asymptotic, cheaper is better

**Duration score:** `1.0 / (1.0 + duration_minutes / 60.0)` — shorter is better

**Preference score** (per flight):

| Signal | Value | Condition |
|---|---|---|
| Preferred airline | +0.15 | Any flight airline matches `preferred_airlines` |
| Preferred time | +0.10 | Departure hour in preferred time category |
| Blocked time | -0.30 | Departure hour in blocked time category |

Time categories: Morning (5-11h), Afternoon (12-17h), Night (18-4h)

**Preference score** (per hotel):

| Signal | Value | Condition |
|---|---|---|
| Chain match | +0.15 | Hotel chain in `hotel_chain_prefs` |
| Chain in name | +0.10 | Chain name appears in hotel name (fallback) |
| Facility match | +0.03 each (max 0.10) | Amenity in `hotel_facilities` |
| City hotel match | +0.20 | Hotel name in `preferred_hotels_by_city` for that city |

### Bundle Selection

After scoring all combinations, the ranker picks **up to 10 diverse bundles**:

1. Cheapest total
2. Most expensive (frame of reference)
3. Fastest (min duration)
4. Best refundable
5. Cheapest direct (no layovers)
6. Premium lodging (highest hotel spend)
7. Premium flights (highest flight spend)
8. Budget flights + premium stay
9. Premium flights + budget stay
10. Remaining slots filled by highest overall score

## 3. Preference Collection

**Function:** `_collect_preferences(intent)`

Aggregates preferences from all travelers in `intent.party`:

```
intent.party.travelers_by_id[tid].preferences_id → pref_id
intent.party.preferences_by_id[pref_id] → preference object
```

Preferences are **unioned** across travelers (if Alice prefers LATAM and Bob prefers GOL, both get boosted).

## 4. Data Flow

```
User message
  → generate_plan.py
    → _inject_policies_into_intent()     # merge policies (most restrictive)
    → _inject_preferences_per_traveler() # attach per-traveler prefs
    → intent stored in workspace

  → specialist.py (ReAct loop)
    → loads preferences from intent
    → injects as system message to LLM
    → LLM calls search_flights / search_hotels

  → search_flights.py
    → fetches flight options from API
    → calls PolicyFilter to remove violations
    → returns filtered options

  → specialist calls trip_option_ranker
    → TripOptionRanker.run()
      → _collect_preferences(intent)
      → _build_bundles_simple() or _build_bundles_multi()
      → returns ranked bundles

  → bundles shown to user for selection
```

## 5. Usage Examples

### Running the PolicyFilter

```python
from noma.handlers.policy_filter import PolicyFilter

pf = PolicyFilter()
result = pf.run({
    "intent": {
        "preferences": {
            "flight": {"max_budget": 500, "max_class": "economy"},
            "hotel": {"max_daily_rate": 300},
        },
        "constraints": {"enabled_services": ["flights", "hotels"]},
    },
    "flight_options": [flight1, flight2, flight3],
    "hotel_options": [hotel1, hotel2],
})

compliant_flights = result["output"]["flight_options"]
violations = result["output"]["violations"]
```

### Running the Ranker

```python
from noma.handlers.trip_option_ranker import TripOptionRanker

ranker = TripOptionRanker()
result = ranker.run({
    "intent": {
        "party": {
            "travelers_by_id": {"t-1": {"preferences_id": "pref-1"}},
            "preferences_by_id": {
                "pref-1": {
                    "preferred_airlines": ["LATAM"],
                    "preferred_travel_time": "Morning",
                    "blocked_travel_time": ["Night"],
                    "hotel_chain_pref": ["Marriott"],
                    "hotel_facilities": ["Pool"],
                }
            },
        },
    },
    "flight_options": filtered_flights,
    "hotel_options": filtered_hotels,
    "ranking_policy": {"weights": {"price": 0.5, "preference": 0.3}},
})

bundles = result["output"]["bundles"]
# bundles[0] = best option with bundle_id, estimated_total, why_this_bundle, etc.
```

### Full Pipeline

```python
from noma.handlers.policy_filter import PolicyFilter
from noma.handlers.trip_option_ranker import TripOptionRanker

# Step 1: Filter
filtered = PolicyFilter().run({
    "intent": intent,
    "flight_options": raw_flights,
    "hotel_options": raw_hotels,
})

# Step 2: Rank
ranked = TripOptionRanker().run({
    "intent": intent,
    "flight_options": filtered["output"]["flight_options"],
    "hotel_options": filtered["output"]["hotel_options"],
    "ranking_policy": {},
})

bundles = ranked["output"]["bundles"]
```

## 6. Adding New Ranking Criteria

### Step 1: Add a scoring function

In `trip_option_ranker.py`, create a new function following the pattern:

```python
def _flight_YOUR_CRITERION_score(opt: Dict, data: Dict) -> float:
    """Score range: describe min to max."""
    score = 0.0
    # your logic here
    return score
```

### Step 2: Add a weight

In `_build_bundles_simple()` and `_build_bundles_multi()`, add:

```python
w_your_criterion = float(weights.get("your_criterion", 0.1))  # default weight
```

### Step 3: Include in the score calculation

```python
score = (
    w_price * (...)
    + w_duration * (...)
    + ...
    + w_your_criterion * your_score  # add here
)
```

### Step 4: Collect the data

If your criterion needs data from the intent (like preferences), add a collector function similar to `_collect_preferences()`, and call it in `TripOptionRanker.run()`.

### Step 5: Write tests

Add tests in `pes_noma/tests/` following the existing pattern:
- Test the scoring function in isolation with known inputs
- Test it integrates into the bundle scoring
- Test zero-shot (no data available = score 0)

### Running Tests

```bash
cd extensions/pes_noma/package
python -m pytest pes_noma/tests/ -v
```
