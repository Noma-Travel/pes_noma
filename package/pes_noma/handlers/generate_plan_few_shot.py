"""
Few-shot plan generator — MVP replacement for generate_plan.
Single LLM call with 3 fixed examples + structured output.
No intent generation. Output is a Plan compatible with commit_plan/specialist.
"""
import json
import uuid
import datetime
from typing import Any, Dict
import re

from renglo.common import load_config
from renglo.agent.agent_utilities import AgentUtilities


# ── Structured output schema (strict mode for gpt-4.1) ──────────────────────

def _strip_null_inputs(inputs: Any) -> Dict[str, Any]:
    if not isinstance(inputs, dict) or not inputs:
        return {}
    return {k: v for k, v in inputs.items() if v is not None}


def _strip_null_inputs_from_steps(steps: Any) -> Any:
    """Post-process LLM steps: remove input keys with None values."""
    if not isinstance(steps, list):
        return steps
    out = []
    for s in steps:
        if not isinstance(s, dict):
            out.append(s)
            continue
        out.append({**s, "inputs": _strip_null_inputs(s.get("inputs"))})
    return out


PLAN_JSON_SCHEMA = {
    "name": "travel_plan_steps",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "step_id", "action", "depends_on", "enter_guard",
                        "next_step", "success_criteria", "title", "inputs",
                    ],
                    "properties": {
                        "step_id": {"type": "integer"},
                        "action": {"type": "string", "enum": ["quote_flight", "quote_hotel"]},
                        "depends_on": {"type": "array", "items": {"type": "integer"}},
                        "enter_guard": {"type": "string"},
                        "next_step": {"type": ["integer", "null"]},
                        "success_criteria": {"type": "string"},
                        "title": {"type": "string"},
                        "inputs": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "leg", "from_airport_code", "to_airport_code",
                                "departure_date", "passengers", "city",
                                "check_in_date", "number_of_nights",
                                "number_of_guests", "area", "traveler_ids",
                            ],
                            "properties": {
                                "leg": {"type": ["integer", "null"]},
                                "from_airport_code": {"type": ["string", "null"]},
                                "to_airport_code": {"type": ["string", "null"]},
                                "departure_date": {"type": ["string", "null"]},
                                "passengers": {"type": ["integer", "null"]},
                                "city": {"type": ["string", "null"]},
                                "check_in_date": {"type": ["string", "null"]},
                                "number_of_nights": {"type": ["string", "null"]},
                                "number_of_guests": {"type": ["string", "null"]},
                                "area": {"type": ["string", "null"]},
                                "traveler_ids": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        },
    },
}

SCHEMA_TEXT = json.dumps(PLAN_JSON_SCHEMA["schema"], indent=2)

MODEL = "gpt-4.1"
IATA_CODE_RE = re.compile(r"^[A-Z]{3}$")
PLACEHOLDER_VALUES = {"not informed", "n/a", "tbd", "unknown", "nao informado"}


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.date.fromisoformat(value)
        return True
    except Exception:
        return False


def _validate_plan(steps: Any) -> tuple[bool, str]:
    if not isinstance(steps, list) or len(steps) == 0:
        return False, "Plan has no steps"

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, f"Step {idx} is not an object"

        action = step.get("action")
        inputs = step.get("inputs") or {}
        if not isinstance(inputs, dict):
            return False, f"Step {idx} has invalid inputs"

        # Placeholder guard
        for key, value in inputs.items():
            if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES:
                return False, f"Step {idx} has placeholder value in {key}"

        if action == "quote_flight":
            for required in ("from_airport_code", "to_airport_code", "departure_date"):
                if not inputs.get(required):
                    return False, f"Step {idx} missing {required}"

            if not _is_iso_date(inputs.get("departure_date")):
                return False, f"Step {idx} has invalid departure_date format"

            from_code = inputs.get("from_airport_code")
            to_code = inputs.get("to_airport_code")
            if not (isinstance(from_code, str) and IATA_CODE_RE.match(from_code)):
                return False, f"Step {idx} has invalid from_airport_code"
            if not (isinstance(to_code, str) and IATA_CODE_RE.match(to_code)):
                return False, f"Step {idx} has invalid to_airport_code"

            traveler_ids = inputs.get("traveler_ids")
            passengers = inputs.get("passengers")
            if isinstance(traveler_ids, list) and isinstance(passengers, int) and passengers != len(traveler_ids):
                return False, f"Step {idx} passengers mismatch traveler_ids"

        elif action == "quote_hotel":
            for required in ("city", "check_in_date", "number_of_nights"):
                if not inputs.get(required):
                    return False, f"Step {idx} missing {required}"
            if not _is_iso_date(inputs.get("check_in_date")):
                return False, f"Step {idx} has invalid check_in_date format"

    return True, ""


# ── System prompt ────────────────────────────────────────────────────────────

def _system_prompt() -> str:
    today = datetime.date.today().isoformat()
    return f"""Today's date is {today}.

We are in 2026. All dates will be in the future.

You convert free-form corporate travel requests into an ordered list of executable plan steps.

A plan is a JSON object with one key: "steps". Each step is one of two actions:

- quote_flight: a single flight leg between two airports on one date for a set of travelers.
- quote_hotel: a single hotel stay in one city for a check-in date and a number of nights.

Output MUST conform to this JSON schema exactly:

{SCHEMA_TEXT}

Rules:
- step_id is a sequential integer starting at 0.
- depends_on is a list with the previous step_id (linear chain). The first step has [].
- next_step is the following step_id, or null for the last step.
- enter_guard is always the literal string "True".
- success_criteria is always the literal string "len(result) > 0".
- For quote_flight: set leg=0 for outbound, leg=1 for return; for multi-city, increment leg per leg of the same group. Set city, check_in_date, number_of_nights, number_of_guests, area to null. from_airport_code and to_airport_code are 3-letter UPPERCASE IATA codes (e.g. "São Paulo"->"GRU", "New York"->"JFK", "London"->"LHR"). departure_date is ISO YYYY-MM-DD. passengers is the integer count of traveler_ids.
- For quote_hotel: set city to the name of destination city, check_in_date as ISO date, number_of_nights and number_of_guests as STRINGS (e.g. "3", "2"). Set leg, from_airport_code, to_airport_code, departure_date, passengers to null. area is usually null.
- traveler_ids are short stable strings like "t1", "t2", "t3". Use t1..tN in the order travelers appear in the text.
- title for flights: "<FROM> to <TO> flight". For hotels: "<CITY> hotel <N> nights (<G> guests)".
- Emit a hotel step for every accommodation mentioned. Emit a flight step for every distinct leg. If a subgroup flies separately, emit a separate flight step for that subgroup with only their traveler_ids.
- Never invent extra steps. Never omit required fields (use null where allowed)."""


# ── Few-shot examples ────────────────────────────────────────────────────────

EXAMPLES = [
    {
        "user": "Preciso de voos de ida e volta GRU-JFK para Ana Silva e Bruno Costa, saindo 10/05/2026, voltando 15/05/2026, econômica. Hotel em NY nessas datas, 1 quarto.",
        "plan": {
            "steps": [
                {"step_id": 0, "action": "quote_flight", "depends_on": [], "enter_guard": "True", "next_step": 1, "success_criteria": "len(result) > 0", "title": "GRU to JFK flight", "inputs": {"leg": 0, "from_airport_code": "GRU", "to_airport_code": "JFK", "departure_date": "2026-05-10", "passengers": 2, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t1", "t2"]}},
                {"step_id": 1, "action": "quote_flight", "depends_on": [0], "enter_guard": "True", "next_step": 2, "success_criteria": "len(result) > 0", "title": "JFK to GRU flight", "inputs": {"leg": 1, "from_airport_code": "JFK", "to_airport_code": "GRU", "departure_date": "2026-05-15", "passengers": 2, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t1", "t2"]}},
                {"step_id": 2, "action": "quote_hotel", "depends_on": [1], "enter_guard": "True", "next_step": None, "success_criteria": "len(result) > 0", "title": "New York hotel 5 nights (2 guests)", "inputs": {"leg": None, "from_airport_code": None, "to_airport_code": None, "departure_date": None, "passengers": None, "city": "New York", "check_in_date": "2026-05-10", "number_of_nights": "5", "number_of_guests": "2", "area": None, "traveler_ids": ["t1", "t2"]}},
            ]
        },
    },
    {
        "user": "Multi-city trip for Carlos Mendes: São Paulo to Lisbon on June 3rd, then Lisbon to Madrid on June 7th, then Madrid back to São Paulo on June 12th. Book hotel in Lisbon Jun 3-7 and hotel in Madrid Jun 7-12. Business class.",
        "plan": {
            "steps": [
                {"step_id": 0, "action": "quote_flight", "depends_on": [], "enter_guard": "True", "next_step": 1, "success_criteria": "len(result) > 0", "title": "GRU to LIS flight", "inputs": {"leg": 0, "from_airport_code": "GRU", "to_airport_code": "LIS", "departure_date": "2026-06-03", "passengers": 1, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t1"]}},
                {"step_id": 1, "action": "quote_flight", "depends_on": [0], "enter_guard": "True", "next_step": 2, "success_criteria": "len(result) > 0", "title": "LIS to MAD flight", "inputs": {"leg": 1, "from_airport_code": "LIS", "to_airport_code": "MAD", "departure_date": "2026-06-07", "passengers": 1, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t1"]}},
                {"step_id": 2, "action": "quote_flight", "depends_on": [1], "enter_guard": "True", "next_step": 3, "success_criteria": "len(result) > 0", "title": "MAD to GRU flight", "inputs": {"leg": 2, "from_airport_code": "MAD", "to_airport_code": "GRU", "departure_date": "2026-06-12", "passengers": 1, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t1"]}},
                {"step_id": 3, "action": "quote_hotel", "depends_on": [2], "enter_guard": "True", "next_step": 4, "success_criteria": "len(result) > 0", "title": "Lisbon hotel 4 nights (1 guests)", "inputs": {"leg": None, "from_airport_code": None, "to_airport_code": None, "departure_date": None, "passengers": None, "city": "Lisbon", "check_in_date": "2026-06-03", "number_of_nights": "4", "number_of_guests": "1", "area": None, "traveler_ids": ["t1"]}},
                {"step_id": 4, "action": "quote_hotel", "depends_on": [3], "enter_guard": "True", "next_step": None, "success_criteria": "len(result) > 0", "title": "Madrid hotel 5 nights (1 guests)", "inputs": {"leg": None, "from_airport_code": None, "to_airport_code": None, "departure_date": None, "passengers": None, "city": "Madrid", "check_in_date": "2026-06-07", "number_of_nights": "5", "number_of_guests": "1", "area": None, "traveler_ids": ["t1"]}},
            ]
        },
    },
    {
        "user": "Team meeting in Miami July 10-13. Diana flies from São Paulo, Eduardo and Fernanda from Rio. All three stay at the same hotel, 2 rooms. Return same day July 13.",
        "plan": {
            "steps": [
                {"step_id": 0, "action": "quote_flight", "depends_on": [], "enter_guard": "True", "next_step": 1, "success_criteria": "len(result) > 0", "title": "GRU to MIA flight", "inputs": {"leg": 0, "from_airport_code": "GRU", "to_airport_code": "MIA", "departure_date": "2026-07-10", "passengers": 1, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t1"]}},
                {"step_id": 1, "action": "quote_flight", "depends_on": [0], "enter_guard": "True", "next_step": 2, "success_criteria": "len(result) > 0", "title": "GIG to MIA flight", "inputs": {"leg": 0, "from_airport_code": "GIG", "to_airport_code": "MIA", "departure_date": "2026-07-10", "passengers": 2, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t2", "t3"]}},
                {"step_id": 2, "action": "quote_flight", "depends_on": [1], "enter_guard": "True", "next_step": 3, "success_criteria": "len(result) > 0", "title": "MIA to GRU flight", "inputs": {"leg": 1, "from_airport_code": "MIA", "to_airport_code": "GRU", "departure_date": "2026-07-13", "passengers": 1, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t1"]}},
                {"step_id": 3, "action": "quote_flight", "depends_on": [2], "enter_guard": "True", "next_step": 4, "success_criteria": "len(result) > 0", "title": "MIA to GIG flight", "inputs": {"leg": 1, "from_airport_code": "MIA", "to_airport_code": "GIG", "departure_date": "2026-07-13", "passengers": 2, "city": None, "check_in_date": None, "number_of_nights": None, "number_of_guests": None, "area": None, "traveler_ids": ["t2", "t3"]}},
                {"step_id": 4, "action": "quote_hotel", "depends_on": [3], "enter_guard": "True", "next_step": None, "success_criteria": "len(result) > 0", "title": "Miami hotel 3 nights (3 guests)", "inputs": {"leg": None, "from_airport_code": None, "to_airport_code": None, "departure_date": None, "passengers": None, "city": "Miami", "check_in_date": "2026-07-10", "number_of_nights": "3", "number_of_guests": "3", "area": None, "traveler_ids": ["t1", "t2", "t3"]}},
            ]
        },
    },
]


def _few_shot_messages() -> list:
    msgs = []
    for ex in EXAMPLES:
        msgs.append({"role": "user", "content": f"Extract the travel plan from this request:\n\n<<<\n{ex['user']}\n>>>"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["plan"], ensure_ascii=False)})
    return msgs


# ── Handler ──────────────────────────────────────────────────────────────────

class GeneratePlanFewShot:

    def __init__(self):
        self.config = load_config()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        function = 'run > generate_plan_few_shot'

        portfolio = payload.get('_portfolio') or payload.get('portfolio')
        if not portfolio:
            return {'success': False, 'function': function, 'input': payload, 'output': 'No portfolio provided'}

        org = payload.get('_org') or payload.get('org') or '_all'

        entity_type = payload.get('_entity_type') or 'some_entity_type'
        entity_id = payload.get('_entity_id') or 'some_entity_id'
        thread = payload.get('_thread') or 'some_thread'

        user_message = payload.get('message', '').strip()
        if not user_message:
            return {'success': False, 'function': function, 'input': payload, 'output': 'No message provided'}

        try:
            agu = AgentUtilities(self.config, portfolio, org, entity_type, entity_id, thread)

            base_messages = (
                [{"role": "system", "content": _system_prompt()}]
                + _few_shot_messages()
                + [{"role": "user", "content": f"Extract the travel plan from this request:\n\n<<<\n{user_message}\n>>>"}]
            )

            response = agu.llm({
                "model": MODEL,
                "messages": base_messages,
                "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": PLAN_JSON_SCHEMA},
            })

            if not response:
                return {'success': False, 'function': function, 'input': payload, 'output': 'LLM call failed'}

            steps_data = json.loads(response.content)
            steps = _strip_null_inputs_from_steps(steps_data.get("steps", []))
            is_valid, validation_error = _validate_plan(steps)

            if not is_valid:
                retry_messages = base_messages + [{
                    "role": "system",
                    "content": (
                        "Your previous output failed validation. "
                        f"Validation error: {validation_error}. "
                        "Regenerate the plan and strictly fix these issues. "
                        "Return only JSON compliant with the schema."
                    ),
                }]
                retry_response = agu.llm({
                    "model": MODEL,
                    "messages": retry_messages,
                    "temperature": 0,
                    "response_format": {"type": "json_schema", "json_schema": PLAN_JSON_SCHEMA},
                })
                if not retry_response:
                    return {
                        'success': False,
                        'function': function,
                        'input': payload,
                        'output': f'Plan validation failed and retry LLM call failed: {validation_error}'
                    }

                retry_steps_data = json.loads(retry_response.content)
                steps = _strip_null_inputs_from_steps(retry_steps_data.get("steps", []))
                is_valid, validation_error = _validate_plan(steps)
                if not is_valid:
                    return {
                        'success': False,
                        'function': function,
                        'input': payload,
                        'output': f'Plan validation failed after retry: {validation_error}'
                    }

            last_step_id = steps[-1]["step_id"] if steps else -1
            post_exec_step_id = last_step_id + 1
            if steps:
                steps[-1]["next_step"] = post_exec_step_id
            steps.append({
                "step_id": post_exec_step_id,
                "action": "post_execution",
                "depends_on": [last_step_id] if steps else [],
                "enter_guard": "True",
                "next_step": None,
                "success_criteria": "len(result) > 0",
                "title": "Finalizar e enviar para advogado",
                "inputs": {},
            })

            plan = {
                "id": str(uuid.uuid4()),
                "steps": steps,
                "meta": {"strategy": "few_shot", "model": MODEL},
            }

            output = {"plan": plan, "intent": None}

            return {'success': True, 'interface': 'plan', 'input': payload, 'output': output}

        except Exception as e:
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@generate_plan_few_shot/run: {str(e)}'}
