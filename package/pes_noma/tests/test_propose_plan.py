import sys
import types


def _install_renglo_stubs():
    if "renglo" in sys.modules:
        return

    renglo = types.ModuleType("renglo")
    agent = types.ModuleType("renglo.agent")
    agent_utilities = types.ModuleType("renglo.agent.agent_utilities")
    common = types.ModuleType("renglo.common")
    data = types.ModuleType("renglo.data")
    data_controller = types.ModuleType("renglo.data.data_controller")
    blueprint = types.ModuleType("renglo.blueprint")
    blueprint_controller = types.ModuleType("renglo.blueprint.blueprint_controller")

    class AgentUtilities: ...
    class DataController: ...
    class BlueprintController: ...

    def load_config():
        return {}

    agent_utilities.AgentUtilities = AgentUtilities
    common.load_config = load_config
    data_controller.DataController = DataController
    blueprint_controller.BlueprintController = BlueprintController

    sys.modules["renglo"] = renglo
    sys.modules["renglo.agent"] = agent
    sys.modules["renglo.agent.agent_utilities"] = agent_utilities
    sys.modules["renglo.common"] = common
    sys.modules["renglo.data"] = data
    sys.modules["renglo.data.data_controller"] = data_controller
    sys.modules["renglo.blueprint"] = blueprint
    sys.modules["renglo.blueprint.blueprint_controller"] = blueprint_controller


_install_renglo_stubs()

from pes_noma.handlers.generate_plan import ActionSpec
from pes_noma.handlers.propose_plan import ProposePlan


def test_compose_plan_light_preserves_all_segments_for_multi_city_without_stays():
    planner = ProposePlan.__new__(ProposePlan)
    action_catalog = [
        ActionSpec(key="quote_flight", description="Quote a flight"),
        ActionSpec(key="quote_train_bus", description="Quote a train or bus"),
    ]
    intent = {
        "party": {
            "travelers": {"adults": 1, "children": 0, "infants": 0},
            "traveler_ids": ["t1"],
        },
        "itinerary": {
            "trip_type": "multi_city",
            "segments": [
                {
                    "origin": {"type": "airport", "code": "GRU"},
                    "destination": {"type": "airport", "code": "PRG"},
                    "depart_date": "2026-03-27",
                    "transport_mode": "flight",
                    "passengers": "1",
                    "traveler_ids": ["t1"],
                },
                {
                    "origin": {"type": "city", "code": "PRG"},
                    "destination": {"type": "city", "code": "BTS"},
                    "depart_date": "2026-04-10",
                    "transport_mode": "train",
                    "passengers": "1",
                    "traveler_ids": ["t1"],
                },
                {
                    "origin": {"type": "airport", "code": "BTS"},
                    "destination": {"type": "airport", "code": "GRU"},
                    "depart_date": "2026-04-15",
                    "transport_mode": "flight",
                    "passengers": "1",
                    "traveler_ids": ["t1"],
                },
            ],
            "lodging": {
                "needed": False,
                "stays": [],
            },
        },
    }

    plan = planner.compose_plan_light(intent, action_catalog)

    assert [step.action for step in plan.steps] == [
        "quote_flight",
        "quote_train_bus",
        "quote_flight",
    ]
    assert [step.inputs["leg"] for step in plan.steps] == [0, 1, 2]
    assert plan.steps[0].inputs["from_airport_code"] == "GRU"
    assert plan.steps[0].inputs["to_airport_code"] == "PRG"
    assert plan.steps[1].inputs["departure_city"] == "PRG"
    assert plan.steps[1].inputs["arrival_city"] == "BTS"
    assert plan.steps[2].inputs["from_airport_code"] == "BTS"
    assert plan.steps[2].inputs["to_airport_code"] == "GRU"
    assert [step.step_id for step in plan.steps] == [0, 1, 2]
    assert [step.next_step for step in plan.steps] == [1, 2, None]
    assert [step.depends_on for step in plan.steps] == [[], [0], [1]]
