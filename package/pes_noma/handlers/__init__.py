"""
PES Handlers

All custom handlers for the PES application.
Each handler should implement a `run(payload)` method.

Note: Handlers are imported lazily to avoid import errors when
optional dependencies are not installed.
"""

__all__ = [
    "GeneratePlan",
    "CommitPlan",
    "ModifyPlan",
    "ExecutePlan",
    "PesAgent",
    "Specialist",
]


def __getattr__(name: str):
    """
    Lazy import handler classes on demand.

    This keeps import-time side effects minimal and avoids import errors when
    optional dependencies are missing.
    """
    if name == "GeneratePlan":
        from pes_noma.handlers.generate_plan import GeneratePlan

        return GeneratePlan
    if name == "CommitPlan":
        from pes_noma.handlers.commit_plan import CommitPlan

        return CommitPlan
    if name == "ModifyPlan":
        from pes_noma.handlers.modify_plan import ModifyPlan

        return ModifyPlan
    if name == "ExecutePlan":
        from pes_noma.handlers.execute_plan import ExecutePlan

        return ExecutePlan
    if name == "PesAgent":
        from pes_noma.handlers.pes_agent import PesAgent

        return PesAgent
    if name == "Specialist":
        from pes_noma.handlers.specialist import Specialist

        return Specialist

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")




