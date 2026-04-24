"""
ReAct specialist for a single plan step: interpret → act → verify loop.

User-visible text is tool-gated: intermediate assistant reasoning uses msg_type internal.
Consent (preview + confirm) applies only to high-risk tools listed in CONSENT_REQUIRED_TOOLS.

Blueprint source of truth:
  Actions and tools are loaded at runtime from Dynamo/DataController::
    ``schd_actions`` / ``schd_tools`` for the current portfolio/org.
  Those indexes are populated from blueprint packages such as
  ``noma_backend/extensions/backend/package/noma/actions/*.json`` and
  ``.../tools/*.json`` — not read from disk directly on each request.
"""

from __future__ import annotations

import json
import logging
import random
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from renglo.debug_json import djson

_logger_spec = logging.getLogger("agent.specialist")
_logger_verify = logging.getLogger("agent.verify")

CONSENT_REQUIRED_TOOLS: Set[str] = frozenset(
    {
        "send_email_to_lawyer",
        "book_flights_rextur",
    }
)

MAX_REACT_ITERATIONS = 8
_PENDING_CACHE_KEY = "irn:specialist_pending_tool"


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super().default(obj)


class Specialist:
    def __init__(self, agu):
        self.AGU = agu

    @staticmethod
    def _single_leg_dict_from_step_inputs(step_inputs: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
        One Rextur search segment from the **current plan step** inputs only.
        Used to override LLM tool args (legacy prompts often pass all trip legs).
        """
        if not isinstance(step_inputs, dict):
            return None
        fr = (step_inputs.get("from_airport_code") or "").strip()
        to = (step_inputs.get("to_airport_code") or "").strip()
        dt = (
            step_inputs.get("departure_date")
            or step_inputs.get("outbound_date")
            or ""
        ).strip()
        if not fr or not to or not dt:
            return None
        return [{"origin": fr, "destination": to, "date": dt}]

    def _force_single_leg_search_payload(
        self, params: Dict[str, Any], step_inputs: Optional[Dict[str, Any]]
    ) -> None:
        """Mutates params in place so search_flights_rextur always receives exactly one leg when step_inputs allow."""
        one = self._single_leg_dict_from_step_inputs(step_inputs or {})
        if not one:
            return
        params["legs"] = json.dumps(one)
        params.pop("return_date", None)

    # ------------------------------------------------------------------
    # Loading action / tools (same pattern as AgentUtilities.interpret)
    # ------------------------------------------------------------------
    def _load_action_doc(self, action_key: str) -> Dict[str, Any]:
        try:
            response = self.AGU.DAC.get_a_b(self.AGU.portfolio, self.AGU.org, "schd_actions")
            if not response.get("items"):
                return {}
            for a in response["items"]:
                if a.get("key") == action_key:
                    return a
        except Exception as e:
            _logger_spec.error("load_action_failed | %s", e)
        return {}

    def _load_all_tools(self) -> List[Dict[str, Any]]:
        try:
            response = self.AGU.DAC.get_a_b(self.AGU.portfolio, self.AGU.org, "schd_tools")
            return list(response.get("items") or [])
        except Exception as e:
            _logger_spec.error("load_tools_failed | %s", e)
        return []

    def _approved_tool_keys(self, action_doc: Dict[str, Any]) -> List[str]:
        ref = (action_doc.get("tools_reference") or "").strip()
        if not ref or ref in ("_", "-", ".", ""):
            return []
        return [k.strip() for k in ref.split(",") if k.strip()]

    def _build_openai_tools(
        self, list_tools_raw: List[Dict[str, Any]], approved_keys: List[str]
    ) -> List[Dict[str, Any]]:
        available_tools: List[Dict[str, Any]] = []
        approved = set(approved_keys)
        for t in list_tools_raw:
            attrs = t.get("attributes", {}) if isinstance(t, dict) else {}
            tool_key = t.get("key") or attrs.get("key")
            if tool_key not in approved:
                continue
            tool_goal = t.get("goal") or attrs.get("goal", "")
            tool_input_raw = t.get("input") or attrs.get("input", "[]")
            try:
                tool_input = json.loads(tool_input_raw)
            except (json.JSONDecodeError, TypeError):
                tool_input = []

            dict_params: Dict[str, Any] = {}
            required_params: List[str] = []

            if isinstance(tool_input, list):
                for param in tool_input:
                    if isinstance(param, dict) and "name" in param and "hint" in param:
                        pn = param["name"]
                        dict_params[pn] = {"type": "string", "description": param.get("hint", "")}
                        if param.get("required"):
                            required_params.append(pn)
            elif isinstance(tool_input, dict):
                for key, val in tool_input.items():
                    dict_params[key] = {"type": "string", "description": val}
                    required_params.append(key)

            available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_key or "",
                        "description": tool_goal,
                        "parameters": {
                            "type": "object",
                            "properties": dict_params,
                            "required": required_params,
                        },
                    },
                }
            )
        return available_tools

    def _meta_language_directive(self) -> str:
        lang = str(self.AGU.config.get("AGENT_LANGUAGE", "pt-BR") or "pt-BR")
        if lang.lower().startswith("en"):
            return "IMPORTANT: Respond ONLY in English for any user-visible message."
        return (
            "IMPORTANTE: Responda APENAS em Português do Brasil nas mensagens visíveis ao usuário."
        )

    def _trim_tool_output_for_history(self, tool_name: str, inner_output: Any) -> Any:
        """Shrink tool results for LLM chat history; full payload stays in workspace cache."""
        if tool_name == "search_flights_rextur" and isinstance(inner_output, dict):
            index = inner_output.get("index") if isinstance(inner_output.get("index"), dict) else {}
            slim = {
                "success": True,
                "segment_keys_by_price": index.get("fares", []),
                "total": index.get("total", 0),
                "agent_message": inner_output.get("agent_message"),
                "search_key": inner_output.get("search_key"),
                "current_leg": inner_output.get("current_leg"),
                "total_legs": inner_output.get("total_legs"),
                "hint": (
                    "Use `key` when calling show_options / add_flight_rextur. "
                    "`display` carries the price breakdown per fare family. "
                    "List is already sorted by price ascending."
                ),
            }
            warning = self._suspicious_search_warning(inner_output)
            if warning:
                slim["WARNING"] = warning
            return slim
        # show_options: do NOT trim — the full output (flights dict) is saved to the chat store
        # and is what the frontend carousel widget reads. The specialist always returns
        # `awaiting` before this content could reach the LLM messages array, so trimming
        # here only breaks the UI with zero LLM benefit.
        return inner_output

    @staticmethod
    def _suspicious_search_warning(inner_output: Dict[str, Any]) -> Optional[str]:
        """Flag when the first shown result is much pricier than the full-set minimum.

        Heuristic: if top-shown > 3x index.min_price, the filter is likely too tight
        and the model should re-search with looser args.
        """
        try:
            index = inner_output.get("index") or {}
            fares = index.get("fares") or []
            if not isinstance(fares, list) or not fares:
                return None
            first = fares[0]
            if not isinstance(first, dict):
                return None
            top_price = float(first.get("price") or 0)
            min_price = float(index.get("min_price") or 0)
            if min_price <= 0 or top_price <= 0:
                return None
            if top_price > 3 * min_price:
                return (
                    f"WARNING: top displayed option costs R${int(top_price)} but cheapest in the full "
                    f"result set is R${int(min_price)} — filter may be too narrow. Consider re-running "
                    "search_flights_rextur with relaxed constraints (looser time window, include more airlines)."
                )
        except (TypeError, ValueError):
            return None
        return None

    def _consent_message_body(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        preview_html = ""
        preview_plain = ""
        if tool_name == "send_email_to_lawyer":
            try:
                from noma.handlers.send_email_to_lawyer import preview_email_html, preview_email_plain

                merged = {
                    **arguments,
                    "_portfolio": self.AGU.portfolio,
                    "_org": self.AGU.org,
                    "_entity_type": self.AGU.entity_type,
                    "_entity_id": self.AGU.entity_id,
                }
                preview_html = preview_email_html(merged)[:12000]
                preview_plain = preview_email_plain(merged)[:8000]
            except Exception as e:
                _logger_spec.warning("email_preview_failed | %s", e)
                preview_plain = str(arguments)
        else:
            preview_plain = json.dumps(arguments, ensure_ascii=False, indent=2)

        content_parts = [
            f"Confirme a execução da ferramenta **{tool_name}**.",
            "",
            preview_plain[:6000],
        ]
        if preview_html and tool_name == "send_email_to_lawyer":
            content_parts.append("\n--- Pré-visualização HTML (resumo) ---\n")
            content_parts.append(preview_html[:4000])

        return {
            "role": "assistant",
            "content": "\n".join(content_parts),
            "preview_html": preview_html,
            "pending_tool": tool_name,
            "pending_arguments": arguments,
        }

    def _get_pending_from_workspace(self, workspace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache = workspace.get("cache") or {}
        raw = cache.get(_PENDING_CACHE_KEY)
        if isinstance(raw, dict) and raw.get("plan_id") is not None:
            return raw
        return None

    def _set_pending_tool(self, pending: Optional[Dict[str, Any]], workspace_id: Optional[str]) -> None:
        self.AGU.mutate_workspace(
            {"cache": {_PENDING_CACHE_KEY: pending}},
            workspace_id=workspace_id,
        )

    # ------------------------------------------------------------------
    def _act(
        self,
        execution_request: Dict[str, Any],
        list_tools_raw: List[Dict[str, Any]],
        extra: Optional[Dict[str, Any]] = None,
        trim_for_chat: bool = True,
    ) -> Dict[str, Any]:
        action = "act"
        list_handlers: Dict[str, str] = {}
        list_inits: Dict[str, Any] = {}
        for t in list_tools_raw:
            list_handlers[t.get("key", "")] = t.get("handler", "")
            init_value = t.get("init", {})
            if isinstance(init_value, str):
                try:
                    init_value = json.loads(init_value)
                except (json.JSONDecodeError, ValueError):
                    init_value = {}
            list_inits[t.get("key", "")] = init_value if isinstance(init_value, dict) else {}

        tool_name = execution_request["tool_calls"][0]["function"]["name"]
        params = execution_request["tool_calls"][0]["function"]["arguments"]
        if isinstance(params, str):
            params = json.loads(params)
        tid = execution_request["tool_calls"][0]["id"]

        if not tool_name:
            raise ValueError("No tool name in tool_calls")
        hidden = {"_portfolio", "_org", "_entity_type", "_entity_id", "_thread", "_init"}
        user_params = {k: v for k, v in dict(params).items() if k not in hidden}
        _logger_spec.info("calling tool=%s params=%s", tool_name, user_params)
        self.AGU.print_chat(f"Calling tool {tool_name} with parameters {params}", "transient")

        if tool_name not in list_handlers or not list_handlers[tool_name]:
            raise ValueError(f"No handler for tool '{tool_name}'")

        handler_route = list_handlers[tool_name]
        parts = handler_route.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid handler route for {tool_name}")

        handler_init = list_inits.get(tool_name, {})
        if not isinstance(handler_init, dict):
            handler_init = {}

        params = dict(params)
        params["_portfolio"] = self.AGU.portfolio
        params["_org"] = self.AGU.org
        params["_entity_type"] = self.AGU.entity_type
        params["_entity_id"] = self.AGU.entity_id
        params["_thread"] = self.AGU.thread
        params["_init"] = handler_init
        if extra and isinstance(extra, dict):
            params.update(extra)

        step_inputs = extra.get("_step_inputs") if isinstance(extra, dict) else None
        plan_ctx = extra.get("_plan_context") if isinstance(extra, dict) else None

        if tool_name == "search_flights_rextur":
            self._force_single_leg_search_payload(params, step_inputs)

        if tool_name == "search_flights_rextur" and isinstance(step_inputs, dict):
            if step_inputs.get("leg") is not None and "leg" not in params:
                params["leg"] = step_inputs.get("leg")

        # Force leg from plan step inputs for add_flight_rextur so the return leg is always
        # stored under the correct key (e.g. "1") even when the model passes leg=0.
        if tool_name == "add_flight_rextur" and isinstance(step_inputs, dict):
            si_leg = step_inputs.get("leg")
            if si_leg is not None:
                params["leg"] = si_leg
                _logger_spec.debug("add_flight_rextur leg overridden from step_inputs | leg=%s", si_leg)
            ob = step_inputs.get("outbound_date") or step_inputs.get("departure_date")
            if ob and not params.get("outbound_date") and not params.get("departure_date"):
                params["outbound_date"] = ob
                params["departure_date"] = ob
            for ab in (
                "from_airport_code",
                "to_airport_code",
                "passengers",
                "traveler_ids",
            ):
                if step_inputs.get(ab) is not None and params.get(ab) in (None, ""):
                    params[ab] = step_inputs[ab]
            if isinstance(plan_ctx, dict):
                ages = plan_ctx.get("trip_ages") or plan_ctx.get("ages")
                if ages and not params.get("ages"):
                    params["ages"] = ages

        for k in (
            "_step_inputs",
            "_plan_context",
            "plan",
            "intent",
            "_continuity_plan_id",
            "_continuity_plan_step",
        ):
            params.pop(k, None)

        response = self.AGU.SHC.handler_call(self.AGU.portfolio, self.AGU.org, parts[0], parts[1], params)

        if not response.get("success"):
            _logger_spec.error(
                "tool=%s handler_fail | success=False | detail_keys=%s",
                tool_name,
                list(response.keys()),
            )
            return {"success": False, "action": action, "input": params, "output": response}

        response_meta = {k: v for k, v in response.items() if k not in ("stack", "output")}
        response_meta["output_size"] = len(str(response.get("output", "")))
        _logger_spec.info(
            "tool=%s returned success=True details=%s",
            tool_name,
            response_meta,
        )

        full_output = response.get("output")
        interface = response.get("interface")
        if isinstance(full_output, dict) and full_output.get("interface"):
            interface = interface or full_output.get("interface")

        store_for_llm = (
            self._trim_tool_output_for_history(tool_name, full_output)
            if trim_for_chat
            else full_output
        )
        clean_output_str = json.dumps(store_for_llm, cls=DecimalEncoder)

        tool_out = {
            "role": "tool",
            "tool_call_id": tid,
            "content": clean_output_str,
            "tool_calls": False,
        }

        next_c_id = None
        if extra and isinstance(extra, dict):
            pid = extra.get("_continuity_plan_id")
            pstep = extra.get("_continuity_plan_step")
            if pid is not None and str(pstep).strip() != "":
                nonce = random.randint(100000, 999999)
                next_c_id = f"irn:c_id:{pid}:{pstep}:{tool_name}:5:{nonce}"

        if interface:
            self.AGU.save_chat(
                tool_out,
                interface=interface,
                connection_id=self.AGU.connection_id,
                next=next_c_id,
            )
        else:
            self.AGU.save_chat(
                tool_out,
                connection_id=self.AGU.connection_id,
                next=next_c_id,
            )

        index = f"irn:tool_rs:{handler_route}"
        tool_input_obj = json.loads(params) if isinstance(params, str) else params
        ws_id = getattr(self.AGU, "workspace_id", None)
        self.AGU.mutate_workspace(
            {"cache": {index: {"input": tool_input_obj, "output": full_output}}},
            workspace_id=ws_id,
        )

        _logger_spec.info("tool=%s done success=True interface=%s", tool_name, interface)
        djson(
            "specialist_last_tool_meta.json",
            {"tool": tool_name, "interface": interface, "handler_route": handler_route},
        )

        return {
            "success": True,
            "action": action,
            "input": execution_request,
            "output": tool_out,
            "_handler_interface": interface,
        }

    def _verify(self, action_name: str, payload_extra: Dict[str, Any]) -> Dict[str, Any]:
        action_doc = self._load_action_doc(action_name)
        verifier_key = (action_doc.get("verification") or "").strip()
        if not verifier_key:
            return {"success": True, "output": "no_verifier", "verified": True}

        tools = self._load_all_tools()
        handler_route = ""
        for t in tools:
            if t.get("key") == verifier_key:
                handler_route = t.get("handler") or ""
                break
        if not handler_route or "/" not in handler_route:
            return {"success": False, "output": f"Verifier {verifier_key} not found"}

        parts = handler_route.split("/")
        ws = self.AGU.get_active_workspace()
        plan_id = payload_extra.get("plan_id")
        plan = ws.get("plan", {}).get(plan_id) if plan_id else None
        sm = ws.get("state_machine", {}).get(plan_id) if plan_id else None

        vpayload = {
            "portfolio": self.AGU.portfolio,
            "org": self.AGU.org,
            "_entity_type": self.AGU.entity_type,
            "_entity_id": self.AGU.entity_id,
            "_thread": self.AGU.thread,
            "plan_id": plan_id,
            "plan_step": str(payload_extra.get("plan_step")),
            "plan": plan,
            "state_machine": sm,
        }
        resp = self.AGU.SHC.handler_call(self.AGU.portfolio, self.AGU.org, parts[0], parts[1], vpayload)
        return resp

    def _verification_succeeded(self, vres: Dict[str, Any]) -> bool:
        """True if verifier handler reported success and no failed step in stacked output."""
        if not vres.get("success"):
            return False
        inner = vres.get("output")
        if isinstance(inner, list):
            for item in inner:
                if isinstance(item, dict) and item.get("success") is False:
                    return False
        elif isinstance(inner, dict) and inner.get("success") is False:
            return False
        return True

    # ------------------------------------------------------------------
    def interpret_iteration(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        no_tools: bool = False,
    ):
        prompt = {
            "model": self.AGU.AI_2_MODEL,
            "messages": messages,
            "temperature": 0,
            "tool_choice": "auto",
        }
        if tools and not no_tools:
            prompt["tools"] = tools
            # OpenAI: at most one function call per assistant message (avoids orphan tool_call_ids).
            prompt["parallel_tool_calls"] = False

        prompt = self.AGU.sanitize(prompt)
        djson("specialist_last_prompt.json", prompt)
        response = self.AGU.llm(prompt)
        if not response:
            _logger_spec.error("interpret_failed | llm_empty_response")
            return {"success": False, "output": "LLM failure"}
        validation = self.AGU.validate_interpret_openai_llm_response(response)
        if not validation.get("success"):
            _logger_spec.error("interpret_failed | validation=%s", validation)
            return {"success": False, "output": validation}
        return {"success": True, "output": validation["output"]}

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            workspace = self.AGU.get_active_workspace()
            ws_id = workspace.get("_id")
            setattr(self.AGU, "workspace_id", ws_id)
            plan_id = payload.get("plan_id")
            step_id = str(payload.get("step_id"))
            action_name = payload.get("action") or ""
            _logger_spec.info(
                "specialist started action=%s step_id=%s plan_id=%s",
                action_name,
                step_id,
                plan_id,
            )
            step_inputs = payload.get("inputs") or {}
            title = payload.get("title") or ""

            trip_ctx = {}
            ages = None
            trip_doc_flights = None
            try:
                parts_eid = (self.AGU.entity_id or "").split("-")
                trip_id = "-".join(parts_eid[1:]) if len(parts_eid) > 1 else None
                if trip_id and self.AGU.portfolio and self.AGU.org:
                    trip_doc = self.AGU.DAC.get_a_b_c(
                        self.AGU.portfolio, self.AGU.org, "noma_travels", trip_id
                    )
                    if isinstance(trip_doc, dict):
                        ages = trip_doc.get("ages")
                        trip_ctx["trip_ages"] = ages
                        trip_ctx["trip_doc_present"] = True
                        if isinstance(trip_doc.get("flights"), list):
                            trip_doc_flights = trip_doc["flights"]
                            trip_ctx["trip_flights"] = trip_doc_flights
            except Exception:
                pass

            pend = self._get_pending_from_workspace(workspace)
            if (
                pend
                and str(pend.get("plan_id")) == str(plan_id)
                and str(pend.get("plan_step")) == step_id
            ):
                assistant_cmd = pend.get("assistant_command")
                if assistant_cmd:
                    tools_raw = self._load_all_tools()
                    extra = {
                        "_step_inputs": step_inputs,
                        "_plan_context": trip_ctx,
                        "plan": workspace.get("plan", {}).get(plan_id),
                        "intent": workspace.get("intent"),
                        "_continuity_plan_id": plan_id,
                        "_continuity_plan_step": step_id,
                    }
                    act_res = self._act(assistant_cmd, tools_raw, extra=extra)
                    self._set_pending_tool(None, ws_id)
                    if not act_res.get("success"):
                        return {
                            "success": False,
                            "output": {"status": "error", "detail": act_res},
                        }
                    vres = self._verify(
                        action_name,
                        {"plan_id": plan_id, "plan_step": step_id},
                    )
                    if not vres.get("success"):
                        return {
                            "success": False,
                            "output": {"status": "error", "detail": vres},
                        }
                    if self._verification_succeeded(vres):
                        return {"success": True, "output": {"status": "completed"}}
                    return {"success": False, "output": {"status": "error", "detail": vres}}

            action_doc = self._load_action_doc(action_name)
            approved_keys = self._approved_tool_keys(action_doc)
            list_tools_raw = self._load_all_tools()
            openai_tools = self._build_openai_tools(list_tools_raw, approved_keys)

            try:
                mh = self.AGU.get_message_history(include_internal=True)
            except TypeError:
                # Older AgentUtilities without include_internal flag
                mh = self.AGU.get_message_history()
            message_list = mh.get("output") or []
            message_list = self.AGU.strip_orphan_tool_messages(message_list)
            message_list = self.AGU.ensure_tool_responses_after_assistant(message_list)
            message_list = self.AGU.clear_tool_message_content(message_list, recent_tool_messages=1)

            search_leg_note = (
                "Flight search (search_flights_rextur): this step quotes **one leg only**. "
                "Pass `legs` as a **single** segment matching the step inputs below. "
                "Do **not** bundle outbound+return into one search — the next plan step runs the other leg. "
                "**At most one tool call per assistant message** — never emit two search_flights_rextur (or any two tools) "
                "in the same turn; run one tool, read the tool result, then continue in the next turn if needed. "
                "Use **show_options** / **ask_user** (see tools_reference) so the secretary can choose "
                "before **add_flight_rextur**."
            )
            react_trace_note = (
                "REACT SCRATCHPAD REQUIRED:\n"
                "- Before each tool call, first send a short assistant message (1–2 lines, no tool_calls) "
                "stating what you intend to do and why. The runtime stores this as internal reasoning and "
                "replays it back to you on the next turn — prefixed with `[reasoning]`.\n"
                "- After each tool result, send another short assistant message evaluating whether the result "
                "looks sensible (e.g. cheapest matches expected range; airlines/time match the ask). If it "
                "looks wrong, plan a retry with relaxed/adjusted parameters before exposing anything to the user.\n"
                "- If the runtime injects a message beginning with `WARNING:` into a tool result, treat it as "
                "evidence that the tool output is suspect — reason about it and re-call the tool with better "
                "arguments rather than passing the suspect data to show_options/add_flight.\n"
                "- Never respond with empty content or placeholders like `🤖🤖`. Always either call a tool or "
                "emit real reasoning / a user-facing message."
            )
            instruction_parts = [
                self._meta_language_directive(),
                action_doc.get("prompt_3_reasoning_and_planning") or "",
                action_doc.get("goal") or "",
                search_leg_note,
                react_trace_note,
                f"Step title (goal): {title}",
                f"Structured step inputs (belief): {json.dumps(step_inputs, ensure_ascii=False)}",
                "Use tools to progress. Do not paste large flight tables in assistant text; rely on tools.",
                "For user-visible questions use ask_user / show_options tools when appropriate.",
            ]
            if action_name == "post_execution" and trip_doc_flights:
                # Count segments per leg across all flight entries
                legs_shortlist_counts: Dict[str, int] = {}
                for fe in trip_doc_flights:
                    if not isinstance(fe, dict):
                        continue
                    for leg_k, segs in (fe.get("legs_shortlist") or {}).items():
                        cnt = len(segs) if isinstance(segs, list) else (1 if segs else 0)
                        legs_shortlist_counts[str(leg_k)] = legs_shortlist_counts.get(str(leg_k), 0) + cnt
                has_multiple = any(v > 1 for v in legs_shortlist_counts.values())
                send_rule = (
                    "DECISION RULE: there are multiple flight options per leg (legs_shortlist counts: "
                    f"{legs_shortlist_counts}). Use send_email_to_lawyer — do NOT book."
                ) if has_multiple else (
                    "DECISION RULE: exactly one option per leg detected. "
                    "You MAY book directly (book_flights_rextur) if the secretary explicitly confirms, "
                    "but send_email_to_lawyer remains the safe default."
                )
                instruction_parts.append(
                    "TRIP FLIGHTS DATA (from trip document):\n"
                    + json.dumps(trip_doc_flights, ensure_ascii=False, cls=DecimalEncoder)
                    + f"\n\n{send_rule}"
                )
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": "\n\n".join(instruction_parts)},
            ]
            for msg in message_list:
                messages.append(msg)

            iteration = 0

            while iteration < MAX_REACT_ITERATIONS:
                iteration += 1
                _logger_spec.debug(
                    "specialist loop iteration=%s/%s",
                    iteration,
                    MAX_REACT_ITERATIONS,
                )
                # Before each LLM turn: if this action has a verifier and the trip already
                # satisfies it (e.g. quote segment present), exit — skip further tools/chat.
                # Skip when verification is unset: _verify would otherwise no-op success.
                if (action_doc.get("verification") or "").strip():
                    early_v = self._verify(
                        action_name,
                        {"plan_id": plan_id, "plan_step": step_id},
                    )
                    if self._verification_succeeded(early_v):
                        _logger_verify.info(
                            "verify_result | phase=before_turn | result=SUCCESS | stop_loop=True"
                        )
                        return {"success": True, "output": {"status": "completed"}}

                interp = self.interpret_iteration(
                    messages,
                    openai_tools,
                    no_tools=False,
                )
                if not interp["success"]:
                    return {
                        "success": False,
                        "output": {"status": "error", "detail": interp},
                    }

                assistant_msg = interp["output"]

                if assistant_msg.get("tool_calls"):
                    raw_tcs = assistant_msg["tool_calls"]
                    if len(raw_tcs) > 1:
                        _logger_spec.warning(
                            "multiple tool_calls in one turn (%s); executing first only",
                            len(raw_tcs),
                        )
                    tc = raw_tcs[0]
                    tool_name = tc["function"]["name"]
                    _logger_spec.info("tool call selected=%s", tool_name)

                    # Persist only the tool call we execute — avoids orphan tool_call_ids vs one tool row.
                    cmd = {
                        "role": "assistant",
                        "tool_calls": [tc],
                        "content": assistant_msg.get("content") or "",
                    }
                    self.AGU.save_chat(cmd, connection_id=self.AGU.connection_id)

                    raw_args = tc["function"].get("arguments") or "{}"
                    args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                    if tool_name in CONSENT_REQUIRED_TOOLS:
                        nonce = random.randint(10000, 99999)
                        consent_body = self._consent_message_body(tool_name, args_obj)
                        self.AGU.save_chat(
                            consent_body,
                            interface="binary_consent",
                            msg_type="consent",
                            connection_id=self.AGU.connection_id,
                        )
                        self._set_pending_tool(
                            {
                                "plan_id": plan_id,
                                "plan_step": step_id,
                                "assistant_command": cmd,
                                "nonce": nonce,
                                "tool_name": tool_name,
                            },
                            ws_id,
                        )
                        self.AGU.mutate_workspace(
                            {
                                "action_log": {
                                    "plan_id": plan_id,
                                    "plan_step": step_id,
                                    "tool": tool_name,
                                    "status": "3",
                                    "nonce": nonce,
                                    "message": consent_body.get("content", ""),
                                    "type": "consent_rq",
                                }
                            }
                        )
                        return {"success": True, "output": {"status": "awaiting"}}

                    extra = {
                        "_step_inputs": step_inputs,
                        "_plan_context": trip_ctx,
                        "plan": workspace.get("plan", {}).get(plan_id),
                        "intent": workspace.get("intent"),
                        "_continuity_plan_id": plan_id,
                        "_continuity_plan_step": step_id,
                    }
                    act_res = self._act(cmd, list_tools_raw, extra=extra)
                    if not act_res.get("success"):
                        return {
                            "success": False,
                            "output": {"status": "error", "detail": act_res},
                        }

                    tool_payload = act_res.get("output") or {}
                    iface = act_res.get("_handler_interface")
                    try:
                        content_obj = json.loads(tool_payload.get("content", "{}"))
                        if isinstance(content_obj, dict):
                            iface = iface or content_obj.get("interface")
                            outn = content_obj.get("output")
                            if isinstance(outn, dict) and outn.get("interface"):
                                iface = iface or outn.get("interface")
                    except (json.JSONDecodeError, TypeError):
                        content_obj = {}

                    # Detect pause flag — tools can opt out to let the model chain another call.
                    pause_flag = True
                    if isinstance(content_obj, dict):
                        inner_out = content_obj.get("output") if isinstance(content_obj.get("output"), dict) else None
                        if isinstance(inner_out, dict) and "pause" in inner_out:
                            pause_flag = bool(inner_out.get("pause"))
                        elif "pause" in content_obj:
                            pause_flag = bool(content_obj.get("pause"))

                    # show_options uses interface=flights_rextur for the carousel UI; still pause for tap/reply
                    # unless the tool explicitly emitted pause=false.
                    if tool_name == "show_options" and pause_flag:
                        _logger_spec.info(
                            "specialist pause | tool=show_options interface=%s -> awaiting user",
                            iface,
                        )
                        return {"success": True, "output": {"status": "awaiting"}}

                    awaiting_iface = {"awaiting", "options", "awaiting_lawyer_reply"}
                    if iface in awaiting_iface and pause_flag:
                        if tool_name == "ask_user":
                            inner_out = content_obj.get("output") if isinstance(content_obj, dict) else None
                            question_text = (
                                (isinstance(inner_out, dict) and inner_out.get("message"))
                                or (isinstance(content_obj, dict) and content_obj.get("message"))
                                or ""
                            )
                            if question_text:
                                self.AGU.save_chat(
                                    {"role": "assistant", "content": str(question_text)},
                                    connection_id=self.AGU.connection_id,
                                )
                        return {"success": True, "output": {"status": "awaiting"}}

                    tool_content = tool_payload.get("content", "")
                    messages.append(cmd)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": tool_content}
                    )

                    vres = self._verify(
                        action_name,
                        {"plan_id": plan_id, "plan_step": step_id},
                    )
                    if self._verification_succeeded(vres):
                        return {"success": True, "output": {"status": "completed"}}
                    continue

                # No tool calls: internal reasoning unless user-facing allowed
                content = assistant_msg.get("content") or ""
                internal_msg = {
                    "role": "assistant",
                    "content": content,
                }
                self.AGU.save_chat(internal_msg, msg_type="internal", connection_id=self.AGU.connection_id)

                if iteration >= MAX_REACT_ITERATIONS - 1:
                    user_visible = {
                        "role": "assistant",
                        "content": content
                        or "Não consegui concluir esta etapa automaticamente; envie mais detalhes.",
                    }
                    self.AGU.save_chat(user_visible, connection_id=self.AGU.connection_id)
                    return {"success": True, "output": {"status": "awaiting"}}

                messages.append(internal_msg)

            return {"success": True, "output": {"status": "awaiting"}}

        except Exception as e:
            _logger_spec.exception("specialist_run_failed | %s", e)
            self.AGU.print_chat(f"Specialist error: {e}", "error")
            return {"success": False, "output": {"status": "error", "message": str(e)}}
