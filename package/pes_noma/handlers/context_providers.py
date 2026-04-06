"""
Context Providers for the Specialist agent.

Each provider injects domain-specific context into the Specialist's LLM prompt.
Providers are registered by name and activated per-action via the 'context_modules'
field in the action definition (schd_actions).

Usage in specialist.py interpret():
    for module_name in action.get('context_modules', []):
        provider = CONTEXT_PROVIDERS.get(module_name)
        if provider:
            ctx = provider.get_context(workspace, continuity, agu)
            if ctx:
                messages.append({"role": "system", "content": ctx})
"""

import json


class ContextProvider:
    """Base class for context modules injected into the Specialist prompt."""

    def get_label(self) -> str:
        """Return the module name for logging."""
        raise NotImplementedError

    def get_context(self, workspace: dict, continuity: dict, agu) -> str:
        """Return a string to inject as a system message into the Specialist prompt.

        Args:
            workspace: The active workspace dict (contains state, cache, plan, state_machine, etc.)
            continuity: The continuity dict (plan_id, plan_step, action_step, tool_step)
            agu: AgentUtilities instance for data access (DAC, CHC, etc.)

        Returns:
            str: Context text to inject, or empty string to skip.
        """
        raise NotImplementedError


class TravelersContextProvider(ContextProvider):
    """Injects traveler information (names, documents, preferences) into the prompt."""

    def get_label(self) -> str:
        return 'travelers'

    def get_context(self, workspace: dict, continuity: dict, agu) -> str:
        try:
            intent = workspace.get('intent', {})
            party = intent.get('party', {})
            travelers = party.get('travelers', {})

            if not travelers:
                return ''

            lines = ["## Traveler Information"]

            # Structured traveler counts
            adults = travelers.get('adults', 0)
            children = travelers.get('children', 0)
            infants = travelers.get('infants', 0)
            total = adults + children + infants
            if total > 0:
                lines.append(f"Total passengers: {total} (adults: {adults}, children: {children}, infants: {infants})")

            # Named travelers (if available)
            traveler_list = travelers.get('list', [])
            if traveler_list:
                for i, t in enumerate(traveler_list):
                    name = t.get('name', t.get('full_name', f'Traveler {i+1}'))
                    doc_type = t.get('document_type', '')
                    doc_number = t.get('document_number', '')
                    prefs = t.get('preferences', {})
                    line = f"- {name}"
                    if doc_type and doc_number:
                        line += f" ({doc_type}: {doc_number})"
                    if prefs:
                        pref_str = ', '.join(f"{k}: {v}" for k, v in prefs.items() if v)
                        if pref_str:
                            line += f" | Preferences: {pref_str}"
                    lines.append(line)

            return '\n'.join(lines) if len(lines) > 1 else ''

        except Exception as e:
            print(f"TravelersContextProvider error: {e}")
            return ''


class FlightContextProvider(ContextProvider):
    """Injects already-searched/selected flight data from workspace cache into the prompt."""

    def get_label(self) -> str:
        return 'flights'

    def get_context(self, workspace: dict, continuity: dict, agu) -> str:
        try:
            cache = workspace.get('cache', {})
            flight_entries = {k: v for k, v in cache.items() if 'flight' in k.lower()}

            if not flight_entries:
                return ''

            lines = ["## Previous Flight Search Results"]
            for key, entry in flight_entries.items():
                inp = entry.get('input', {})
                out = entry.get('output', {})
                origin = inp.get('from_airport_code', inp.get('origin', '?'))
                dest = inp.get('to_airport_code', inp.get('destination', '?'))
                date = inp.get('outbound_date', inp.get('date', '?'))
                lines.append(f"- Search: {origin} -> {dest} on {date}")

                # Include selection if available
                if isinstance(out, dict):
                    selected = out.get('selected', out.get('selection', None))
                    if selected:
                        lines.append(f"  Selected: {json.dumps(selected, default=str)[:200]}")
                elif isinstance(out, list) and len(out) > 0:
                    lines.append(f"  Results: {len(out)} options found")

            return '\n'.join(lines) if len(lines) > 1 else ''

        except Exception as e:
            print(f"FlightContextProvider error: {e}")
            return ''


class HotelContextProvider(ContextProvider):
    """Injects already-searched/selected hotel data from workspace cache into the prompt."""

    def get_label(self) -> str:
        return 'hotels'

    def get_context(self, workspace: dict, continuity: dict, agu) -> str:
        try:
            cache = workspace.get('cache', {})
            hotel_entries = {k: v for k, v in cache.items() if 'hotel' in k.lower()}

            if not hotel_entries:
                return ''

            lines = ["## Previous Hotel Search Results"]
            for key, entry in hotel_entries.items():
                inp = entry.get('input', {})
                out = entry.get('output', {})
                city = inp.get('city', inp.get('destination', '?'))
                checkin = inp.get('check_in', inp.get('checkin', '?'))
                checkout = inp.get('check_out', inp.get('checkout', '?'))
                lines.append(f"- Search: {city}, {checkin} to {checkout}")

                if isinstance(out, dict):
                    selected = out.get('selected', out.get('selection', None))
                    if selected:
                        lines.append(f"  Selected: {json.dumps(selected, default=str)[:200]}")
                elif isinstance(out, list) and len(out) > 0:
                    lines.append(f"  Results: {len(out)} options found")

            return '\n'.join(lines) if len(lines) > 1 else ''

        except Exception as e:
            print(f"HotelContextProvider error: {e}")
            return ''


class FullHistoryProvider(ContextProvider):
    """Injects the full conversation history (not filtered by step) into the prompt."""

    def get_label(self) -> str:
        return 'full_history'

    def get_context(self, workspace: dict, continuity: dict, agu) -> str:
        try:
            history_response = agu.get_message_history(target='llm')
            if not history_response.get('success'):
                return ''

            messages = history_response.get('output', [])
            if not messages:
                return ''

            lines = ["## Full Conversation History (summary)"]
            # Include only user and assistant text messages to keep it concise
            for m in messages:
                role = m.get('role', '')
                content = m.get('content', '')
                if role in ('user', 'assistant') and isinstance(content, str) and content.strip():
                    # Truncate long messages
                    truncated = content[:300] + '...' if len(content) > 300 else content
                    lines.append(f"[{role}]: {truncated}")

            return '\n'.join(lines) if len(lines) > 1 else ''

        except Exception as e:
            print(f"FullHistoryProvider error: {e}")
            return ''


# Registry of available context providers
CONTEXT_PROVIDERS = {
    'travelers': TravelersContextProvider(),
    'flights': FlightContextProvider(),
    'hotels': HotelContextProvider(),
    'full_history': FullHistoryProvider(),
}
