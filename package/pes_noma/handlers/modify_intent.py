from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import importlib.resources
import json
import time
from decimal import Decimal

try:
    import yaml
except ImportError:
    yaml = None

from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config
from renglo.data.data_controller import DataController

from contextvars import ContextVar

from pes_noma.handlers import utilities as plan_utilities
from pes_noma.handlers.intent_invalidation import (
    apply_working_memory_invalidations_for_intent_modification,
)
from pes_noma.handlers.intent_revision import IntentRevisor, resolve_intent_revision_rules


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)

@dataclass
class RequestContext:
    portfolio: str = ''
    org: str = ''
    search_results: Dict[str, Any] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    case_group: str = ''
    init: Dict[str, Any] = field(default_factory=dict)
    trip: Dict[str, Any] = field(default_factory=dict)


request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

'''
ModifyIntent: intent + modification request -> revised intent (IntentRevisor) -> plan (ProposePlan).
'''

class ModifyIntent:
    def __init__(self, prompts: Optional[Dict[str, str]] = None):
        self.config = load_config()

        self.AGU = None
        self.DAC = DataController(config=self.config)

        self.prompts = prompts or {}

    def _get_context(self) -> RequestContext:
        return request_context.get()

    def _set_context(self, context: RequestContext):
        request_context.set(context)

    def _update_context(self, **kwargs):
        context = self._get_context()
        for key, value in kwargs.items():
            setattr(context, key, value)
        self._set_context(context)

    def _load_prompts(self, portfolio: str, org: str, prompt_ring: str = "pes_prompts", case_group: str = None) -> Dict[str, str]:
        prompts = {
            'to_intent': '',
            'modify_intent_delta': '',
            'compose_plan_light': '',
            'modify_intent': '',
            'intent_revision_rules': '',
        }

        try:
            if not case_group:
                raise Exception('No case group')

            query = {
                'portfolio': portfolio,
                'org': org,
                'ring': prompt_ring,
                'value': case_group,
                'limit': 99,
                'operator': 'begins_with',
                'lastkey': None,
                'sort': 'asc'
            }
            response = self.DAC.get_a_b_query(query)

            if response and 'items' in response:
                for item in response['items']:
                    key = item.get('key', '').lower()
                    prompt_text = item.get('prompt', '')
                    if prompt_text:
                        prompt_text = prompt_text.lstrip()
                    if key == 'to_intent':
                        prompts['to_intent'] = prompt_text
                    elif key == 'modify_intent_delta':
                        prompts['modify_intent_delta'] = prompt_text
                    elif key == 'compose_plan_light':
                        prompts['compose_plan_light'] = prompt_text
                    elif key == 'modify_intent':
                        prompts['modify_intent'] = prompt_text
                    elif key == 'intent_revision_rules':
                        prompts['intent_revision_rules'] = prompt_text
        except Exception as e:
            print(f'Warning: Could not load prompts from database: {str(e)}')

        return prompts

    def _load_prompts_from_package(self, package_route: str) -> Dict[str, str]:
        print('Using prompts from package')

        prompts = {
            'to_intent': '',
            'modify_intent_delta': '',
            'compose_plan_light': '',
            'modify_intent': '',
            'intent_revision_rules': '',
        }
        if not yaml:
            print('Warning: PyYAML not installed. Cannot load prompts from package.')
            return prompts
        try:
            parts = package_route.split('.')
            if not parts:
                raise ValueError('package_route must be non-empty (e.g. noma.prompts.pes)')
            package_name = parts[0]
            path_parts = parts[1:] if len(parts) > 1 else []
            resource_root = importlib.resources.files(package_name)
            for p in path_parts:
                resource_root = resource_root / p
            if not resource_root.is_dir():
                raise FileNotFoundError(f'Package resource path does not exist: {package_route}')
            for entry in resource_root.iterdir():
                if entry.name.endswith('.yaml') or entry.name.endswith('.yml'):
                    try:
                        with importlib.resources.as_file(entry) as f:
                            with open(f, 'r') as fp:
                                raw = yaml.safe_load(fp)
                        if not isinstance(raw, dict):
                            continue
                        key = (raw.get('key') or '').lower()
                        prompt_text = raw.get('prompt', '') or ''
                        if prompt_text:
                            prompt_text = prompt_text.lstrip()
                        if key in prompts:
                            prompts[key] = prompt_text
                    except Exception as e:
                        print(f'Warning: Could not load prompt from {entry.name}: {e}')
        except Exception as e:
            print(f'Warning: Could not load prompts from package {package_route}: {e}')
        return prompts

    def run(self, payload):

        '''
        Payload
        {
            "portfolio":"",
            "org":"",
            "case_group":"",
            "message":"",  # Human-readable paragraph describing the modifications to make
            "plan": {},    # Existing plan to modify (Plan object with id, steps, meta)
            "state_machine": {},  # Optional: Execution state; used for in-place reconcile (Option C)
            "trip": {},    # Optional: Trip document showing existing reservations
            "_init": {},   # Optional: preserve_plan_id (default false) — true reuses current plan id
        }
        '''
        function = 'run > modify_intent'
        context = RequestContext()

        if 'portfolio' in payload:
            context.portfolio = payload['portfolio']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No portfolio provided'}

        if 'org' in payload:
            context.org = payload['org']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No org provided'}

        if 'case_group' in payload:
            context.case_group = payload['case_group']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No case group provided'}

        if 'message' not in payload:
            return {'success':False,'function':function,'input':payload,'output':'No modification request (message) provided'}

        if 'plan' not in payload:
            return {'success':False,'function':function,'input':payload,'output':'No existing plan provided'}

        if '_init' in payload:
            raw = payload['_init']
            context.init = json.loads(raw) if isinstance(raw, str) else raw
        else:
            context.init = {}

        if 'trip' in payload:
            raw_trip = payload['trip']
            context.trip = json.loads(raw_trip) if isinstance(raw_trip, str) else raw_trip
        else:
            context.trip = {}

        try:
            self._set_context(context)

            self.AGU = AgentUtilities(
                self.config,
                context.portfolio,
                context.org,
                'some_entity_type',
                'some_entity_id',
                'some_thread'
            )

            print('Initializing PES>ModifyIntent')

            base_intent = payload.get('intent')
            modification_request = payload['message']
            print(
                "ModifyIntent.run(): payload summary "
                f"message_len={len(str(modification_request or ''))}, "
                f"intent_type={type(base_intent).__name__}, "
                f"intent_keys={list(base_intent.keys()) if isinstance(base_intent, dict) else 'n/a'}, "
                f"has_plan={'plan' in payload}, has_state_machine={'state_machine' in payload}"
            )
            if not base_intent or not isinstance(base_intent, dict):
                return {'success': False, 'function': function, 'input': payload, 'output': 'ERROR:@modify_intent/run: No intent provided. Plan modification requires intent from generate_plan cache.'}
            try:
                prompt_route = (context.init or {}).get('prompt_route') if isinstance(context.init, dict) else None
                print(
                    "ModifyIntent.run(): resolve prompts "
                    f"prompt_route={prompt_route!r}, "
                    f"portfolio={context.portfolio!r}, org={context.org!r}, case_group={context.case_group!r}"
                )
                if prompt_route:
                    prompts = self._load_prompts_from_package(prompt_route)
                else:
                    prompts = self._load_prompts(context.portfolio, context.org, case_group=context.case_group)
                rules_text, extra_rules = resolve_intent_revision_rules(
                    config=self.config,
                    prompts=prompts,
                    init=context.init if isinstance(context.init, dict) else None,
                )
                revisor_kw: Dict[str, Any] = {}
                if rules_text:
                    revisor_kw["rules_text"] = rules_text
                revisor = IntentRevisor(self.AGU, **revisor_kw)
                print(
                    "ModifyIntent.run(): invoking IntentRevisor.revise "
                    f"extra_rules_len={len(str(extra_rules or ''))}, rules_text_len={len(str(rules_text or ''))}"
                )
                updated_intent = revisor.revise(
                    portfolio=context.portfolio,
                    org=context.org,
                    case_group=context.case_group,
                    current_intent=base_intent,
                    user_message=modification_request,
                    extra_rules=extra_rules,
                )
                print(
                    "ModifyIntent.run(): revise result "
                    f"type={type(updated_intent).__name__}, "
                    f"keys={list(updated_intent.keys()) if isinstance(updated_intent, dict) else 'n/a'}"
                )
                if not updated_intent:
                    return {'success': False, 'function': function, 'input': payload, 'output': 'ERROR:@modify_intent/run: Could not update intent from modification request.'}
                try:
                    apply_working_memory_invalidations_for_intent_modification(
                        base_intent, updated_intent
                    )
                except Exception:
                    pass
                preserve_plan_id = False
                if isinstance(context.init, dict) and 'preserve_plan_id' in context.init:
                    preserve_plan_id = bool(context.init.get('preserve_plan_id'))

                raw_plan = payload.get('plan')
                existing_plan: Optional[Dict[str, Any]] = None
                if raw_plan is not None:
                    existing_plan = (
                        json.loads(raw_plan) if isinstance(raw_plan, str) else raw_plan
                    )
                    if not isinstance(existing_plan, dict):
                        existing_plan = None

                target_plan_id: Optional[str] = None
                if preserve_plan_id and existing_plan:
                    target_plan_id = str(existing_plan.get('id') or '').strip() or None

                raw_sm = payload.get('state_machine')
                existing_state: Optional[Dict[str, Any]] = None
                if raw_sm is not None:
                    existing_state = (
                        json.loads(raw_sm) if isinstance(raw_sm, str) else raw_sm
                    )
                    if not isinstance(existing_state, dict):
                        existing_state = None

                from pes_noma.handlers.propose_plan import ProposePlan
                propose_payload = {
                    'portfolio': context.portfolio,
                    'org': context.org,
                    'case_group': context.case_group,
                    'intent': updated_intent,
                    '_init': context.init,
                }
                if target_plan_id:
                    propose_payload['target_plan_id'] = target_plan_id
                response = ProposePlan().run(propose_payload)
                if response.get('success') and response.get('output'):
                    plan_dict = response['output']['plan']
                    if not isinstance(plan_dict, dict):
                        plan_dict = {}
                    if target_plan_id:
                        plan_dict['id'] = target_plan_id
                        meta = plan_dict.setdefault('meta', {})
                        if isinstance(meta, dict):
                            try:
                                rev = int(meta.get('in_place_revision', 0) or 0) + 1
                            except (TypeError, ValueError):
                                rev = 1
                            meta['in_place_revision'] = rev
                            meta['in_place_revised_at'] = time.strftime(
                                '%Y-%m-%dT%H:%M:%SZ', time.gmtime()
                            )

                    canonical: Dict[str, Any] = {
                        'plan': plan_dict,
                        'intent': updated_intent,
                    }
                    if target_plan_id and existing_plan and isinstance(existing_plan, dict):
                        base_state = existing_state if isinstance(existing_state, dict) else {}
                        canonical['state_machine'] = (
                            plan_utilities.reconcile_plan_state_in_place(
                                existing_plan,
                                plan_dict,
                                base_state,
                            )
                        )
                    return {
                        'success': True,
                        'interface': 'plan',
                        'input': payload,
                        'output': canonical,
                        'stack': [response],
                    }
                return {'success': False, 'function': function, 'input': payload, 'output': 'ERROR:@modify_intent/run: Could not generate plan from updated intent.'}
            except Exception as e:
                print(f'Intent-based regeneration failed: {e}')
                return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@modify_intent/run: {str(e)}'}

        except Exception as e:
            print(f'Error during execution: {str(e)}')
            import traceback
            traceback.print_exc()
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@modify_intent/run: {str(e)}'}
