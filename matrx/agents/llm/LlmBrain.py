"""
LlmBrain

It composes the agent's infrastructure mixins (perception, execution,
communication, navigation, async LLM calls) and wires their lifecycle:
    * Perception    — filter raw observations and build the agent's world view.
    * Execution     — validate + dispatch a chosen action.
    * Communication — connect to message channel and process messages.
    * Navigation    — A* navigation state + per-tick stepping.
    * LLM calls     — submit non-blocking LLM calls.
"""

import logging
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

from matrx.helpers.logic_helpers import _chebyshev_distance
from matrx.agents.agent_utils.state import State

from matrx.actions.CustomActions import Idle as _Idle

from matrx.agents.llm.modules.agent_infra.async_model_prompting import AsyncLLMCalls
from matrx.agents.llm.modules.agent_infra.communication_module import Communication
from matrx.agents.llm.modules.agent_infra.execution_module import Execution
from matrx.agents.llm.modules.agent_infra.navigation_module import Navigation
from matrx.agents.llm.modules.agent_infra.perception_module import Perception
from matrx.helpers.logic_module import ActionValidator
from matrx.agents.ArtificialBrain import ArtificialBrain
from matrx.agents.llm.modules.memory_module import MemoryModule, SharedMemory
from matrx.metrics.agent_metrics import AgentMetricsTracker
from matrx.grid_world_creation.environment_info import EnvironmentInformation

logger = logging.getLogger('LlmBrain')

class LlmBrain(ArtificialBrain, Perception, Execution, Communication, Navigation, AsyncLLMCalls):
    """Infrastructure base class for LLM-driven rescue agents.
    """

    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str,
        include_human: bool = True,
        shared_memory: Optional[SharedMemory] = None,
        api_base: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        capability_knowledge: str = 'informed',
        env_info: Optional[EnvironmentInformation] = None,
    ) -> None:
        super().__init__(slowdown, condition, name, folder)

        self._slowdown = slowdown
        self._llm_model = llm_model
        self._api_base = api_base
        self._include_human = include_human
        self._partner_name = name
        self.teammates: set = set()

        self.env_info: EnvironmentInformation = env_info or EnvironmentInformation()

        self._capabilities = capabilities
        self._capability_knowledge = capability_knowledge

        self._validator = ActionValidator(
            capabilities=capabilities,
            capability_knowledge=capability_knowledge,
            grid_size=self.env_info.grid_size,
            valid_areas=frozenset(self.env_info.areas.keys()) if self.env_info.areas else None,
            env_info=self.env_info,
        )

        self.memory = MemoryModule()
        self.shared_memory: Optional[SharedMemory] = shared_memory

        self._pending_future: Optional[Future] = None
        self._last_validation_error: str = ''

        self.metrics: Optional[AgentMetricsTracker] = None
        self._tick_count: int = 0

        self.WORLD_STATE: Dict = {}

    def initialize(self) -> None:
        """Called once before the simulation starts."""
        self.init_navigation(agent_id=self.agent_id, action_set=self.action_set)
        self.metrics = AgentMetricsTracker(agent_id=self.agent_id)

        from matrx.metrics.simulation_metrics import SimulationMetrics
        SimulationMetrics.get_or_create().register(self)
        self.init_global_state()
        self.init_communication()
        logger.info('[%s] LlmBrain ready (model=%s)', self.agent_id, self._llm_model)


    def filter_observations(self, state: State) -> State:
        """Restrict observations to the agent's vision radius + doors + self.
        """
        agent_loc = state[self.agent_id]['location']
        self.state_for_navigation = state.copy()
        filtered = state.copy()
        self.teammates = set()

        if self.shared_memory and self.agent_id:
            self.shared_memory.add_to_set('registered_agents', self.agent_id)

        vision_radius = self._vision_radius

        keep = {self.agent_id, 'World'}
        if self._include_human:
            keep.add(self._partner_name)
            _partner_raw = state.get(self._partner_name)
            _partner_loc = tuple(_partner_raw.get('location', [0, 0])) if isinstance(_partner_raw, dict) else (0, 0)
            self.teammates.add((self._partner_name, _partner_loc))

        for obj_id, obj_data in filtered.items():
            if obj_id in keep:
                continue
            if isinstance(obj_id, str) and obj_id.startswith('rescuebot'):
                keep.add(obj_id)
                _raw = state.get(obj_id)
                _loc = _raw.get('location', [0, 0]) if isinstance(_raw, dict) else [0, 0]
                self.teammates.add((obj_id, tuple(_loc)))
            if not isinstance(obj_data, dict):
                keep.add(obj_id)
                continue
            loc = obj_data.get('location')
            if loc is not None and _chebyshev_distance(agent_loc, loc) <= vision_radius:
                keep.add(obj_id)
            if 'door' in str(obj_id).lower():
                keep.add(obj_id)

        if self.shared_memory:
            registered = self.shared_memory.retrieve('registered_agents') or []
            known_ids = {tid for tid, _ in self.teammates}
            for _reg_id in registered:
                if _reg_id != self.agent_id and _reg_id not in known_ids:
                    _raw = state.get(_reg_id)
                    loc = _raw.get('location', [0, 0]) if isinstance(_raw, dict) else [0, 0]
                    self.teammates.add((_reg_id, tuple(loc)))

        for obj_id in list(filtered.keys()):
            if obj_id not in keep:
                filtered.remove(obj_id)

        return filtered

    def update_knowledge(self, filtered_state: State) -> None:
        """Update state tracker and perception. Call at the top of decide_on_actions."""
        self._tick_count += 1
        self._state_tracker.update(self.state_for_navigation)
        self.WORLD_STATE = self.percept_state(
            filtered_state, agent_id=self.agent_id, teammates=self.teammates
        )
        self.update_world_belief(filtered_state)

        raw_loc = filtered_state[self.agent_id]['location']
        self.update_area_exploration(
            (int(raw_loc[0]), int(raw_loc[1])), vision_radius=self._vision_radius
        )

        if self.metrics:
            loc = self.WORLD_STATE.get('agent', {}).get('location')
            if loc:
                self.metrics.record_location(self._tick_count, loc[0], loc[1])

        if self.metrics:
            known_ids = {v['victim_id'] for v in self.metrics.victims_found}
            for v in self.WORLD_STATE.get('victims', []):
                vid = v.get('obj_id', '')
                if vid and vid not in known_ids:
                    severity = 'critical' if 'critical' in vid else 'mild'
                    vloc = v.get('location', (0, 0))
                    self.metrics.record_victim_found(self._tick_count, vid, severity, tuple(vloc))

        peer_msgs = self.received_messages
        self.process_messages(peer_msgs)

        if self.metrics:
            for msg in peer_msgs:
                if getattr(msg, 'from_id', '') == self.agent_id:
                    continue
                content = msg.content if hasattr(msg, 'content') else msg
                if isinstance(content, dict):
                    self.metrics.record_message_received(
                        self._tick_count,
                        getattr(msg, 'from_id', ''),
                        content.get('message_type', 'message'),
                        content.get('text', ''),
                    )

    # Execution: MATRX entry point

    def decide_on_action(self, state):
        """MATRX entry point (called once per tick).
        """
        act, params = self.decide_on_actions(state)
        params.setdefault('grab_range', 1)
        params.setdefault('max_objects', 1)

        # Own location, looked up by this agent's id (not a hard-coded name).
        me = state[self.agent_id]
        my_loc = me['location'] if me else None

        # Water slows movement to 13 ticks, except on doormat tiles.
        water_locs = []
        water_objs = state[{"name": "water"}]
        if water_objs:
            for water in water_objs:
                if water['location'] not in water_locs:
                    water_locs.append(water['location'])
        doormats = [(3, 5), (9, 5), (15, 5), (21, 5), (3, 6), (9, 6), (15, 6),
                    (3, 17), (9, 17), (15, 17), (3, 18), (9, 18), (15, 18), (21, 18)]
        if my_loc in water_locs and my_loc not in doormats:
            params['action_duration'] = 13
        else:
            params['action_duration'] = self._slowdown

        obj_id = params.get('object_id', '') or ''
        if act == 'RemoveObject' and 'stone' in obj_id:
            params['action_duration'] = 200
        if act == 'RemoveObject' and 'tree' in obj_id:
            params['action_duration'] = 100
        if act == 'CarryObject' and 'mild' in obj_id:
            params['action_duration'] = 150

        return act, params

    def call_llm(
        self,
        messages: List[Dict],
        tools: Optional[List] = None,
        tool_choice: str = 'auto',
        max_tokens: Optional[int] = None,
    ) -> None:
        """Submit an async LLM call.
        """
        if self.metrics:
            self.metrics.record_llm_call_start()
        self._pending_future = self.submit_llm_call(
            llm_model=self._llm_model,
            messages=messages,
            max_token_num=max_tokens,  # None → configured DEFAULT_SAMPLING
            tools=tools,
            tool_choice=tool_choice if tools else 'none',
            api_base=self._api_base,
        )

    # Utilities

    @property
    def agent_location(self) -> Tuple[int, int]:
        """Current agent (x, y) from world state."""
        if isinstance(self.WORLD_STATE, dict):
            return tuple(self.WORLD_STATE.get('agent', {}).get('location', (0, 0)))
        return (0, 0)

    def _idle(self, reason: str = 'idle') -> Tuple[str, Dict]:
        """Convenience: return an Idle action and record it in metrics."""
        if self.metrics:
            self.metrics.record_idle(self._tick_count, reason)
        return _Idle.__name__, {'duration_in_ticks': 1}
