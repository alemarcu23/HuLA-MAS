import json
import logging
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from matrx.helpers.logic_helpers import _chebyshev_distance, extract_action_json
from matrx.helpers.navigation_helpers import apply_navigation
from matrx.agents.agent_utils.state import State
from matrx.messages.message import Message

from matrx.agents.llm.modules.agent_infra.async_model_prompting import _strip_thinking
from matrx.agents.llm.modules.profile_module import (
    Profile,
    MSG_CAPABILITY_CLAIM,
    ROLE_CLAIM_MSG_TYPE,
    get_capability_summary,
)
from matrx.agents.llm.LlmBrain import LlmBrain
from matrx.agents.llm.modules.joint_action import JointAction, MSG_ASK_HELP, MSG_HELP
from matrx.agents.llm.modules.planning_module import Planning
from matrx.agents.llm.modules.reasoning_module import (
    REASONING_STRATEGY_REGISTRY,
    FollowupRequest,
    ActionCommit,
    build_reasoning_strategy,
)
from matrx.agents.llm.tool_registry import build_tool_schemas
from matrx.agents.llm.modules.memory_module import SharedMemory
from matrx.metrics.agent_metrics import COOPERATIVE_ACTIONS
from matrx.grid_world_creation.environment_info import EnvironmentInformation

logger = logging.getLogger('LlmAgent')


def _human_readable_action(name: str, args: dict) -> str:
    """Return a concise human-readable description of a MATRX action."""
    if name == 'MoveTo':
        return f"Moving to ({args.get('x')}, {args.get('y')})"
    if name == 'CarryObject':
        return f"Picking up {args.get('object_id', 'victim')}"
    if name == 'CarryObjectTogether':
        return (f"Coop-carrying {args.get('object_id', 'victim')} "
                f"with {args.get('partner_id', 'teammate')}")
    if name == 'Drop':
        return "Dropping victim at drop zone"
    if name == 'RemoveObject':
        return f"Removing obstacle {args.get('object_id', '')}"
    if name == 'RemoveObjectTogether':
        return (f"Coop-removing {args.get('object_id', 'obstacle')} "
                f"with {args.get('partner_id', 'teammate')}")
    if name == 'SearchArea':
        return f"Searching {args.get('area_id') or args.get('area', 'area')}"
    if name == 'SendMessage':
        mtype = args.get('message_type', 'message')
        to    = args.get('send_to', 'team')
        return f"Sending {mtype} to {to}"
    if name == 'Idle':
        return "Idle (waiting)"
    return f"Executing {name}"


class PipelineStage(Enum):
    IDLE = 'idle'
    PLANNING = 'planning'
    REASONING = 'reasoning'
    EXECUTE = 'execute'
    COMM_DISPATCH = 'comm_dispatch'


class LlmAgent(JointAction, LlmBrain):
    def __init__(
        self,
        slowdown: int,
        condition: str,
        name: str,
        folder: str,
        llm_model: str = 'ollama/llama3',
        strategy: str = 'react',
        include_human: bool = True,
        shared_memory: Optional[SharedMemory] = None,
        planning_strategy: str = 'io',
        api_base: Optional[str] = None,
        capabilities: Optional[Dict] = None,
        capability_knowledge: str = 'informed',
        env_info: Optional[EnvironmentInformation] = None,
        initial_role: Optional[str] = None,
        role_goal: str = '',
        game_rules: str = '',
        joint_action_ask_trigger: str = 'auto_bridge',
    ) -> None:
        super().__init__(
            slowdown=slowdown,
            condition=condition,
            name=name,
            folder=folder,
            llm_model=llm_model,
            include_human=include_human,
            shared_memory=shared_memory,
            api_base=api_base,
            capabilities=capabilities,
            capability_knowledge=capability_knowledge,
            env_info=env_info,
        )
        self._init_joint_action(ask_trigger=joint_action_ask_trigger)

        self.planner = Planning(strategy=planning_strategy)
        self._strategy = strategy if strategy in REASONING_STRATEGY_REGISTRY else 'io'
        self.reasoning = build_reasoning_strategy(self._strategy)

        self.init_area_tracker(self.env_info.get_area_cells())
        self.tools_by_name, self.tool_schemas = build_tool_schemas()

        # Pipeline state
        self._pipeline_stage: PipelineStage = PipelineStage.IDLE
        self._pipeline_context: Dict[str, Any] = {}

        self._comm_msg_cursor: int = 0
        self._episode_msg_cursor: int = 0

        self._current_task: Optional[str] = None        # atomic next plan
        self._current_plan: Optional[str] = None   # planning
        self._current_action: Optional[Dict[str, Any]] = None  # reasoning
        self._last_plan: Optional[str] = None           # _current_plan from previous cycle
        self._last_action: Optional[Dict[str, Any]] = None     # _current_action from previous cycle
        self._last_action_outcome: str = ''

        self._game_rules = game_rules.replace('{drop_zone}', str(tuple(self.env_info.drop_zone)))

        self.profile = Profile(
            self._capabilities,
            roles=[initial_role] if initial_role else [],
            knowledge=self._capability_knowledge,
            goal=role_goal,
        )
        self._capabilities = self.profile.as_dict()
        self._high_level_task: str = self.profile.role_goal()
        self._role_published: bool = False
        self._team_roles: Dict[str, str] = {} 
        self._team_capabilities: Dict[str, Dict[str, Any]] = {}
        self._team_plans: Dict[str, Dict[str, str]] = {}

        self._recent_actions: deque = deque(maxlen=3)

        self._supervisor_cmd_history: List[Dict[str, Any]] = []

        print(
            f'[LlmAgent] Created '
            f'(model={llm_model}, strategy={self._strategy}, '
            f'plan_strategy={planning_strategy}, '
            f'caps={capabilities})'
        )

    # Perception

    def update_knowledge(self, filtered_state: State) -> None:
        super().update_knowledge(filtered_state)
        self._ingest_team_messages(self.received_messages)
        self._persist_new_messages_to_memory()
        self._ingest_joint_messages(self.received_messages, self._tick_count)

    def _ingest_team_messages(self, messages) -> None:
        """Update team belief (roles, plans, capabilities) from incoming messages."""
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else {}
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    content = {}
            if not isinstance(content, dict):
                continue
            sender = getattr(msg, 'from_id', '')
            if not sender or sender == self.agent_id:
                continue
            mtype = content.get('message_type')

            if mtype == ROLE_CLAIM_MSG_TYPE:
                role = content.get('role', '')
                if role:
                    self._team_roles[sender] = role
            elif mtype == 'plan_update':
                task = content.get('task', '')
                if task:
                    mot = (content.get('motivation', '') or '')[:50]
                    self._team_plans[sender] = {'task': task, 'motivation': mot}
            elif mtype == MSG_CAPABILITY_CLAIM:
                caps = content.get('capabilities')
                if isinstance(caps, dict):
                    self._team_capabilities[sender] = caps

    def _persist_new_messages_to_memory(self) -> None:
        """Save messages received since last tick to base memory (cursor-tracked)."""
        all_msgs = self.all_messages_raw
        new_msgs = all_msgs[self._comm_msg_cursor:]
        for msg in new_msgs:
            self.memory.base.update('received_message', {
                'from': msg.get('from'),
                'type': msg.get('message_type'),
                'text': msg.get('text'),
                'tick': self._tick_count,
            })
        self._comm_msg_cursor = len(all_msgs)

    # Execution: action dispatch

    def decide_on_actions(self, filtered_state: State) -> Tuple[Optional[str], Dict]:
        self.update_knowledge(filtered_state)

        # Infrastructure: carry retry, navigation
        action = self._run_infra(filtered_state)
        if action is not None:
            action_name = action[0] if isinstance(action, tuple) else action
            if (self.metrics and action_name in COOPERATIVE_ACTIONS
                    and self.previous_action != action_name):
                loc = self.WORLD_STATE.get('agent', {}).get('location', (0, 0))
                kwargs = action[1] if isinstance(action, tuple) else {}
                self.metrics.record_action(self._tick_count, action_name, kwargs, tuple(loc))
            return action

        # Poll pending LLM future
        if self._pending_future is not None:
            try:
                result = self.get_llm_result(self._pending_future)
            except Exception as exc:
                logger.warning('[%s] LLM future raised: %s', self.agent_id, exc)
                self._pending_future = None
                self._pipeline_stage = PipelineStage.IDLE
                return self._idle()
            if result is None:
                return self._idle(reason='llm_wait')
            self._pending_future = None
            return self._on_llm_result(result)

        # Advance pipeline
        return self._advance_pipeline()

    def _open_episode_cycle(self) -> None:
        # save all cycle metrics and data
        prev = self.memory.episode.get_open_episode()
        if prev is not None and not prev.closed:
            succeeded = None
            reason = None
            if self.previous_action_result is not None:
                succeeded = bool(self.previous_action_result.succeeded)
                reason = str(getattr(self.previous_action_result, 'result', ''))
            self.memory.episode.close_episode(
                tick=self._tick_count,
                succeeded=succeeded,
                reason=reason,
            )

        self._last_plan = self._current_plan
        self._last_action = self._current_action
        self._current_plan = None
        self._current_action = None

        self.memory.episode.open_episode(
            tick=self._tick_count,
            agent_id=self.agent_id,
            task=self._current_task or '',
            role=self.profile.role_str(),
        )

        all_msgs = list(self.all_messages_raw)
        new_msgs = all_msgs[self._episode_msg_cursor:]
        self._episode_msg_cursor = len(all_msgs)
        self.memory.episode.set_received_messages([
            {
                'from': m.get('from'),
                'type': m.get('message_type'),
                'text': m.get('text'),
            }
            for m in new_msgs
        ])

    def _advance_pipeline(self) -> Tuple[Optional[str], Dict]:
        if self._joint_active():
            return self._idle(reason='joint_action_in_progress')

        if self._pipeline_stage == PipelineStage.IDLE:
            self._broadcast_profile_once()
            self._pipeline_context = {}
            self._open_episode_cycle()
            self._pipeline_stage = PipelineStage.PLANNING
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self.plan()
        if self._pipeline_stage == PipelineStage.REASONING:
            return self.reason()
        if self._pipeline_stage == PipelineStage.EXECUTE:
            return self.execute()
        if self._pipeline_stage == PipelineStage.COMM_DISPATCH:
            return self.communicate()

        return self._idle()

    def _on_llm_result(self, result) -> Tuple[Optional[str], Dict]:
        if self.metrics:
            self.metrics.record_llm_call_end()
        if self._pipeline_stage == PipelineStage.PLANNING:
            return self._handle_planning_result(result)
        if self._pipeline_stage == PipelineStage.REASONING:
            return self._handle_reasoning_result(result)
        return self._idle()


    def _broadcast_profile_once(self) -> None:
        if self._role_published:
            return
        self._role_published = True

        self.send_message(Message(
            content={'message_type': ROLE_CLAIM_MSG_TYPE, 'role': self.profile.role_str()},
            from_id=self.agent_id,
            to_id=None,
        ))
        # Send capabilities to teammates
        if self._capabilities:
            self.send_message(Message(
                content={
                    'message_type': MSG_CAPABILITY_CLAIM,
                    'capabilities': dict(self._capabilities),
                },
                from_id=self.agent_id,
                to_id=None,
            ))

    def _build_common_context(self) -> Dict[str, Any]:
        """Return the game rules and capability text shared by the LLM stages."""
        return {
            'game_rules': self._game_rules,
            'agent_capabilities': self.profile.capability_prompt(),
        }

    # PLANNING stage

    def plan(self) -> Tuple[Optional[str], Dict]:
        agent_info = self.WORLD_STATE.get('agent', {})
        common_ctx = self._build_common_context()

        teammates_enriched = []
        for t in self.WORLD_STATE.get('teammates', []):
            tp = self._team_plans.get(t['object_id'], {})
            if isinstance(tp, str): # from old code
                tp = {'task': tp, 'motivation': ''}
            t_caps = self._team_capabilities.get(t['object_id'])
            teammates_enriched.append({
                'id': t['object_id'],
                'location': [t.get('x'), t.get('y')],
                'role': self._team_roles.get(t['object_id'], 'unknown'),
                'capabilities': get_capability_summary(t_caps) if t_caps else 'unknown',
                'current_plan': tp.get('task', ''),
                'plan_motivation': tp.get('motivation', ''),
            })

        # OBSERVATION
        observation = {
            'victims': self.WORLD_STATE.get('victims', []) or [],
            'teammates': self.WORLD_STATE.get('teammates', []) or [],
            'obstacles': self.WORLD_STATE.get('obstacles', []) or [],
        }

        messages = self.get_messages(limit=10)

        # supervisor stuff
        try:
            from matrx.agents.llm.modules import supervisor_channel as _sup_ch
            new_sup_cmds = _sup_ch.get_unconsumed(self.agent_id)
        except Exception:
            new_sup_cmds = []
        if new_sup_cmds:
            self._supervisor_cmd_history.extend(new_sup_cmds)
            for _cmd in new_sup_cmds:
                self.memory.base.update(
                    'supervisor_command',
                    {'text': _cmd.get('text', ''), 'tick': _cmd.get('tick'),
                     'target': _cmd.get('target_agent', 'all')},
                    tick=self._tick_count,
                )
            if self.shared_memory:
                try:
                    _log = list(self.shared_memory.retrieve('supervisor_commands_log') or [])
                    _log.extend(new_sup_cmds)
                    self.shared_memory.update('supervisor_commands_log', _log)
                except Exception:
                    pass
        latest_sup_cmds = self._supervisor_cmd_history[-8:]

        urgent_abandon = self._pending_help_abandon
        self._pending_help_abandon = None

        last_validation_error = self._last_validation_error or ''
        self._last_validation_error = ''

        planning_inputs = {
            'context': {
                'agent': self.agent_id,
                'role': self.profile.role_str(),
                'position': agent_info.get('location'),
                'carrying': agent_info.get('carrying') or 'nothing',
                'capabilities': common_ctx['agent_capabilities'],
                'teammates': teammates_enriched,
            },
            'high_level_task': self._high_level_task or '',
            'last_plan': self._last_plan or '',
            'last_action': self._last_action or {},
            'observation': observation,
            'world_state_belief': self._world_state_belief,
            'memory': self.memory.episode.to_prompt_memory(n=10),
            'messages': messages,
            'urgent_abandon': urgent_abandon,
            'last_validation_error': last_validation_error,
            'supervisor_commands': latest_sup_cmds,
            'area_summaries': [
                {**s, 'door': self.env_info.get_door(int(s['name'].split()[-1]))}
                for s in self.get_area_summaries()
            ],
        }

        prompt = self.planner.get_planning_prompt(planning_inputs)
        self.call_llm(prompt)
        return self._idle()

    def _handle_planning_result(self, result) -> Tuple[Optional[str], Dict]:
        text = _strip_thinking(getattr(result[0], 'content', '') or '') or ''

        parsed = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = extract_action_json(text)

        motivation = ''
        task = ''
        if isinstance(parsed, dict):
            task = (parsed.get('next_plan') or parsed.get('next_task') or '').strip()
            motivation = (parsed.get('motivation') or '').strip()
        else:
            task = text.strip()

        critic_result: Optional[Dict[str, Any]] = None
        if self.planner.emits_critic and isinstance(parsed, dict):
            critic_raw = parsed.get('critic')
            if isinstance(critic_raw, dict):
                critic_result = {
                    'success': bool(critic_raw.get('success', True)),
                    'critique': critic_raw.get('critique', '') or '',
                }
            else:
                critic_result = {
                    'success': bool(parsed.get('success', True)),
                    'critique': parsed.get('critique', '') or '',
                }

        self._pipeline_context['critic_result'] = critic_result
        self._pipeline_context['motivation'] = motivation
        if critic_result is not None:
            self.memory.episode.set_critic_on_last_closed(critic_result)
            status = 'SUCCESS' if critic_result['success'] else f"FAILED: {critic_result['critique']}"
            self.memory.base.update('plan_status', f"[tick {self._tick_count}] Last action: {status}")

        self._pipeline_context['planned_task'] = task
        self.memory.base.update('planned_task', {'task': task, 'tick': self._tick_count})
        self.memory.episode.set_planned_task(task)
        self.memory.episode.set_motivation(motivation)

        if task:
            short_motivation = (motivation or '').strip()[:50]
            self.send_message(Message(
                content={
                    'message_type': 'plan_update',
                    'text': f'plan: {task}',
                    'motivation': short_motivation,
                },
                from_id=self.agent_id,
                to_id=None,
            ))
            self._current_task = task
            self._current_plan = task

            try:
                from matrx.agents.llm.modules.supervisor_channel import push_agent_status
                push_agent_status(
                    agent_id=self.agent_id,
                    message_type='plan_human',
                    text=f'Plan: {task}',
                    task=task,
                    motivation=(motivation or '').strip(),
                    tick=self._tick_count,
                )
            except Exception:
                pass

        self._pipeline_stage = PipelineStage.REASONING
        return self._advance_pipeline()

    # REASONING stage

    def reason(self) -> Tuple[Optional[str], Dict]:
        if getattr(self.reasoning, '_phase', 'main') in ('main', 'done'):
            self.reasoning.reset_phase()
        self.memory.base.compress()

        agent_info = self.WORLD_STATE.get('agent', {})
        observation = {
            'victims': self.WORLD_STATE.get('victims', []) or [],
            'teammates': self.WORLD_STATE.get('teammates', []) or [],
            'obstacles': self.WORLD_STATE.get('obstacles', []) or [],
        }

        recent_actions_list = list(self._recent_actions)
        critic = self._pipeline_context.get('critic_result') or {}
        last_critique = critic.get('critique', '')
        motivation = self._pipeline_context.get('motivation', '')

        common = self._build_common_context()

        teammates_for_reasoning = [
            {
                'id': t['object_id'],
                'location': [t.get('x'), t.get('y')],
                'role': self._team_roles.get(t['object_id'], 'unknown'),
            }
            for t in self.WORLD_STATE.get('teammates', [])
        ]

        ###### REASONING PROMPT INPUTS ######
        reasoning_inputs = {
            'agent_id': self.agent_id,
            'current_role': self.profile.role_str(),
            'agent_capabilities': common['agent_capabilities'],
            'game_rules': common['game_rules'],
            'position': agent_info.get('location'),
            'carrying': agent_info.get('carrying') or 'nothing',
            'current_plan': self._current_plan or self._pipeline_context.get('planned_task', ''),
            'motivation': motivation,
            'last_critique': last_critique,
            'observation': observation,
            'teammates': teammates_for_reasoning,
            'messages': self.get_messages(limit=7),
            'recent_actions': recent_actions_list,
            'last_action': (
                {
                    'name': self._last_action.get('name'),
                    'args': self._last_action.get('args', {}),
                    'outcome': self._last_action_outcome,
                }
                if self._last_action else None
            ),
            'tools_available': list(self.tools_by_name.keys()),
            'last_validation_error': self._last_validation_error or '',
            'supervisor_commands': self._supervisor_cmd_history[-8:],
        }

        area_info = []
        for s in self.get_area_summaries():
            try:
                area_id = int(s['name'].split()[-1])
                door = self.env_info.get_door(area_id)
            except (ValueError, AttributeError):
                door = None
            area_info.append({**s, 'door': door})
        reasoning_inputs['area_summaries'] = area_info

        # Anti-loop
        if len(recent_actions_list) == 3 and len(set(
            json.dumps(a, sort_keys=True) for a in recent_actions_list
        )) == 1:
            loop_msg = (
                f'LOOP DETECTED: last 3 actions are identical ({recent_actions_list[0]}). '
                f'You MUST try a completely different action type to make progress.'
            )
            self.memory.base.update('loop_warning', {'warning': loop_msg, 'tick': self._tick_count})
            self.memory.episode.set_loop_warning(loop_msg)
            reasoning_inputs['critic_feedback'] = {'success': False, 'critique': loop_msg, 'loop_warning': loop_msg}
        elif not critic.get('success', True):
            reasoning_inputs['critic_feedback'] = critic

        prompt = self.reasoning.get_reasoning_prompt(reasoning_inputs)
        self.call_llm(prompt, tools=self.tool_schemas)
        return self._idle()

    def _handle_reasoning_result(self, llm_response) -> Tuple[Optional[str], Dict]:
        message = llm_response[0]

        # for the strategies that require multiple LLM calls
        hook_result = None
        try:
            hook_result = self.reasoning.on_llm_result(message, self)
        except Exception as exc:
            logger.warning('[%s] reasoning.on_llm_result raised: %s', self.agent_id, exc)

        if isinstance(hook_result, FollowupRequest):
            self.call_llm(
                hook_result.messages,
                tools=hook_result.tools,
                tool_choice=hook_result.tool_choice,
            )
            self._pipeline_stage = PipelineStage.REASONING
            return self._idle()

        if isinstance(hook_result, ActionCommit):
            name, args = hook_result.name, hook_result.args
            self._pipeline_context['action_name'] = name
            self._pipeline_context['action_args'] = args
            self._pipeline_context['_reasoning_retries'] = 0
            self._current_action = {'name': name, 'args': args}
            self._pipeline_stage = (
                PipelineStage.COMM_DISPATCH
                if name == 'SendMessage'
                else PipelineStage.EXECUTE
            )
            return self._advance_pipeline()

        # Path A: structured tool_call
        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls is None:
            content = getattr(message, 'content', '') or ''

            # Path B: plain-text 
            extracted = extract_action_json(content)
            if extracted and extracted.get('name') in self.tools_by_name:
                name = extracted['name']
                args = extracted.get('args', extracted.get('arguments', {}))
                print(
                    f'[{self.agent_id}] Fallback JSON parse succeeded: '
                    f'{name}({args})'
                )
                self._pipeline_context['action_name'] = name
                self._pipeline_context['action_args'] = args
                self._pipeline_context['_reasoning_retries'] = 0
                self._current_action = {'name': name, 'args': args}
                self._pipeline_stage = (
                    PipelineStage.COMM_DISPATCH
                    if name == 'SendMessage'
                    else PipelineStage.EXECUTE
                )
                return self._advance_pipeline()

            # Path C: wrong. retry with a hard cap of 3 attempts
            retries = self._pipeline_context.get('_reasoning_retries', 0) + 1
            self._pipeline_context['_reasoning_retries'] = retries
            print(
                f'[{self.agent_id}] Reasoning produced no tool call '
                f'(attempt {retries}/3). '
                f'content={content[:120]!r}'
            )
            logger.warning(
                '[%s] Reasoning result missing tool_calls (attempt %d/3): %s',
                self.agent_id, retries, message,
            )
            if retries >= 3:
                print(
                    f'[{self.agent_id}] Reasoning retry cap reached — '
                    f'resetting pipeline to IDLE'
                )
                self._pipeline_context['_reasoning_retries'] = 0
                self._pipeline_stage = PipelineStage.IDLE
            else:
                self._pipeline_stage = PipelineStage.REASONING
            return self._idle()
        
        tc = tool_calls[0]
        name = tc.function.name
        args_raw = tc.function.arguments
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        print(f'[{self.agent_id}] REASONING Tool call: {name}({args})')

        if len(tool_calls) >= 2:
            print(
                f'[{self.agent_id}] Ignoring {len(tool_calls) - 1} extra tool '
                f'call(s); only the first ({name}) is executed.'
            )

        self._pipeline_context['action_name'] = name
        self._pipeline_context['action_args'] = args
        self._current_action = {'name': name, 'args': args}

        if name == 'SendMessage':
            self._pipeline_stage = PipelineStage.COMM_DISPATCH
        else:
            self._pipeline_stage = PipelineStage.EXECUTE
        return self._advance_pipeline()

    def _infer_help_victim(self) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Pick the nearest visible heavy victim as the target of an ask_help.
        """
        agent_info = self.WORLD_STATE.get('agent', {}) or {}
        loc = agent_info.get('location') or (0, 0)
        ax, ay = int(loc[0]), int(loc[1])
        victims = self.WORLD_STATE.get('victims', []) or []
        if not victims:
            return None

        def _dist(v):
            vl = v.get('location') or (v.get('x', 0), v.get('y', 0))
            return abs(int(vl[0]) - ax) + abs(int(vl[1]) - ay)

        heavy = [v for v in victims if 'critical' in (v.get('obj_id', '') or v.get('object_id', ''))]
        pool = heavy or victims
        chosen = min(pool, key=_dist)
        vid = chosen.get('obj_id') or chosen.get('object_id') or ''
        vl = chosen.get('location') or (chosen.get('x', 0), chosen.get('y', 0))
        return (vid, (int(vl[0]), int(vl[1])))

    # EXECUTE stage

    def communicate(self) -> Tuple[Optional[str], Dict]:
        args = self._pipeline_context.get('action_args', {})
        send_to = args.get('send_to', 'all')
        message_type = args.get('message_type', 'message')
        text = args.get('message', '')

        if (message_type == MSG_HELP and send_to in (None, 'all')
                and (text or '').strip().lower() not in ('yes', 'no')):
            message_type = MSG_ASK_HELP

        target = None if send_to == 'all' else send_to

        # No self msg.
        if target is not None and target == self.agent_id:
            print(f'[{self.agent_id}] Dropping message addressed to self')
            self.memory.base.update('action_failure',
                               'You cannot send a message to yourself. '
                               'Send to a teammate or to "all".')
            self._pipeline_stage = PipelineStage.IDLE
            return self._idle()

        decision, extra = self._joint_on_outgoing_message(send_to, message_type, text, args)
        if decision == 'suppress':
            self.memory.base.update('help_blocked',
                                    {'reason': 'joint_action_in_progress', 'tick': self._tick_count})
            self._pipeline_stage = PipelineStage.IDLE
            return self._idle()
        if 'text' in extra:
            text = extra['text']

        content_payload = {'message_type': message_type, 'text': text}
        for k, v in extra.items():
            if k != 'text':
                content_payload[k] = v

        self.send_message(Message(content=content_payload, from_id=self.agent_id, to_id=target))
        self.memory.base.update('sent_message', {
            'entry_type': 'sent_message',
            'from': self.agent_id,
            'to': send_to,
            'message_type': message_type,
            'text': text,
            'tick': self._tick_count,
        })
        if self.metrics:
            self.metrics.record_message_sent(self._tick_count, send_to, message_type, text)

        self._pipeline_stage = PipelineStage.IDLE
        return self._idle()

    def execute(self) -> Tuple[Optional[str], Dict]:
        name = self._pipeline_context['action_name']
        args = self._pipeline_context['action_args']

        if name == 'MoveTo':
            planned = self._pipeline_context.get('planned_task') or self._current_task or ''
            target_str = f"({args.get('x')}, {args.get('y')})"
            if str(args.get('x')) not in planned and str(args.get('y')) not in planned:
                print(
                    f'[{self.agent_id}] MISMATCH: MoveTo{target_str} not found in '
                    f'planned task: {planned[:120]!r}'
                )

        if name in ('CarryObjectTogether', 'RemoveObjectTogether'):
            decision, info = self._joint_on_action(name, args)
            if decision == 'error' and info:
                self._last_validation_error = info
                self._pipeline_context['critic_result'] = {'success': False, 'critique': info}
                print(f'[{self.agent_id}] {name} blocked: {info}')
            self._pipeline_stage = PipelineStage.IDLE
            return self._idle('joint_action_delegated')

        check = self._validate_action(name, args)
        if check is not None:
            retries = self._pipeline_context.get('_validation_retries', 0) + 1
            self._pipeline_context['_validation_retries'] = retries
            self._recent_actions.append({'name': name, 'args': args, 'result': 'validation_failed'})
            self._last_action_outcome = 'rejected_by_validator'
            self._pipeline_context['critic_result'] = {
                'success': False,
                'critique': self._last_validation_error or f"Action {name} failed validation.",
            }
            print(
                f'[{self.agent_id}] Validation rejected {name}({args}) '
                f'(attempt {retries}/3): {self._last_validation_error[:120]!r}'
            )
            if retries >= 3:
                print(
                    f'[{self.agent_id}] Validation retry cap reached — '
                    f'returning to PLANNING for a new task'
                )
                self._pipeline_context['_validation_retries'] = 0
                self._pipeline_stage = PipelineStage.IDLE
            else:
                self._pipeline_stage = PipelineStage.REASONING
            return check

        self._pipeline_context['_validation_retries'] = 0

        try:
            from matrx.agents.llm.modules.supervisor_channel import push_agent_status
            push_agent_status(
                agent_id=self.agent_id,
                message_type='action_human',
                text=_human_readable_action(name, args),
                task=self._current_plan or '',
                motivation=self._pipeline_context.get('motivation', ''),
                action_name=name,
                action_args=args,
                tick=self._tick_count,
            )
        except Exception:
            pass

        action_name, kwargs = self.execute_action(name, args)

        if name in ('RemoveObject', 'RemoveObjectTogether'):
            obj_id = args.get('object_id', '')
            if obj_id:
                self.WORLD_STATE_GLOBAL['obstacles'] = [
                    o for o in self.WORLD_STATE_GLOBAL.get('obstacles', [])
                    if o.get('object_id') != obj_id
                ]

        if action_name == 'Drop' and self.shared_memory:
            drop_zone = tuple(self.env_info.drop_zone)
            if _chebyshev_distance(self.agent_location, drop_zone) <= 1:
                carrying = self.WORLD_STATE.get('agent', {}).get('carrying', [])
                if carrying:
                    rescued_id = carrying[0]
                    rescued = self.shared_memory.retrieve('rescued_victims') or []
                    if not any(v['victim_id'] == rescued_id for v in rescued):
                        rescued = rescued + [{
                            'victim_id': rescued_id,
                            'tick': self._tick_count,
                            'agent': self.agent_id,
                            'method': 'solo',
                        }]
                        self.shared_memory.update('rescued_victims', rescued)
                        print(
                            f'[{self.agent_id}] Recorded solo rescue of '
                            f'{rescued_id} (total rescued: {len(rescued)})'
                        )

        self.memory.base.update('action', {'action': action_name, 'args': kwargs})
        self.memory.episode.set_action(action_name, kwargs, self._tick_count)
        self._recent_actions.append({'name': action_name, 'args': kwargs})
        self._last_action_outcome = 'dispatched'

        if self.metrics:
            loc = self.WORLD_STATE.get('agent', {}).get('location', (0, 0))
            self.metrics.record_action(self._tick_count, action_name, kwargs, tuple(loc))

        action, updates = apply_navigation(action_name, kwargs, navigator=self._navigator, state_tracker=self._state_tracker, env_info=self.env_info, memory=self.memory.base)
        if 'nav_target' in updates:
            self._nav_target = updates['nav_target']
        self._pipeline_stage = PipelineStage.IDLE
        return action

