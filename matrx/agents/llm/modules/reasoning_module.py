import ast
import json
import re
from typing import Dict, List, Any, Optional
from matrx.helpers.toon_utils import to_toon

MAX_MESSAGES = 7

_PHASE_MAIN = 'main'
_PHASE_REFINE = 'refine'
_PHASE_DONE = 'done'

REASONING_PROMPT_CORE = """
You are a Search and Rescue agent. Your job each cycle is to execute the `current_plan` by emitting ONE tool call."""

REASONING_BODY = """
The plan given to you will include all necessary details (can include: victim_id, location [x, y], obstacle_id, partner_id). You do NOT re-plan. If the plan looks unreachable, choose the tool call that makes the most direct progress toward it.
The tool calls that you make require precise argument names and types — follow the schema exactly. If you are unsure about the correct arguments, pick the best guess but be prepared for the validator to reject it and force you to try again with a different action.
Your belief of the world and memory should help you choose the best action to advance the current plan and the parameters.
ALWAYS PREFER TO DO AN ACTION, and not a MoveTo.

HUMAN SUPERVISOR COMMANDS:
- If the user-message contains a "== SUPERVISOR COMMAND ==" block, follow it as soon as possible, since it comes from a human supervisor. Ignore other plans, but use your information since the feedback might be vague.

CAPABILITY LIMITATIONS & JOINT ACTIONS:
- Some actions need two agents (e.g. carrying a critically injured victim, or removing a big rock): the JOINT actions CarryObjectTogether and RemoveObjectTogether.
- To start one, send an "ask_help" naming the target id and its [x,y] and wait for a teammate to reply "yes". After that, getting into position, firing the joint action, and delivering to the drop zone are AUTOMATIC — you do not navigate or fire them yourself.
- If a teammate sends an "ask_help" you can fulfill, reply "yes" — the system then drives you to the target and completes the joint action.

MESSAGING:
- SendMessage to teammates to share information or delegate tasks. Use it often to share discoveries, ask a teammate to take over a task, or offer your own capabilities to assist others.
"""

_COT_ADDENDUM = (
    "[ChainOfThought] Solve the task step by step. Before choosing an output, reason privately "
    "about the current world state, the active plan, and the likely consequences "
    "of each candidate.\n\n"
)

_REFLEXION_ADDENDUM = (
    "[Reflexion] Before acting, reflect on what you have done and what failed. "
    "If a previous action failed (see critic_feedback.critique and reflexion block), "
    "you MUST try a completely different approach this turn — do not repeat the same "
    "action.\n\n"
)

_SELF_REFINE_ADDENDUM = (
    "[Self-Refine] Choose the best tool call for the active plan. "
    "A second verification pass will check your call for syntactic and semantic "
    "correctness against the tool schema, so be precise with argument names and types.\n\n"
)

_REFINE_FOLLOWUP_TEMPLATE = (
    "You are verifying a proposed tool call for a Search and Rescue agent.\n"
    "Proposed call: {tool_name}({args_json})\n"
    "Active plan: {subtask}\n"
    "Agent position: {position}\n"
    "Available tool names: {tool_names}\n\n"
    "Check (a) the tool name is in the available list, (b) required arguments are "
    "present and of the correct type (coordinates are ints, object_ids are strings), "
    "(c) the call advances or completes the plan, (d) it is NOT a no-op (e.g. MoveTo to your "
    "own position is invalid).\n\n"
    "Respond with exactly one line in one of these two formats:\n"
    "  correct\n"
    "  error, revised: {{\"name\": \"<tool>\", \"args\": {{...}}}}\n"
)


REASONING_USER_TEMPLATE = """\
{supervisor_command}== INFORMATION ABOUT YOU ==
Name: {agent}
Role: {role}
Capabilities: {capabilities}
Position: {position}
Carrying: {carrying}

{last_action}{validator_rejection}== CURRENT PLAN ==
Plan:       "{current_plan}"
Motivation: "{motivation}"{critique}

== TEAMMATES (use these ids as partner_id for cooperative actions / message recipients) ==
{teammates}

== CURRENT OBSERVATION ==
{observation}

== MESSAGES (incoming only, newest first; [HELP REQUEST] entries are critical) ==
{messages}

{area_info}== RECENT ACTIONS ==
{recent_actions}"""


def _format_supervisor_command(sup_cmds: List[Dict[str, Any]]) -> str:
    """SUPERVISOR COMMAND banner (with trailing blank line). Empty when none."""
    if not sup_cmds:
        return ""
    rows = ["== SUPERVISOR COMMAND (SOURCE OF TRUTH — OVERRIDES ALL OTHER INPUTS) =="]
    for c in sup_cmds:
        tick = c.get('tick')
        tick_str = f"tick {tick}" if tick is not None else "now"
        rows.append(f"  [{tick_str}] {c.get('text', '')}")
    rows.append("Treat the above as a direct order from the human supervisor.")
    rows.append("It overrides current_plan, teammate messages, and your own beliefs.")
    rows.append("Pick the tool call that most directly fulfills it.")
    rows.append("=======================================================================")
    return "\n".join(rows) + "\n\n"


def _format_last_action(last_action: Optional[Dict[str, Any]]) -> str:
    """LAST ACTION block (with trailing blank line). Empty when none recorded."""
    if not last_action:
        return ""
    name = last_action.get('name', '?')
    args = last_action.get('args', {})
    outcome = last_action.get('outcome', '')
    rows = [
        "== LAST ACTION ==",
        f"Name: {name}",
        f"Args: {to_toon(args) if args else '{}'}",
        f"Outcome: {outcome}" if outcome else "Outcome: (not recorded)",
    ]
    return "\n".join(rows) + "\n\n"


def _format_validator_rejection(last_validation_error: str) -> str:
    """VALIDATOR rejection banner (with trailing blank line). Empty when none."""
    if not last_validation_error:
        return ""
    rows = [
        "== LAST ACTION REJECTED BY VALIDATOR ==",
        last_validation_error,
        "Your last tool call was rejected! Do NOT repeat the same tool call again. Understand the reason for rejection and choose a different action that still advances the current plan.",
        "====================================",
    ]
    return "\n".join(rows) + "\n\n"


def _format_area_info(area_summaries: List[Dict[str, Any]]) -> str:
    """SEARCH PROGRESS block (with trailing blank line). Empty when unknown."""
    if not area_summaries:
        return ""
    rows = ["== SEARCH PROGRESS AND AREA INFORMATION =="]
    for a in area_summaries:
        door_str = str(a['door']) if a.get('door') else "unknown"
        rows.append(f"  {a['name']}: door={door_str}")
    return "\n".join(rows) + "\n\n"


def _format_reasoning_user_content(information: Dict[str, Any]) -> str:
    observation = information.get('observation', {}) or {}
    teammates = information.get('teammates') or []
    messages = (information.get('messages') or [])[:MAX_MESSAGES]
    recent_actions = information.get('recent_actions', []) or []
    critique = information.get('last_critique', '') or ''

    return REASONING_USER_TEMPLATE.format(
        supervisor_command=_format_supervisor_command(information.get('supervisor_commands') or []),
        agent=information.get('agent_id', 'unknown'),
        role=information.get('current_role', 'unassigned'),
        capabilities=information.get('agent_capabilities', ''),
        position=information.get('position', '?'),
        carrying=information.get('carrying', 'nothing'),
        last_action=_format_last_action(information.get('last_action')),
        validator_rejection=_format_validator_rejection(information.get('last_validation_error', '')),
        current_plan=information.get('current_plan', '') or 'none',
        motivation=information.get('motivation', '') or '',
        critique=f'\nCritique of last action (from planner): "{critique}"' if critique else '',
        teammates=to_toon(teammates) if teammates else "none",
        observation=to_toon({
            'victims': observation.get('victims', []),
            'obstacles': observation.get('obstacles', []),
        }),
        messages=to_toon(messages) if messages else "none",
        area_info=_format_area_info(information.get('area_summaries') or []),
        recent_actions=to_toon(recent_actions) if recent_actions else "none",
    )


class FollowupRequest:
    __slots__ = ('messages', 'tools', 'tool_choice')

    def __init__(self, messages: List[Dict[str, str]], tools=None, tool_choice: str = 'none'):
        self.messages = messages
        self.tools = tools
        self.tool_choice = tool_choice


class ActionCommit:
    __slots__ = ('name', 'args')

    def __init__(self, name: str, args: Dict[str, Any]):
        self.name = name
        self.args = args


class ReasoningBase:

    def __init__(self) -> None:
        self._phase: str = _PHASE_MAIN
        self._pending_action: Optional[Dict[str, Any]] = None

    def reset_phase(self) -> None:
        self._phase = _PHASE_MAIN
        self._pending_action = None

    def _strategy_addendum(self) -> str:
        return ""

    def get_reasoning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        system_content = REASONING_PROMPT_CORE + self._strategy_addendum() + REASONING_BODY
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": _format_reasoning_user_content(information)},
        ]

    def on_llm_result(self, message, agent) -> Optional[Any]:
        return None


class ReasoningIO(ReasoningBase):
    pass


class ReasoningCoT(ReasoningBase):
    def _strategy_addendum(self) -> str:
        return _COT_ADDENDUM


class ReasoningReflexion(ReasoningBase):
    def _strategy_addendum(self) -> str:
        return _REFLEXION_ADDENDUM


def _try_ast_literal_eval(text: str):
    return ast.literal_eval(text)


def _coerce_action(parsed: Any, tools_by_name: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(parsed, dict) and parsed.get('name') in tools_by_name:
        return {
            'name': parsed['name'],
            'args': parsed.get('args', parsed.get('arguments', {})),
        }
    return None


def _extract_action_from_message(message, tools_by_name: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tool_calls = getattr(message, 'tool_calls', None)
    if tool_calls:
        tc = tool_calls[0]
        name = tc.function.name
        args_raw = tc.function.arguments
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        if name in tools_by_name:
            return {'name': name, 'args': args}

    content = getattr(message, 'content', '') or ''
    if not content:
        return None

    m = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if m:
        try:
            action = _coerce_action(json.loads(m.group(1)), tools_by_name)
            if action:
                return action
        except (json.JSONDecodeError, ValueError):
            pass

    m = re.search(r'\{.*\}', content, re.DOTALL)
    if m:
        for loader in (json.loads, _try_ast_literal_eval):
            try:
                action = _coerce_action(loader(m.group(0)), tools_by_name)
                if action:
                    return action
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
    return None


def _parse_refine_verdict(text: str, tools_by_name: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if text.lower().lstrip().startswith('correct'):
        return None
    m = re.search(r'revised:\s*(\{.*\})', text, re.DOTALL)
    if not m:
        return None
    for loader in (json.loads, _try_ast_literal_eval):
        try:
            action = _coerce_action(loader(m.group(1)), tools_by_name)
            if action:
                return action
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
    return None


class ReasoningSelfRefine(ReasoningBase):

    def _strategy_addendum(self) -> str:
        return _SELF_REFINE_ADDENDUM

    def on_llm_result(self, message, agent) -> Optional[Any]:
        if self._phase == _PHASE_MAIN:
            extracted = _extract_action_from_message(message, agent.tools_by_name)
            if extracted is None:
                return None
            self._pending_action = extracted

            subtask = (
                agent._pipeline_context.get('planned_task')
                or agent._current_task
                or ''
            )
            position = agent.WORLD_STATE.get('agent', {}).get('location')
            verify_user = _REFINE_FOLLOWUP_TEMPLATE.format(
                tool_name=extracted['name'],
                args_json=json.dumps(extracted['args']),
                subtask=subtask,
                position=position,
                tool_names=', '.join(sorted(agent.tools_by_name.keys())),
            )
            self._phase = _PHASE_REFINE
            return FollowupRequest(
                messages=[
                    {"role": "system", "content": "You are a strict verifier of tool calls."},
                    {"role": "user", "content": verify_user},
                ],
                tools=None,
                tool_choice='none',
            )

        if self._phase == _PHASE_REFINE:
            content = (getattr(message, 'content', '') or '').strip()
            pending = self._pending_action or {}
            revised = _parse_refine_verdict(content, agent.tools_by_name)
            final = revised or pending
            self._phase = _PHASE_DONE
            self._pending_action = None
            if not final.get('name'):
                return None
            return ActionCommit(final['name'], final.get('args', {}))

        return None


REASONING_STRATEGY_REGISTRY: Dict[str, type] = {
    'io': ReasoningIO,
    'cot': ReasoningCoT,
    'reflexion': ReasoningReflexion,
    'self_refine': ReasoningSelfRefine,
}


def build_reasoning_strategy(name: str) -> ReasoningBase:
    cls = REASONING_STRATEGY_REGISTRY.get(name, ReasoningIO)
    return cls()
