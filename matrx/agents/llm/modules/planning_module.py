from typing import Dict, List, Any
from matrx.helpers.toon_utils import to_toon

try:
    from engine.parsing_utils import load_few_shot
except ImportError:
    def load_few_shot(key):
        return []

DEFAULT_STRATEGY = 'io'

SYSTEM_ROLE_PROMPT = (
    "You are the planner for a Search and Rescue agent. You must emit ONE atomic plan for the agent to execute. The plan should be in natural language but contain explicit details (e.g. victim IDs, coordinates)."
    "Return a single JSON object — no text outside the JSON. "
)

CRITIC_PROMPT = """
Judge if `last_action` completed `last_plan`, using all available information.

1. If `last_action.name is "SendMessage"`, ALWAYS emit: success = true, critique = ""
2. If `last_action` is empty/null (first cycle), emit success = true, critique = "".
3. A `MoveTo` whose target [x, y] is not the same as the agent's current position. Critique must suggest checking for obstacles blocking the path.
4. Otherwise judge based on the current OBSERVATION, WORLD STATE BELIEF, and the YOU block:
   - Carry plan => success if `carrying` now matches the targeted victim.
   - Move-to-[x,y] plan => success if position now equals [x,y].
   - Remove-obstacle plan => success if the obstacle is GONE from OBSERVATION and WORLD STATE BELIEF.
   - Drop-at-zone plan => success if `carrying` is empty.

The critique (if failure) MUST be ONE piece of actionable feedback.
"""

PLANNING_INSTRUCTIONS = """
Your `high_level_task` is your role-based directive for the entire run. Every plan you emit MUST be a concrete step toward that high-level goal. Before choosing a plan, ask: "Does this plan advance my high-level task?" If the answer is no, pick a different plan. The only exceptions are URGENT help requests and genuine blockers that must be cleared first (e.g. an obstacle preventing you from reaching your target).
PREFER planning something at your current location instead of moving.

You should expect to find victims inside the areas that have different severity levels. Prioritize rescuing victims with higher severity levels first, as they are in more critical condition and require immediate attention.
You should expect obstacles to block entrances to areas and paths to victims. Removing these obstacles is necessary to access the victims and to search the areas.

Use all available information before planning: OBSERVATION, WORLD STATE BELIEF, MEMORY, other TEAMMATES' plans, and MESSAGES. Keep in mind your HISTORY of actions and plans, and plan accordingly. Keep in mind your CAPABILITIES when planning the task. Your next plan should be predictable to a teammate who has been following your work so far. The AREA COVERAGE block shows the % of each area that has been searched. Use this to track your progress and plan next steps. An area is searched only when AREA COVERAGE shows 100%. Only move on to a different area after coverage shows 100%.
DO NOT output the same plan or goal every time - use the MEMORY. If something did not work, try again later or delegate the task to a teammate.

Some tasks need two agents (e.g. carrying a critically injured victim, or removing a big rock): a JOINT action (CarryObjectTogether / RemoveObjectTogether). To start one, plan to send a "Send message" with intent = "ask_help" naming the target id and its [x,y], then wait for a teammate to reply "yes". Once they do, walking to the target, performing the joint action, and carrying to the drop zone all happen AUTOMATICALLY — do not plan those steps yourself.

## COORDINATING WITH TEAMMATES
- If you receive an `ask_help` you can fulfill, plan to reply "yes" or "no". After you reply "yes", the system drives you to the target and completes the joint action — you do NOT plan any further steps for it.
- Send Messages to teammates when you want to share information, delegate a task you cannot perform, or coordinate. This is the only way to influence your teammates' plans.
"""

OUTPUT_CRITIC = """
## OUTPUT
Return a single valid JSON object — and nothing else:
{
  "critic": {
    "success": true | false,
    "critique": "actionable feedback or empty string if success"
  },
  "next_plan": "one atomic sub-task using the shapes above, with explicit IDs and coordinates so the Reasoning stage needs no extra context",
  "motivation": "<=50 words explaining how this plan advances your high_level_task and why it is consistent with your recent history"
}
"""

OUTPUT_PLANNER = """
## OUTPUT
Return a single valid JSON object — and nothing else:
{
  "next_plan": "one atomic sub-task using the shapes above, with explicit IDs and coordinates so the Reasoning stage needs no extra context",
  "motivation": "<=50 words explaining how this plan advances your high_level_task and why it is consistent with your recent history"
}
"""

_STRATEGY_ADDENDA: Dict[str, str] = {
    'io': "",
    'deps': (
        "[DEPS] Treat the high-level task as a chain of dependent sub-goals. "
        "Prefer the sub-task that directly depends on the last plan having been "
        "completed. If the last plan failed, the next plan should unblock it. \n\n"
    ),
    'critic': CRITIC_PROMPT,
    'cot': (
        "[ChainOfThought] Lets think step by step. Before choosing a plan, reason privately "
        "about the current world state, the memory, the current needs, and what each teammate is doing.\n\n"
    ),
}


class PlanningBase:
    prefix: str = _STRATEGY_ADDENDA['io']
    output_schema: str = OUTPUT_PLANNER

    emits_critic: bool = False

    def decorate_system_prompt(self, base_prompt: str) -> str:
        decorated = (self.prefix + base_prompt) if self.prefix else base_prompt
        return decorated + "\n" + self.output_schema


class PlanningIO(PlanningBase):
    prefix = _STRATEGY_ADDENDA['io']


class PlanningDEPS(PlanningBase):
    prefix = _STRATEGY_ADDENDA['deps']


class PlanningCritic(PlanningBase):
    prefix = _STRATEGY_ADDENDA['critic']
    output_schema = OUTPUT_CRITIC
    emits_critic = True


class PlanningCot(PlanningBase):
    prefix = _STRATEGY_ADDENDA['cot']


PLANNING_STRATEGY_REGISTRY: Dict[str, type] = {
    'io': PlanningIO,
    'deps': PlanningDEPS,
    'critic': PlanningCritic,
    'cot': PlanningCot,
}


def build_planning_strategy(name: str) -> PlanningBase:
    cls = PLANNING_STRATEGY_REGISTRY.get(name, PlanningIO)
    return cls()


def _format_action(action: Any) -> str:
    if not isinstance(action, dict) or not action.get('name'):
        return 'none'
    raw_args = action.get('args')
    if raw_args is None:
        raw_args = {k: v for k, v in action.items() if k != 'name'}
    args_str = ', '.join(f'{k}={v}' for k, v in (raw_args or {}).items())
    return f"{action['name']}({args_str})"


PLANNING_USER_TEMPLATE = """\
{alerts}== YOU ==
Name: {agent}
Role: {role}
Capabilities:
{capabilities}
Position: {position}
Carrying: {carrying}

== MOST RECENT WORK ==
High-level task: "{high_level_task}"
Last plan: "{last_plan}"
Last action: {last_action}

== OBSERVATION (vision range only) ==
{observation}

== WORLD STATE BELIEF ==
{world_state_belief}

== AREA COVERAGE ==
{area_coverage}

== TEAM ==
{teammates}

== MEMORY (past episodes, newest first; each carries motivation + outcome) ==
{memory}

== MESSAGES (incoming only, newest first; [HELP REQUEST] entries are critical) ==
{messages}

== INSTRUCTIONS ==
{instructions}"""


def _format_alerts(information: Dict[str, Any]) -> str:
    """One-shot banners shown above the YOU block. Empty string when none apply."""
    blocks: List[str] = []

    urgent_abandon = information.get('urgent_abandon')
    if urgent_abandon:
        blocks.append(
            "== URGENT: ABANDON CURRENT TASK ==\n"
            f"{urgent_abandon}\n"
            "Your help request did not work out. Stop waiting, pick a completely "
            "different objective, and continue your mission."
        )

    last_validation_error = information.get('last_validation_error', '')
    if last_validation_error:
        blocks.append(
            "== LAST ACTION REJECTED BY VALIDATOR ==\n"
            f"{last_validation_error}"
        )

    return ''.join(f"{block}\n\n" for block in blocks)


def _format_area_coverage(area_summaries: List[Dict[str, Any]]) -> str:
    """AREA COVERAGE block (with trailing blank line). Empty string when unknown."""
    if not area_summaries:
        return ""
    rows = ["== AREA COVERAGE =="]
    for a in area_summaries:
        door_str = f"door={a['door']}" if a.get('door') else "door=unknown"
        pct = int(a.get('coverage', 0) * 100)
        rows.append(f"  {a['name']}: {door_str}, {pct}% searched ({a.get('status', '?')})")
    return "\n".join(rows) + "\n\n"


def _format_planning_user_content(information: Dict[str, Any]) -> str:
    ctx = information.get('context', {})
    observation = information.get('observation', {}) or {}
    world_state_belief = information.get('world_state_belief', {}) or {}
    teammates = ctx.get('teammates', [])
    memory = information.get('memory', []) or []
    messages = information.get('messages', []) or []

    return PLANNING_USER_TEMPLATE.format(
        alerts=_format_alerts(information),
        agent=ctx.get('agent', 'unknown'),
        role=ctx.get('role', 'unassigned'),
        capabilities=ctx.get('capabilities', ''),
        position=ctx.get('position', '?'),
        carrying=ctx.get('carrying', 'nothing'),
        high_level_task=information.get('high_level_task', '') or 'none',
        last_plan=information.get('last_plan', '') or 'none',
        last_action=_format_action(information.get('last_action')),
        observation=to_toon({
            'victims': observation.get('victims', []),
            'obstacles': observation.get('obstacles', []),
        }),
        world_state_belief=to_toon({
            'victims': world_state_belief.get('victims', {}),
            'obstacles': world_state_belief.get('obstacles', {}),
        }),
        area_coverage=_format_area_coverage(information.get('area_summaries') or []),
        teammates=to_toon(teammates) if teammates else "none",
        memory=to_toon(memory) if memory else "none",
        messages=to_toon(messages) if messages else "none",
        instructions=PLANNING_INSTRUCTIONS.strip(),
    )


class Planning:
    def __init__(self, strategy: str = DEFAULT_STRATEGY) -> None:
        self.strategy = build_planning_strategy(strategy)
        self.emits_critic = self.strategy.emits_critic

    def get_planning_prompt(self, information: Dict[str, Any]) -> List[Dict[str, str]]:
        system_content = self.strategy.decorate_system_prompt(SYSTEM_ROLE_PROMPT)
        messages = [{"role": "system", "content": system_content}]

        try:
            examples = load_few_shot('planning_next_task')
            for ex in examples:
                if 'user' in ex and 'assistant' in ex:
                    messages.append({"role": "user", "content": ex['user'].strip()})
                    messages.append({"role": "assistant", "content": ex['assistant'].strip()})
        except Exception:
            pass

        messages.append({"role": "user", "content": _format_planning_user_content(information)})
        return messages
