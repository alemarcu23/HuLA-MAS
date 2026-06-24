"""Shared test harness for the joint-action coordinator.

These tests exercise `JointAction` (matrx/agents/llm/modules/joint_action.py)
with NO LLM and NO MATRX world — a lightweight fake host supplies exactly the
attributes the mixin reads, so the deterministic state machine, target
resolution, and navigation logic can be asserted directly and fast.
"""
import pytest

from matrx.agents.llm.modules.joint_action import JointAction
from matrx.agents.llm.modules.memory_module import SharedMemory


class FakeNavigator:
    """navigate_to() only calls these three; it returns a fixed move."""
    def reset_full(self): pass
    def add_waypoints(self, wps): pass
    def get_move_action(self, state_tracker): return 'MoveNorth'


class Result:
    """Stand-in for a MATRX ActionResult."""
    def __init__(self, succeeded): self.succeeded = succeeded


class Msg:
    """Stand-in for matrx.messages.message.Message as delivered to an agent."""
    def __init__(self, content, from_id):
        self.content = content
        self.from_id = from_id


class FakeAgent(JointAction):
    """Minimal host exposing just what JointAction reads."""

    def __init__(self, agent_id='RescueBot', trigger='auto_bridge', roster=('RescueBot', 'RescueBot1')):
        self.agent_id = agent_id
        self._navigator = FakeNavigator()
        self._state_tracker = object()
        self._tick_count = 0
        self._nav_target = None
        self._pending_future = None
        self._include_human = False
        self._partner_name = None
        self.shared_memory = SharedMemory()
        if roster:
            self.shared_memory.update('registered_agents', list(roster))
        self.WORLD_STATE = {'agent': {'location': (8, 8), 'carrying': []}, 'teammates': []}
        self.WORLD_STATE_GLOBAL = {'victims': [], 'obstacles': []}
        self.state_for_navigation = {}
        self.previous_action = None
        self.previous_action_result = None
        self.sent = []

        class _EnvInfo:
            drop_zone = (23, 8)
            grid_size = (25, 25)
        self.env_info = _EnvInfo()
        self._init_joint_action(trigger)

    # ── host attributes the mixin expects ──────────────────────────────
    @property
    def agent_location(self):
        return tuple(self.WORLD_STATE['agent']['location'])

    def _idle(self, reason='idle'):
        return ('Idle', {'duration_in_ticks': 1, 'reason': reason})

    def _handle_navigation_tick(self):
        return None

    def send_message(self, msg):
        self.sent.append(msg.content)

    # ── test conveniences ──────────────────────────────────────────────
    def at(self, loc):
        self.WORLD_STATE['agent']['location'] = loc
        return self

    def tick(self, t):
        self._tick_count = t
        return self

    def set_world(self, victims=None, obstacles=None, nav=None):
        self.WORLD_STATE_GLOBAL = {'victims': victims or [], 'obstacles': obstacles or []}
        if nav is not None:
            self.state_for_navigation = nav
        return self

    def result(self, action_name, ok=True):
        self.previous_action = action_name
        self.previous_action_result = Result(ok)
        return self

    def receive(self, content, from_id):
        """Feed one incoming message (content dict) through the ingest path."""
        self._ingest_joint_messages([Msg(content, from_id)], self._tick_count)
        return self

    def last_sent(self, message_type):
        for c in reversed(self.sent):
            if isinstance(c, dict) and c.get('message_type') == message_type:
                return c
        return None


@pytest.fixture
def make_agent():
    def _factory(agent_id='RescueBot', trigger='auto_bridge', roster=('RescueBot', 'RescueBot1')):
        return FakeAgent(agent_id, trigger, roster)
    return _factory
