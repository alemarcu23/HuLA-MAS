"""
Per-tick message routing for the GridWorld game loop.

``MessageManager`` (message_manager.py) owns message bookkeeping: preprocessing,
chatrooms, and exposing messages to the API. ``MessageRouter`` is the thin
orchestration layer that the game loop drives each tick:

    1. collect   — gather each agent's outgoing messages (brain + API) and
                   preprocess them (per agent, inside the agent loop).
    2. buffer    — group this tick's preprocessed messages by receiver.
    3. deliver   — hand each receiver its messages, then clear the buffer.

Pulling this out of ``GridWorld.__step`` keeps the messaging behaviour in one
findable place without touching the existing MessageManager/API wiring (the
router wraps the same MessageManager instance the API already points at).
"""
import copy


class MessageRouter:
    """Routes messages between agents within a single tick.

    Wraps a :class:`MessageManager` and owns the per-tick receiver buffer.
    """

    def __init__(self, message_manager):
        self.message_manager = message_manager
        self._buffer = {}  # receiver_id -> list[Message] for the current tick

    def collect_and_preprocess(self, tick, agent_obj, all_agent_ids, teams,
                               api_received=None):
        """Gather one agent's outgoing messages this tick and preprocess them.

        Parameters
        ----------
        tick : int
            The current tick number.
        agent_obj : AgentBody
            The agent whose outgoing messages are collected.
        all_agent_ids : iterable
            Ids of every registered agent (needed to resolve broadcast targets).
        teams : dict
            Team-name -> agent-ids mapping (needed to resolve team targets).
        api_received : list, optional
            Messages this agent sent via the API this tick (already popped by
            the caller). Merged with the brain's outgoing messages.
        """
        agent_messages = agent_obj.get_messages_func(all_agent_ids)
        if api_received:
            agent_messages += copy.copy(api_received)
        self.message_manager.preprocess_messages(tick, agent_messages,
                                                 all_agent_ids, teams)

    def buffer_tick(self, tick):
        """Group this tick's preprocessed messages by receiver into the buffer."""
        for mssg in self.message_manager.preprocessed_messages.get(tick, []):
            self._buffer.setdefault(mssg.to_id, []).append(mssg)

    def deliver(self, registered_agents):
        """Deliver buffered messages to existing receivers, then clear the buffer."""
        for receiver_id, messages in self._buffer.items():
            if receiver_id in registered_agents:
                registered_agents[receiver_id].set_messages_func(messages)
        self._buffer = {}
