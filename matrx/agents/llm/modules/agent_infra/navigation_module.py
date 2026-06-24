"""
Navigation Module — A* navigation state + tick-by-tick steps.
"""

from typing import Dict, Optional, Tuple

from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker


class Navigation:
    """A* navigation state and stepping."""

    def init_navigation(self, agent_id: str, action_set) -> None:
        """Create the navigator + state tracker. Call once from `initialize."""
        self._state_tracker: Optional[StateTracker] = StateTracker(agent_id=agent_id)
        self._navigator: Optional[Navigator] = Navigator(
            agent_id=agent_id,
            action_set=action_set,
            algorithm=Navigator.A_STAR_ALGORITHM,
        )
        self.state_for_navigation: Optional[State] = None

        self._nav_target: Optional[Tuple[int, int]] = None

    def _handle_navigation_tick(self) -> Optional[Tuple[str, Dict]]:
        """Continue A* navigation if a target is set."""
        if self._nav_target is None:
            return None
        move = self._navigator.get_move_action(self._state_tracker)
        if move is not None:
            return move, {}
        self._nav_target = None
        return None
