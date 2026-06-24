"""
Execution Module — maps LLM action names + args to MATRX (action, kwargs) pairs.
"""

import logging
from typing import Dict, Any, Tuple, Optional

from matrx.actions.CustomActions import Idle as _Idle
from matrx.actions.CustomActions import CarryObject as _CarryObject
from matrx.actions.CustomActions import Drop as _Drop
from matrx.actions.CustomActions import CarryObjectTogether as _CarryObjectTogether
from matrx.actions.CustomActions import DropObjectTogether as _DropObjectTogether
from matrx.actions.CustomActions import RemoveObjectTogether as _RemoveObjectTogether
from matrx.actions.object_actions import RemoveObject as _RemoveObject

logger = logging.getLogger('action_dispatch')

_MOVE_ACTIONS = frozenset({'MoveNorth', 'MoveSouth', 'MoveEast', 'MoveWest'})

def execute_action(
    name: str,
    args: Dict[str, Any],
    agent_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:

    log_prefix = f"[{agent_id}] " if agent_id else ""

    print(f"{log_prefix}Dispatching action '{name}' with args {args} and partner '{args.get('partner_id', 'None')}'")

    if name in _MOVE_ACTIONS:
        return name, {}

    if name == 'MoveTo':
        return 'MoveTo', {'x': args.get('x', 0), 'y': args.get('y', 0)}

    if name == "MoveToArea":
        return 'MoveToArea', {'area': args.get('area', 1)}

    if name == "EnterArea":
        return 'EnterArea', {'area': args.get('area', 1)}

    if name == "SearchArea":
        return 'SearchArea', {'area': args.get('area', 1)}

    if name == 'NavigateToDropZone':
        return 'NavigateToDropZone', {}

    if name == 'SendMessage':
        return 'SendMessage', {
            'message': args.get('message', "Empty"),
            'send_to': args.get('send_to', "all"),
            'message_type': args.get('message_type', 'message'),
        }

    if name == 'CarryObject':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sCarryObject called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        return _CarryObject.__name__, {'object_id': obj_id}

    if name == 'Drop':
        return _Drop.__name__, {}

    if name == 'CarryObjectTogether':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sCarryObjectTogether called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        resolved_partner = args.pop('partner_id', '')
        return _CarryObjectTogether.__name__, {'object_id': obj_id, 'partner_name': resolved_partner}

    if name == 'RemoveObjectTogether':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sRemoveObjectTogether called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        resolved_partner = args.pop('partner_id', '')
        return _RemoveObjectTogether.__name__, {
            'object_id': obj_id,
            'remove_range': 1,
            'partner_name': resolved_partner,
        }

    if name == 'RemoveObject':
        obj_id = args.get('object_id', '')
        if not obj_id:
            logger.warning("%sRemoveObject called without object_id", log_prefix)
            return _Idle.__name__, {'duration_in_ticks': 1}
        return _RemoveObject.__name__, {'object_id': obj_id, 'remove_range': 1}

    if name == 'Idle':
        ticks = int(args.get('duration_in_ticks', 1))
        return _Idle.__name__, {'duration_in_ticks': ticks}

    logger.warning("%sUnknown action '%s', defaulting to Idle", log_prefix, name)
    return _Idle.__name__, {'duration_in_ticks': 1}


class Execution:

    def execute_action(
        self, name: str, args: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        return execute_action(name, args, getattr(self, 'agent_id', None))

    def _validate_action(
        self, name: str, args: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict]]:
        """Validate an action before dispatching to MATRX.

        Returns None when the action is valid otherwise saves failure
        (memory + episode + metrics) and returns Idle.
        """
        result = self._validator.validate(name, args, self.WORLD_STATE, self.teammates)
        if result.valid:
            return None
        self.memory.base.update("action_failure", result.feedback)
        self._last_validation_error = result.feedback
        ep = self.memory.episode
        ep.set_action(name, args, self._tick_count)
        open_ep = ep.get_open_episode()
        if open_ep is not None:
            open_ep.validation_failure = result.feedback
        if self.metrics:
            self.metrics.record_validation_failure(self._tick_count, name, result.feedback)
        return self._idle()
