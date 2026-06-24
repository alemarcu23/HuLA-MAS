from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

LOG_MAXLEN = 2000
TARGET_ALL = 'all'

_LOCK = threading.Lock()
_COMMANDS: deque = deque(maxlen=LOG_MAXLEN)
_NEXT_ID: int = 0
_CONSUMED: Dict[str, int] = {}

AGENT_STATUS_LOG: deque = deque(maxlen=LOG_MAXLEN)
_STATUS_LOCK = threading.Lock()


def push_command(
    text: str,
    tick: Optional[int] = None,
    target_agent: Optional[str] = None,
) -> Optional[dict]:
    global _NEXT_ID
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    target = (target_agent or TARGET_ALL).strip() or TARGET_ALL

    with _LOCK:
        entry = {
            'id': _NEXT_ID,
            'text': text,
            'tick': tick,
            'ts': time.time(),
            'target_agent': target,
        }
        _COMMANDS.append(entry)
        _NEXT_ID += 1

    try:
        from matrx.agents.llm.modules.agent_infra import communication_module as cm
        cm.GLOBAL_MESSAGE_LOG.append({
            'tick': tick,
            'from': 'SUPERVISOR',
            'to': target,
            'message_type': 'supervisor_command',
            'text': text,
            'ts': entry['ts'],
        })
    except Exception:
        pass

    return entry


def get_unconsumed(agent_id: str) -> List[dict]:
    if not agent_id:
        return []
    with _LOCK:
        last_seen = _CONSUMED.get(agent_id, 0)
        out = [
            c for c in _COMMANDS
            if c['id'] >= last_seen
            and (c.get('target_agent', TARGET_ALL) in (TARGET_ALL, agent_id))
        ]
        if _COMMANDS:
            _CONSUMED[agent_id] = _COMMANDS[-1]['id'] + 1
        return list(out)


def push_agent_status(
    agent_id: str,
    message_type: str,
    text: str,
    task: str = '',
    motivation: str = '',
    action_name: str = '',
    action_args: Optional[Dict[str, Any]] = None,
    tick: Optional[int] = None,
) -> None:
    entry: Dict[str, Any] = {
        'from': agent_id,
        'to': 'HUMAN',
        'message_type': message_type,
        'text': text,
        'task': task,
        'motivation': motivation,
        'action_name': action_name,
        'action_args': action_args or {},
        'tick': tick,
        'ts': time.time(),
    }
    with _STATUS_LOCK:
        AGENT_STATUS_LOG.append(entry)
