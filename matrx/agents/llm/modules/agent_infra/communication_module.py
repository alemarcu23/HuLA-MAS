"""
Communication Module — environment-sided message processing.
"""

import re
import time
from collections import deque
from typing import List, Optional, Tuple
from matrx.helpers.communication_helpers import _extract_message

GLOBAL_MESSAGE_LOG: deque = deque(maxlen=500)

_COORD_PATTERNS = (
    re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]'),
    re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'),
    re.compile(r'\bat\s+(\d+)\s*,\s*(\d+)\b'),
)


def extract_coords_from_text(text: str) -> Optional[Tuple[int, int]]:
    """Pull the first [x,y] or (x,y) coordinate pair from free-form text."""
    if not text:
        return None
    for pat in _COORD_PATTERNS:
        m = pat.search(text)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None


class Communication:

    def init_communication(self) -> None:
        """Initialise per-agent message state. Idempotent re-init resets state."""
        self._messages: List[dict] = []
        self._processed_count: int = 0

    def process_messages(self, received_messages: list) -> None:
        new_count = len(received_messages)
        if new_count <= self._processed_count:
            return

        for msg in received_messages[self._processed_count:]:
            entry = _extract_message(msg, self.agent_id)
            if entry is None:
                continue
            self._messages.append(entry)

            key = (entry.get('from'), entry.get('to'),
                   entry.get('message_type'), entry.get('text'),
                   entry.get('request_id', ''))
            if not any(
                (m.get('from'), m.get('to'), m.get('message_type'),
                 m.get('text'), m.get('request_id', '')) == key
                for m in list(GLOBAL_MESSAGE_LOG)[-20:]
            ):
                GLOBAL_MESSAGE_LOG.append({'ts': time.time(), **entry})

        self._processed_count = new_count

    def get_messages(self, limit: int = 10) -> List[dict]:
        """Return the most recent messages for the LLM prompt newest first.

        Drops this agent's own messages,
        and tags `ask_help` entries with a [HELP REQUEST].
        """
        incoming = [
            m for m in self._messages
            if m.get('from') != self.agent_id
        ]
        result = []
        for m in reversed(incoming[-limit:]):  # newest first
            entry = dict(m)
            if entry.get('message_type') == 'ask_help':
                entry['text'] = f"[HELP REQUEST] {entry.get('text', '')}"
            result.append(entry)
        return result

    @property
    def all_messages_raw(self) -> List[dict]:
        """All processed messages (for metrics export)."""
        return list(self._messages)
