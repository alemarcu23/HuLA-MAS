from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional


OUTCOME_OK = 'OK'
OUTCOME_FAIL = 'FAIL'
OUTCOME_UNKNOWN = 'UNKNOWN'
DEFAULT_MAXLEN = 50          # max entries in BaseMemory
DEDUP_LOOKBACK = 5 
COMPRESS_THRESHOLD = 40    
COMPRESS_KEEP_RECENT = 10  
MAX_EPISODES = 50            # max episodes EpisodeMemory
PROMPT_MEMORY_LIMIT = 5      # default number of recent episodes in the prompt

_NOTABLE_KINDS = ('loop_warning', 'critic_feedback', 'planned_task')


def _outcome_label(succeeded: Optional[bool]) -> str:
    if succeeded is True:
        return OUTCOME_OK
    if succeeded is False:
        return OUTCOME_FAIL
    return OUTCOME_UNKNOWN


class BaseMemory:

    def __init__(self, maxlen: int = DEFAULT_MAXLEN) -> None:
        self.storage: deque = deque(maxlen=maxlen)

    def update(self, key: str, information: Any, tick: Optional[int] = None) -> None:
        if tick is not None and isinstance(information, dict) and 'tick' not in information:
            information = {**information, 'tick': tick}

        for last in list(self.storage)[-DEDUP_LOOKBACK:]:
            if information == last:
                return
        self.storage.append(information)

    def retrieve_all(self) -> List[Any]:
        return list(self.storage)

    def compress(self, threshold: int = COMPRESS_THRESHOLD, keep_recent: int = COMPRESS_KEEP_RECENT) -> None:
        entries = list(self.storage)
        if len(entries) <= threshold:
            return
        old, recent = entries[:-keep_recent], entries[-keep_recent:]
        action_counts: dict = {}
        notable = []
        for e in old:
            if not isinstance(e, dict):
                continue
            kind = e.get('kind') or e.get('entry_type') or e.get('action')
            if kind:
                action_counts[kind] = action_counts.get(kind, 0) + 1
            if e.get('kind') in _NOTABLE_KINDS:
                notable.append(e)
        summary: dict = {'kind': 'summary', 'compressed': len(old), 'action_counts': action_counts}
        if notable:
            summary['notable'] = notable[-3:]
        self.storage.clear()
        self.storage.append(summary)
        for e in recent:
            self.storage.append(e)


@dataclass
class EpisodeRecord:
    episode_id: str
    agent_id: str
    tick_open: int
    task: str
    role: str
    closed: bool = False

    received_messages: List[Dict[str, Any]] = field(default_factory=list)
    critic_feedback: Optional[Dict[str, Any]] = None
    planned_task: Optional[str] = None
    motivation: Optional[str] = None

    action_name: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    tick_action: Optional[int] = None
    validation_failure: Optional[str] = None

    loop_warning: Optional[str] = None

    tick_close: Optional[int] = None
    outcome_succeeded: Optional[bool] = None
    outcome_reason: Optional[str] = None

    collaboration: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        status = 'OPEN' if not self.closed else _outcome_label(self.outcome_succeeded)

        critic_ok = (
            self.critic_feedback is not None
            and self.critic_feedback.get('success', True)
        )
        critique_text = (self.critic_feedback or {}).get('critique', '')[:80]

        lines = [
            f'── Episode {self.episode_id} [{status}] ──',
            f'  tick:     {self.tick_open} -> {self.tick_close or "?"}',
            f'  agent:    {self.agent_id}  role={self.role}',
            f'  task:     {self.task}',
            f'  plan:     {self.planned_task or "-"}',
            f'  critic:   {"GOOD" if critic_ok else "BAD"} {critique_text}',
            (
                f'  action:   {self.action_name}({json.dumps(self.action_args, default=str)})'
                if self.action_name
                else '  action:  '
            ),
            f'  outcome:  {self.outcome_reason or "-"}',
        ]
        if self.loop_warning:
            lines.append(f'  LOOP:   {self.loop_warning[:80]}')
        if self.validation_failure:
            lines.append(f'  FAIL:  {self.validation_failure[:80]}')
        if self.received_messages:
            lines.append(f'  msg_in:   {len(self.received_messages)} messages')
        return '\n'.join(lines)


class EpisodeMemory:

    def __init__(self) -> None:
        self._episodes: deque = deque(maxlen=MAX_EPISODES)
        self._open_episode: Optional[EpisodeRecord] = None

    def open_episode(self, tick: int, agent_id: str, task: str, role: str) -> EpisodeRecord:
        self._open_episode = EpisodeRecord(
            episode_id=f'ep_{agent_id}_{tick}',
            agent_id=agent_id,
            tick_open=tick,
            task=task,
            role=role,
        )
        return self._open_episode

    def set_received_messages(self, messages: List[Dict[str, Any]]) -> None:
        if self._open_episode is not None:
            self._open_episode.received_messages = list(messages)

    def set_critic_on_last_closed(self, critic_result: Dict[str, Any]) -> None:
        if self._episodes:
            self._episodes[-1].critic_feedback = critic_result

    def set_planned_task(self, task: str) -> None:
        if self._open_episode is not None:
            self._open_episode.planned_task = task

    def set_motivation(self, motivation: str) -> None:
        if self._open_episode is not None:
            self._open_episode.motivation = motivation

    def set_action(self, action_name: str, action_args: Dict[str, Any], tick_action: int) -> None:
        if self._open_episode is not None:
            self._open_episode.action_name = action_name
            self._open_episode.action_args = action_args
            self._open_episode.tick_action = tick_action

    def set_loop_warning(self, warning: str) -> None:
        if self._open_episode is not None:
            self._open_episode.loop_warning = warning

    def set_collaboration(self, collab: Dict[str, Any]) -> None:
        if self._open_episode is not None:
            self._open_episode.collaboration = dict(collab)

    def close_episode(self, tick: int, succeeded: Optional[bool], reason: Optional[str]) -> Optional[EpisodeRecord]:
        ep = self._open_episode
        if ep is None or ep.closed:
            return ep
        ep.tick_close = tick
        ep.outcome_succeeded = succeeded
        ep.outcome_reason = reason
        ep.closed = True
        self._episodes.append(ep)
        print(str(ep))
        self._open_episode = None
        return ep

    def get_open_episode(self) -> Optional[EpisodeRecord]:
        return self._open_episode

    def get_closed_episodes(self, n: int = 5) -> List[EpisodeRecord]:
        episodes = list(self._episodes)
        recent = episodes[-n:] if n < len(episodes) else episodes
        return list(reversed(recent))

    def to_prompt_memory(self, n: int = PROMPT_MEMORY_LIMIT) -> List[Dict[str, Any]]:
        result = []
        for ep in self.get_closed_episodes(n):
            entry = {
                'tick': ep.tick_open,
                'task': ep.task,
                'planned_task': ep.planned_task or '',
                'motivation': ep.motivation or '',
                'action': ep.action_name or '',
                'action_args': ep.action_args or {},
                'outcome': _outcome_label(ep.outcome_succeeded),
            }
            critique = (ep.critic_feedback or {}).get('critique', '')
            if critique:
                entry['critique'] = critique
            result.append(entry)
        return result


class MemoryModule:

    def __init__(self, maxlen: int = DEFAULT_MAXLEN) -> None:
        self.base = BaseMemory(maxlen=maxlen)
        self.episode = EpisodeMemory()


class SharedMemory:

    def __init__(self) -> None:
        self.storage: Dict[str, Any] = {}
        self.lock = Lock()

    def update(self, key: str, information: Any) -> None:
        with self.lock:
            self.storage[key] = information

    def retrieve(self, key: str) -> Any:
        with self.lock:
            return self.storage.get(key)

    def retrieve_all(self) -> Dict[str, Any]:
        with self.lock:
            return self.storage.copy()

    def add_to_set(self, key: str, value: Any) -> None:
        with self.lock:
            lst = self.storage.get(key, [])
            if value not in lst:
                self.storage[key] = lst + [value]

    def try_start_rendezvous(self, key: str, entry: Dict[str, Any], dedupe_key: str) -> bool:
        with self.lock:
            existing = self.storage.get(key)
            if (existing is not None
                    and existing.get(dedupe_key) == entry.get(dedupe_key)):
                return False
            self.storage[key] = entry
            return True

    def add_unique_record(self, key: str, record: Dict[str, Any], dedupe_field: str) -> bool:
        with self.lock:
            lst = self.storage.get(key, [])
            target = record.get(dedupe_field)
            if any(r.get(dedupe_field) == target for r in lst):
                return False
            self.storage[key] = lst + [record]
            return True

    def clear_if_initiator(self, key: str, agent_id: str) -> bool:
        with self.lock:
            existing = self.storage.get(key)
            if existing is not None and existing.get('initiator') == agent_id:
                self.storage[key] = None
                return True
            return False
