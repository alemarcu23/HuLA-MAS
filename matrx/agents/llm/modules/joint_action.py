r"""
JointAction — deterministic coordination of two-agent (joint) actions.
"""

from typing import Any, Dict, List, Optional, Tuple

from matrx.messages.message import Message
from matrx.helpers.logic_helpers import _chebyshev_distance
from matrx.helpers.navigation_helpers import navigate_to
from matrx.agents.llm.modules.agent_infra.communication_module import extract_coords_from_text

from matrx.actions.CustomActions import CarryObjectTogether as _CarryObjectTogether
from matrx.actions.CustomActions import RemoveObjectTogether as _RemoveObjectTogether
from matrx.actions.CustomActions import DropObjectTogether as _DropObjectTogether

# Message types
MSG_ASK_HELP = 'ask_help' # to request help
MSG_HELP = 'help' # to accept help
MSG_HELP_DONE = 'help_done' # to signal that help is done

REPLY_YES = 'yes'
REPLY_NO = 'no'

KIND_CARRY = 'carry'
KIND_REMOVE = 'remove'

# to check who started the joint action and who is the partner
ROLE_FIRER = 'firer'
ROLE_POSITIONER = 'positioner'

# ask-trigger mode (one does it auto one requires the LLM to confirm)
TRIGGER_AUTO = 'auto_bridge'      # simulation will pick a partner
TRIGGER_EXPLICIT = 'llm_explicit'  # more difficult for LLM, but it also gives the LLM more control over when to ask for help

ADJACENT = 1
CONSENT_TIMEOUT = 250  # ticks to wait for a "yes" before abandoning the request
GOTO_TIMEOUT = 300     # ticks to reach the target


class JointAction:
    """All joint-action logic. The agent only routes data in and acts on the
    returned value"""

    # ── init ───────────────────────────────────────────────────────────────
    def _init_joint_action(self, ask_trigger: str = TRIGGER_AUTO) -> None:
        self._joint: Optional[Dict[str, Any]] = None
        self._known_request: Optional[Dict[str, Any]] = None
        self._ask_trigger = ask_trigger if ask_trigger in (TRIGGER_AUTO, TRIGGER_EXPLICIT) else TRIGGER_AUTO
        self._pending_help_abandon: Optional[str] = None
        self._joint_processed_msgs: int = 0
        self._in_position_announced: bool = False

    def _joint_active(self) -> bool:
        """agent stops LLM calls and infra is taking control."""
        return self._joint is not None

    def _run_infra(self, filtered_state) -> Optional[Tuple[str, Dict]]:
        action = self._joint_infra(filtered_state)
        if action is not None:
            return action
        return self._handle_navigation_tick()

    def _ingest_joint_messages(self, received_messages: List[Any], tick: int) -> None:
        """Process new protocol messages and translated human button presses."""
        new = received_messages[self._joint_processed_msgs:]
        self._joint_processed_msgs = len(received_messages)
        for raw in new:
            content = getattr(raw, 'content', raw)
            sender = getattr(raw, 'from_id', '') or (
                content.get('from', '') if isinstance(content, dict) else '')
            if not sender or sender == self.agent_id:
                continue

            if self._is_human_partner(sender) and self._adapt_human_message(content, sender, tick):
                continue
            if not isinstance(content, dict):
                continue
            mtype = content.get('message_type', '')
            if mtype == MSG_ASK_HELP:
                self._on_ask_help(content, sender, tick)
            elif mtype == MSG_HELP:
                self._on_help_reply(content, sender, tick)
            elif mtype == MSG_HELP_DONE:
                self._on_help_done(content, sender)

    def _joint_on_outgoing_message(
        self, send_to: Any, message_type: str, text: str, args: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Decide what to do with the LLM SendMessage.
        """
        stripped = (text or '').strip().lower()

        if message_type == MSG_ASK_HELP:
            if self._joint is not None:
                return 'suppress', {}
            target_id, loc, kind = self._resolve_ask_target(args, text)
            if not target_id or not loc:
                return 'suppress', {}
            self._begin_request_state(target_id, loc, kind)
            return 'send', {
                'text': text or f"I need help to {kind} {target_id} at {tuple(loc)}.",
                'target_id': target_id, 'target_location': list(loc), 'kind': kind,
            }

        if (message_type == MSG_HELP and send_to not in (None, 'all')
                and stripped in (REPLY_YES, REPLY_NO)):
            if stripped == REPLY_YES:
                if self._joint is not None:
                    return 'send', {'text': REPLY_NO}
                kr = self._known_request
                if kr and kr['requester'] == send_to:
                    self._begin_help_state(kr)
            return 'send', {'text': stripped}

        return 'send', {}

    def _joint_on_action(self, name: str, args: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Decide what to do with the LLM's joint-action tool call.
        """
        if self._joint is not None:
            return 'idle', 'joint action already in progress'

        kind = KIND_CARRY if name == 'CarryObjectTogether' else KIND_REMOVE
        target_id, loc, kind = self._resolve_target(
            args.get('object_id', ''),
            args.get('location') or args.get('victim_location') or args.get('object_location'),
            kind)
        if not target_id or not loc:
            return 'error', (
                f"Cannot start {name}: no {kind} target is in view. "
                "Observe or SearchArea to find one first."
            )

        if self._ask_trigger == TRIGGER_AUTO:
            self._begin_request_state(target_id, loc, kind)
            self._send(MSG_ASK_HELP, f"I need help to {kind} {target_id} at {tuple(loc)}.",
                       to_id=None, target_id=target_id, target_location=list(loc), kind=kind)
            return 'idle', 'auto-requested help (auto_bridge)'

        return 'error', (
            f"Cannot {name} yet — no teammate has agreed to help. First send "
            f"SendMessage(message_type='ask_help', ...) naming {target_id} and its location "
            f"{tuple(loc)}, then wait for a teammate to reply 'yes'."
        )

    def _joint_infra(self, filtered_state) -> Optional[Tuple[str, Dict]]:
        self._check_success()
        j = self._joint
        if j is None:
            return None
        if j['phase'] == 'await_consent':
            if self._tick_count - j['started_tick'] > CONSENT_TIMEOUT:
                self._abandon('No teammate accepted your help request in time. '
                              'Choose a different objective.')
            return self._idle('awaiting_help_consent')
        if j['phase'] == 'deliver':
            return self._drive_deliver()
        return self._drive_to_target()  # phase == 'go'

    def _drive_to_target(self) -> Optional[Tuple[str, Dict]]:
        j = self._joint
        target_id, role = j['target_id'], j['role']

        # Refresh the live location; detect the target leaving the world.
        loc = self._locate(target_id)
        if loc is not None:
            j['target_loc'] = loc
        elif self._target_gone(target_id):
            if role == ROLE_POSITIONER and not j['partner_is_human']:
                # An AI firer will broadcast help_done after delivery — wait for it.
                return self._idle('joint_waiting_for_done')
            # Firer (pre-fire) or human-partner positioner: the job is over here.
            self._finish(broadcast=False)
            return None

        target_loc = j['target_loc']
        if target_loc is None:
            return self._idle('joint_no_target_loc')
        if self._tick_count - j['started_tick'] > GOTO_TIMEOUT:
            self._abandon('Could not reach the joint-action target in time. '
                          'Choose a different objective.')
            return None

        if _chebyshev_distance(tuple(self.agent_location), tuple(target_loc)) > ADJACENT:
            # Navigate to a reachable tile NEXT TO the target, never the target's
            # own tile: an obstacle tile is intraversable, so A* would return no
            # path and the agent would idle forever ("stuck behind the rock").
            if self._nav_target is None:
                goal = self._nav_goal(target_loc)
                nav_target, action = navigate_to(
                    goal, navigator=self._navigator, state_tracker=self._state_tracker)
                self._nav_target = nav_target
                return action
            return None  # already navigating — let _handle_navigation_tick advance

        # Adjacent to the target.
        if role == ROLE_POSITIONER:
            if not self._in_position_announced:
                self._send(MSG_HELP,
                           f"In position next to {target_id} at {tuple(target_loc)} — go ahead.",
                           to_id=j['partner_id'])
                self._in_position_announced = True
            return self._idle('joint_in_position')

        # Firer: fire only once the partner is also adjacent (sim makes the final call).
        pl = self._partner_loc(j['partner_id'])
        if pl is None or _chebyshev_distance(pl, tuple(target_loc)) > ADJACENT:
            return self._idle('joint_waiting_for_partner')
        action = _CarryObjectTogether.__name__ if j['kind'] == KIND_CARRY else _RemoveObjectTogether.__name__
        print(f"[{self.agent_id}] JOINT firing {action} on '{target_id}' with {j['partner_id']} "
              f"(both adjacent at {tuple(target_loc)})")
        if j['kind'] == KIND_CARRY:
            return action, {'object_id': target_id, 'partner_name': j['partner_id']}
        return action, {'object_id': target_id, 'remove_range': ADJACENT, 'partner_name': j['partner_id']}

    def _drive_deliver(self) -> Optional[Tuple[str, Dict]]:
        j = self._joint
        dz = tuple(self.env_info.drop_zone)
        if _chebyshev_distance(tuple(self.agent_location), dz) > ADJACENT:
            if self._nav_target != dz:
                nav_target, action = navigate_to(
                    dz, navigator=self._navigator, state_tracker=self._state_tracker)
                self._nav_target = nav_target
                return action
            return None
        return _DropObjectTogether.__name__, {
            'object_id': j['target_id'], 'partner_name': j['partner_id']}

    def _check_success(self) -> None:
        """Advance the state machine when the last action this agent fired landed."""
        j = self._joint
        if j is None or self.previous_action_result is None \
                or not getattr(self.previous_action_result, 'succeeded', False):
            return
        pa = self.previous_action
        if j['phase'] == 'go' and j['role'] == ROLE_FIRER:
            if j['kind'] == KIND_CARRY and pa == _CarryObjectTogether.__name__:
                j['phase'] = 'deliver'
                j['started_tick'] = self._tick_count
                self._nav_target = None
            elif j['kind'] == KIND_REMOVE and pa == _RemoveObjectTogether.__name__:
                self._finish(broadcast=True)
        elif j['phase'] == 'deliver' and pa == _DropObjectTogether.__name__:
            self._record_rescue(j['target_id'], j['partner_id'])
            self._finish(broadcast=True)

    # ── state transitions ───────────────────────────────────────────────────
    def _begin_request_state(self, target_id: str, loc: Tuple[int, int], kind: str) -> None:
        self._joint = {
            'kind': kind, 'target_id': target_id, 'target_loc': tuple(loc),
            'partner_id': None, 'partner_is_human': False,
            'role': None, 'phase': 'await_consent', 'started_tick': self._tick_count,
        }
        self._reset_nav_flags()
        print(f"[{self.agent_id}] JOINT ask_help: {kind} '{target_id}' at {tuple(loc)} — awaiting a teammate's yes")

    def _begin_help_state(self, kr: Dict[str, Any]) -> None:
        # The consenter is always a POSITIONER: in AI–AI the requester fires; in
        # human–AI the human fires.
        self._joint = {
            'kind': kr['kind'], 'target_id': kr['target_id'],
            'target_loc': self._coerce_loc(kr['target_loc']),
            'partner_id': kr['requester'],
            'partner_is_human': self._is_human_partner(kr['requester']),
            'role': ROLE_POSITIONER, 'phase': 'go', 'started_tick': self._tick_count,
        }
        self._known_request = None
        self._reset_nav_flags()
        print(f"[{self.agent_id}] JOINT consented to {self._joint['partner_id']}'s "
              f"{self._joint['kind']} of '{self._joint['target_id']}' — heading there as positioner")

    def _finish(self, broadcast: bool) -> None:
        j = self._joint
        self._joint = None
        self._reset_nav_flags()
        if broadcast and j is not None:
            print(f"[{self.agent_id}] JOINT {j['kind']} of '{j['target_id']}' COMPLETE")
            self._send(MSG_HELP_DONE, f"Joint {j['kind']} of {j['target_id']} complete.",
                       to_id=None, target_id=j['target_id'])

    def _abandon(self, reason: str) -> None:
        if self._joint is not None:
            print(f"[{self.agent_id}] JOINT abandoned ({self._joint.get('kind')} "
                  f"'{self._joint.get('target_id')}'): {reason}")
        self._joint = None
        self._reset_nav_flags()
        self._pending_help_abandon = reason

    def _reset_nav_flags(self) -> None:
        self._nav_target = None
        self._in_position_announced = False
        # Discard any in-flight LLM call so a stale result can't act mid-engagement.
        self._pending_future = None

    def _record_rescue(self, victim_id: str, partner_id: str) -> None:
        if self.shared_memory and victim_id:
            added = self.shared_memory.add_unique_record('rescued_victims', {
                'victim_id': victim_id, 'tick': self._tick_count,
                'agent': self.agent_id, 'partner': partner_id, 'method': 'cooperative',
            }, dedupe_field='victim_id')
            if added:
                print(f"[{self.agent_id}] JOINT cooperative rescue recorded: '{victim_id}' "
                      f"(with {partner_id})")

    # ── incoming message handlers ────────────────────────────────────────────
    def _on_ask_help(self, content: Dict[str, Any], sender: str, tick: int) -> None:
        target_id = content.get('target_id') or content.get('victim_id') or ''
        if not target_id:
            return
        loc = (self._coerce_loc(content.get('target_location') or content.get('victim_location'))
               or extract_coords_from_text(content.get('text', ''))
               or self._locate(target_id))
        kind = (content.get('kind') or self._target_kind(target_id)).lower()
        self._known_request = {
            'requester': sender, 'target_id': target_id,
            'target_loc': loc, 'kind': kind, 'tick': tick,
        }

    def _on_help_reply(self, content: Dict[str, Any], sender: str, tick: int) -> None:
        j = self._joint
        if j is None or j['phase'] != 'await_consent':
            return
        if (content.get('text') or '').strip().lower() != REPLY_YES:
            return  # 'no' / other → keep waiting for another teammate
        human = self._is_human_partner(sender)
        j['partner_id'] = sender
        j['partner_is_human'] = human
        j['role'] = ROLE_POSITIONER if human else ROLE_FIRER
        j['phase'] = 'go'
        j['started_tick'] = tick
        self._reset_nav_flags()
        who = 'human fires' if human else 'I fire'
        print(f"[{self.agent_id}] JOINT {sender} accepted — {j['kind']} of '{j['target_id']}' "
              f"at {j['target_loc']} ({who}); driving into position")

    def _on_help_done(self, content: Dict[str, Any], sender: str) -> None:
        j = self._joint
        if j is not None and content.get('target_id') and content['target_id'] == j['target_id']:
            self._finish(broadcast=False)

    # ── human GUI → protocol adapter ─────────────────────────────────────────
    def _adapt_human_message(self, content: Any, sender: str, tick: int) -> bool:
        """Translate a human's fixed chat-button press into the protocol.

        The human's button carries no target id/location, so a *consent* is
        enriched from our own pending request and an *initiation* infers the
        target from the human's surroundings. Returns True when handled.
        """
        text = content.get('text', '') if isinstance(content, dict) else (
            content if isinstance(content, str) else '')
        phrase = (text or '').strip().lower()
        if not phrase:
            return False
        accept = phrase in ('rescue together', 'remove together')
        decline = phrase in ('rescue alone', 'remove alone', 'continue')
        initiate_remove = phrase == 'help remove' or phrase.startswith('remove:')
        if not (accept or decline or initiate_remove):
            return False

        awaiting = (self._joint is not None and self._joint['phase'] == 'await_consent'
                    and self._joint['partner_id'] in (None, sender))
        if awaiting and (accept or decline):
            self._on_help_reply({'text': REPLY_YES if accept else REPLY_NO}, sender, tick)
            return True

        if accept or initiate_remove:
            kind = KIND_REMOVE if 'remove' in phrase else KIND_CARRY
            target = self._infer_target(kind, self._partner_loc(sender))
            if target:
                tid, loc, kind = target
                self._on_ask_help(
                    {'target_id': tid, 'target_location': list(loc), 'kind': kind, 'text': text},
                    sender, tick)
        return True  # 'continue'/decline with nothing pending → consumed

    def _joint_candidates(self, kind: Optional[str]) -> List[Tuple[str, Tuple[int, int], str]]:
        """Real (id, loc, kind) carry/remove targets, from this agent's world
        belief plus its live full-vision state (deduped by id). Restricted to
        ``kind`` when given. Drawing from both is what makes resolution robust to
        a stale belief or an object that has just entered view."""
        seen: Dict[str, Tuple[Tuple[int, int], str]] = {}
        pools = []
        if kind != KIND_REMOVE:
            pools.append(('victims', KIND_CARRY, True))
        if kind != KIND_CARRY:
            pools.append(('obstacles', KIND_REMOVE, False))
        for pool_key, k, skip_healthy in pools:
            for o in (self.WORLD_STATE_GLOBAL.get(pool_key, []) or []):
                oid = o.get('object_id') or o.get('obj_id') or ''
                oloc = o.get('location')
                if oid and oloc is not None and not (skip_healthy and 'healthy' in oid):
                    seen[oid] = ((int(oloc[0]), int(oloc[1])), k)
        st = self.state_for_navigation
        if st is not None and hasattr(st, 'items'):
            for oid, od in st.items():
                if oid == self.agent_id or oid in seen or not isinstance(od, dict):
                    continue
                oloc = od.get('location')
                if oloc is None:
                    continue
                is_victim = bool(od.get('is_collectable'))
                # Only rock/stone/tree are joint-removable — never walls/doors,
                # which are also intraversable and non-collectable on the real map.
                is_obstacle = (od.get('is_traversable') is False
                               and not od.get('isAgent') and not is_victim
                               and any(k in oid.lower() for k in ('rock', 'stone', 'tree')))
                if is_victim and 'healthy' in oid.lower():
                    continue
                if (kind == KIND_REMOVE and not is_obstacle) or (kind == KIND_CARRY and not is_victim):
                    continue
                if kind is None and not (is_victim or is_obstacle):
                    continue
                seen[oid] = ((int(oloc[0]), int(oloc[1])), KIND_CARRY if is_victim else KIND_REMOVE)
        return [(oid, loc, k) for oid, (loc, k) in seen.items()]

    def _infer_target(self, kind: Optional[str],
                      origin: Optional[Tuple[int, int]] = None
                      ) -> Optional[Tuple[str, Tuple[int, int], str]]:
        """Nearest matching object to ``origin`` (default: this agent)."""
        origin = origin or tuple(self.agent_location)
        cands = self._joint_candidates(kind)
        if not cands:
            return None
        return min(cands, key=lambda c: abs(c[1][0] - origin[0]) + abs(c[1][1] - origin[1]))

    def _object_at(self, loc: Tuple[int, int], kind: Optional[str]
                   ) -> Optional[Tuple[str, Tuple[int, int], str]]:
        """A real object at/adjacent to ``loc`` (LLM ids are vague but its
        coordinates are usually right)."""
        near = [c for c in self._joint_candidates(kind)
                if abs(c[1][0] - loc[0]) + abs(c[1][1] - loc[1]) <= 1]
        if not near:
            return None
        return min(near, key=lambda c: abs(c[1][0] - loc[0]) + abs(c[1][1] - loc[1]))

    def _resolve_target(self, target_id: str, loc: Any, kind: Optional[str]
                        ) -> Tuple[str, Optional[Tuple[int, int]], str]:
        """Turn a possibly-vague (id, loc, kind) into a concrete on-grid target.

        Weak LLMs routinely pass a teammate's name, a generic word ("rock"), or
        the wrong kind, but usually a usable location. An id we can actually
        locate wins; else we match the given location to a real object; else we
        take the nearest matching object. This is what lets joint actions fire
        despite imperfect LLM requests.
        """
        kind = (kind or '').strip().lower() or None
        loc = self._coerce_loc(loc)
        if target_id:
            exact = self._locate(target_id)
            if exact is not None:
                return target_id, exact, (kind or self._target_kind(target_id))
        if loc is not None:
            by_loc = self._object_at(loc, kind)
            if by_loc:
                return by_loc
        inferred = self._infer_target(kind)
        if inferred:
            return inferred
        return target_id, loc, (kind or KIND_CARRY)

    # ── small helpers ────────────────────────────────────────────────────────
    def _resolve_ask_target(
        self, args: Dict[str, Any], text: str,
    ) -> Tuple[str, Optional[Tuple[int, int]], str]:
        target_id = args.get('target_id') or args.get('victim_id') or ''
        loc = (self._coerce_loc(args.get('target_location') or args.get('victim_location'))
               or extract_coords_from_text(text or ''))
        return self._resolve_target(target_id, loc, args.get('kind') or '')

    def _is_human_partner(self, agent_id: str) -> bool:
        if not agent_id or agent_id == self.agent_id:
            return False
        if getattr(self, '_include_human', False) and agent_id == getattr(self, '_partner_name', None):
            return True
        roster = (self.shared_memory.retrieve('registered_agents') if self.shared_memory else None) or []
        return agent_id not in roster

    def _target_kind(self, target_id: str) -> str:
        t = (target_id or '').lower()
        return KIND_REMOVE if any(k in t for k in ('rock', 'stone', 'tree', 'obstacle')) else KIND_CARRY

    def _coerce_loc(self, loc: Any) -> Optional[Tuple[int, int]]:
        if loc and len(loc) == 2:
            try:
                return (int(loc[0]), int(loc[1]))
            except (TypeError, ValueError):
                return None
        return None

    def _locate(self, target_id: str) -> Optional[Tuple[int, int]]:
        if not target_id:
            return None
        st = self.state_for_navigation
        if st is not None:
            obj = st.get(target_id)
            if isinstance(obj, dict) and obj.get('location') is not None:
                return tuple(obj['location'])
        for key in ('victims', 'obstacles'):
            for o in (self.WORLD_STATE_GLOBAL.get(key, []) or []):
                if (o.get('object_id') or o.get('obj_id')) == target_id and o.get('location') is not None:
                    return (int(o['location'][0]), int(o['location'][1]))
        return None

    def _target_gone(self, target_id: str) -> bool:
        st = self.state_for_navigation
        return bool(st is not None and target_id and st.get(target_id) is None)

    def _partner_loc(self, partner_id: str) -> Optional[Tuple[int, int]]:
        if not partner_id:
            return None
        st = self.state_for_navigation
        if st is not None:
            obj = st.get(partner_id)
            if isinstance(obj, dict) and obj.get('location') is not None:
                return tuple(obj['location'])
        for t in (self.WORLD_STATE.get('teammates', []) or []):
            if t.get('object_id') == partner_id and t.get('x') is not None:
                return (int(t['x']), int(t['y']))
        return None

    def _blocked_tiles(self) -> set:
        """Tiles an agent cannot stand on: intraversable objects (walls, the
        obstacle itself) and other agents. Read from the live full-vision state."""
        blocked = set()
        st = self.state_for_navigation
        if st is not None and hasattr(st, 'items'):
            for oid, od in st.items():
                if not isinstance(od, dict):
                    continue
                loc = od.get('location')
                if loc is None:
                    continue
                if od.get('is_traversable') is False or od.get('isAgent'):
                    blocked.add((int(loc[0]), int(loc[1])))
        return blocked

    def _in_bounds(self, t: Tuple[int, int]) -> bool:
        gs = getattr(self.env_info, 'grid_size', None)
        if not gs:
            return True
        return 0 <= t[0] < gs[0] and 0 <= t[1] < gs[1]

    def _nav_goal(self, target_loc: Tuple[int, int]) -> Tuple[int, int]:
        """A reachable tile to actually walk to for a joint action.

        If the target tile is standable (a traversable, free victim tile) we go
        onto it; otherwise (an intraversable obstacle, or an occupied tile) we go
        to the nearest free tile beside it — A* cannot path onto a blocked tile,
        so targeting it directly leaves the agent stuck."""
        target_loc = (int(target_loc[0]), int(target_loc[1]))
        blocked = self._blocked_tiles()
        if target_loc not in blocked:
            return target_loc
        me = tuple(self.agent_location)
        cands = [(target_loc[0] + dx, target_loc[1] + dy)
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 if (dx or dy)]
        cands = [t for t in cands if self._in_bounds(t) and t not in blocked]
        if not cands:
            return target_loc  # fully surrounded — let the navigator try (and time out)
        return min(cands, key=lambda t: abs(t[0] - me[0]) + abs(t[1] - me[1]))

    def _send(self, message_type: str, text: str, to_id: Optional[str] = None, **extra: Any) -> None:
        content = {'message_type': message_type, 'from': self.agent_id, 'text': text}
        content.update(extra)
        self.send_message(Message(content=content, from_id=self.agent_id, to_id=to_id))
