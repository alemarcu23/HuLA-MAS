import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from matrx.helpers.perception_helpers import _serialize_agent, _serialize_nearby, _classify_type
from matrx.helpers.logic_helpers import _chebyshev_distance
from matrx.agents.agent_utils.state import State


_VISION_RADIUS = {'low': 1, 'medium': 2, 'high': 3}
PERCEPTION_REGISTRY: Dict[str, Dict[str, Any]] = {}

def count_cells(
    top_left: Tuple[int, int],
    width: int,
    height: int,
) -> Set[Tuple[int, int]]:
    x0, y0 = top_left
    cells: Set[Tuple[int, int]] = set()
    for x in range(x0 + 1, x0 + width - 1):
        for y in range(y0 + 1, y0 + height - 1):
            cells.add((x, y))
    return cells


def precompute_all_areas(
    areas_config: List[Dict[str, Any]],
) -> Dict[str, Set[Tuple[int, int]]]:
    result: Dict[str, Set[Tuple[int, int]]] = {}
    for cfg in areas_config:
        area_id = cfg["id"]
        if area_id == "world_bounds":
            continue
        name = f"area {area_id}"
        result[name] = count_cells(cfg["pos"], cfg["w"], cfg["h"])
    return result


@dataclass
class _AreaState:
    name: str
    inside_cells: Set[Tuple[int, int]]
    explored_cells: Set[Tuple[int, int]] = field(default_factory=set)

    @property
    def total(self) -> int:
        return len(self.inside_cells)

    @property
    def explored_count(self) -> int:
        return len(self.explored_cells)

    @property
    def unexplored_count(self) -> int:
        return self.total - self.explored_count

    @property
    def coverage(self) -> float:
        return self.explored_count / self.total if self.total else 1.0

    @property
    def remaining_cells(self) -> List[List[int]]:
        return sorted([list(c) for c in self.inside_cells - self.explored_cells])

    @property
    def is_complete(self) -> bool:
        return self.unexplored_count == 0

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "coverage": round(self.coverage, 2),
            "status": "complete" if self.is_complete else
                      "in_progress" if self.explored_count > 0 else
                      "not_started",
        }


class Perception:
    """Converts a filtered MATRX WorldState dict to dict."""

    def percept_state(
        self,
        state: Dict[str, Any],
        agent_id: str,
        teammates: Optional[set] = None,
    ) -> Dict[str, Any]:
        # Convert a filtered MATRX WorldState dict to a perception dict.
        teammate_ids: set = {t[0] for t in teammates} if teammates else set()

        victims, obstacles, walls = _serialize_nearby(state, agent_id, teammate_ids)
        result = {
            "agent": _serialize_agent(state, agent_id),
            "victims": victims,
            "obstacles": obstacles,
            "walls": walls,
            "teammates": [
                {"object_id": t[0], "x": int(t[1][0]), "y": int(t[1][1])}
                for t in teammates
            ] if teammates else [],
        }
        return result

    @property
    def _vision_radius(self) -> int:
        """Vision range (in grid blocks) for THIS agent, from its capability
        preset. Single source of truth: every site that limits what the agent
        reads to its vision (filter, prune, area-exploration) uses this.
        """
        caps = getattr(self, '_capabilities', None) or {}
        return _VISION_RADIUS.get(caps.get('vision', 'medium'), 2)

    def init_global_state(self) -> None:
        self.WORLD_STATE_GLOBAL: Dict[str, Any] = {
            'victims':   [],   # [{'object_id', 'severity', 'location'}]
            'obstacles': [],   # [{'object_id', 'type', 'location'}]
        }

        self._world_state_belief: Dict[str, Dict[str, Dict[str, Any]]] = {
            'victims': {}, 'obstacles': {},
        }

        self._area_states: Dict[str, _AreaState] = {}

        aid = getattr(self, 'agent_id', None)
        if aid:
            PERCEPTION_REGISTRY[aid] = self.WORLD_STATE_GLOBAL

    def init_area_tracker(
        self,
        area_cells: Dict[str, Set[Tuple[int, int]]],
    ) -> None:
        """Set up area exploration belief.
        """
        self._area_states = {
            name: _AreaState(name=name, inside_cells=frozenset(cells))
            for name, cells in area_cells.items()
        }

        for aname, astate in sorted(self._area_states.items()):
            if astate.inside_cells:
                xs = [x for x, _y in astate.inside_cells]
                ys = [y for _x, y in astate.inside_cells]
                print(
                    f'  {aname}: {astate.total} inside cells  '
                    f'x=[{min(xs)},{max(xs)}] y=[{min(ys)},{max(ys)}]'
                )
            else:
                print(
                    f'  {aname}: 0 inside cells '
                    f'(EMPTY — check room dimensions in the preset!)'
                )

    def update_area_exploration(
        self,
        agent_location: Tuple[int, int],
        vision_radius: int = 1,
    ) -> None:
        """Mark the cells currently within vision as explored."""
        ax, ay = agent_location
        observed: Set[Tuple[int, int]] = {
            (ax + dx, ay + dy)
            for dx in range(-vision_radius, vision_radius + 1)
            for dy in range(-vision_radius, vision_radius + 1)
        }
        for area in self._area_states.values():
            newly_seen = observed & area.inside_cells
            if newly_seen:
                area.explored_cells |= newly_seen

    def get_area_summary(self, area_name: str) -> Optional[Dict[str, Any]]:
        area = self._area_states.get(area_name)
        return area.summary() if area else None

    def get_area_summaries(self) -> List[Dict[str, Any]]:
        return [a.summary() for a in self._area_states.values()]

    def is_area_complete(self, area_name: str) -> bool:
        area = self._area_states.get(area_name)
        return area.is_complete if area else False

    def update_world_belief(self, state) -> Dict[str, Any]:
        # Update the agent's global world belief with new observations.
        if not hasattr(self, 'WORLD_STATE_GLOBAL'):
            self.init_global_state()

        aid = getattr(self, 'agent_id', None)
        if aid and PERCEPTION_REGISTRY.get(aid) is not self.WORLD_STATE_GLOBAL:
            PERCEPTION_REGISTRY[aid] = self.WORLD_STATE_GLOBAL
        if state is not None:
            self._remove_disappeared(state)
            self.add_new_obs(state)

        rescued_ids: set = set()
        shared = getattr(self, 'shared_memory', None)
        if shared:
            rescued_ids = {
                v['victim_id']
                for v in (shared.retrieve('rescued_victims') or [])
            }
        victims = {
            v['object_id']: {'location': v.get('location'), 'severity': v.get('severity')}
            for v in self.WORLD_STATE_GLOBAL.get('victims', [])
            if v.get('object_id') and v['object_id'] not in rescued_ids
        }
        obstacles = {
            o['object_id']: {'location': o.get('location'), 'type': o.get('type')}
            for o in self.WORLD_STATE_GLOBAL.get('obstacles', [])
            if o.get('object_id')
        }
        self._world_state_belief = {'victims': victims, 'obstacles': obstacles}

        return self.WORLD_STATE_GLOBAL

    def _remove_disappeared(self, state) -> None:
        """Remove objects from ``WORLD_STATE_GLOBAL`` that are no longer in state."""
        agent_data = state.get(self.agent_id) if self.agent_id else None
        agent_loc = agent_data.get('location') if isinstance(agent_data, dict) else None
        if agent_loc is None:
            return

        vision_radius = self._vision_radius

        visible_ids = {
            oid for oid, d in state.items()
            if isinstance(d, dict) and isinstance(d.get('location'), (list, tuple))
        }

        for bucket in ('victims', 'obstacles'):
            kept = []
            for o in self.WORLD_STATE_GLOBAL.get(bucket, []):
                loc = o.get('location')
                if loc is None:
                    kept.append(o)
                    continue
                in_vision = _chebyshev_distance(agent_loc, loc) <= vision_radius
                if in_vision and o.get('object_id') not in visible_ids:
                    continue
                kept.append(o)
            self.WORLD_STATE_GLOBAL[bucket] = kept

    def add_new_obs(self, state) -> None:
        """Merge state into ``WORLD_STATE_GLOBAL``."""
        if state is None:
            return

        partner_name = getattr(self, '_partner_name', None)
        include_human = getattr(self, '_include_human', False)
        
        teammate_ids: set = {t[0] for t in getattr(self, 'teammates', set())}

        #  Nearby objects 
        skip_ids = {self.agent_id, 'World'} | teammate_ids
        if include_human and partner_name:
            skip_ids.add(partner_name)

        for obj_id, obj_data in state.items():
            if obj_id in skip_ids:
                continue
            if not isinstance(obj_data, dict):
                continue
            loc = obj_data.get('location')
            if loc is None:
                continue

            typ = _classify_type(obj_id, obj_data, teammate_ids)
            if typ is None:
                continue

            pos = [int(c) for c in loc]

            # Victims
            if typ == 'victim':
                img = str(obj_data.get('img_name', '')).lower()
                if 'critical' in img:
                    severity = 'critical'
                elif 'mild' in img:
                    severity = 'mild'
                else:
                    severity = 'healthy'
                existing = next((v for v in self.WORLD_STATE_GLOBAL['victims'] if v['object_id'] == obj_id), None)
                if existing is None:
                    self.WORLD_STATE_GLOBAL['victims'].append(
                        {'object_id': obj_id, 'severity': severity, 'location': pos}
                    )
                else:
                    existing.update(location=pos, severity=severity)

            # Obstacles
            elif typ in ('rock', 'stone', 'tree'):
                existing = next((o for o in self.WORLD_STATE_GLOBAL['obstacles'] if o['object_id'] == obj_id), None)
                if existing is None:
                    self.WORLD_STATE_GLOBAL['obstacles'].append(
                        {'object_id': obj_id, 'type': typ, 'location': pos}
                    )
                else:
                    existing.update(location=pos, type=typ)