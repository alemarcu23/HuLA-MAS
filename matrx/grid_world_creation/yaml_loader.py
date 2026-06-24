"""
Load a :class:`WorldPreset` from a YAML map file.

See ``maps/official.yaml`` and ``maps/demo.yaml`` for examples.
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import yaml

from matrx.grid_world_creation.world_model import (
    ALL_VICTIMS, VICTIM_POOL, OBSTACLE_POOL,
    VictimDef, ObstacleDef, DropZoneDef, PropDef, RoomDef, WorldPreset,
    interior_cells, compute_door_bottom, compute_door_top, pack_rooms,
    collect_all_victims, victims_in_category, victim_category, normalize_category,
    img_for_victim,
)

def load_preset(path: str) -> WorldPreset:
    """Parse a YAML map file into a fully-resolved :class:`WorldPreset`."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    name = cfg.get('name', os.path.splitext(os.path.basename(path))[0])
    grid = cfg.get('grid')
    if not grid or len(grid) != 2:
        raise ValueError(f"Map '{name}': 'grid' must be [width, height].")
    grid_w, grid_h = int(grid[0]), int(grid[1])

    seed = cfg.get('seed')
    rng = random.Random(seed)

    rooms = _build_rooms(cfg.get('rooms', []), grid_w, grid_h)
    if not rooms:
        raise ValueError(f"Map '{name}': no rooms defined.")

    # Obstacles
    _resolve_obstacles(rooms, cfg.get('obstacles'), rng)
    _resolve_victims(rooms, cfg.get('victims') or {}, rng)

    # Drop zone
    dz_cfg = cfg.get('drop_zone') or {}
    dz_loc = tuple(dz_cfg.get('location', (grid_w - 2, max(1, grid_h // 2 - 4))))
    ghost = _resolve_goal(rooms, cfg.get('goal'))
    dz_height = dz_cfg.get('height')
    if dz_height is None:
        dz_height = max(len(ghost), 1)
    else:
        dz_height = int(dz_height)
        ghost = ghost[:dz_height]
    drop_zone = DropZoneDef(location=(int(dz_loc[0]), int(dz_loc[1])), height=dz_height)

    # Decorations
    dec = cfg.get('decorations') or {}
    props = [PropDef(name=p.get('name', 'prop'), img=p['img'],
                     location=tuple(p['pos']), size=float(p.get('size', 1.0)),
                     traversable=bool(p.get('traversable', True)))
             for p in (dec.get('props') or [])]
    keyboard = tuple(dec['keyboard_sign']) if dec.get('keyboard_sign') else None

    return WorldPreset(
        name=name,
        grid_width=grid_w,
        grid_height=grid_h,
        rooms=rooms,
        drop_zone=drop_zone,
        ghost_victims=ghost,
        props=props,
        auto_roof=bool(dec.get('auto_roof', True)),
        auto_streets=bool(dec.get('auto_streets', True)),
        auto_signs=bool(dec.get('auto_signs', True)),
        keyboard_sign=keyboard,
        seed=seed,
    )


def _build_rooms(spec: Any, grid_w: int, grid_h: int) -> List[RoomDef]:
    # Auto-layout: {auto: {count, size, columns}}
    if isinstance(spec, dict) and 'auto' in spec:
        a = spec['auto']
        return pack_rooms(
            count=int(a['count']),
            size=tuple(a.get('size', (5, 4))),
            columns=int(a.get('columns', 3)),
            grid_w=grid_w, grid_h=grid_h,
            x_start=int(a.get('x_start', 1)),
            y_start=int(a.get('y_start', 1)),
            x_gap=int(a.get('x_gap', 1)),
            y_gap=int(a.get('y_gap', 2)),
        )

    rooms: List[RoomDef] = []
    for item in spec:
        rid = item['id']
        x, y = item['pos']
        w, h = item.get('size', (5, 4))
        door_spec = item.get('door', 'bottom')
        if door_spec == 'bottom':
            door, mat, enter = compute_door_bottom(x, y, w, h)
        elif door_spec == 'top':
            door, mat, enter = compute_door_top(x, y, w, h)
        elif isinstance(door_spec, (list, tuple)):
            door = tuple(door_spec)
            mat = tuple(item['doormat']) if item.get('doormat') else (door[0], door[1] + 1)
            enter = item.get('enter', 'North')
        else:
            raise ValueError(f"Room {rid}: 'door' must be 'top', 'bottom', or [x, y].")
        rooms.append(RoomDef(id=rid, pos=(x, y), width=w, height=h,
                             door=door, doormat=mat, enter_direction=enter))
    return rooms


class _VictimDrawer:
    """adds victims."""

    def __init__(self, pool_name: str, rng: random.Random):
        self.rng = rng
        if pool_name in ('injured', 'critical_mild'):
            self.base = list(VICTIM_POOL)
        else:
            self.base = list(ALL_VICTIMS)
        self.used: set = set()

    def draw(self, category: Optional[str] = None) -> Optional[Tuple[str, str]]:
        if category is not None:
            canonical = normalize_category(category)
            candidates = [v for v in self.base if victim_category(v[0]) == canonical]
        else:
            candidates = list(self.base)
        if not candidates:
            return None
        fresh = [v for v in candidates if v[0] not in self.used]
        vic = self.rng.choice(fresh if fresh else candidates)
        self.used.add(vic[0])
        return vic


def _place_victim(room: RoomDef, vic: Tuple[str, str], rng: random.Random) -> None:
    cells = interior_cells(room)
    used = {v.location for v in room.victims} | {o.location for o in room.obstacles}
    free = [c for c in cells if c not in used and c != room.door]
    if not free:
        free = cells or [room.doormat]
    loc = rng.choice(free)
    room.victims.append(VictimDef(name=vic[0], img=vic[1], location=loc,
                                  area=f'area {room.id}'))


def _count_category(rooms: List[RoomDef], category: str) -> int:
    canonical = normalize_category(category)
    return sum(1 for r in rooms for v in r.victims
               if victim_category(v.name) == canonical)


def _place_explicit_victims(rooms: List[RoomDef], items: List[Dict[str, Any]]) -> None:
    """Place victims at exact positions: [{type, area, pos: [x, y]}, ...]."""
    by_id = {r.id: r for r in rooms}
    for item in items:
        room = by_id.get(item.get('area'))
        if room is None:
            warnings.warn(f"victims references unknown area {item.get('area')!r}; ignored.")
            continue
        vtype = item['type']
        room.victims.append(VictimDef(
            name=vtype, img=item.get('img') or img_for_victim(vtype),
            location=tuple(item['pos']), area=f'area {room.id}'))


def _resolve_victims(rooms: List[RoomDef], spec, rng: random.Random) -> None:
    # {type, area, pos}.
    if isinstance(spec, list):
        _place_explicit_victims(rooms, spec)
        return

    by_id = {r.id: r for r in rooms}
    drawer = _VictimDrawer(spec.get('pool', 'default'), rng)

    def place(room: RoomDef, category: Optional[str] = None) -> None:
        vic = drawer.draw(category)
        if vic is not None:
            _place_victim(room, vic, rng)

    # areas X, Y, Z must each contain a category victim.
    for cat, area_ids in (spec.get('by_area') or {}).items():
        for aid in area_ids:
            if aid in by_id:
                place(by_id[aid], cat)
            else:
                warnings.warn(f"victims.by_area references unknown area {aid!r}; ignored.")

    target = int(spec.get('per_room', 0) or 0)
    for room in rooms:
        while len(room.victims) < target:
            place(room)

    for cat, n in (spec.get('min_by_category') or {}).items():
        while _count_category(rooms, cat) < int(n):
            place(min(rooms, key=lambda r: len(r.victims)), cat)

    total = spec.get('total')
    if total is not None:
        while sum(len(r.victims) for r in rooms) < int(total):
            place(min(rooms, key=lambda r: len(r.victims)))


def _obstacle_by_type(otype: Optional[str], rng: random.Random) -> Tuple[str, str]:
    if otype:
        for name, img in OBSTACLE_POOL:
            if name == otype:
                return (name, img)
        warnings.warn(f"Unknown obstacle type {otype!r}; picking a random one.")
    return rng.choice(OBSTACLE_POOL)


def _resolve_obstacles(rooms: List[RoomDef], spec: Any, rng: random.Random) -> None:
    if not spec:
        return

    # [{area: 2, type: rock, pos: [x, y]}, ...]
    if isinstance(spec, list):
        by_id = {r.id: r for r in rooms}
        for item in spec:
            room = by_id.get(item.get('area'))
            if room is None:
                warnings.warn(f"obstacles references unknown area {item.get('area')!r}; ignored.")
                continue
            name, img = _obstacle_by_type(item.get('type'), rng)
            loc = tuple(item['pos']) if item.get('pos') else room.door
            room.obstacles.append(ObstacleDef(name=name, img=img, location=loc))
        return

    at_doors = spec.get('at_doors')
    if at_doors:
        prob = float(at_doors.get('probability', 0.5))
        for room in rooms:
            if rng.random() < prob:
                name, img = _obstacle_by_type(at_doors.get('type'), rng)
                room.obstacles.append(ObstacleDef(name=name, img=img, location=room.door))


def _resolve_goal(rooms: List[RoomDef], spec: Any) -> List[Tuple[str, str]]:
    # Only injured victims can be rescued for points 
    injured = [(n, img) for (n, img) in collect_all_victims(rooms)
               if victim_category(n) != 'healthy']
    if not isinstance(spec, dict):
        return injured
    if spec.get('ghosts'):
        return [(t, img_for_victim(t)) for t in spec['ghosts']]
    rescue = spec.get('rescue', 'all')
    if rescue in ('all', None):
        return injured
    n = int(rescue)
    if n > len(injured):
        warnings.warn(
            f"goal.rescue={n} exceeds the {len(injured)} injured victim(s) "
            f"placed on this map; using all {len(injured)}. Add more injured "
            f"victims or lower 'rescue' to silence this.")
    return injured[:n]
