"""
Helpers for easy world generation using YAML files.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VICTIM_POOL: List[Tuple[str, str]] = [
    ('critically injured girl', '/images/critically injured girl.svg'),
    ('critically injured elderly woman', '/images/critically injured elderly woman.svg'),
    ('critically injured man', '/images/critically injured man.svg'),
    ('critically injured dog', '/images/critically injured dog.svg'),
    ('mildly injured boy', '/images/mildly injured boy.svg'),
    ('mildly injured elderly man', '/images/mildly injured elderly man.svg'),
    ('mildly injured woman', '/images/mildly injured woman.svg'),
    ('mildly injured cat', '/images/mildly injured cat.svg'),
]

HEALTHY_VICTIM_POOL: List[Tuple[str, str]] = [
    ('healthy boy', '/images/healthy boy.svg'),
    ('healthy woman', '/images/healthy woman.svg'),
    ('healthy elderly man', '/images/healthy elderly man.svg'),
    ('healthy girl', '/images/healthy girl.svg'),
    ('healthy man', '/images/healthy man.svg'),
    ('healthy elderly woman', '/images/healthy elderly woman.svg'),
    ('healthy cat', '/images/healthy cat.svg'),
    ('healthy dog', '/images/healthy dog.svg'),
]

ALL_VICTIMS: List[Tuple[str, str]] = VICTIM_POOL + HEALTHY_VICTIM_POOL

OBSTACLE_POOL: List[Tuple[str, str]] = [
    ('rock', '/images/stone.svg'),
    ('stone', '/images/stone-small.svg'),
    ('tree', '/images/tree-fallen2.svg'),
]

CATEGORY_ALIASES: Dict[str, str] = {
    'red': 'critical', 'critical': 'critical', 'critically_injured': 'critical',
    'yellow': 'mild', 'mild': 'mild', 'mildly_injured': 'mild',
    'green': 'healthy', 'healthy': 'healthy',
}


def victim_category(name: str) -> str:
    """Return the category ('critical' / 'mild' / 'healthy') of a victim name."""
    n = name.lower()
    if n.startswith('critically injured'):
        return 'critical'
    if n.startswith('mildly injured'):
        return 'mild'
    return 'healthy'


def normalize_category(cat: str) -> str:
    """Resolve a user-facing category alias (e.g. 'red') to a canonical one."""
    key = str(cat).strip().lower().replace(' ', '_')
    if key not in CATEGORY_ALIASES:
        raise ValueError(
            f"Unknown victim category {cat!r}. "
            f"Valid options: {sorted(set(CATEGORY_ALIASES))}"
        )
    return CATEGORY_ALIASES[key]


def victims_in_category(category: Optional[str]) -> List[Tuple[str, str]]:
    """Return the (name, img) pool for a category, or all victims if None."""
    if category is None:
        return list(ALL_VICTIMS)
    canonical = normalize_category(category)
    return [v for v in ALL_VICTIMS if victim_category(v[0]) == canonical]


VICTIM_IMG: Dict[str, str] = {name: img for name, img in ALL_VICTIMS}


def img_for_victim(name: str) -> str:
    """Resolve a victim type name to its image path (used for explicit placement)."""
    base = name.split(' in area')[0].strip()
    if base in VICTIM_IMG:
        return VICTIM_IMG[base]
    return f"/images/{base}.svg"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class VictimDef:
    name: str                     # victim type
    img: str
    location: Tuple[int, int]
    area: str                     # e.g. 'area 2'

    @property
    def full_name(self) -> str:
        """Object name as placed in the world, e.g. 'critically injured girl in area 2'."""
        return f"{self.name} in {self.area}"


@dataclass
class ObstacleDef:
    name: str
    img: str
    location: Tuple[int, int]


@dataclass
class DropZoneDef:
    location: Tuple[int, int]
    height: int


@dataclass
class PropDef:
    """A purely decorative / scenery object (helicopter, ambulance, plant, ...)."""
    name: str
    img: str
    location: Tuple[int, int]
    size: float = 1.0
    traversable: bool = True


@dataclass
class RoomDef:
    id: int
    pos: Tuple[int, int]          # top-left corner (x, y), walls included
    width: int                    # including walls
    height: int                   # including walls
    door: Tuple[int, int]
    doormat: Tuple[int, int]
    enter_direction: str          # 'North' or 'South'
    victims: List[VictimDef] = field(default_factory=list)
    obstacles: List[ObstacleDef] = field(default_factory=list)


@dataclass
class WorldPreset:
    name: str
    grid_width: int
    grid_height: int
    rooms: List[RoomDef]
    drop_zone: DropZoneDef
    ghost_victims: List[Tuple[str, str]] = field(default_factory=list)
    props: List[PropDef] = field(default_factory=list)
    auto_roof: bool = True
    auto_streets: bool = True
    auto_signs: bool = True
    keyboard_sign: Optional[Tuple[int, int]] = None
    seed: Optional[int] = None


# ── Geometry helpers ───────────────────────────────────────────────────────────

def interior_cells(room: RoomDef) -> List[Tuple[int, int]]:
    """All cells strictly inside the room walls."""
    x0, y0 = room.pos
    return [(x, y)
            for x in range(x0 + 1, x0 + room.width - 1)
            for y in range(y0 + 1, y0 + room.height - 1)]


def compute_door_bottom(x0: int, y0: int, w: int, h: int):
    """Door on bottom wall; agent enters from the North (walks south)."""
    door_x = x0 + w // 2
    door_y = y0 + h - 1
    return (door_x, door_y), (door_x, door_y + 1), 'North'


def compute_door_top(x0: int, y0: int, w: int, h: int):
    """Door on top wall; agent enters from the South (walks north)."""
    door_x = x0 + w // 2
    door_y = y0
    return (door_x, door_y), (door_x, door_y - 1), 'South'


def rects_overlap(ax, ay, aw, ah, bx, by, bw, bh, margin: int = 0) -> bool:
    """Whether two axis-aligned rectangles overlap (with optional margin)."""
    return not (ax + aw + margin <= bx or bx + bw + margin <= ax or
                ay + ah + margin <= by or by + bh + margin <= ay)


def collect_all_victims(rooms: List[RoomDef]) -> List[Tuple[str, str]]:
    """Gather (name, img) for every victim across all rooms (order preserved)."""
    return [(v.name, v.img) for room in rooms for v in room.victims]


# ── Auto room layout ───────────────────────────────────────────────────────────

def pack_rooms(count: int, size: Tuple[int, int], columns: int,
               grid_w: int, grid_h: int,
               x_start: int = 1, y_start: int = 1,
               x_gap: int = 1, y_gap: int = 2) -> List[RoomDef]:
    """Arrange ``count`` identically-sized rooms in a regular grid.

    Doors face the horizontal mid-corridor: rooms in the top half get a
    bottom door, rooms in the bottom half get a top door. Raises ValueError if
    the requested layout does not fit in ``grid_w`` x ``grid_h``.
    """
    w, h = size
    rooms: List[RoomDef] = []
    for i in range(count):
        col = i % columns
        row = i // columns
        x = x_start + col * (w + x_gap)
        y = y_start + row * (h + y_gap)
        if x + w > grid_w - 1 or y + h > grid_h - 1:
            raise ValueError(
                f"Room {i + 1} at ({x},{y}) size {w}x{h} does not fit in a "
                f"{grid_w}x{grid_h} grid. Increase 'grid', reduce 'count'/'size', "
                f"or add more 'columns'."
            )
        if y + h / 2 < grid_h / 2:
            door, mat, enter = compute_door_bottom(x, y, w, h)
        else:
            door, mat, enter = compute_door_top(x, y, w, h)
        rooms.append(RoomDef(i + 1, (x, y), w, h, door, mat, enter))
    return rooms


# ── Decoration generators (auto, derived from room geometry) ───────────────────

def generate_roof_tiles(rooms: List[RoomDef]) -> List[Tuple[int, int]]:
    """Roof tiles covering each room's wall ring."""
    tiles = []
    for room in rooms:
        x0, y0 = room.pos
        for x in range(x0, x0 + room.width):
            for y in range(y0, y0 + room.height):
                on_wall = (x in (x0, x0 + room.width - 1) or
                           y in (y0, y0 + room.height - 1))
                if on_wall:
                    tiles.append((x, y))
    return tiles


def generate_street_tiles(rooms: List[RoomDef], drop_zone: DropZoneDef,
                          grid_w: int, grid_h: int) -> List[Tuple[int, int]]:
    """A mid-height corridor, vertical paths from each doormat to it, and a
    path from the corridor to the drop zone. Tiles inside rooms are removed."""
    tiles = set()
    corridor_y = grid_h // 2
    for x in range(1, grid_w - 1):
        tiles.add((x, corridor_y))
    for room in rooms:
        mat_x, mat_y = room.doormat
        for y in range(min(mat_y, corridor_y), max(mat_y, corridor_y) + 1):
            tiles.add((mat_x, y))
    dz_x, dz_y = drop_zone.location
    for y in range(min(corridor_y, dz_y), max(corridor_y, dz_y + drop_zone.height) + 1):
        tiles.add((dz_x - 1, y))

    room_cells = set()
    for room in rooms:
        x0, y0 = room.pos
        for x in range(x0, x0 + room.width):
            for y in range(y0, y0 + room.height):
                room_cells.add((x, y))
    return list(tiles - room_cells)


def generate_area_signs(rooms: List[RoomDef]) -> List[Tuple[Tuple[int, int], int]]:
    """(location, area_id) for a sign placed at each room's doormat."""
    return [(room.doormat, room.id) for room in rooms]


# ── Agent spawn locations ──────────────────────────────────────────────────────

def agent_spawn_locations(preset: WorldPreset, count: int) -> List[Tuple[int, int]]:
    """Spawn cells just left of the drop zone, stacked vertically and centred.

    Falls back to staying inside the grid bounds if the drop zone sits at an
    edge. Returns exactly ``count`` cells.
    """
    dz_x, dz_y = preset.drop_zone.location
    spawn_x = max(1, min(dz_x - 1, preset.grid_width - 2))
    centre = dz_y + preset.drop_zone.height // 2
    locs: List[Tuple[int, int]] = []
    for i in range(count):
        y = centre + i
        y = max(1, min(y, preset.grid_height - 2))
        locs.append((spawn_x, y))
    return locs
