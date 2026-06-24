"""
World-building constants for HuLA-MAS.
"""

import numpy as np

from matrx.actions.move_actions import MoveNorth, MoveEast, MoveSouth, MoveWest
from matrx.actions.CustomActions import (RemoveObjectTogether, CarryObject, Drop,
                                         CarryObjectTogether, DropObjectTogether)
from matrx.actions.object_actions import RemoveObject

# ── Simulation speed ────────────────────────────────────────────────────────────
# Fallback seed used when neither config.yaml (world.seed) nor the map YAML
# specifies one. Prefer setting world.seed in config.yaml for reproducibility.
DEFAULT_RANDOM_SEED = 1

VERBOSE = False

# Seconds per tick. 0.1 = 10 ticks/second. Leave at 0.1 for evaluations.
TICK_DURATION = 0.1

# ── Visuals ─────────────────────────────────────────────────────────────────────
WALL_COLOR = "#8a8a8a"
DROP_OFF_COLOR = "#1F262A"
OBJECT_SIZE = 0.9

# ── Perception ranges ───────────────────────────────────────────────────────────
# These are calibrated for the official task — change with caution.
AGENT_SENSE_RANGE = 2    # range with which agents detect other agents
OBJECT_SENSE_RANGE = 1   # range with which agents detect blocks
OTHER_SENSE_RANGE = np.inf  # range with which agents detect walls/doors

FOV_OCCLUSION = True

# ── Map decoration ──────────────────────────────────────────────────────────────
# Sign images only exist for areas 1–14.
MAX_SIGN_AREA = 14

# ── Keyboard controls (human agent) ─────────────────────────────────────────────
KEY_ACTION_MAP = {
    'ArrowUp': MoveNorth.__name__,
    'ArrowRight': MoveEast.__name__,
    'ArrowDown': MoveSouth.__name__,
    'ArrowLeft': MoveWest.__name__,
    'q': CarryObject.__name__,
    'w': Drop.__name__,
    'd': RemoveObjectTogether.__name__,
    'a': CarryObjectTogether.__name__,
    's': DropObjectTogether.__name__,
    'e': RemoveObject.__name__,
}
