"""Obstacle navigation — the fix for agents getting stuck *beside* the obstacle
they need joint help to remove. Verified against the REAL MATRX A* pathfinder.

Root cause: AStarPlanner.plan only succeeds if it lands ON the goal tile and
skips intraversable neighbours, so routing to an obstacle's own (intraversable)
tile returns [start] (no move). The fix routes to a reachable FREE tile adjacent
to the target instead.
"""
import numpy as np

from matrx.agents.agent_utils.navigator import AStarPlanner

ROCK = (3, 4)


def _state_with_walled_rock():
    """A rock fills the only gap in a wall column at x=3 (left side unreachable);
    both agents live on the right. Mirrors a rock blocking a doorway."""
    W = H = 8
    state = {'World': {'grid_shape': (W, H)}}
    for y in range(H):
        if (3, y) != ROCK:
            state[f'wall_3_{y}'] = {'location': (3, y), 'is_traversable': False}
    state['rock_5'] = {'location': ROCK, 'is_traversable': False, 'is_collectable': False}
    state['RescueBot'] = {'location': (6, 4), 'isAgent': True, 'is_traversable': False}
    state['RescueBot1'] = {'location': (6, 5), 'isAgent': True, 'is_traversable': False}
    occ = np.zeros((W, H), dtype=int)
    for oid, od in state.items():
        if oid != 'World' and od.get('is_traversable') is False:
            x, y = od['location']
            occ[x][y] = 1
    return state, occ


# ── _nav_goal / _blocked_tiles ────────────────────────────────────────────────
def test_nav_goal_walks_onto_a_traversable_target(make_agent):
    a = make_agent('RescueBot')
    a.state_for_navigation = {'man_1': {'location': (10, 10), 'is_traversable': True,
                                        'is_collectable': True}}
    assert a._nav_goal((10, 10)) == (10, 10)


def test_nav_goal_avoids_obstacle_tile(make_agent):
    a = make_agent('RescueBot').at((8, 4))
    state, _ = _state_with_walled_rock()
    a.state_for_navigation = state
    goal = a._nav_goal(ROCK)
    assert goal != ROCK
    assert max(abs(goal[0] - ROCK[0]), abs(goal[1] - ROCK[1])) == 1  # adjacent
    assert goal == (4, 4)  # nearest free tile to the agent


def test_blocked_tiles_includes_obstacles_walls_and_agents(make_agent):
    a = make_agent('RescueBot')
    state, _ = _state_with_walled_rock()
    a.state_for_navigation = state
    blk = a._blocked_tiles()
    assert ROCK in blk           # the rock
    assert (3, 3) in blk         # a wall
    assert (6, 5) in blk         # the other agent


def test_drive_to_target_issues_a_move_not_stuck_idle(make_agent):
    a = make_agent('RescueBot').at((6, 4))
    state, _ = _state_with_walled_rock()
    a.set_world(obstacles=[{'object_id': 'rock_5', 'location': ROCK}], nav=state)
    a._joint = {'kind': 'remove', 'target_id': 'rock_5', 'target_loc': ROCK,
                'partner_id': 'RescueBot1', 'partner_is_human': False,
                'role': 'firer', 'phase': 'go', 'started_tick': 0}
    act = a._joint_infra(None)
    assert act is not None and act[0] == 'MoveNorth'           # moving, not idle
    assert a._nav_target != ROCK                               # not the rock tile
    assert max(abs(a._nav_target[0] - ROCK[0]), abs(a._nav_target[1] - ROCK[1])) == 1


# ── against the REAL A* pathfinder ────────────────────────────────────────────
def test_real_astar_cannot_path_onto_obstacle_tile():
    _, occ = _state_with_walled_rock()
    planner = AStarPlanner(['MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'], {})
    start = (6, 4)
    # THE BUG: routing to the rock's own intraversable tile → "stay put".
    assert planner.plan(start, ROCK, occ) == [start]


def test_real_astar_reaches_adjacent_goal_from_nav_goal(make_agent):
    state, occ = _state_with_walled_rock()
    a = make_agent('RescueBot').at((6, 4))
    a.state_for_navigation = state
    goal = a._nav_goal(ROCK)
    planner = AStarPlanner(['MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'], {})
    path = planner.plan((6, 4), goal, occ)
    assert goal != ROCK
    assert path and path != [(6, 4)]                              # a real, non-empty path
    assert max(abs(path[-1][0] - ROCK[0]), abs(path[-1][1] - ROCK[1])) <= 1  # ends adjacent


def test_real_astar_full_drive_reaches_adjacency(make_agent):
    state, occ = _state_with_walled_rock()
    a = make_agent('RescueBot')
    a.state_for_navigation = state
    planner = AStarPlanner(['MoveNorth', 'MoveEast', 'MoveSouth', 'MoveWest'], {})
    cur = (6, 4)
    for _ in range(30):
        if max(abs(cur[0] - ROCK[0]), abs(cur[1] - ROCK[1])) <= 1:
            break
        a.at(cur)
        step = planner.plan(cur, a._nav_goal(ROCK), occ)
        if not step or step == [cur]:
            break
        cur = step[0]
    assert max(abs(cur[0] - ROCK[0]), abs(cur[1] - ROCK[1])) <= 1  # arrived, not stuck
