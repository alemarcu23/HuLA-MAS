"""
YAML world builder for the HuLA-MAS.
"""

import os
import numpy as np
from datetime import datetime

from matrx import WorldBuilder
from matrx.agents import SenseCapability
from matrx.grid_world import AgentBody
from matrx.objects import EnvObject
from matrx.goals import WorldGoal
from matrx.agents.HumanBrain import HumanBrain
from matrx.logger.ActionLogger import ActionLogger

from matrx.grid_world_creation.yaml_loader import load_preset
from matrx.grid_world_creation.world_model import (
    generate_roof_tiles, generate_street_tiles, generate_area_signs,
    agent_spawn_locations,
)
from matrx.grid_world_creation.constants import (
    DEFAULT_RANDOM_SEED, VERBOSE, TICK_DURATION, KEY_ACTION_MAP,
    WALL_COLOR, DROP_OFF_COLOR, OBJECT_SIZE,
    AGENT_SENSE_RANGE, OBJECT_SENSE_RANGE, OTHER_SENSE_RANGE,
    FOV_OCCLUSION, MAX_SIGN_AREA,
)


def _pad(values, n, default):
    """Return exactly ``n`` entries, repeating the last value (or ``default``)."""
    values = list(values) if values else [default]
    while len(values) < n:
        values.append(values[-1])
    return values[:n]


# ── Map resolution ──────────────────────────────────────────────────────────────

def resolve_map_path(map_name, folder):
    """Resolve a map name/path to a YAML file.

    Accepts a bare name ('official' -> <repo>/maps/official.yaml), or a path to
    a .yaml/.yml file (absolute, or relative to the run folder).
    """
    if map_name.endswith('.yaml') or map_name.endswith('.yml'):
        if os.path.isabs(map_name) and os.path.exists(map_name):
            return map_name
        cand = os.path.join(folder, map_name)
        return cand if os.path.exists(cand) else map_name

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for base in (repo_root, folder):
        cand = os.path.join(base, 'maps', f'{map_name}.yaml')
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Map '{map_name}' not found. Expected {os.path.join(repo_root, 'maps', map_name + '.yaml')}."
    )


# ── World population from a preset ───────────────────────────────────────────────

def _add_rooms(builder, preset):
    for room in preset.rooms:
        builder.add_room(
            top_left_location=room.pos, width=room.width, height=room.height,
            name=f'area {room.id}', door_locations=[room.door], doors_open=True,
            wall_visualize_colour=WALL_COLOR, with_area_tiles=True,
            area_visualize_colour='#0008ff', area_visualize_opacity=0.0,
            door_open_colour='#9a9083',
            area_custom_properties={'doormat': room.doormat})


def _add_obstacles(builder, preset):
    for room in preset.rooms:
        for obs in room.obstacles:
            builder.add_object(tuple(obs.location), obs.name, ObstacleObject,
                               visualize_shape='img', img_name=obs.img)


def _add_victims(builder, preset):
    for room in preset.rooms:
        for v in room.victims:
            builder.add_object(tuple(v.location), v.full_name,
                               callable_class=CollectableBlock,
                               visualize_shape='img', img_name=v.img)


def _add_ghost_blocks(builder, preset):
    dz_x, dz_y = preset.drop_zone.location
    for i, (_name, img) in enumerate(preset.ghost_victims):
        builder.add_object((dz_x, dz_y + i), name="Collect Block",
                           callable_class=GhostBlock, visualize_shape='img',
                           img_name=img, drop_zone_nr=0)


def _add_decorations(builder, preset):
    door_cells = {room.door for room in preset.rooms}
    if preset.auto_roof:
        for loc in generate_roof_tiles(preset.rooms):
            if loc in door_cells:
                continue  # don't roof over a doorway
            builder.add_object(loc, 'roof', EnvObject, is_traversable=True,
                               is_movable=False, visualize_shape='img',
                               img_name="/images/roof-final5.svg")
    if preset.auto_streets:
        for loc in generate_street_tiles(preset.rooms, preset.drop_zone,
                                         preset.grid_width, preset.grid_height):
            builder.add_object(loc, 'street', EnvObject, is_traversable=True,
                               is_movable=False, visualize_shape='img',
                               img_name="/images/paving-final20.svg", visualize_size=1)
    if preset.auto_signs:
        for loc, area_id in generate_area_signs(preset.rooms):
            if not isinstance(area_id, int) or not (1 <= area_id <= MAX_SIGN_AREA):
                continue  # sign images only exist for areas 1-14
            builder.add_object(location=list(loc), is_traversable=True, is_movable=False,
                               name=f"area {area_id:02d} sign",
                               img_name=f"/images/sign{area_id:02d}.svg",
                               visualize_depth=110, visualize_size=0.55)
    for prop in preset.props:
        builder.add_object(tuple(prop.location), prop.name, EnvObject,
                           is_traversable=prop.traversable, is_movable=False,
                           visualize_shape='img', img_name=prop.img,
                           visualize_size=prop.size)
    if preset.keyboard_sign:
        builder.add_object(location=list(preset.keyboard_sign), is_traversable=True,
                           name="keyboard sign", img_name="/images/keyboard-final.svg",
                           visualize_depth=110, visualize_size=15)


def _add_drop_off_zone(builder, preset):
    dz_x, dz_y = preset.drop_zone.location
    builder.add_area((dz_x, dz_y), width=1, height=preset.drop_zone.height,
                     name="Drop off 0", visualize_opacity=0.5,
                     visualize_colour=DROP_OFF_COLOR, drop_zone_nr=0,
                     is_drop_zone=True, is_goal_block=False, is_collectable=False)


def _preset_to_env_info(preset):
    """Build :class:`EnvironmentInformation` (areas, drop zone, grid, victim
    count) from a :class:`WorldPreset`, for the LLM agents' spatial reasoning."""
    from matrx.grid_world_creation.environment_info import EnvironmentInformation
    areas_raw = [
        {"id": r.id, "pos": tuple(r.pos), "w": r.width, "h": r.height,
         "door": tuple(r.door) if r.door else None,
         "mat": tuple(r.doormat) if r.doormat else None,
         "enter": r.enter_direction}
        for r in preset.rooms
    ]
    num_victims = sum(len(r.victims) for r in preset.rooms)
    return EnvironmentInformation.build(
        areas_raw=areas_raw,
        drop_zone=tuple(preset.drop_zone.location),
        drop_zone_height=preset.drop_zone.height,
        grid_size=(preset.grid_width, preset.grid_height),
        num_victims=num_victims,
    )


def _new_world_from_preset(preset, task_type, condition, seed=None):
    """Create a MATRX builder and populate it (rooms, victims, decorations, drop
    zone, goal) from ``preset`` — everything except the agents."""
    # Reproducibility: explicit seed override > map's own seed > module default.
    np.random.seed(seed if seed is not None
                   else (preset.seed if preset.seed is not None else DEFAULT_RANDOM_SEED))

    dz = preset.drop_zone
    total_victims = sum(len(r.victims) for r in preset.rooms)
    goal = CollectionGoal(max_nr_ticks=np.inf,
                          drop_zone_location=tuple(dz.location),
                          drop_zone_height=dz.height,
                          total_victims=total_victims)
    builder = WorldBuilder(shape=[preset.grid_width, preset.grid_height],
                           tick_duration=TICK_DURATION, run_matrx_api=True,
                           run_matrx_visualizer=False, verbose=VERBOSE,
                           simulation_goal=goal, visualization_bg_clr='#9a9083')

    # World bounds
    builder.add_room(top_left_location=(0, 0), width=preset.grid_width,
                     height=preset.grid_height, name="world_bounds",
                     wall_visualize_colour=DROP_OFF_COLOR)

    # Action logging during the official task
    if task_type == "official":
        current_exp_folder = datetime.now().strftime(
            "exp_" + condition + "_at_time_%Hh-%Mm-%Ss_date_%dd-%mm-%Yy")
        logger_save_folder = os.path.join("logs", current_exp_folder)
        builder.add_logger(ActionLogger, log_strategy=1,
                           save_path=logger_save_folder, file_name_prefix="actions_")

    _add_rooms(builder, preset)
    _add_obstacles(builder, preset)
    _add_victims(builder, preset)
    _add_ghost_blocks(builder, preset)
    _add_decorations(builder, preset)
    _add_drop_off_zone(builder, preset)
    return builder


# ── Agents ────────────────────────────────────────────────────────────────────

def add_agents(builder, condition, name, folder, preset, env_info, *,
               num_agents, agent_model, api_base, agent_presets, agent_roles,
               capability_knowledge, reasoning_strategies, planning_strategies,
               presets=None, role_goals=None, game_rules='', include_human=False,
               joint_action_ask_trigger='auto_bridge'):
    """Add ``num_agents`` LLM-driven agent and, if ``include_human``,
    one keyboard-controlled human teammate named ``name``.

    All LLM agents share a single :class:`SharedMemory` blackboard (rendezvous,
    rescued victims, registered-agent roster) and one read-only
    :class:`EnvironmentInformation`. Per-agent lists are padded to ``num_agents``.

    ``presets`` (name->capability levels), ``role_goals`` (role->directive), and
    ``game_rules`` (task description) all come from config.yaml.
    """
    from matrx.agents.llm.LlmAgent import LlmAgent
    from matrx.agents.llm.modules.memory_module import SharedMemory
    from matrx.agents.llm.modules.profile_module import resolve_capabilities
    from matrx.agents.capabilities.capability import VisionCapability

    presets = presets or {}
    role_goals = role_goals or {}
    agent_presets = _pad(agent_presets, num_agents, 'generalist')
    agent_roles = _pad(agent_roles, num_agents, 'generalist')
    reasoning_strategies = _pad(reasoning_strategies, num_agents, 'io')
    planning_strategies = _pad(planning_strategies, num_agents, 'io')

    shared_memory = SharedMemory()  # one blackboard shared across the whole team
    spawns = agent_spawn_locations(preset, num_agents + (1 if include_human else 0))

    for i in range(num_agents):
        caps = resolve_capabilities(agent_presets[i], presets)
        # Per-agent perception range scales with the capability preset's vision.
        vision_range = VisionCapability.RADIUS.get(caps.get('vision'), 2)
        sense_capability_agent = SenseCapability({
            AgentBody: AGENT_SENSE_RANGE, CollectableBlock: vision_range,
            None: OTHER_SENSE_RANGE, ObstacleObject: vision_range})
        brain = LlmAgent(
            slowdown=8, condition=condition, name=name, folder=folder,
            llm_model=agent_model, strategy=reasoning_strategies[i],
            include_human=include_human, shared_memory=shared_memory,
            planning_strategy=planning_strategies[i],
            api_base=api_base, capabilities=caps,
            capability_knowledge=capability_knowledge,
            env_info=env_info,
            initial_role=agent_roles[i],
            role_goal=role_goals.get(agent_roles[i], ''),
            game_rules=game_rules,
            joint_action_ask_trigger=joint_action_ask_trigger,
        )
        print(f"[WorldBuilder] RescueBot{i} (LlmAgent, caps={caps}, "
              f"role={agent_roles[i]})")
        # Unique names that all lowercase to a 'rescuebot' prefix: the first is
        # exactly 'rescuebot' (the goal looks it up by that id) and every agent
        # is detected as a teammate via the 'rescuebot' id prefix.
        agent_name = "RescueBot" if i == 0 else f"RescueBot{i}"
        builder.add_agent(spawns[i], brain, team="Team 0", name=agent_name,
                          customizable_properties=['score'], score=0,
                          # Store the resolved capabilities on the agent BODY so the
                          # environment actions (CustomActions / RemoveObject) can
                          # enforce per-agent strength/medical constraints at runtime.
                          capabilities=caps,
                          sense_capability=sense_capability_agent, is_traversable=True,
                          img_name="/images/robot-final4.svg")

    if include_human:
        max_carry = np.inf if condition == 'strong' else 1
        obstacle_sense = 10 if condition == 'strong' else 1
        brain = HumanBrain(max_carry_objects=max_carry, grab_range=1, drop_range=0,
                           remove_range=1, fov_occlusion=FOV_OCCLUSION,
                           strength=condition, name=name)
        human_sense = SenseCapability(
            {AgentBody: AGENT_SENSE_RANGE, CollectableBlock: OBJECT_SENSE_RANGE,
             None: OTHER_SENSE_RANGE, ObstacleObject: obstacle_sense})
        builder.add_human_agent(spawns[num_agents], brain, team="Team 0", name=name,
                                key_action_map=KEY_ACTION_MAP,
                                sense_capability=human_sense, is_traversable=True,
                                img_name="/images/rescue-man-final3.svg",
                                visualize_when_busy=True)


# ── Builder entry point ──────────────────────────────────────────────────────────

def create_builder(task_type, condition, name, folder, map_name='official',
                   num_agents=2, agent_model='qwen2.5:3b', api_base=None,
                   agent_presets=None, agent_roles=None,
                   capability_knowledge='informed',
                   reasoning_strategies=None, planning_strategies=None,
                   presets=None, role_goals=None, game_rules='',
                   world_seed=None,
                   include_human=False, log_dir=None,
                   joint_action_ask_trigger='auto_bridge'):
    """Build a MATRX WorldBuilder from a YAML map with LLM agents.

    Parameters
    ----------
    map_name : str
        A bare map name (resolved to ``maps/<name>.yaml``) or a path to a YAML file.
    num_agents : int
        Number of LLM Agents.
    include_human : bool
        Add one keyboard-controlled human teammate named ``name``.

    Per-agent lists (``agent_presets``, ``agent_roles``, ``*_strategies``) are
    padded to ``num_agents`` if shorter, so callers may pass partial lists.
    """
    preset = load_preset(resolve_map_path(map_name, folder))
    builder = _new_world_from_preset(preset, task_type, condition, seed=world_seed)
    env_info = _preset_to_env_info(preset)
    add_agents(builder, condition, name, folder, preset, env_info,
               num_agents=num_agents, agent_model=agent_model, api_base=api_base,
               agent_presets=agent_presets, agent_roles=agent_roles,
               capability_knowledge=capability_knowledge,
               reasoning_strategies=reasoning_strategies,
               planning_strategies=planning_strategies,
               presets=presets, role_goals=role_goals, game_rules=game_rules,
               include_human=include_human,
               joint_action_ask_trigger=joint_action_ask_trigger)
    return builder


# ── Object / goal classes ─────────────────────────────────────────────────────────

class CollectableBlock(EnvObject):
    '''Objects (victims) that can be collected by agents.'''
    def __init__(self, location, name, visualize_shape, img_name, **kwargs):
        super().__init__(location, name, is_traversable=True, is_movable=True,
                         visualize_shape=visualize_shape, img_name=img_name,
                         visualize_size=OBJECT_SIZE, class_callable=CollectableBlock,
                         is_drop_zone=False, is_goal_block=False, is_collectable=True,
                         **kwargs)


class ObstacleObject(EnvObject):
    '''Obstacles that can be removed by agents.'''
    def __init__(self, location, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=False, is_movable=True,
                         visualize_shape=visualize_shape, img_name=img_name,
                         visualize_size=1.25, class_callable=ObstacleObject,
                         is_drop_zone=False, is_goal_block=False, is_collectable=False)


class GhostBlock(EnvObject):
    '''Objects on the drop zone that cannot be carried by agents.'''
    def __init__(self, location, drop_zone_nr, name, visualize_shape, img_name):
        super().__init__(location, name, is_traversable=True, is_movable=False,
                         visualize_shape=visualize_shape, img_name=img_name,
                         visualize_size=OBJECT_SIZE, class_callable=GhostBlock,
                         visualize_depth=110, drop_zone_nr=drop_zone_nr, visualize_opacity=0.5,
                         is_drop_zone=False, is_goal_block=True, is_collectable=False)


class CollectionGoal(WorldGoal):
    '''Determines when the simulation stops and tracks the team score.

    Only *injured* victims (critical = +6, mild = +3) score; healthy victims are
    ignored. A rescued victim is removed from the grid and its score is broadcast
    to every ``rescuebot*`` agent.
    '''
    def __init__(self, max_nr_ticks, drop_zone_location=(23, 8),
                 drop_zone_height=8, total_victims=8):
        super().__init__()
        self.max_nr_ticks = max_nr_ticks
        self._dz_x = drop_zone_location[0]
        self._dz_y_min = drop_zone_location[1]
        self._dz_y_max = drop_zone_location[1] + drop_zone_height - 1
        self._total_victims = total_victims
        self.__drop_off = {}
        self.__drop_off_zone = {}
        self.__progress = 0
        self.__score = 0

    def score(self, grid_world):
        return self.__score

    def goal_reached(self, grid_world):
        if grid_world.current_nr_ticks >= self.max_nr_ticks:
            return True
        return self.isVictimPlaced(grid_world)

    def isVictimPlaced(self, grid_world):
        '''@return true if all victims have been rescued'''
        if self.__drop_off == {}:
            self.__find_drop_off_locations(grid_world)
        is_satisfied, progress = self.__check_completion(grid_world)
        self.__progress = progress / sum([len(goal_vics) for goal_vics in self.__drop_off.values()])
        return is_satisfied

    def progress(self, grid_world):
        if self.__drop_off == {}:
            self.__find_drop_off_locations(grid_world)
        is_satisfied, progress = self.__check_completion(grid_world)
        self.__progress = progress / sum([len(goal_vics) for goal_vics in self.__drop_off.values()])
        return self.__progress

    def __find_drop_off_locations(self, grid_world):
        goal_vics = {}
        all_objs = grid_world.environment_objects
        for obj_id, obj in all_objs.items():  # go through all objects
            if "drop_zone_nr" in obj.properties.keys():  # part of a drop zone?
                zone_nr = obj.properties["drop_zone_nr"]
                if obj.properties["is_goal_block"]:  # a ghostly goal victim?
                    if zone_nr in goal_vics.keys():
                        goal_vics[zone_nr].append(obj)
                    else:
                        goal_vics[zone_nr] = [obj]

        self.__drop_off_zone = {}
        self.__drop_off = {}
        for zone_nr in goal_vics.keys():
            self.__drop_off_zone[zone_nr] = {}
            self.__drop_off[zone_nr] = {}
            vics = goal_vics[zone_nr].copy()
            max_rank = len(vics)
            # Find the 'bottom' location
            bottom_loc = (-np.inf, -np.inf)
            for vic in vics:
                if vic.location[1] > bottom_loc[1]:
                    bottom_loc = vic.location
            # Loop through victim lists and add them to their appropriate ranks
            for rank in range(max_rank):
                loc = (bottom_loc[0], bottom_loc[1] - rank)
                for vic in vics:
                    if vic.location == loc:
                        self.__drop_off_zone[zone_nr][rank] = [loc, vic.properties['img_name'][8:-4], None]
                        for i in self.__drop_off_zone.keys():
                            self.__drop_off[i] = {}
                            vals = list(self.__drop_off_zone[i].values())
                            vals.reverse()
                            for j in range(len(self.__drop_off_zone[i].keys())):
                                self.__drop_off[i][j] = vals[j]

    def __check_completion(self, grid_world):
        curr_tick = grid_world.current_nr_ticks

        # Track victims already scored to avoid double-counting.
        if not hasattr(self, '_scored_victims'):
            self._scored_victims = set()
        # Track last known carrier per victim so we can report who rescued it
        # even after the agent has dropped it and walked away.
        if not hasattr(self, '_victim_last_carrier'):
            self._victim_last_carrier = {}

        all_objs = grid_world.environment_objects
        for oid, o in all_objs.items():
            if getattr(o, 'carried_by', None):
                vk = f"{oid}_{o.properties.get('img_name', '')}"
                self._victim_last_carrier[vk] = o.carried_by[0]

        # Find all collectable victims sitting in the drop zone column.
        for obj_id, obj in all_objs.items():
            if not obj.properties.get("is_collectable", False):
                continue
            obj_loc = obj.location
            if obj_loc[0] == self._dz_x and self._dz_y_min <= obj_loc[1] <= self._dz_y_max:
                victim_key = f"{obj_id}_{obj.properties['img_name']}"
                if victim_key in self._scored_victims:
                    continue

                img_name = obj.properties['img_name'][8:-4]
                # Only injured victims score.
                if 'healthy' in img_name.lower():
                    continue

                rescuer = (obj.carried_by[0] if getattr(obj, 'carried_by', None)
                           else self._victim_last_carrier.get(victim_key, 'unknown'))
                if 'critical' in img_name.lower():
                    self.__score += 6
                    print(f"VICTIM '{img_name}' SAVED by {rescuer}! +6 pts (Total: {self.__score})", flush=True)
                elif 'mild' in img_name.lower():
                    self.__score += 3
                    print(f"VICTIM '{img_name}' SAVED by {rescuer}! +3 pts (Total: {self.__score})", flush=True)

                self._scored_victims.add(victim_key)

                # Mark the matching goal block as complete.
                for zone_nr, goal_vics in self.__drop_off.items():
                    for rank, vic_data in goal_vics.items():
                        if vic_data[1] == img_name and vic_data[2] is None:
                            self.__drop_off[zone_nr][rank][2] = curr_tick
                            break

                # Remove the rescued victim from the world (hide from UI).
                grid_world.remove_from_grid(obj_id, remove_from_carrier=True)

        # Check if all goal victims are collected.
        is_satisfied = True
        progress = 0
        for zone_nr, goal_vics in self.__drop_off.items():
            ticks = [goal_vics[r][2] for r in range(len(goal_vics))]
            progress += sum(1 for t in ticks if t is not None)
            if None in ticks:
                is_satisfied = False

        # Broadcast the team score to every rescue agent.
        for agent_id, agent_body in grid_world.registered_agents.items():
            if agent_id.startswith('rescuebot'):
                agent_body.change_property('score', self.__score)

        return is_satisfied, progress
