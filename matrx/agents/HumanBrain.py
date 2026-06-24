import warnings
import copy
import numpy as np
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.door_actions import OpenDoorAction, CloseDoorAction
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.agents import HumanAgentBrain
from matrx.messages import Message
from matrx.actions.move_actions import MoveNorth, MoveNorthEast, MoveEast, MoveSouthEast, MoveSouth, MoveSouthWest, MoveWest, MoveNorthWest
from matrx.actions.CustomActions import RemoveObjectTogether, Idle, CarryObject, CarryObjectTogether, DropObjectTogether, Drop, RemoveObject

class HumanBrain(HumanAgentBrain):
    """An agent controlled by a human."""
    def __init__(self, memorize_for_ticks=None, fov_occlusion=False, max_carry_objects=3, grab_range=1, drop_range=1, door_range=1, remove_range=1, strength='normal', name='human'):
        super().__init__(memorize_for_ticks=memorize_for_ticks)
        self.__fov_occlusion = fov_occlusion
        if fov_occlusion:
            warnings.warn("FOV Occlusion is not yet fully implemented. "
                          "Setting fov_occlusion to True has no effect.")
        self.__max_carry_objects = max_carry_objects
        self.__remove_range = remove_range
        self.__grab_range = grab_range
        self.__drop_range = drop_range
        self.__door_range = door_range
        self.__remove_range = remove_range
        self.__strength = strength
        self.__name = name

    def _factory_initialise(self, agent_name, agent_id, action_set,
                            sense_capability, agent_properties,
                            rnd_seed, callback_is_action_possible,
                            customizable_properties=None, key_action_map=None):
        """Private MATRX function. The WorldFactory wires up the agent. Do NOT override.

        key_action_map maps pressed keys to actions.
        """
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.action_set = action_set

        self.rnd_seed = rnd_seed
        self._set_rnd_seed(seed=rnd_seed)

        self.sense_capability = sense_capability
        self.agent_properties = agent_properties

        # which properties this agent may change; the rest only change via actions
        self.keys_of_agent_writable_props = customizable_properties if customizable_properties is not None else list(agent_properties.keys())
        self.__callback_is_action_possible = callback_is_action_possible

        self.key_action_map = key_action_map if key_action_map is not None else {}
        self._init_state()

    def _get_action(self, state, agent_properties, agent_id, user_input):
        """Private MATRX function. Like the normal _get_action but acts on user input. Do NOT override."""
        self.agent_properties = agent_properties
        self.state.state_update(state.as_dict())
        self.state = self.filter_observations(self.state)
        usrinput = self.filter_user_input(user_input)  # keep only input bound to an action
        action, action_kwargs = self.decide_on_action(self.state, usrinput)
        self.previous_action = action
        return self.state, self.agent_properties, action, action_kwargs

    def decide_on_action(self, state, user_input):
        """Map the user's key press to an action and build its kwargs."""
        action = None
        action_kwargs = {}

        # no keys pressed -> do nothing
        if user_input is None or user_input == []:
            return None, {}

        # use the latest pressed key
        pressed_keys = user_input[-1]
        action = self.key_action_map[pressed_keys]

        if action == CarryObjectTogether.__name__:
            action_kwargs['grab_range'] = self.__grab_range
            action_kwargs['max_objects'] = self.__max_carry_objects
            action_kwargs['human_name'] = self.__name

            obj_id = self.__select_random_obj_in_range(state,
                                                  range_=self.__grab_range,
                                                  property_to_check="is_movable")
            action_kwargs['strength'] = self.__strength
            if obj_id and 'critical' in obj_id:
                action_kwargs['object_id'] = obj_id
            if obj_id and 'mild' in obj_id:
                action_kwargs['object_id'] = obj_id

        elif action == DropObjectTogether.__name__:
            action_kwargs['strength'] = self.__strength
            action_kwargs['drop_range'] = self.__drop_range
            action_kwargs['human_name'] = self.__name
            pass

        if action == CarryObject.__name__:
            action_kwargs['grab_range'] = self.__grab_range
            action_kwargs['max_objects'] = self.__max_carry_objects
            action_kwargs['object_id'] = None
            action_kwargs['strength'] = self.__strength
            action_kwargs['human_name'] = self.__name

            obj_id = \
                self.__select_random_obj_in_range(state,
                                                  range_=self.__grab_range,
                                                  property_to_check="is_movable")
            if obj_id and self.__strength!='weak':
                action_kwargs['object_id'] = obj_id
            action_kwargs['action_type'] = 'alone'

        elif action == Drop.__name__:
            action_kwargs['strength'] = self.__strength
            action_kwargs['drop_range'] = self.__drop_range
            action_kwargs['human_name'] = self.__name
            pass

        elif action == RemoveObjectTogether.__name__:
            action_kwargs['remove_range'] = self.__remove_range
            action_kwargs['human_name'] = self.__name

            obj_id = \
                self.__select_random_obj_in_range(state,
                                                  range_=self.__remove_range,
                                                  property_to_check="is_movable")
            action_kwargs['object_id'] = obj_id
            if obj_id and 'stone' in obj_id:
                action_kwargs['action_duration'] = 25
            if obj_id and 'rock' in obj_id:
                action_kwargs['action_duration'] = 50

        elif action == RemoveObject.__name__:
            action_kwargs['remove_range'] = self.__remove_range
            action_kwargs['human_name'] = self.__name

            obj_id = \
                self.__select_random_obj_in_range(state,
                                                  range_=self.__remove_range,
                                                  property_to_check="is_movable")
            if obj_id and 'stone' in obj_id and self.__strength!='weak':
                action_kwargs['object_id'] = obj_id
                action_kwargs['action_duration'] = 200

        elif action == OpenDoorAction.__name__ \
                or action == CloseDoorAction.__name__:
            action_kwargs['door_range'] = self.__door_range
            action_kwargs['object_id'] = None

            objects = list(state.keys())
            doors = [obj for obj in objects if 'is_open' in state[obj]]

            # doors within range
            doors_in_range = []
            for object_id in doors:
                dist = int(np.ceil(np.linalg.norm(
                    np.array(state[object_id]['location']) - np.array(
                        state[self.agent_id]['location']))))
                if dist <= action_kwargs['door_range']:
                    doors_in_range.append(object_id)

            if len(doors_in_range) > 0:
                action_kwargs['object_id'] = \
                    self.rnd_gen.choice(doors_in_range)

        elif action in [MoveNorth.__name__, MoveNorthEast.__name__, MoveEast.__name__, MoveSouthEast.__name__, MoveSouth.__name__, MoveSouthWest.__name__, MoveWest.__name__, MoveNorthWest.__name__]:
            water_locs = []
            if state[{"name": "water"}]:
                for water in state[{"name": "water"}]:
                    if water['location'] not in water_locs:
                        water_locs.append(water['location'])
            if state[{"name": self.__name}]['location'] in water_locs and state[{"name": self.__name}]['location'] not in [(3,5),(9,5),(15,5),(21,5),(3,6),(9,6),(15,6),(3,17),(9,17),(15,17),(3,18),(9,18),(15,18),(21,18)]:
                action == Idle.__name__
                action_kwargs['duration_in_ticks'] = 5

        return action, action_kwargs

    def filter_observations(self, state):
        """Observe phase: filter the state to what the agent should see. Override to hide objects."""
        return state

    def filter_user_input(self, user_input):
        """Keep only key presses that map to an action."""
        # clear received messages
        for message in list(self.received_messages):
            self.received_messages.remove(message)

        if user_input is None:
            return []
        possible_key_presses = list(self.key_action_map.keys())
        return list(set(possible_key_presses) & set(user_input))

    def create_context_menu_for_self(self, clicked_object_id, click_location,
                                     self_selected):
        """Context menu options when the controlling user right-clicks an object/location."""
        print("Context menu self with self selected:", self_selected)

        context_menu = []

        for action in self.action_set:

            context_menu.append({
                "OptionText": f"Do action: {action}",
                "Message": Message(content=action, from_id=self.agent_id,
                                   to_id=self.agent_id)
            })

        return context_menu

    def create_context_menu_for_other(self, agent_id_who_clicked,
                                      clicked_object_id, click_location):
        """Context menu options when another human agent right-clicks while controlling this agent."""
        print("Context menu other")
        context_menu = []

        # one option per action
        for action in self.action_set:
            context_menu.append({
                "OptionText": f"Do action: {action}",
                "Message": Message(content=action, from_id=clicked_object_id,
                                   to_id=self.agent_id)
            })
        return context_menu

    def __select_random_obj_in_range(self, state, range_,
                                     property_to_check=None):
        # all perceived objects minus world, self and other agents
        object_ids = list(state.keys())
        object_ids.remove("World")
        object_ids.remove(self.agent_id)
        object_ids = [obj_id for obj_id in object_ids if "AgentBrain" not in
                      state[obj_id]['class_inheritance'] and
                      "AgentBody" not in state[obj_id]['class_inheritance']]

        # objects in range (optionally matching property_to_check)
        object_in_range = []
        for object_id in object_ids:
            dist = int(np.ceil(np.linalg.norm(
                np.array(state[object_id]['location'])
                - np.array(state[self.agent_id]['location']))))
            if dist <= range_:
                if property_to_check is not None:
                    if property_to_check in state[object_id] \
                            and state[object_id][property_to_check]:
                        object_in_range.append(object_id)
                else:
                    object_in_range.append(object_id)

        if object_in_range:
            object_id = self.rnd_gen.choice(object_in_range)
        else:
            object_id = None

        return object_id