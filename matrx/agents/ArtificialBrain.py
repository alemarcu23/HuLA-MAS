import copy
import warnings
import numpy as np
from abc import  ABC, abstractmethod
from matrx.actions.CustomActions import RemoveObjectTogether
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_brain import AgentBrain
from matrx.agents.agent_brain import AgentBrain
from matrx.actions import GrabObject, RemoveObject, OpenDoorAction, CloseDoorAction
from matrx.agents.agent_utils.state import State
from matrx.messages import Message


class ArtificialAgentBrain(AgentBrain):
    """MATRX AgentBrain copy with a tweak in _set_messages so we can identify message senders."""

    def __init__(self,memorize_for_ticks=None):
        self.previous_action = None
        self.previous_action_result = None

        self.messages_to_send = []
        self.received_messages = []
        self.received_messages_content = []

        # filled by the WorldFactory during _factory_initialise()
        self.agent_id = None
        self.agent_name = None
        self.action_set = None
        self.sense_capability = None
        self.rnd_gen = None
        self.rnd_seed = None
        self.agent_properties = {}
        self.keys_of_agent_writable_props = []
        self.__memorize_for_ticks = memorize_for_ticks

        self._state = None

    def initialize(self):
        """Called by the world on start. Reset per-run state here."""
        self.previous_action = None
        self.previous_action_result = None
        self.messages_to_send = []
        self.received_messages = []
        self.received_messages_content = []
        self._init_state()

    def filter_observations(self, state):
        """Filter the world state before deciding. Override to hide objects/properties."""
        return state

    def decide_on_action(self, state):
        """Pick an action and its kwargs. Default brain picks a random action."""
        if self.action_set:
            action = self.rnd_gen.choice(self.action_set)
        else:
            action = None

        action_kwargs = {}

        if action == RemoveObject.__name__:
            action_kwargs['object_id'] = None

            objects = list(state.keys())
            objects.remove(self.agent_properties["obj_id"])
            objects = [obj for obj in objects if 'agent' not in obj]  # don't remove agents
            if objects:
                object_id = self.rnd_gen.choice(objects)
                action_kwargs['object_id'] = object_id
                # range just big enough to reach the object
                remove_range = int(np.ceil(np.linalg.norm(
                    np.array(state[object_id]['location']) - np.array(
                        state[self.agent_properties["obj_id"]]['location']))))
                remove_range = max(remove_range, 0)
                action_kwargs['remove_range'] = remove_range
            else:
                action_kwargs['object_id'] = None
                action_kwargs['remove_range'] = 0

        elif action == GrabObject.__name__:
            grab_range = 1
            max_objects = 3
            action_kwargs['grab_range'] = grab_range
            action_kwargs['max_objects'] = max_objects

            objects = list(state.keys())
            objects.remove(self.agent_properties["obj_id"])
            objects = [obj for obj in objects if 'agent' not in obj]  # don't grab agents

            object_in_range = []
            for object_id in objects:
                dist = int(np.ceil(np.linalg.norm(
                    np.array(state[object_id]['location']) - np.array(
                        state[self.agent_properties["obj_id"]]['location']))))
                if dist <= grab_range and state[object_id]["is_movable"]:
                    object_in_range.append(object_id)

            if object_in_range:
                action_kwargs['object_id'] = self.rnd_gen.choice(object_in_range)
            else:
                action_kwargs['object_id'] = None

        elif action == OpenDoorAction.__name__ or action == CloseDoorAction.__name__:
            action_kwargs['door_range'] = 1
            action_kwargs['object_id'] = None

            objects = list(state.keys())
            doors = [obj for obj in objects
                     if 'class_inheritance' in state[obj] and state[obj]['class_inheritance'][0] == "Door"]
            if len(doors) > 0:
                action_kwargs['object_id'] = self.rnd_gen.choice(doors)

        return action, action_kwargs

    def get_log_data(self):
        """Data to relay to a Logger. Override to log internal state."""
        return {}

    def send_message(self, message):
        """Queue a Message for sending. to_id may be one agent, a list, or None for all."""
        self.__check_message(message, self.agent_id)
        self.messages_to_send.append(message)

    def is_action_possible(self, action, action_kwargs):
        """Check whether an action is currently possible. Returns (succeeded, ActionResult)."""
        action_result = self.__callback_is_action_possible(self.agent_id, action, action_kwargs)

        return action_result.succeeded, action_result

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):

        # warn if we're replacing the State object (loses stored memory)
        if new_state is not self.state:
            warnings.warn(f"Overwriting State object of {self.agent_id}. This "
                          f"will cause any stored memory to be gone for good "
                          f"as this was stored in the previous State object.")

        if isinstance(new_state, dict):
            raise TypeError(f"The new state should of type State, is of "
                            f"type {new_state.__class__}")

        self._state = new_state

    @property
    def memorize_for_ticks(self):
        return self.__memorize_for_ticks

    def create_context_menu_for_other(self, agent_id_who_clicked, clicked_object_id, click_location):
        """Context menu options when another human agent right-clicks while controlling this agent."""
        print("Context menu other")
        context_menu = []

        # one option per action
        for action in self.action_set:
            context_menu.append({
                "OptionText": f"Do action: {action}",
                "Message": Message(content=action, from_id=clicked_object_id, to_id=self.agent_id)
            })
        return context_menu

    def _factory_initialise(self, agent_name, agent_id, action_set, sense_capability, agent_properties,
                            rnd_seed, callback_is_action_possible, customizable_properties=None):
        """Private MATRX function. The WorldBuilder calls this to wire up the brain."""
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.action_set = action_set

        self.rnd_seed = rnd_seed
        self._set_rnd_seed(seed=rnd_seed)

        self._init_state()
        self.sense_capability = sense_capability
        self.agent_properties = agent_properties

        # which properties this agent may change; the rest only change via actions
        self.keys_of_agent_writable_props = customizable_properties if customizable_properties is not None else list(agent_properties.keys())
        self.__callback_is_action_possible = callback_is_action_possible

    def _get_action(self, state, agent_properties, agent_id):
        """Private MATRX function the environment calls to get this agent's action. Do NOT override."""
        self.agent_properties = agent_properties
        self.state.state_update(state.as_dict())
        self.state = self.filter_observations(self.state)
        action, action_kwargs = self.decide_on_action(self.state)
        self.previous_action = action
        return self.state, self.agent_properties, action, action_kwargs

    def _fetch_state(self, state):
        self.state.state_update(state.as_dict())
        filtered_state = self.filter_observations(self.state)
        return filtered_state

    def _get_log_data(self):
        return self.get_log_data()

    def _set_action_result(self, action_result):
        """Private MATRX function. Stores the result of the last action. Do NOT override."""
        self.previous_action_result = action_result

    def _set_rnd_seed(self, seed):
        """Private MATRX function. Seed this agent's RNG. Do NOT override."""
        self.rnd_seed = seed
        self.rnd_gen = np.random.RandomState(self.rnd_seed)

    def _get_messages(self, all_agent_ids):
        """Private MATRX function. Hand queued messages to the GridWorld and clear them. Do NOT override."""
        send_messages = copy.copy(self.messages_to_send)
        self.messages_to_send = []
        return send_messages

    def _set_messages(self, messages=None):
        """Private MATRX function. Store messages received this tick. Do NOT override."""
        for mssg in messages:
            ArtificialAgentBrain.__check_message(mssg, self.agent_id)
            # each message is wrapped in a Message; content holds the real payload
            received_message = mssg.content
            self.received_messages.append(mssg)
            self.received_messages_content.append(mssg.content)


    def _init_state(self):
        self._state = State(memorize_for_ticks=self.memorize_for_ticks,
                            own_id=self.agent_id)

    @staticmethod
    def __check_message(mssg, this_agent_id):
        if not isinstance(mssg, Message):
            raise Exception(f"A message to {this_agent_id} is not, nor inherits from, the class {Message.__name__}."
                            f" This is required for agents to be able to send and receive them.")



class ArtificialBrain(ArtificialAgentBrain, ABC):
    """
    This class is the obligatory base class for the agents.
    Agents must implement decide_on_action
    """
    def __init__(self, slowdown, condition, name, folder):
        # slowdown sets action_duration: 1 = one action/tick, 3 = one action every 3 ticks
        self.__slowdown = slowdown
        self.__condition = condition
        self.__name = name
        self.__folder = folder
        super().__init__()
    
    def decide_on_action(self, state:State):
        """Wraps decide_on_actions and sets action durations (water, stone, tree, victims)."""
        act,params = self.decide_on_actions(state)
        params['grab_range']=1
        params['max_objects']=1
        # tiles covered by water
        water_locs = []
        if state[{"name": "water"}]:
            for water in state[{"name": "water"}]:
                if water['location'] not in water_locs:
                    water_locs.append(water['location'])
        # slow down in water, except on doormats
        if state[{"name": "RescueBot"}]['location'] in water_locs and state[{"name": "RescueBot"}]['location'] not in [(3,5),(9,5),(15,5),(21,5),(3,6),(9,6),(15,6),(3,17),(9,17),(15,17),(3,18),(9,18),(15,18),(21,18)]:
            params['action_duration'] = 13
        else:
            params['action_duration'] = self.__slowdown
        if act == 'RemoveObject' and 'stone' in params['object_id']:
            params['action_duration'] = 200
        if act == 'RemoveObject' and 'tree' in params['object_id']:
            params['action_duration'] = 100
        if act == 'CarryObject' and 'mild' in params['object_id']:
            params['action_duration'] = 150

        return act,params

    @abstractmethod
    def decide_on_actions(self, state:State):
        """Agent decision logic. Override this. Returns (action_name:str, action_args:dict)."""
        pass
    