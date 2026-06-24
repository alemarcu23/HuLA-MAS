import sys, random, enum, ast, time
from matrx import grid_world
from matrx.agents.ArtificialBrain import ArtificialBrain
from matrx.actions.CustomActions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from matrx.actions.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop

class Phase(enum.Enum):
    INTRO0=0,
    INTRO1=1,
    INTRO2=2,
    INTRO3=3,
    INTRO4=4,
    INTRO5=5,
    INTRO6=6,
    INTRO7=7,
    INTRO8=8,
    INTRO9=9,
    INTRO10=10,
    INTRO11=11,
    FIND_NEXT_GOAL=12,
    PICK_UNSEARCHED_ROOM=13,
    PLAN_PATH_TO_ROOM=14,
    FOLLOW_PATH_TO_ROOM=15,
    PLAN_ROOM_SEARCH_PATH=16,
    FOLLOW_ROOM_SEARCH_PATH=17,
    PLAN_PATH_TO_VICTIM=18,
    FOLLOW_PATH_TO_VICTIM=19,
    TAKE_VICTIM=20,
    PLAN_PATH_TO_DROPPOINT=21,
    FOLLOW_PATH_TO_DROPPOINT=22,
    DROP_VICTIM=23,
    WAIT_FOR_HUMAN=24,
    WAIT_AT_ZONE=25,
    FIX_ORDER_GRAB=26,
    FIX_ORDER_DROP=27,
    REMOVE_OBSTACLE_IF_NEEDED=28,
    ENTER_ROOM=29
    
class TutorialAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        self._slowdown = slowdown
        self._humanName = name
        self._folder = folder
        self._phase=Phase.INTRO0
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._maxTicks = 9600
        self._sendMessages = []
        self._currentDoor=None 
        self._condition = condition
        self._providedExplanations = []   
        self._teamMembers = []
        self._carryingTogether = False
        self._remove = False
        self._goalVic = None
        self._goalLoc = None
        self._humanLoc = None
        self._distanceHuman = None
        self._distanceDrop = None
        self._agentLoc = None
        self._todo = []
        self._answered = False
        self._tosearch = []
        self._tutorial = True
        self._recentVic = None

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        return state

    def decide_on_actions(self, state):
        # collect team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member!=agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        self._processMessages(state, self._teamMembers)

        # distance to the human, from last known areas when out of sight
        if state[{'is_human_agent':True}]:
            self._distanceHuman = 'close'
        if not state[{'is_human_agent':True}]:
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'far'
            if self._agentLoc in [1, 2, 3, 4, 5, 6, 7] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'close'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [1, 2, 3, 4, 5, 6, 7]:
                self._distanceHuman = 'far'
            if self._agentLoc in [8, 9, 10, 11, 12, 13, 14] and self._humanLoc in [8, 9, 10, 11, 12, 13, 14]:
                self._distanceHuman = 'close'

        # distance to the drop zone from last known area
        if self._agentLoc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distanceDrop = 'far'
        if self._agentLoc in [3, 4, 7, 10, 13, 14]:
            self._distanceDrop = 'close'

        # track victims being carried together
        for info in state.values():
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying'])>0 and 'critical' in info['is_carrying'][0]['obj_id']:
                self._collectedVictims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carryingTogether = True
            if 'is_human_agent' in info and self._humanName in info['name'] and len(info['is_carrying'])==0:
                self._carryingTogether = False
        # while carrying together the human drives the joint action, so stay idle
        if self._carryingTogether == True:
            return None, {}

        # hidden score message used for display and logging, DO NOT REMOVE
        self._sendMessage('Our score is ' + str(state['rescuebot']['score']) +'.', 'RescueBot')

        # main loop: behaviour is driven by the current phase
        while True:
            if Phase.INTRO0==self._phase:
                self._sendMessage('Hello! My name is RescueBot. During this task we will collaborate with each other to search and rescue the victims at the drop zone on our right. \
                For this tutorial there are 4 victims and 3 injury types, during the official task there will be 8 victims to rescue. \
                The red color refers to critically injured victims, yellow to mildly injured victims, and green to healthy victims. Healthy victims do not need to be rescued. \
                The 8 victims are a girl (critically injured girl/mildly injured girl/healthy girl), boy (critically injured boy/mildly injured boy/healthy boy), \
                woman (critically injured woman/mildly injured woman/healthy woman), man (critically injured man/mildly injured man/healthy man), \
                elderly woman (critically injured elderly woman/mildly injured elderly woman/healthy elderly woman), \
                elderly man (critically injured elderly man/mildly injured elderly man/healthy elderly man), dog (critically injured dog/mildly injured dog/healthy dog), \
                and a cat (critically injured cat/mildly injured cat/healthy cat). The environment will also contain different obstacle types with varying removal times. \
                At the top of the world you can find the keyboard controls, for moving you can use the arrow keys. \
                Press the "Continue" button to start the tutorial explaining everything.', 'RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO1
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO1==self._phase:
                self._sendMessage('Lets try out the controls first. You can move with the arrow keys. If you move down twice, you will notice that you can now no longer see me. \
                So you can only see as far as 2 grid cells. Therefore, it is important to search the areas well. If you moved down twice, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO2
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO2==self._phase:
                self._sendMessage('Lets move to area 3 now. When you are going to search an area, it is recommended to inform me about this.  \
                You can do this using the button "03". This way, we can collaborate more efficiently. \
                If you pressed the button "03" and moved to the area entrance, press the "Continue" button.', 'RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO3
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO3==self._phase:
                self._sendMessage('If you search area 3, you will find one of the victims to rescue: critically injured elderly woman. \
                There will be 3 different versions of the official task, manipulating your capabilities and resulting in different interdependence relationships between us. \
                However, in all conditions the critically injured victims have to be carried together. \
                So, let us carry critically injured elderly woman together! To do so, inform me that you found this victim by using the buttons below "I have found:" and selecting "critically injured elderly woman in 03". \
                If you found critically injured elderly woman and informed me about it, press the "Continue" button. I will then come over to help.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.FIND_NEXT_GOAL
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO4==self._phase:
                self._sendMessage('Let us carry ' + self._goalVic + ' together. To do this, move yourself on top, above, or next to ' + self._goalVic + '. \
                Now, press "A" on your keyboard (all keyboard controls can be found at the top of the world). \
                Transport ' + self._goalVic + ' to the drop zone and move yourself on top of the image of '+ self._goalVic + '. \
                Next, press "S" on your keyboard to drop '+ self._goalVic + '. \
                If you completed these steps, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO5
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO5==self._phase:
                self._sendMessage('Nice job! Lets move to area 5 next. Remember to inform me about this. \
                If you are in front of area 5, you see that it is blocked by rock. This is one of the three obstacle types, and can only be removed together. \
                So, let us remove rock together! To do so, inform me that you found this obstacle by using the button "Help remove" and selecting "at 05". \
                I will then come over to help. If you informed me and I arrived at area 5 to help, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO6
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO6==self._phase:
                self._sendMessage('Let us remove rock together now! To do so, remain in front of rock and press "D" on your keyboard. \
                Now, you will see a small busy icon untill rock is successfully removed. If the entrance is cleared, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO7
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO7==self._phase:
                self._sendMessage('Lets move to area 4 next. Remember to inform me about this. \
                If you are in front of area 4, you see that it is blocked by tree. This is another obstacle type, and tree can only be removed by me. \
                So, let me remove tree for you! To do so, inform me that you need help with removing by using the button "Help remove" and selecting "at 04". \
                I will then come over to remove tree for you.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO8
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO8==self._phase:
                self._sendMessage('In area 4 you will find mildly injured elderly man. If you find mildly injured victims, it is recommended to inform me about this. \
                You can do this using the buttons below "I have found:" and selecting "mildly injured elderly man in 04". \
                Depending on the condition of the official task, you can rescue mildly injured victims alone or require my help. In this tutorial, you will carry mildly injured elderly man alone. \
                If you decide to carry mildly injured victims, it is recommended to inform me about it. \
                You can do this using the buttons below "I will pick up:" and selecting "mildly injured elderly man in 04." \
                Next, you can pick up mildly injured elderly man by moving yourself on top, above, or next to mildly injured elderly man. \
                Now, press "Q" on your keyboard and transport mildly injured elderly man to the drop zone. \
                Drop mildly injured elderly man by moving on top of the image and pressing "W" on your keyboard. \
                If you completed these steps, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO9
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO9==self._phase:
                self._sendMessage('Nice job! Lets move to area 8 now. Remember to inform me about this. \
                If you are in front of area 8, you see that it is blocked by stones. \
                Depending on the condition of the official task, you might remove stones alone, require my help, or use my help to remove stones faster than doing it alone. \
                However, when I find stones, removing them together will always be faster than when I remove stones alone. For this tutorial, you will remove stones alone. \
                You can remove stones by pressing "E" on your keyboard. Now, you will see a small busy icon untill stones is successfully removed. \
                When you are busy removing, you can send messages but they will only appear once the action is finished. \
                So, no need to keep clicking buttons! If the entrance is cleared, press the "Continue" button.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                    self._phase=Phase.INTRO10
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None,{}

            if Phase.INTRO10==self._phase:
                self._sendMessage('This concludes the tutorial! You can now start the real task.','RescueBot')
                if self.received_messages_content and self.received_messages_content[-1]=='Found: critically injured girl in 5':
                    self._phase=Phase.FIND_NEXT_GOAL
                    self.received_messages_content=[]
                    self.received_messages=[]
                else:
                    return None, {}
            
            if Phase.FIND_NEXT_GOAL==self._phase:
                self._answered = False
                self._goalVic = None
                self._goalLoc = None
                remainingZones = []
                remainingVics = []
                remaining = {}
                # victims still to rescue and where to drop them
                zones = self._getDropZones(state)
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remainingZones.append(info)
                        remainingVics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remainingZones:
                    self._remainingZones = remainingZones
                    self._remaining = remaining
                # nothing left to rescue -> idle
                if not remainingZones:
                    return None,{}

                # go for an already-found victim
                for vic in remainingVics:
                    if vic in self._foundVictims and vic not in self._todo:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]
                        # exact location known -> go to victim, else just to the area
                        if 'location' in self._foundVictimLocs[vic].keys():
                            self._phase=Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__,{'duration_in_ticks':25}
                        if 'location' not in self._foundVictimLocs[vic].keys():
                            self._phase=Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__,{'duration_in_ticks':25}
                # no known victim -> search a new area
                self._phase=Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM==self._phase:
                agent_location = state[self.agent_id]['location']
                unsearchedRooms=[room['room_name'] for room in state.values()
                if 'class_inheritance' in room
                and 'Door' in room['class_inheritance']
                and room['room_name'] not in self._searchedRooms
                and room['room_name'] not in self._tosearch]
                # all areas searched but task unfinished -> start over
                if self._remainingZones and len(unsearchedRooms) == 0:
                    self._tosearch = []
                    self._todo = []
                    self._searchedRooms = []
                    self._sendMessages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._searchedRooms.append(self._door['room_name'])
                    self._sendMessage('Going to re-search all areas.','RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                # otherwise pick the closest unsearched area
                else:
                    if self._currentDoor==None:
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,agent_location))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state,unsearchedRooms,agent_location))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':  # bug workaround
                            self._doormat = (3,5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    if self._currentDoor!=None:
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,self._currentDoor))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state, unsearchedRooms,self._currentDoor))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3,5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM==self._phase:
                self._navigator.reset_full()
                # head to the area where the human found the victim, else the area to search
                if self._goalVic and self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    self._door = state.get_room_doors(self._foundVictimLocs[self._goalVic]['room'])[0]
                    self._doormat = state.get_room(self._foundVictimLocs[self._goalVic]['room'])[-1]['doormat']
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3,5)
                    doorLoc = self._doormat
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3,5)
                    doorLoc = self._doormat
                self._navigator.add_waypoints([doorLoc])
                self._phase=Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM==self._phase:
                # re-plan if the target victim was rescued, found elsewhere, or the area was searched
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._goalVic and self._goalVic in self._foundVictims and self._door['room_name']!=self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    # tell the human why we're heading there
                    if self._goalVic in self._foundVictims and str(self._door['room_name']) == self._foundVictimLocs[self._goalVic]['room'] and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic+'.', 'RescueBot')
                    if self._goalVic not in self._foundVictims and not self._remove or not self._goalVic and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.', 'RescueBot')
                    self._currentDoor=self._door['location']
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        # clear stones blocking the path
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9,7),(9,19),(21,19)]:
                                return RemoveObject.__name__,{'object_id':info['obj_id']}
                        return action,{}
                    self._phase=Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED==self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                # what is blocking the entrance?
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info['obj_id']:
                        objects.append(info)
                        if self._tutorial and self.received_messages_content and self.received_messages_content[-1]=='Continue':
                            self._phase=Phase.INTRO6
                            self.received_messages_content=[]
                            self.received_messages=[]
                        else:
                            return None,{}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info['obj_id']:
                        objects.append(info)
                        self.received_messages_content=[]
                        self.received_messages=[]
                        self._remove = False
                        self._phase=Phase.INTRO8
                        return RemoveObject.__name__,{'object_id':info['obj_id']}  # agent removes trees

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id']:
                        objects.append(info)
                        return None, {}  # stones handled by the human
                # entrance clear -> enter
                if len(objects)==0:
                    self._answered = False
                    self._remove = False
                    self._phase = Phase.ENTER_ROOM

            if Phase.ENTER_ROOM==self._phase:
                self._answered = False
                # re-plan if the target victim was rescued, found elsewhere, or the area was searched
                if self._goalVic in self._collectedVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._goalVic in self._foundVictims and self._door['room_name']!=self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH==self._phase:
                self._agentLoc = int(self._door['room_name'].split()[-1])
                # all tiles of this area
                roomTiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info
                    and 'AreaTile' in info['class_inheritance']
                    and 'room_name' in info
                    and info['room_name'] == self._door['room_name']]
                self._roomtiles=roomTiles
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(roomTiles))
                self._roomVics=[]
                self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    # look for victims while walking through the area
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            if vic not in self._roomVics:
                                self._roomVics.append(vic)

                            # fill in the exact location for a victim the human reported
                            if vic in self._foundVictims and 'location' not in self._foundVictimLocs[vic].keys():
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                                if vic == self._goalVic:
                                    self._sendMessage('Found '+ vic + ' in ' + self._door['room_name'] + ' because you told me '+vic+ ' was located here.', 'RescueBot')
                                    self._searchedRooms.append(self._door['room_name'])
                                    self._phase=Phase.FIND_NEXT_GOAL

                            # newly found injured victim
                            if 'healthy' not in vic and vic not in self._foundVictims:
                                self._recentVic = vic
                                self._foundVictims.append(vic)
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                    return action,{}

                # searched the whole area but the reported victim wasn't there
                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room']==self._door['room_name']:
                    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic+'.', 'RescueBot')
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []
                    self.received_messages = []
                    self.received_messages_content = []
                self._searchedRooms.append(self._door['room_name'])
                self._recentVic = None
                self._phase=Phase.FIND_NEXT_GOAL
                return Idle.__name__,{'duration_in_ticks':25}

            if Phase.PLAN_PATH_TO_VICTIM==self._phase:
                if 'mild' in self._goalVic:
                    self._sendMessage('Picking up ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'] + '.', 'RescueBot')
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocs[self._goalVic]['location']])
                self._phase=Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM==self._phase:
                # human already rescued it -> find a new goal
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM==self._phase:
                objects=[]
                # critically injured victims are carried together: wait for the human
                for info in state.values():
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        self._collectedVictims.append(self._goalVic)
                        self._phase=Phase.INTRO4
                        if not self._humanName in info['name']:
                            return None, {}
                if len(objects)==0 and 'critical' in self._goalVic:
                    self._collectedVictims.append(self._goalVic)
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                # mildly injured victims are picked up alone
                if 'mild' in self._goalVic:
                    self._phase=Phase.PLAN_PATH_TO_DROPPOINT
                    self._collectedVictims.append(self._goalVic)
                    return CarryObject.__name__,{'object_id':self._foundVictimLocs[self._goalVic]['obj_id'], 'human_name': self._humanName}

            if Phase.PLAN_PATH_TO_DROPPOINT==self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalLoc])
                self._phase=Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT==self._phase:
                self._state_tracker.update(state)
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                self._phase=Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                if 'mild' in self._goalVic:
                    self._sendMessage('Delivered '+ self._goalVic + ' at the drop zone.', 'RescueBot')
                self._phase=Phase.FIND_NEXT_GOAL
                self._currentDoor = None
                self._tick = state['World']['nr_ticks']
                return Drop.__name__,{'human_name': self._humanName}

            
    def _getDropZones(self,state:State):
        """Drop zones (full dicts), ordered so the first one is dropped first."""
        places=state[{'is_goal_block':True}]
        places.sort(key=lambda info:info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr']==0:
                zones.append(place)
        return zones

    def _processMessages(self, state, teamMembers):
        """Update memory from team members' messages (Search/Found/Collect/Remove)."""
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        for mssgs in receivedMessages.values():
            for msg in mssgs:
                # area searched by a team member
                if msg.startswith("Search:"):
                    area = 'area '+ msg.split()[-1]
                    if area not in self._searchedRooms:
                        self._searchedRooms.append(area)
                # victim found by a team member
                if msg.startswith("Found:"):
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area '+ msg.split()[-1]
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    if foundVic not in self._foundVictims:
                        self._foundVictims.append(foundVic)
                        self._foundVictimLocs[foundVic] = {'room':loc}
                    if foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] != loc:
                        self._foundVictimLocs[foundVic] = {'room':loc}
                    if 'mild' in foundVic:
                        self._todo.append(foundVic)
                # victim rescued by a team member
                if msg.startswith('Collect:'):
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    if collectVic not in self._foundVictims:
                        self._foundVictims.append(collectVic)
                        self._foundVictimLocs[collectVic] = {'room':loc}
                    if collectVic in self._foundVictims and self._foundVictimLocs[collectVic]['room'] != loc:
                        self._foundVictimLocs[collectVic] = {'room':loc}
                    if collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                # human asks for help removing an obstacle -> go there
                if msg.startswith('Remove:'):
                    area = 'area ' + msg.split()[-1]
                    self._door = state.get_room_doors(area)[0]
                    self._doormat = state.get_room(area)[-1]['doormat']
                    if area in self._searchedRooms:
                        self._searchedRooms.remove(area)
                    self.received_messages = []
                    self.received_messages_content = []
                    self._remove = True
                    self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.', 'RescueBot')
                    self._phase = Phase.PLAN_PATH_TO_ROOM
            # remember the human's last reported area
            if mssgs and mssgs[-1].split()[-1] in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']:
                self._humanLoc = int(mssgs[-1].split()[-1])

    def _sendMessage(self, mssg, sender):
        """Send a message to the team, skipping duplicates (score messages always go)."""
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'score' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        # hidden score message, DO NOT REMOVE
        if 'score' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        """Return the area closest to the agent (or to currentDoor if set)."""
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj]=state.get_room_doors(obj)[0]['location']
        dists = {}
        for room,loc in locs.items():
            if currentDoor!=None:
                dists[room]=utils.get_distance(currentDoor,loc)
            if currentDoor==None:
                dists[room]=utils.get_distance(agent_location,loc)

        return min(dists,key=dists.get)

    def _efficientSearch(self, tiles):
        """Zig-zag waypoints to sweep an area without visiting every tile."""
        x=[]
        y=[]
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i%2==0:
                locs.append((x[i],min(y)))
            else:
                locs.append((x[i],max(y)))
        return locs