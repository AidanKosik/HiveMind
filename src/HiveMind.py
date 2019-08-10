from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib import actions
from pysc2.agents import base_agent
from pysc2.lib import features, units

import numpy as np
import random
import math
import pandas as pd

import os

from src.QLearningTable import QLearningTable

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_OVERLORD = actions.FUNCTIONS.Train_Overlord_quick.id
_BUILD_SPAWNING_POOL = actions.FUNCTIONS.Build_SpawningPool_screen.id
_TRAIN_ZERGLING = actions.FUNCTIONS.Train_Zergling_quick.id
_TRAIN_DRONE = actions.FUNCTIONS.Train_Drone_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_ZERG_HATCHERY = 86
_ZERG_DRONE = 104 
_ZERG_OVERLORD = 106
_ZERG_LARVA = 151
_ZERG_ZERGLING = 105
_ZERG_SPAWNING_POOL = 89
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


DATA_FILE = 'random_agent_data'


ACTION_DO_NOTHING = 'donothing'
ACTION_TRAIN_DRONE = 'traindrone'
ACTION_TRAIN_OVERLORD = 'trainoverlord'
ACTION_BUILD_SPAWNING_POOL = 'buildspawningpool'
ACTION_TRAIN_ZERGLING = 'trainzergling'
ACTION_ATTACK = 'attack'

SMART_ACTIONS = []

_ATTACK_COORDINATES = []

KILL_UNIT_REWARD = 0.25
KILL_BUILDING_REWARD = 0.5

for mm_x in range(0,128):
    for mm_y in range(0,148):
        SMART_ACTIONS.append("15" +'_' + str(mm_x) + '_' + str(mm_y))

for func in actions._FUNCTIONS:
    string = str(func)
    ls = string.split("/")
    func_id = str(ls[0])
    if func_id != "15":
        SMART_ACTIONS.append(func_id)


class HiveMind(base_agent.BaseAgent):

    def __init__(self):
        super(HiveMind, self)
        self.qlearn = QLearningTable(actions=list(range(len(SMART_ACTIONS))))

        self.previous_action = None
        self.previous_state = None

        self.base_top_left = None
        
        self.hatch_x = None
        self.hatch_y = None

        self.move_number = 0

        self.attack_coordinates = None

        self.steps = 0
        self.episodes = 0

        self.reward = 0

        # if os.path.isfile(DATA_FILE + '.gz'):
        #     self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x+x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 74 - y]
        return [x,y]

    def splitAction(self, action_id):
        print("Len Attacks: " +str(len(SMART_ACTIONS)) + " Action ID: " + str(action_id))
        smart_action = SMART_ACTIONS[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
        if smart_action == 'attack':
            smart_action = _ATTACK_MINIMAP
        return (int(smart_action), x, y)

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self,obs):
        super(HiveMind, self).step(obs)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() <= 31 else 0

        # Gather State Information
        current_state = np.zeros(10)
        player_info = obs.observation['player']
        current_state[0] = player_info[1]       # Minerals
        current_state[1] = player_info[2]       # Vespene
        current_state[2] = player_info[3]       # Supply
        current_state[3] = player_info[4]       # Worker Supply
        current_state[4] = player_info[8]       # Army Count
        current_state[5] = self.base_top_left   # Where is our base?

        # Calculate enemy positions for state
        hot_squares = np.zeros(4)
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 32))
            x = int(math.ceil((enemy_x[i] + 1) / 32))
            hot_squares[((y - 1) * 2) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0,4):
            current_state[i+5] = hot_squares[i]

        for ob in obs.observation:
            print(ob)

        if self.previous_action is not None:
            
            self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

         # Choose Action
        rl_action = self.qlearn.choose_action(str(current_state))

        self.previous_state = current_state
        self.previous_action = rl_action

        smart_action, x, y = self.splitAction(self.previous_action)
        print(str(smart_action))
        print("-------------------------------------------")
        print(str(obs.observation.available_actions))

        if smart_action == _ATTACK_MINIMAP:
            if 13 in obs.observation.available_actions:
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-1, 1)
                return actions.FunctionCall(13, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])

        elif smart_action in obs.observation.available_actions:
            args = [[np.random.randint(0, size) for size in arg.sizes] for arg in self.action_spec.functions[smart_action].args]
            return actions.FunctionCall(smart_action, args)

        return actions.FUNCTIONS.no_op()