## Setup environment
from enum import Enum
import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as timeStep

class ActionResult(Enum):
    VALID_MOVE = 1
    ILLEGAL_MOVE = 2
    FOUND_BONE = 3
    FOUND_ROBOT = 4
    GAME_COMPLETE = 5
    SNEAK_ATTACK = 6

class DogAdventure():
    def __init__(self):
        self._state = np.zeros((36,),dtype=np.int32)
        self._robot_locations = [10,14,25,28]
        self._bone_locations = [5,7,9,16,19,26,30]
        self._state[self._robot_locations] = 2
        self._state[self._bone_locations] = 3
        self._state[0] = 1
        self._state[1] = 4
        self._game_ended = False

    def reset(self):
        self._state = np.zeros((36,),dtype=np.int32)
        self._state[self._robot_locations] = 2
        self._state[self._bone_locations] = 3
        self._state[0] = 1
        self._state[1] = 4
        self._game_ended = False

    def __is_spot_last(self, position):
        return position == 35

    def move_dog(self, current_position, next_position, attack, player):

        if self.__is_spot_last(next_position):
          if player == 2:
            return ActionResult.ILLEGAL_MOVE
          else:
            self._state[current_position] = 0
            self._state[next_position] = 1

            self._game_ended = True
            return ActionResult.GAME_COMPLETE
        
        if next_position < 0 or next_position > (len(self._state) - 1):
          return ActionResult.ILLEGAL_MOVE

        if self._state[next_position] == 1 or self._state[next_position] == 4:
          return ActionResult.ILLEGAL_MOVE

        if self._state[next_position] == 2:
          if attack:
            if player == 2:
              self._state[current_position] = 0
              self._state[next_position] = 4
              return ActionResult.SNEAK_ATTACK
            else:
              self._state[current_position] = 0
              self._state[next_position] = 1
              return ActionResult.SNEAK_ATTACK
          else:
            self._game_ended = True
            return ActionResult.FOUND_ROBOT

        if self._state[next_position] == 3:
          if attack:
            return ActionResult.ILLEGAL_MOVE
          else:
            if player == 2:
              self._state[current_position] = 0
              self._state[next_position] = 4
              return ActionResult.FOUND_BONE
            else:
              self._state[current_position] = 0
              self._state[next_position] = 1
              return ActionResult.FOUND_BONE

        if self._state[next_position] == 0 and attack:
          return ActionResult.ILLEGAL_MOVE
          
        if player == 2:
          self._state[current_position] = 0
          self._state[next_position] = 4
          return ActionResult.VALID_MOVE
        
        self._state[current_position] = 0
        self._state[next_position] = 1
        return ActionResult.VALID_MOVE

    def game_ended(self):
        return self._game_ended
  
    def game_state(self):
        return self._state


class DogAdventureEnvironment(py_environment.PyEnvironment):

    def __init__(self, game):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=15, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(36,), dtype=np.int32, minimum=0, maximum=4, name='observation')

        # 0=>Left, 1=>Right, 2=>Down, 3=>Up 
        # 4=>attack_up_right, 5=>attack_up_left, 6=>attack_down_right, 7=>attack_down_left
        # same order for player 2, index 8 and onwards
        self._action_values = {0:-1,1:1,2:-6,3:6,4:7,5:5,6:-7,7:-5,8:-1,9:1,10:-6,11:6,12:7,13:5,14:-7,15:-5}
        self._game = game

        # make sure the environment is okay
        utils.validate_py_environment(self, episodes=5)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._game.reset()
        return timeStep.restart(self._game.game_state())
  
    def _step(self, action):    

        if self._game.game_ended():
            return self.reset()

        action = action.item()
        attack = False
        player = 1

        if 4 <= action <= 7 or 12 <= action <= 15:
          attack = True

        if 8 <= action <= 15:
          player = 2

        next_agent_position_direction = self._action_values.get(action)
        
        if player == 2:
          current_agent_position = np.where(self._game.game_state() == 4)[0].item()
        else:
          current_agent_position = np.where(self._game.game_state() == 1)[0].item()

        new_agent_position = current_agent_position + next_agent_position_direction
        response = self._game.move_dog(current_agent_position,new_agent_position, attack, player)

        if response == ActionResult.GAME_COMPLETE:
            return timeStep.termination(self._game.game_state(), 10)

        elif response == ActionResult.ILLEGAL_MOVE:
            return timeStep.transition(self._game.game_state(), reward=-0.5, discount=1.0)

        elif response == ActionResult.FOUND_ROBOT:
            return timeStep.termination(self._game.game_state(), -0.8)
        
        elif response == ActionResult.SNEAK_ATTACK:
            return timeStep.transition(self._game.game_state(), reward=2, discount=1.0)

        elif response == ActionResult.FOUND_BONE:
            return timeStep.transition(self._game.game_state(), reward=1, discount=1.0)

        return timeStep.transition(self._game.game_state(), reward=-0.3, discount=1.0)



def environment_setup(dogEnvironment):
  
  # setup training and evaluation environment
  train_env = tf_py_environment.TFPyEnvironment(dogEnvironment)
  eval_env = tf_py_environment.TFPyEnvironment(dogEnvironment)
  return train_env, eval_env