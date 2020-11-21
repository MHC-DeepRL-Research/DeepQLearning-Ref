## Setup environment
from enum import Enum
from enum import IntEnum

import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as timeStep

# possible consequences from action
class ActionResult(Enum):
    VALID_MOVE = 1
    ILLEGAL_MOVE = 2
    FOUND_BONE = 3
    FOUND_ROBOT = 4
    GAME_COMPLETE = 5
    SNEAK_ATTACK = 6

class StateFlag(IntEnum):
    EMPTY = 0
    PLAYER1 = 1
    ROBOT_ZONE = 2
    BONE_ZONE = 3
    PLAYER2 = 4
    GOAL = 5

class DogAdventureGame():

    def __init__(self):
        self._robot_locations = [10,14,25,28]
        self._bone_locations = [5,7,9,16,19,26,30]
        self._n_states = 36
        self.reset()

    def reset(self):
        self._state = np.zeros((self._n_states,),dtype=np.int32)
        self._state[self._robot_locations] = StateFlag.ROBOT_ZONE
        self._state[self._bone_locations] = StateFlag.BONE_ZONE
        self._state[0] = StateFlag.PLAYER1
        self._state[1] = StateFlag.PLAYER2
        self._state[self._n_states-1] = StateFlag.GOAL
        self.update_reward_to_goal()
        self._game_ended = False

    def is_goal_reached(self, position):
        return self._state[position] == StateFlag.GOAL

    def typical_state_update(self, current_position, next_position, player):
        self._state[current_position] = StateFlag.EMPTY
        if player == 2:
          self._state[next_position] = StateFlag.PLAYER2
        else:
          self._state[next_position] = StateFlag.PLAYER1

    def update_reward_to_goal(self):
        goal = self._n_states-1
        player1 = np.where(self._state == StateFlag.PLAYER1)[0].item()
        dist2goal = (goal-player1) % 6 + (goal-player1) // 6
        self._reward_to_goal = 1.0 / (dist2goal+1.0)
        return self._reward_to_goal

    def move_dog(self, current_position, next_position, attack, player):

        # rule1: out of border conditions
        if next_position < 0 or next_position > (self._n_states - 1):
          return ActionResult.ILLEGAL_MOVE, self._reward_to_goal

        if next_position % 6 == 0 and current_position % 6 == 1:
          return ActionResult.ILLEGAL_MOVE, self._reward_to_goal

        if next_position % 6 == 1 and current_position % 6 == 0:
          return ActionResult.ILLEGAL_MOVE, self._reward_to_goal

        # rule2: reach goal conditions
        if self.is_goal_reached(next_position):
          if player == 2:
            return ActionResult.ILLEGAL_MOVE, self._reward_to_goal
          else:
            self._state[current_position] = StateFlag.EMPTY
            self._state[next_position] = StateFlag.PLAYER1
            self._game_ended = True
            self.update_reward_to_goal()
            return ActionResult.GAME_COMPLETE, self._reward_to_goal

        # rule3: run into player conditions
        if self._state[next_position] == StateFlag.PLAYER1 or self._state[next_position] == StateFlag.PLAYER2:
          return ActionResult.ILLEGAL_MOVE, self._reward_to_goal

        # rule4: run into robot conditions
        if self._state[next_position] == StateFlag.ROBOT_ZONE:
          if attack:
            self.typical_state_update(current_position, next_position, player)
            self.update_reward_to_goal()
            return ActionResult.SNEAK_ATTACK, self._reward_to_goal
          else:
            self._game_ended = True
            return ActionResult.FOUND_ROBOT, self._reward_to_goal

        # rule5: run into bone conditions
        if self._state[next_position] == StateFlag.BONE_ZONE:
          if attack:
            return ActionResult.ILLEGAL_MOVE, self._reward_to_goal
          else:
            self.typical_state_update(current_position, next_position, player)
            self.update_reward_to_goal()
            return ActionResult.FOUND_BONE, self._reward_to_goal

        # rule6: attack nothing conditions
        if self._state[next_position] == StateFlag.EMPTY and attack:
          return ActionResult.ILLEGAL_MOVE, self._reward_to_goal

        self.typical_state_update(current_position, next_position, player)
        self.update_reward_to_goal()
        return ActionResult.VALID_MOVE, self._reward_to_goal


    def game_ended(self):
        return self._game_ended
  
    def game_state(self):
        return self._state


class DogAdventureEnvironment(py_environment.PyEnvironment):

    def __init__(self, game):

        # set game
        self._game = game

        # set action range
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=15, name='action')

        # set observation range
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._game._n_states,), dtype=np.int32, minimum=0, maximum=5, name='observation')

        # action table: along the diagnal are attack moves
        # player1      player2  
        # ---------------------
        #  6 2 7      14 10 15
        #  0 . 1       8  .  9
        #  5 3 4      13 11 12

        self._action_values = {0:-1, 1:1,  2:-6,  3:6,  4:7,  5:5,  6:-7,  7:-5,
                               8:-1, 9:1, 10:-6, 11:6, 12:7, 13:5, 14:-7, 15:-5}
        

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

        # set attack flag
        if 4 <= action <= 7 or 12 <= action <= 15:
          attack = True

        # set player flag
        if 8 <= action <= 15:
          player = 2

        # set position change after action
        next_agent_position_direction = self._action_values.get(action)
        
        # set current position of the player
        if player == 2:
          current_agent_position = np.where(self._game.game_state() == StateFlag.PLAYER2)[0].item()
        else:
          current_agent_position = np.where(self._game.game_state() == StateFlag.PLAYER1)[0].item()
          

        # update agent's new position
        new_agent_position = current_agent_position + next_agent_position_direction
        response, reward_to_goal = self._game.move_dog(current_agent_position,new_agent_position, attack, player)

        # game termination handling
        if response == ActionResult.GAME_COMPLETE:
            return timeStep.termination(self._game.game_state(), 10+reward_to_goal)

        elif response == ActionResult.FOUND_ROBOT:
            return timeStep.termination(self._game.game_state(), -0.8+reward_to_goal)

        # game transition handling
        elif response == ActionResult.ILLEGAL_MOVE:
            return timeStep.transition(self._game.game_state(), reward=-2+reward_to_goal, discount=1.0)
        
        elif response == ActionResult.SNEAK_ATTACK:
            return timeStep.transition(self._game.game_state(), reward=2+reward_to_goal, discount=1.0)

        elif response == ActionResult.FOUND_BONE:
            return timeStep.transition(self._game.game_state(), reward=1+reward_to_goal, discount=1.0)

        return timeStep.transition(self._game.game_state(), reward=-0.3+reward_to_goal, discount=1.0)



def environment_setup(dogEnvironment):
  
  # setup training and evaluation environment
  train_env = tf_py_environment.TFPyEnvironment(dogEnvironment)
  eval_env = tf_py_environment.TFPyEnvironment(dogEnvironment)
  return train_env, eval_env