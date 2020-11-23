## Setup environment
import param
import funcs
import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as timeStep

class CamAdventureGame():

    def __init__(self):
        self._n_locs = param.GRIDS_IN_SPACE
        self._n_rots = param.CAM_ROT_OPTIONS
        self.reset()

    def reset(self):
        self._step_counter = 0
        self._state = np.zeros((self._n_locs,self._n_rots),dtype=np.int32)

        for i in range(param.CAM_COUNT):
          angle = 2 * np.pi * i / param.CAM_COUNT
          coord2Dy = round(param.CAM_INIT_DIST * np.sin(angle)/param.GRID_LENGTH) * param.GRID_LENGTH 
          coord2Dx = round(param.CAM_INIT_DIST * np.cos(angle)/param.GRID_LENGTH) * param.GRID_LENGTH 
          coord2D = np.array([coord2Dy, coord2Dx])
          coord1D = funcs.campose_2Dto1D(coord2D)
          self._state[coord1D,param.RotFlag.CENTER] = i+1

    def move_cam(self, curr_pose, next_pose, cam):
        # rule0: up to max iteration
        self._step_counter += 1
        if self._step_counter > param.EVAL_MAX_ITER:
          return param.ActionResult.END_GAME

        # rule1: out of verticle border conditions
        if next_pose[0] < 0 or next_pose[0] > (self._n_locs - 1):
          return param.ActionResult.ILLEGAL_MOVE

        # rule2: out of horizontal border conditions
        if np.abs(next_pose[0] - curr_pose[0]) % param.GRIDS_PER_EDGE > 1.0:
          return param.ActionResult.ILLEGAL_MOVE

        # rule3: out of allowable rotation options
        if next_pose[1] < 0 or next_pose[1] > (self._n_rots - 1):
          return param.ActionResult.ILLEGAL_MOVE

        # rule4: speedy orientation change
        if next_pose[1] != param.RotFlag.CENTER and curr_pose[1] != param.RotFlag.CENTER:
          return param.ActionResult.ILLEGAL_MOVE

        # rule4: run into other cameras conditions
        if np.sum(self._state[next_pose[0],:]) > 0 and np.sum(self._state[next_pose[0],:]) != cam+1:
          return param.ActionResult.ILLEGAL_MOVE

        self._state[curr_pose[0],curr_pose[1]] = 0
        self._state[next_pose[0],next_pose[1]] = cam+1
        return param.ActionResult.VALID_MOVE
  
    def game_state(self):
        return self._state


class CamAdventureEnvironment(py_environment.PyEnvironment):

    def __init__(self, game):

        # set game
        self._game = game

        # set action range
        self.action_count = param.CAM_COUNT * param.CAM_ACT_OPTIONS
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.action_count-1, name='action')

        # set observation range
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._game._n_locs,self._game._n_rots), dtype=np.int32, 
            minimum=0, maximum=param.CAM_COUNT, name='observation')

        # create action dictionary
        self.create_action_dict()

        # make sure the environment is okay
        utils.validate_py_environment(self, episodes=5)

    def create_action_dict(self):
        # action table: along the diagnal are attack moves
        self._action_values = {} # the action table

        for i in range(param.CAM_COUNT):
          self._action_values[i*param.CAM_ACT_OPTIONS + 0] = [                    0,param.RotFlag.CENTER]
          self._action_values[i*param.CAM_ACT_OPTIONS + 1] = [                    0,param.RotFlag.UP]
          self._action_values[i*param.CAM_ACT_OPTIONS + 2] = [                    0,param.RotFlag.DOWN]
          self._action_values[i*param.CAM_ACT_OPTIONS + 3] = [                    0,param.RotFlag.LEFT]
          self._action_values[i*param.CAM_ACT_OPTIONS + 4] = [                    0,param.RotFlag.RIGHT]
          self._action_values[i*param.CAM_ACT_OPTIONS + 5] = [-param.GRIDS_PER_EDGE,param.RotFlag.SAME]
          self._action_values[i*param.CAM_ACT_OPTIONS + 6] = [ param.GRIDS_PER_EDGE,param.RotFlag.SAME]
          self._action_values[i*param.CAM_ACT_OPTIONS + 7] = [                   -1,param.RotFlag.SAME]
          self._action_values[i*param.CAM_ACT_OPTIONS + 8] = [                   +1,param.RotFlag.SAME]
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._game.reset()
        return timeStep.restart(self._game.game_state())
  
    def _step(self, action):    

        action = action.item()

        # set cam flag
        cam = action // param.CAM_ACT_OPTIONS

        # retrieve current pose of the selected camera
        campose = np.where(self._game.game_state() == cam+1)
        current_agent_pose = np.array([campose[0].item(), campose[1].item()])

        # update agent's new pose
        new_agent_pose = self._action_values.get(action)
        new_agent_pose[0] = new_agent_pose[0] + current_agent_pose[0]
        if new_agent_pose[1] == param.RotFlag.SAME:
          new_agent_pose[1] = current_agent_pose[1]

        response = self._game.move_cam(current_agent_pose,new_agent_pose, cam)

        # game transition handling
        if response == param.ActionResult.END_GAME:
            return timeStep.termination(self._game.game_state(),reward=10)
        elif response == param.ActionResult.ILLEGAL_MOVE:
            return timeStep.transition(self._game.game_state(), reward=-2, discount=1.0)

        return timeStep.transition(self._game.game_state(), reward=-0.3, discount=1.0)



def environment_setup(dogEnvironment):
  
  # setup training and evaluation environment
  train_env = tf_py_environment.TFPyEnvironment(dogEnvironment)
  eval_env = tf_py_environment.TFPyEnvironment(dogEnvironment)
  return train_env, eval_env