## Setup environment
import param
import funcs
import numpy as np
import scipy.io

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as timeStep

class CamAdventureGame():

    def __init__(self):
        self._cam_states = param.CAM_COUNT * param.CAM_STATE_DIM 
        self._tool_states = param.TOOL_COUNT * param.TOOL_STATE_DIM
        self._n_states = self._cam_states + self._tool_states 

        self._surgicaldata = scipy.io.loadmat(param.ANIMATION_FILE)

        self._breathdata =  np.array(self._surgicaldata.get('breathing_val'))
        self._tooldata = np.array(self._surgicaldata.get('toolinfo'))
        self.reset()
    

    def reset(self):
        # initialize states
        self._state = np.zeros((self._n_states,),dtype=np.float32)

        # place cameras in init poses
        for i in range(param.CAM_COUNT):
          angle = 2 * np.pi * i / param.CAM_COUNT
          camX = (param.CAM_INIT_DIST * np.sin(angle)//param.GRID_LENGTH) * param.GRID_LENGTH 
          camY = (param.CAM_INIT_DIST * np.cos(angle)//param.GRID_LENGTH) * param.GRID_LENGTH 
          self._state[param.CAM_STATE_DIM*i+0] = camX
          self._state[param.CAM_STATE_DIM*i+1] = camY

        # apply environment changes
        self.env_dynamic_change(first_pass=True)


    def env_dynamic_change(self, first_pass=False):
        # set timer
        if first_pass is True:
          self._step_counter = 0
        else:
          self._step_counter += 1 

        # get dynamic tool information
        for i in range(param.TOOL_COUNT):
          toolinfo = funcs.dynamic_toolinfo_from_data(self._tooldata.copy(), self._step_counter)
          for j in range(param.TOOL_STATE_DIM):
            self._state[param.CAM_STATE_DIM*param.CAM_COUNT+j] = toolinfo[j]

        # apply abdomenal changes due to breathing
        for i in range(param.CAM_COUNT):
          camX = self._state[param.CAM_STATE_DIM*i+0] 
          camY = self._state[param.CAM_STATE_DIM*i+1] 
          camZ = funcs.dynamic_camZ_from_data(self._breathdata.copy(), camX, camY, self._step_counter)
          self._state[param.CAM_STATE_DIM*i+2] = camZ
          self._state[param.CAM_STATE_DIM*i+3] = funcs.calculate_angle(camZ,camX,toolinfo[2],toolinfo[0])
          self._state[param.CAM_STATE_DIM*i+4] = funcs.calculate_angle(camZ,camY,toolinfo[2],toolinfo[1])


    def move_cam(self, curr_pose, next_pose, cam):
        # rule0: encode environment change and update timer
        if self._step_counter > param.EVAL_MAX_ITER:
          return param.ActionResult.END_GAME

        # rule1: out of allowable border conditions
        for i in range(param.CAM_STATE_DIM):
          if next_pose[i] < param.OBS_SPEC_MIN or next_pose[i] > param.OBS_SPEC_MAX:
            print("[WARNING]: illegal move. camera moved out of border. The system will keep running.")
            self.env_dynamic_change()
            return param.ActionResult.ILLEGAL_MOVE

        # rule2: run into other cameras conditions
        for i in range(param.CAM_COUNT):
          if i != cam:
            i_pose = funcs.get_cam_pose(self.game_state(),i)
            if np.sum(i_pose[0:3] - next_pose[0:3]) == 0:
              print("[WARNING]: illegal move. camera collided with another camera. The system will keep running.")
              self.env_dynamic_change()
              return param.ActionResult.ILLEGAL_MOVE

        # set next cam pose
        assert cam >= 0 and  cam < param.CAM_COUNT
        assert (curr_pose == self._state[param.CAM_STATE_DIM*cam:param.CAM_STATE_DIM*(cam+1)]).all()
        self._state[param.CAM_STATE_DIM*cam:param.CAM_STATE_DIM*(cam+1)] = next_pose
        self.env_dynamic_change()
        return param.ActionResult.VALID_MOVE

    def game_data(self):
        return self._surgicaldata.copy()
        
    def game_state(self):
        return self._state.copy()

    def game_step_counter(self):
        return self._step_counter


class CamAdventureEnvironment(py_environment.PyEnvironment):

    def __init__(self, game):

        # set game
        self._game = game

        # set action range
        self.action_count = param.CAM_COUNT * param.MOVE_OPTIONS
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.action_count-1, name='action')

        # set observation range
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._game._n_states,), dtype=np.float32, 
            minimum=param.OBS_SPEC_MIN, maximum=param.OBS_SPEC_MAX, name='observation')

        # create action dictionary
        self.create_action_dict()

        # make sure the environment is okay
        utils.validate_py_environment(self, episodes=5)

    def create_action_dict(self):
        # action table: along the diagnal are attack moves
        self._action_values = {} # the action table

        for i in range(param.CAM_COUNT):
          self._action_values[i*param.MOVE_OPTIONS + 0] = [param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.SAME]
          self._action_values[i*param.MOVE_OPTIONS + 1] = [param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.POS ,param.Move.SAME]
          self._action_values[i*param.MOVE_OPTIONS + 2] = [param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.NEG ,param.Move.SAME]
          self._action_values[i*param.MOVE_OPTIONS + 3] = [param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.POS ]
          self._action_values[i*param.MOVE_OPTIONS + 4] = [param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.NEG ]
          self._action_values[i*param.MOVE_OPTIONS + 5] = [param.Move.POS ,param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.SAME]
          self._action_values[i*param.MOVE_OPTIONS + 6] = [param.Move.NEG ,param.Move.SAME,param.Move.SAME,param.Move.SAME,param.Move.SAME]
          self._action_values[i*param.MOVE_OPTIONS + 7] = [param.Move.SAME,param.Move.POS ,param.Move.SAME,param.Move.SAME,param.Move.SAME]
          self._action_values[i*param.MOVE_OPTIONS + 8] = [param.Move.SAME,param.Move.NEG ,param.Move.SAME,param.Move.SAME,param.Move.SAME]
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_game_data(self):
        return self._game.game_data()

    def _reset(self):
        self._game.reset()
        return timeStep.restart(self._game.game_state())
  
    def _step(self, action):    
        # load selected action values
        action = action.item()
        action_vals = self._action_values.get(action)

        # identify the selected camera index
        cam = action // param.MOVE_OPTIONS

        # retrieve current pose of the selected camera
        curr_cam_pose = funcs.get_cam_pose(self._game.game_state(),cam)

        # update agent new pose
        next_cam_pose = curr_cam_pose.copy()
        for i in range(param.CAM_STATE_DIM):
          next_cam_pose[i] = next_cam_pose[i] + action_vals[i] * param.GRID_LENGTH

        assert (next_cam_pose != curr_cam_pose).any() or (action % param.MOVE_OPTIONS == 0)

        response = self._game.move_cam(curr_cam_pose.copy(), next_cam_pose.copy(), cam)

        # game transition handling
        action_reward = funcs.calculate_action_reward(response)
        reconst_reward = funcs.calculate_reconst_reward(self.get_game_data(),self._game.game_state(),self._game.game_step_counter())

        if response == param.ActionResult.END_GAME:
            feedback = timeStep.termination(self._game.game_state(),reward=action_reward+reconst_reward)
            self.reset()
        else:
            feedback = timeStep.transition(self._game.game_state(), reward=action_reward+reconst_reward, discount=param.QVALUE_DISCOUNT)
        return feedback

def environment_setup(camEnvironment):
  # setup training and evaluation environment
  train_env = tf_py_environment.TFPyEnvironment(camEnvironment)
  eval_env = tf_py_environment.TFPyEnvironment(camEnvironment)
  return train_env, eval_env