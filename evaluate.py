## Evaluation
import os
import datetime
import param
import scipy.io
import numpy as np
import tensorflow as tf
from visualize import observation_viz
from tf_agents.utils import common
from tf_agents.policies import policy_saver
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class Evaluator():
    
    def __init__(self, eval_env, agent=None, replay_buffer=None, train_step=None):
        self._eval_env = eval_env
        self._eval_env.reset()      #reset eval environment
        self._agent = agent
        self._num_episodes = param.EVAL_EPISODE
        self._visual_flag = param.VIZ_FLAG
        self._surgicaldata = scipy.io.loadmat(param.ANIMATION_FILE)
        # for model saving only
        self._replay_buffer = replay_buffer
        self._train_step = train_step

    def evaluate_agent(self, dirname=None):

        if dirname == None:
            trained_policy = self._agent.policy
        else:
            policy_dir = "./content/" + dirname +"/policy"
            trained_policy = tf.saved_model.load(policy_dir)

        total_return = 0.0
        
        for _ in range(self._num_episodes):
            # initialize variables
            episode_return = 0.0
            time_step = self._eval_env.reset()
            # plot the game
            if self._visual_flag:
                step = observation_viz(time_step.observation, self._surgicaldata.copy())
            
            # start eval game
            while not time_step.is_last() and step < param.EVAL_MAX_ITER:
                # calculate action based on current state and policy
                action_step = trained_policy.action(time_step)
                # transition to next state based on chosen action
                time_step = self._eval_env.step(action_step.action)
                # accumulate episode reward
                episode_return += time_step.reward
                # plot the game
                if self._visual_flag:
                    step = observation_viz(time_step.observation, self._surgicaldata.copy(), step, action_step.action, episode_return)   
                # max iteration timeout         
                if step == param.EVAL_MAX_ITER:
                    print("Evaluation ended on max allowed iterations of ", step)
            # accumulate total reward
            total_return += episode_return
        # calculate average performance
        self._avg_return = total_return / self._num_episodes
        self._avg_return = self._avg_return.numpy()[0]
        print("Evaluation performance: ", self._avg_return)

    def save_model(self):
        # get folder directory
        tempdir = "./content/" + str(datetime.datetime.now()) +"/"

        # create directory to save checkpoint
        checkpoint_dir = os.path.join(tempdir, 'checkpoint')
        train_checkpointer = common.Checkpointer(
            ckpt_dir = checkpoint_dir,
            max_to_keep = 1,
            agent = self._agent,
            policy = self._agent.policy,
            replay_buffer = self._replay_buffer,
            global_step = self._train_step)
        # save the checkpoint
        train_checkpointer.save(self._train_step)

        # create directory to save policy
        policy_dir = os.path.join(tempdir, 'policy')
        tf_policy_saver = policy_saver.PolicySaver(self._agent.policy)
        # save the policy
        tf_policy_saver.save(policy_dir)

        # try loading saved policy
        if self._visual_flag:
            loaded_policy = tf.saved_model.load(tempdir+"policy")
            eval_timestep = self._eval_env.reset()
            loaded_action = loaded_policy.action(eval_timestep)
            print("model saved.")
            print("example policy: ", loaded_action)