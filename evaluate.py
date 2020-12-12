## Evaluation
import os
import param
import scipy.io
import numpy as np
import tensorflow as tf
from visualize import observation_viz, animation_viz
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tf_agents.utils import common
from tf_agents.policies import policy_saver
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class Evaluator():
    
    def __init__(self, eval_env, savedir, agent=None, replay_buffer=None, train_step=None):

        self._agent = agent
        self._num_episodes = param.EVAL_EPISODE
        self._visual_flag = param.VIZ_FLAG
        self._surgicaldata = scipy.io.loadmat(param.VISUALIZE_FILE)
        #reset eval environment
        self._eval_env = eval_env
        self._eval_env.reset() 
        # for model saving and visualization only
        self._replay_buffer = replay_buffer
        self._train_step = train_step
        self._ani = []
        # get folder directory
        self._savedir = savedir
        self._vizdir = os.path.join(self._savedir, 'visual/epi')

    def evaluate_agent(self):

        if  self._agent != None:
            trained_policy = self._agent.policy
            saveflag = True
        else:
            policy_dir = os.path.join(self._savedir, 'policy')
            trained_policy = tf.saved_model.load(policy_dir)
            saveflag = False

        total_return = 0.0
        for epi in range(self._num_episodes):
            # initialize variables
            episode_return = 0.0
            time_step = self._eval_env.reset()
            # plot the game
            if self._visual_flag:
                step = observation_viz(time_step.observation, self._surgicaldata.copy(), self._vizdir+str(epi),saveflag)
            else:
                step = 0
            
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
                    step = observation_viz(time_step.observation, self._surgicaldata.copy(), self._vizdir+str(epi), saveflag,
                        step, action_step.action, episode_return)   
                else:
                    step = step + 1
                # max iteration timeout         
                if step == param.EVAL_MAX_ITER:
                    print("Evaluation ended on max allowed iterations of ", step)
            # accumulate total reward
            total_return += episode_return
            #show animation
            if self._visual_flag:
                self._ani.append(animation_viz(step, self._vizdir, epi))
                if saveflag == False:
                    plt.show()

        # calculate average performance
        self._avg_return = total_return / self._num_episodes
        self._avg_return = self._avg_return.numpy()[0]
        print("Evaluation performance: ", self._avg_return)

    def save_model(self):
        # create directory to save checkpoint
        checkpoint_dir = os.path.join(self._savedir, 'checkpoint')
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
        policy_dir = os.path.join(self._savedir, 'policy')
        tf_policy_saver = policy_saver.PolicySaver(self._agent.policy)
        # save the policy
        tf_policy_saver.save(policy_dir)
        print("model saved.")
        
        if self._visual_flag:
            # save the animation
            for epi in range(self._num_episodes):
                f = self._vizdir+str(epi)+"/eval-animation.mp4"
                writervideo = FFMpegWriter(fps=33) 
                self._ani[epi].save(f, writer=writervideo)
                plt.show()
            # try loading saved policy
            loaded_policy = tf.saved_model.load(policy_dir)
            eval_timestep = self._eval_env.reset()
            loaded_action = loaded_policy.action(eval_timestep)
            print("example policy: ", loaded_action)