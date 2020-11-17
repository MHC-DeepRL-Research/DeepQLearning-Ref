## Evaluation
import os
import datetime
import param
import numpy as np
import tensorflow as tf
from visualize import observation_viz
from tf_agents.utils import common
from tf_agents.policies import policy_saver


class Evaluator():
    
    def __init__(self, eval_env, agent, replay_buffer, train_step, episodes=10, visual_flag=False):
        self._eval_env = eval_env
        self._eval_env.reset()      #reset eval environment
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._num_episodes = episodes
        self._visual_flag=visual_flag
        self._train_step = train_step


    def evaluate_agent(self):

        total_return = 0.0
        
        for _ in range(self._num_episodes):
            time_step = self._eval_env.reset()
            episode_return = 0.0
            
            if self._visual_flag:
                step = observation_viz(time_step.observation)

            counter = 0
            while not time_step.is_last() and counter < param.EVAL_MAX_ITER:
                action_step = self._agent.policy.action(time_step)
                time_step = self._eval_env.step(action_step.action)
                episode_return += time_step.reward
                counter += 1
                if self._visual_flag:
                    step = observation_viz(time_step.observation, step, action_step.action, episode_return)            
                if counter == param.EVAL_MAX_ITER:
                    print("Evaluation ended on max allowed iterations of ", counter)
            total_return += episode_return

        self._avg_return = total_return / self._num_episodes
        self._avg_return = self._avg_return.numpy()[0]

        if self._visual_flag:
            print("Evaluation performance: ", self._avg_return)

    def save_model(self):
        tempdir = "./content/" + str(datetime.datetime.now()) +"/"
        checkpoint_dir = os.path.join(tempdir, 'checkpoint')
        train_checkpointer = common.Checkpointer(
            ckpt_dir = checkpoint_dir,
            max_to_keep = 1,
            agent = self._agent,
            policy = self._agent.policy,
            replay_buffer = self._replay_buffer,
            global_step = self._train_step
        )

        train_checkpointer.save(self._train_step)
        #!zip -r saved_checkpoint.zip /content/checkpoint

        policy_dir = os.path.join(tempdir, 'policy')
        tf_policy_saver = policy_saver.PolicySaver(self._agent.policy)

        tf_policy_saver.save(policy_dir)
        #!zip -r saved_policy.zip /content/policy/

        if self._visual_flag:
            loaded_policy = tf.saved_model.load(tempdir+"policy")

            eval_timestep = self._eval_env.reset()
            loaded_action = loaded_policy.action(eval_timestep)
            print("model saved.")
            print("example policy: ", loaded_action)