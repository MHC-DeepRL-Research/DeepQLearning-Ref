#!/usr/bin/env python
from enum import Enum
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as timeStep

from tf_agents.policies import random_tf_policy

from tf_agents.metrics import tf_py_metric
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import py_metric
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.policies import policy_saver


def column(matrix, i):
    return [row[i] for row in matrix]

## Setup environment
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



dogEnvironemt = DogAdventureEnvironment(DogAdventure())
utils.validate_py_environment(dogEnvironemt, episodes=5)


train_env = tf_py_environment.TFPyEnvironment(dogEnvironemt)
eval_env = tf_py_environment.TFPyEnvironment(dogEnvironemt)


## DQN Setup

fc_layer_params = [32,64]

q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params = fc_layer_params
        )

train_step = tf.Variable(0)
update_period = 2
optimizer = tf.keras.optimizers.Adam(lr=2.5e-3, epsilon=0.001)

epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1.0, 
                decay_steps=250000 // update_period,
                end_learning_rate=0.01)

agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=0.99,
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step))

agent.initialize()


## Replay buffer Setup

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=1000000)

replay_buffer_observer = replay_buffer.add_batch


## Metrics Setup

train_metrics = [tf_metrics.AverageReturnMetric(), tf_metrics.AverageEpisodeLengthMetric()]

## Driver Setup
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=7)

## Collect trajectories using Random Policy

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


initial_collect_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

init_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(35000)],
    num_steps=35000)

final_time_step, final_policy_state = init_driver.run()

## Verify collected trajectories
trajectories, buffer_info = replay_buffer.get_next(sample_batch_size=2, num_steps=10)

trajectories._fields



time_steps, action_steps, next_time_steps = trajectory.to_transition(trajectories)
time_steps.observation.shape


## Create Dataset from Replay Buffer

dataset = replay_buffer.as_dataset(sample_batch_size=200, num_steps=2, num_parallel_calls=3).prefetch(3)

## Run it under common function to make it faster
collect_driver.run = common.function(collect_driver.run)
agent.train = common.function(agent.train)

## Train

all_train_loss = []
all_metrics = []

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    iterator = iter(dataset)
    
    for iteration in range(n_iterations):
        current_metrics = []
        
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        
        train_loss = agent.train(trajectories)
        all_train_loss.append(train_loss.loss.numpy())

        for i in range(len(train_metrics)):
            current_metrics.append(train_metrics[i].result().numpy())
            
        all_metrics.append(current_metrics)
        
        if iteration % 500 == 0:
            print("\nIteration: {}, loss:{:.2f}".format(iteration, train_loss.loss.numpy()))
            
            for i in range(len(train_metrics)):
                print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))

train_agent(150000)

avg_return_trained = column(all_metrics,0)
avg_ep_trained = column(all_metrics,1)


fig, axs = plt.subplots(1, 2, figsize=(15,15))

axs[0].plot(range(len(avg_return_trained)), avg_return_trained)
axs[0].set_title('Average Return')

axs[1].plot(range(len(avg_ep_trained)), avg_ep_trained, 'tab:orange')
axs[1].set_title('Average Episode Length')

for ax in axs.flat:
    ax.set(xlabel='Number of Iterations', ylabel='Metric Value')

## Evaluate

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        
        total_return += episode_return

    avg_return = total_return / num_episodes
    
    return avg_return.numpy()[0]

# Reset the train step
agent.train_step_counter.assign(0)

#reset eval environment
eval_env.reset()

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, 5)


avg_return

## Visualize Episode

def observation_viz(observation):
    numpy_obs = observation.numpy()[0]
    string_obs = np.array(np.reshape(numpy_obs, (-1, 6)), dtype=np.unicode_)
    if string_obs[5][5] != "1":
      string_obs[5][5] = "❌"
    string_obs = np.where(string_obs=="1","🐕", string_obs) 
    string_obs = np.where(string_obs=="2","🤖", string_obs)
    string_obs = np.where(string_obs=="3","🦴", string_obs)
    string_obs = np.where(string_obs=="0","⬚", string_obs)
    string_obs = np.where(string_obs=="4","🐈", string_obs)
    observe_2d = pd.DataFrame(string_obs)
    observe_2d.columns = [''] * len(observe_2d.columns)
    observe_2d = observe_2d.to_string(index=False)
    print("\n{}\n".format(observe_2d))


def compute_viz(environment, policy, num_episodes=10):

    total_return = 0.0
    
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        step = 0
        print("Step: 0")
        observation_viz(time_step.observation)
        while not time_step.is_last():
            step += 1
            print("---\nStep: {}".format(step))
            action_step = policy.action(time_step)
            print("Action taken: {}".format(action_step.action))
            time_step = environment.step(action_step.action)
            observation_viz(time_step.observation)
            episode_return += time_step.reward
            print("Reward: {} \n".format(episode_return))
        
        total_return += episode_return

    avg_return = total_return / num_episodes
    
    return avg_return.numpy()[0]


# Reset the train step
agent.train_step_counter.assign(0)

#reset eval environment
eval_env.reset()

# Evaluate the agent's policy once before training.
avg_return = compute_viz(eval_env, agent.policy, 1)

## Checkpoint Saver
import os

tempdir = "./content/"

checkpoint_dir = os.path.join(tempdir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step
)

train_checkpointer.save(train_step)

#!zip -r saved_checkpoint.zip /content/checkpoint

policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

tf_policy_saver.save(policy_dir)
#!zip -r saved_policy.zip /content/policy/

loaded_policy = tf.saved_model.load("./content/policy")

eval_timestep = eval_env.reset()
loaded_action = loaded_policy.action(eval_timestep)
print(loaded_action)
