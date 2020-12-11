## Train agent
import os
import param
import time
import tensorflow as tf
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

from visualize import Progress_viz, metrics_viz, console_viz
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class Trainer():
	
    def __init__(self, train_env):
        self._train_env = train_env
        self._visual_flag = param.VIZ_FLAG
        self._n_iterations = param.TRAIN_ITER

        # create save folders
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self._savedir = "./content/" + timestr
        self._viz_dir = os.path.join(self._savedir,"visual")

    	# two things returned from train function
        self.all_train_loss = []
        self.all_metrics = []
		
        self.create_agent()			# DQN agent Setup		
        self.create_replay_buffer()	# Replay Buffer Setup
        self.create_metrics()		# Metrics Setup
        self.create_driver()		# Driver Setup

    def create_agent(self):
    	# a deep neural network to learn Q(s,a)
    	q_net = q_network.QNetwork(
		    self._train_env.observation_spec(),
		    self._train_env.action_spec(),
            #conv_layer_params= param.QNET_CONV_LAYERS,
		    fc_layer_params = param.QNET_FC_LAYERS)

    	# optional copunter that increments every time the train op is run
    	self._train_step = tf.Variable(0)

    	# an adaptive learning rate for gradient descent
    	optimizer = tf.keras.optimizers.Adam(lr = param.ADAM_LR, epsilon=param.ADAM_EPSILON)

    	# probability of exploration as a function of time steps
    	epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
		    initial_learning_rate=param.DECAY_LR_INIT, 
		    decay_steps= param.DECAY_STEPS // param.DECAY_UPDATE_PERIOD,
		    end_learning_rate=param.DECAY_LR_END)

    	# create the double deep Q learning network agent
    	self._agent = dqn_agent.DdqnAgent(
		    self._train_env.time_step_spec(),
		    self._train_env.action_spec(),
		    q_network=q_net,
		    optimizer=optimizer,
		    # period for soft update of the target networks
		    target_update_period= param.AGENT_UPDATE_PERIOD,
		    # loss function for gradient descent
		    td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
		    # a discount factor for future rewards. 
		    gamma=param.AGENT_GAMMA,
		    # optional copunter that increments every time the train op is run
		    train_step_counter=self._train_step,
		    epsilon_greedy=lambda: epsilon_fn(self._train_step))
    	self._agent.initialize()

    def create_replay_buffer(self):
    	# a batched replay buffer which can be sampled uniformly during training
    	self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    		# the type of data you are collecting
			data_spec=self._agent.collect_data_spec,
			batch_size=self._train_env.batch_size,
			max_length=param.BUFFER_LENGTH)

    	# adds a batch of items to replay_buffer - part of the routine update in dynamic_step_driver
    	self._replay_buffer_observer = self._replay_buffer.add_batch

    def create_metrics(self):
    	# for observations only - will not affect training result
    	self._train_metrics = [tf_metrics.AverageReturnMetric(), 
						tf_metrics.AverageEpisodeLengthMetric()]

    def create_driver(self):
    	# a driver that simulates N steps in an environment 
    	self._collect_driver = dynamic_step_driver.DynamicStepDriver(
			self._train_env,
			#  a policy that can be used to collect data from the environment
			self._agent.collect_policy,
			# a list of observers that are updated after every step in the environment
			observers=[self._replay_buffer_observer] + self._train_metrics,
			# the number of steps simulated - N steps
			num_steps=param.DRIVER_STEPS)
		
	# Collect trajectories using random policy
    def data_generation(self):
		# set up random policy
    	initial_collect_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(), self._train_env.action_spec())
    	# set up a driver that with random policy to collect data
    	init_driver = dynamic_step_driver.DynamicStepDriver(
		    self._train_env,
		    #  a random policy that can be used to collect data from the environment
		    initial_collect_policy,
		    # a list of observers that are updated after every step in the environment
		    observers=[self._replay_buffer_observer, Progress_viz(param.DATASET_STEPS)],
		    # the number of steps in the dataset
		    num_steps=param.DATASET_STEPS)

    	# recording the sequence of state transitions and results in observers
    	final_time_step, final_policy_state = init_driver.run()

		# Verify collected trajectories (optional)
    	if self._visual_flag:
    		trajectories, buffer_info = self._replay_buffer.get_next(sample_batch_size=2, num_steps=10)
    		time_steps, action_steps, next_time_steps = trajectory.to_transition(trajectories)
    		print("trajectories._fields",trajectories._fields)
    		print("time_steps.observation.shape = ", time_steps.observation.shape)
		
		# Create Dataset from Replay Buffer
    	self._dataset = self._replay_buffer.as_dataset(sample_batch_size=param.DATASET_BATCH, 
    		num_steps=param.DATASET_BUFFER_STEP, num_parallel_calls=param.DATASET_PARALLEL).prefetch(param.DATASET_PREFETCH)

    # Run it under common function to make it faster
    def make_common(self):
    	self._collect_driver.run = common.function(self._collect_driver.run)
    	self._agent.train = common.function(self._agent.train)

    # return the save directory for aggregating all results in the same folder
    def get_savedir(self):
        return self._savedir

    def train_agent(self):
    	# make dataset iterable so we can later call next() to retrieve each batch
	    iterator = iter(self._dataset) 
	    # initialize state and timestep
	    time_step = None
	    policy_state = self._agent.collect_policy.get_initial_state(self._train_env.batch_size)
	    
	    # start training for n_iterations
	    for iteration in range(self._n_iterations):
    		current_metrics = []
	        # takes steps in the environment using the policy while updating observers.
    		time_step, policy_state = self._collect_driver.run(time_step, policy_state)
    		trajectories, buffer_info = next(iterator)
	        
	        # train the agent: update the policy by looking at the dataset trajectories
    		train_loss = self._agent.train(trajectories)
    		# save the iteration loss
    		self.all_train_loss.append(train_loss.loss.numpy())

    		# save the iteration performance metrics
    		for i in range(len(self._train_metrics)):
	            current_metrics.append(self._train_metrics[i].result().numpy())
    		self.all_metrics.append(current_metrics)
	        
    		# print console output
    		console_viz(iteration, train_loss.loss.numpy(), self._train_metrics)
    		

	    # show training result in plots
	    if self._visual_flag:
    		metrics_viz(self.all_metrics, self.all_train_loss, self._viz_dir)

    	# all metrics: focuses on how well the collect driver navigates the environment with learned policy
    	# all train loss: focuses on how well the network predicts the reward, despite the action taken
	    return self.all_metrics, self.all_train_loss