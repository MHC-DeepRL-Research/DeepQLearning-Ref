## Train agent
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

from visualize import Progress_viz, metrics_viz

class Trainer():
	
    def __init__(self, train_env, n_iterations, visual_flag=True):
    	self._train_env = train_env
    	self._visual_flag = visual_flag
    	self._n_iterations = n_iterations
    	self.all_train_loss = []
    	self.all_metrics = []
		
    	self.create_agent()			# DQN agent Setup		
    	self.create_metrics()		# Metrics Setup
    	self.create_replay_buffer()	# Replay Buffer Setup
    	self.create_driver()			# Driver Setup

    def create_agent(self):
    	fc_layer_params = [32,64]

    	q_net = q_network.QNetwork(
		            self._train_env.observation_spec(),
		            self._train_env.action_spec(),
		            fc_layer_params = fc_layer_params
		        )

    	self._train_step = tf.Variable(0)
    	update_period = 2
    	optimizer = tf.keras.optimizers.Adam(lr=2.5e-3, epsilon=0.001)

    	epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
		                initial_learning_rate=1.0, 
		                decay_steps=250000 // update_period,
		                end_learning_rate=0.01)

    	self._agent = dqn_agent.DdqnAgent(
		        		self._train_env.time_step_spec(),
		        		self._train_env.action_spec(),
		        		q_network=q_net,
		        		optimizer=optimizer,
		        		target_update_period=2000,
		        		td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
		        		gamma=0.99,
		        		train_step_counter=self._train_step,
		        		epsilon_greedy=lambda: epsilon_fn(self._train_step))
    	self._agent.initialize()


    def create_metrics(self):
    	self._train_metrics = [tf_metrics.AverageReturnMetric(), 
						tf_metrics.AverageEpisodeLengthMetric()]


    def create_replay_buffer(self):
    	self._replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
					    data_spec=self._agent.collect_data_spec,
					    batch_size=self._train_env.batch_size,
					    max_length=1000000)

    	self._replay_buffer_observer = self._replay_buffer.add_batch


    def create_driver(self):
    	self._collect_driver = dynamic_step_driver.DynamicStepDriver(
					    self._train_env,
					    self._agent.collect_policy,
					    observers=[self._replay_buffer_observer] + self._train_metrics,
					    num_steps=7)

		
		
    def data_generation(self):
		# Collect trajectories using Random Policy
    	initial_collect_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(), self._train_env.action_spec())

    	init_driver = dynamic_step_driver.DynamicStepDriver(
		    self._train_env,
		    initial_collect_policy,
		    observers=[self._replay_buffer.add_batch, Progress_viz(3500)],
		    num_steps=3500)

    	final_time_step, final_policy_state = init_driver.run()

		# Verify collected trajectories
    	if self._visual_flag:
    		trajectories, buffer_info = self._replay_buffer.get_next(sample_batch_size=2, num_steps=10)
    		time_steps, action_steps, next_time_steps = trajectory.to_transition(trajectories)
    		print("trajectories._fields",trajectories._fields)
    		print("time_steps.observation.shape = ", time_steps.observation.shape)
		
		# Create Dataset from Replay Buffer
    	self._dataset = self._replay_buffer.as_dataset(sample_batch_size=200, num_steps=2, num_parallel_calls=3).prefetch(3)

    def make_common(self):
		## Run it under common function to make it faster
    	self._collect_driver.run = common.function(self._collect_driver.run)
    	self._agent.train = common.function(self._agent.train)

    def train_agent(self):
	    time_step = None
	    policy_state = self._agent.collect_policy.get_initial_state(self._train_env.batch_size)
	    iterator = iter(self._dataset)
	    
	    for iteration in range(self._n_iterations):
    		current_metrics = []
	        
    		time_step, policy_state = self._collect_driver.run(time_step, policy_state)
    		trajectories, buffer_info = next(iterator)
	        
    		train_loss = self._agent.train(trajectories)
    		self.all_train_loss.append(train_loss.loss.numpy())

    		for i in range(len(self._train_metrics)):
	            current_metrics.append(self._train_metrics[i].result().numpy())
	            
    		self.all_metrics.append(current_metrics)
	        
    		if iteration % 500 == 0:
	            print("\nIteration: {}, loss:{:.2f}".format(iteration, train_loss.loss.numpy()))
	            
	            for i in range(len(self._train_metrics)):
	                print('{}: {}'.format(self._train_metrics[i].name, self._train_metrics[i].result().numpy()))

	    if self._visual_flag:
    		metrics_viz(self.all_metrics, self.all_train_loss)

	    return self.all_metrics, self.all_train_loss