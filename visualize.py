## Visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import funcs
import param

# demonstrate the progress ogf generationg dataset
class Progress_viz:

    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


def console_viz(ite, loss, iter_metrics):
	if ite % 500 == 0:
		print("\n------------------------")
		print("Iteration: {}".format(ite))
		print("Reward Prediction Loss:  {:.2f}".format(ite, loss))
		print("Explore and exploit:")
		for i in range(len(iter_metrics)):
		    print('\t{}: {}'.format(iter_metrics[i].name, iter_metrics[i].result().numpy()))

# demonstrate performance during the training phase
def metrics_viz(all_metrics, all_train_loss):

	avg_return_trained = [row[0] for row in all_metrics]
	avg_ep_trained = [row[1] for row in all_metrics]


	fig, axs = plt.subplots(1, 2, figsize=(15,15))

	axs[0].plot(range(len(avg_return_trained)), avg_return_trained)
	axs[0].set_title('Average Return')

	axs[1].plot(range(len(avg_ep_trained)), avg_ep_trained, 'tab:orange')
	axs[1].set_title('Average Episode Length')

	for ax in axs.flat:
	    ax.set(xlabel='Number of Iterations', ylabel='Metric Value')

	plt.show()

# demonstrate performance during the evaluation phase
def observation_viz(observation, step=None, act=None, epi=None):

	if step == None:
	    step = 0
	    print("------------------------\nStep: {}".format(step))
	else:
	    print("------------------------\nStep: {}".format(step))
	    cam = act // param.MOVE_OPTIONS


	    string_act = " (cam"+str(cam.numpy()[0]+1)
	    if act % param.MOVE_OPTIONS == 0:
	    	string_act = string_act + " stays put.)"
	    elif act % param.MOVE_OPTIONS == 1:
	    	string_act = string_act + " rotates along positive X direction.)\n"
	    elif act % param.MOVE_OPTIONS == 2:
	    	string_act = string_act + " rotates along negative X direction.)\n"
	    elif act % param.MOVE_OPTIONS == 3:
	    	string_act = string_act + " rotates along positive Y direction.)\n"
	    elif act % param.MOVE_OPTIONS == 4:
	    	string_act = string_act + " rotates along negative Y direction.)\n"
	    elif act % param.MOVE_OPTIONS == 5:
	    	string_act = string_act + " translates along positive X direction.)\n"
	    elif act % param.MOVE_OPTIONS == 6:
	    	string_act = string_act + " translates along negative X direction.)\n"
	    elif act % param.MOVE_OPTIONS == 7:
	    	string_act = string_act + " translates along positive Y direction.)\n"
	    elif act % param.MOVE_OPTIONS == 8:
	    	string_act = string_act + " translates along negative Y direction.)\n"
	    print("Action taken: {}".format(act.numpy()[0])+string_act)

	    # summary of the game state
	    print("Observations:")
	    numpy_obs = observation.numpy()[0]
	    for i in range(param.CAM_COUNT):
	    	print("\tcam"+str(i+1)+" pose (x,y,z,Rotx,Roty): ",funcs.get_cam_pose(numpy_obs, i))

	    for i in range(param.TOOL_COUNT):
	    	print("\ttool"+str(i+1)+" pose (x,y,z,Rotx,Roty,Velx,Vely): ",funcs.get_tool_pose(numpy_obs, i))
	    
	    print("\nReward: {} \n".format(epi))

	'''
	
	string_obs = np.array(np.reshape(numpy_obs, (-1, 6)), dtype=np.unicode_)
	if string_obs[5][5] != "1":
	    string_obs[5][5] = "❌"
	string_obs = np.where(string_obs=="1","🐕", string_obs) 
	string_obs = np.where(string_obs=="2","🤖", string_obs)
	string_obs = np.where(string_obs=="3","🍖", string_obs)
	string_obs = np.where(string_obs=="0","⬚", string_obs)
	string_obs = np.where(string_obs=="4","🐈", string_obs)
	observe_2d = pd.DataFrame(string_obs)
	observe_2d.columns = [''] * len(observe_2d.columns)
	observe_2d = observe_2d.to_string(index=False)
	print("\n{}\n".format(observe_2d))
	'''

	return step+1
