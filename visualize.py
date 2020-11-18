## Visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
	    if act < 8:
	        string_act = " 🐕 "
	    else:
	    	string_act = " 🐈 "
	    if act == 0 or act == 8:
	    	string_act = string_act + " moves left."
	    elif act == 1 or act == 9:
	    	string_act = string_act + " moves right."
	    elif act == 2 or act == 10:
	    	string_act = string_act + " moves down."
	    elif act == 3 or act == 11:
	    	string_act = string_act + " moves up."
	    elif act == 4 or act == 12:
	    	string_act = string_act + " attacks top right."
	    elif act == 5 or act == 13:
	    	string_act = string_act + " attacks top left."
	    elif act == 6 or act == 14:
	    	string_act = string_act + " attacks bottom right."
	    else:
	    	string_act = string_act + " attacks bottom left."

	    print("Action taken: {}".format(act)+string_act)
	    print("Reward: {} \n".format(epi))

	numpy_obs = observation.numpy()[0]
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

	return step+1
