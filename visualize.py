## Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# UI menu at the start of the system
def menu_viz():
	response = ''
	response2 = ''
	response3 = ''
	dirname = ''
	while response != '1' and response != '2':
		if response != '':
			print("Your input is invalid. Try again!")
		print("\n\nWelcome to the Autonomous Multicamera Viewpoint Adjustment System.")
		print("------------------------------------------------------------------")
		print("\t1. Train a new model.")
		print("\t2. Evaluate on trained policy.")
		response = input("\nPlease select your intended action today. Your answer: ")

	print("You've selected: "+response+"\n\n")
	if response == '2':
		while (response2 != '1' and response2 != '2') or (response3 != 'y' and response3 != 'Y'):
			if response2 != '' and response2 != '1' and response2 != '2':
				print("\nYour input is invalid. Try again!\n\n")
			elif response3 != '':
				print("\nDo not want to use this trained policy? Let's try again.\n\n")

			print("Next we will load the trained policy.")
			print("------------------------------------------------------------------")
			print("\t1. Enter directory path of the saved model.")
			print("\t2. Use the default one: "+param.EVAL_POLICY_DIR)
			response2 = input("\nPlease select from the options. Your answer: ")

			print("You've selected: "+response2+"\n\n")
			if response2 == '1':
				dirname = input("Enter subfolder name (pick a subfolder under the content folder):")
			elif response2 == '2':
				dirname = param.EVAL_POLICY_DIR
			else:
				continue
			print("\nWill load policy from folder: "+dirname)
			response3 = input("Sound good (y/n)? Your answer:")
	print("\n\n")

	return int(response),dirname

# console output during the training phase
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
def observation_viz(observation, surgicaldata, step=None, act=None, epi=None):

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
	    numpy_obs = observation.numpy()[0].copy()
	    camposes = np.zeros((param.CAM_COUNT,param.CAM_STATE_DIM))
	    for i in range(param.CAM_COUNT):
	    	camposes[i,:] = funcs.get_cam_pose(numpy_obs.copy(), i)
	    	camposes[i,:] = funcs.revert_normalize_cam(camposes[i,:].copy())
	    	np.set_printoptions(precision=3, suppress=True)
	    	print("\tcam"+str(i+1)+" pose (x,y,z,Rotx,Roty): ",camposes[i,:])

	    toolposes = np.zeros((param.TOOL_COUNT,param.TOOL_STATE_DIM))
	    for i in range(param.TOOL_COUNT):
	    	toolposes[i,:] = funcs.get_tool_pose(numpy_obs.copy(), i)
	    	toolposes[i,:] = funcs.revert_normalize_tool(toolposes[i,:].copy())
	    	np.set_printoptions(precision=3, suppress=True)
	    	print("\ttool"+str(i+1)+" pose (x,y,z,Rotx,Roty,Velx,Vely): ",toolposes[i,:])
	    
	    print("\nReward: {} \n".format(epi))
	
	# plot result
	ptLoc = np.array(surgicaldata.get('ptcloud_loc'))
	ptCol = np.array(surgicaldata.get('ptcloud_col')) / 256.0
	ptColMarked = np.array(surgicaldata.get('ptcloud_colmarked')) / 256.0
	loopstep, flipped = funcs.get_loopstep(step)

	fig = plt.figure(2)
	ax = fig.add_subplot(111,projection='3d')

	ax.scatter(ptLoc[loopstep,:,0],ptLoc[loopstep,:,1],ptLoc[loopstep,:,2],c=ptCol[loopstep,:,:])
	ax.set_xlim(np.array(surgicaldata.get('XLimits_all'))[0])
	ax.set_ylim(np.array(surgicaldata.get('YLimits_all'))[0])
	ax.set_zlim(np.array(surgicaldata.get('ZLimits_all'))[0])

	ax.set_xlabel('X (mm)')
	ax.set_ylabel('Y (mm)')
	ax.set_zlabel('Z (mm)')
	
	# TODO: 
	# (1) the camposes and toolposes ready for use. --> dome
	# (2) save animation
	# (3) color marked 
	
	plt.show()
	
	return step+1
