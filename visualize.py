## Visualization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from matplotlib.animation import ArtistAnimation 
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
				dirname = "./content/"+dirname+"/"
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
def metrics_viz(all_metrics, all_train_loss, savedir):

	funcs.mkdir_p(savedir)
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
	fig.savefig('{}/train-result.png'.format(savedir))

# show animation from images
def animation_viz(step, vizdir, epi):
	implots = []     
	fig = plt.figure(num=2,figsize=(20,6))
	fig.clf()
	for i in range(step):
	    img = mpimg.imread(vizdir+str(epi)+"/eval-step"+str(i)+".png")
	    imgplot = plt.imshow(img)
	    implots.append([imgplot])
	animation = ArtistAnimation(fig, implots, interval=33, blit=True, repeat=False)
	plt.ioff()
	return animation

# demonstrate performance during the evaluation phase
def observation_viz(observation, surgicaldata, vizdir, saveflag, step=None, act=None, epi=None):

	if step == None:
	    step = 0
	    print("------------------------\nStep: {}".format(step))
	    funcs.mkdir_p(vizdir)
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
	    print("\nReward: {} \n".format(epi))

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
	    
	if saveflag == True:
		# plot result
		ptLoc = np.array(surgicaldata.get('ptcloud_loc'))
		ptCol = np.array(surgicaldata.get('ptcloud_col')) / 256.0
		ptColMarked = np.array(surgicaldata.get('ptcloud_colmarked')) / 256.0
		loopstep, flipped = funcs.get_loopstep(step)

		fig = plt.figure(num=2,figsize=(20,6))
		fig.clf() # clear graph from previous iteratrion
		ax1 = fig.add_subplot(131,projection='3d')

		ax1.scatter(ptLoc[loopstep,:,0],ptLoc[loopstep,:,1],ptLoc[loopstep,:,2],c=ptCol[loopstep,:,:],s=5)

		domeX = np.arange(-1.0, 1.0, param.GRID_LENGTH)
		domeY = np.arange(-1.0, 1.0, param.GRID_LENGTH)
		domeX, domeY = np.meshgrid(domeX, domeY)
		breathdata = np.array(surgicaldata.get('breathing_val'))
		domeZ = funcs.dynamic_camZ_from_data(breathdata, domeX, domeY, loopstep)
		ax1.plot_surface(domeX*param.BELLY_EDGE_LENGTH, domeY*param.BELLY_EDGE_LENGTH, 
			domeZ*param.BELLY_EDGE_LENGTH, cmap='summer', alpha=0.2, linewidth=5, antialiased=False)

		for i in range(param.CAM_COUNT):
			coneX, coneY, coneZ = funcs.truncated_cam_cone(camposes[i,:],param.CONE_LENGTH,param.CONE_R1)
			ax1.plot_surface(coneX, coneY, coneZ, color='green', alpha=0.5, linewidth=0, antialiased=False)

		ax1.scatter(camposes[:,0],camposes[:,1],camposes[:,2],c='green',s=20)

		ax1.set_xlim([-param.BELLY_EDGE_LENGTH, param.BELLY_EDGE_LENGTH])
		ax1.set_ylim([-param.BELLY_EDGE_LENGTH, param.BELLY_EDGE_LENGTH])
		ax1.set_zlim([np.array(surgicaldata.get('ZLimits_all'))[0,0], param.BELLY_EDGE_LENGTH*1.5])

		ax1.set_xlabel('X (mm)')
		ax1.set_ylabel('Y (mm)')
		ax1.set_zlabel('Z (mm)')
		ax1.set_title('The Surgical Scene')
		ax1.view_init(elev=15., azim=-18.)

		ax2 = fig.add_subplot(132,projection='3d')

		for i in range(param.CAM_COUNT):
			coneX, coneY, coneZ = funcs.truncated_cam_cone(camposes[i,:],10*param.CONE_LENGTH,5*param.CONE_R1)
			ax2.plot_surface(coneX, coneY, coneZ, cmap='summer', alpha=0.2, linewidth=0, antialiased=False)

		ax2.scatter(camposes[:,0],camposes[:,1],camposes[:,2],c='green',s=20)

		ax2.set_xlim([-param.BELLY_EDGE_LENGTH, param.BELLY_EDGE_LENGTH])
		ax2.set_ylim([-param.BELLY_EDGE_LENGTH, param.BELLY_EDGE_LENGTH])
		ax2.set_zlim([np.array(surgicaldata.get('ZLimits_all'))[0,0], param.BELLY_EDGE_LENGTH*1.5])

		ax2.set_xlabel('X (mm)')
		ax2.set_ylabel('Y (mm)')
		ax2.set_zlabel('Z (mm)')
		ax2.set_title('Camera Poses')
		ax2.view_init(elev=15., azim=-18.)

		ax3 = fig.add_subplot(133,projection='3d')

		reconst_list = np.zeros(ptLoc.shape[1])
		vis_list = np.where(reconst_list==1)
		nonvis_list = np.where(reconst_list==0)

		ax3.scatter(ptLoc[loopstep,vis_list,0],ptLoc[loopstep,vis_list,1],
			ptLoc[loopstep,vis_list,2],c=ptColMarked[loopstep,vis_list,:],s=5)
		ax3.scatter(ptLoc[loopstep,nonvis_list,0],ptLoc[loopstep,nonvis_list,1],
			ptLoc[loopstep,nonvis_list,2],c='gray',s=5)

		ax3.set_xlim([-param.BELLY_EDGE_LENGTH, param.BELLY_EDGE_LENGTH])
		ax3.set_ylim([-param.BELLY_EDGE_LENGTH, param.BELLY_EDGE_LENGTH])
		ax3.set_zlim([np.array(surgicaldata.get('ZLimits_all'))[0,0], param.BELLY_EDGE_LENGTH*1.5])

		ax3.set_xlabel('X (mm)')
		ax3.set_ylabel('Y (mm)')
		ax3.set_zlabel('Z (mm)')
		ax3.set_title('Reconstructability')
		ax3.view_init(elev=15., azim=-18.)

		plt.ioff()
		fig.savefig(vizdir+"/eval-step"+str(step)+".png")

	return step+1
