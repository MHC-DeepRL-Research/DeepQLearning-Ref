import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import param


def get_loopstep(timestep):
	# repeat animation when timestep larger than animation length
	loopstep = timestep % (2*param.ANIMATION_LENGTH)
	if loopstep >= param.ANIMATION_LENGTH:
		loopstep = 2*param.ANIMATION_LENGTH - loopstep -1
		flipped = True
	else:
		flipped = False
	return loopstep, flipped


def breath_deform_factor(breathdata, loopstep):
	# decide breathing factor based on timestep
	assert loopstep >= 0 and loopstep < param.ANIMATION_LENGTH
	deform_factor = breathdata[0,loopstep]
	deform_factor = deform_factor/38+1
	return deform_factor

def get_dynamic_camZ(breathdata, normX, normY, timestep):
	# find camera height based on timestep
	loopstep, flipped = get_loopstep(timestep)
	deform_factor = breath_deform_factor(breathdata, loopstep)
	normZ = deform_factor * np.sqrt(- normX*normX - normY*normY + 2)
	return normZ

def get_dynamic_toolinfo(tooldata, timestep):
	loopstep, flipped = get_loopstep(timestep)
	toolinfo = tooldata[loopstep,:]
	
	for i in range(param.TOOL_STATE_DIM):
		if i == 3 or i == 4:
			toolinfo[i] = toolinfo[i]*2 / np.pi      		    # normalize angles
		else:
			toolinfo[i] = toolinfo[i] / param.BELLY_EDGE_LENGTH	# normalize pos and vel

	if flipped is True:
		toolinfo[5] = - toolinfo[5]                             # flip velocity direction
		toolinfo[6] = - toolinfo[6]   
	return toolinfo


def calculate_angle(camZ, camP, toolZ, toolP):
	# calculate the rotation angle in direction P
	angle = np.arctan2(camP - toolP, camZ - toolZ)
	norm_angle = angle*2 / np.pi
	assert norm_angle >= -1.0 and norm_angle <= 1.0

	return norm_angle

def calculate_action_reward(response):
	# calculate part of the reward value based on action consequences
	if response == param.ActionResult.END_GAME:
		action_reward = 10.0
	elif response == param.ActionResult.ILLEGAL_MOVE:
		action_reward = -2.0
	else:
		action_reward =  0.3
	return action_reward

def calculate_reconst_reward(surgicaldata,state,timestep):
	# calculate part of the reward value based on reconstructability
	reconst_reward = 0.0 # TODO
	return reconst_reward