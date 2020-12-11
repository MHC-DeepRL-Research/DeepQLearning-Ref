import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import linalg as la
import param
import random

def unit_vector(vector):
    # Returns the unit vector of the vector
	norm = np.linalg.norm(vector)
	if norm == 0:
		raise ValueError('zero norm')
	else:
		normalized = vector / norm
	return normalized

def angle_between(v1, v2):
    # Returns the angle in radians between vectors 'v1' and 'v2'unit_vector(v1)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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

def dynamic_camZ_from_data(breathdata, normX, normY, timestep):
	# find camera height based on timestep
	loopstep, flipped = get_loopstep(timestep)
	deform_factor = breath_deform_factor(breathdata, loopstep)
	normZ = deform_factor * np.sqrt(- normX*normX - normY*normY + 2)
	return normZ

def dynamic_toolinfo_from_data(tooldata, timestep):
	# get the dynamic tool pose information from the mat file
	loopstep, flipped = get_loopstep(timestep)
	toolinfo = tooldata[loopstep,:]
	for i in range(param.TOOL_STATE_DIM):
		if i == 3 or i == 4:
			toolinfo[i] = toolinfo[i] * 2.0 / np.pi      		        # normalize angles
		else:
			toolinfo[i] = toolinfo[i] * 1.0 / param.BELLY_EDGE_LENGTH	# normalize pos and vel
	if flipped is True:
		toolinfo[5] = - toolinfo[5]                             # flip velocity direction
		toolinfo[6] = - toolinfo[6]   
	return toolinfo


def calculate_angle(camZ, camP, toolZ, toolP):
	# calculate the rotation angle in direction P
	angle = np.arctan2(camP - toolP, camZ - toolZ)
	norm_angle = angle*2 / np.pi
	if norm_angle <= -1.0 and norm_angle >= 1.0:
		print("[WARNING]: camera angle exceeds pi/2 limit. The system will keep running.")

	return norm_angle

def revert_normalize_cam(campose):
	# revert feature normalization for campose
    campose[:3] = campose[:3] * param.BELLY_EDGE_LENGTH
    campose[3:] = campose[3:] * np.pi / 2
    return campose
        
def revert_normalize_tool(toolpose):
	# revert feature normalization for toolpose
    for i in range(param.TOOL_STATE_DIM):
        if i == 3 or i == 4:
        	toolpose[i] = toolpose[i] * np.pi / 2.0     		        # normalize angles
        else:
        	toolpose[i] = toolpose[i] * param.BELLY_EDGE_LENGTH	        # normalize pos and vel
    return toolpose

def get_tool_pose(obs, tool_idx):
	# get the pose of the ith tool from observations
    assert obs.shape[0] == param.CAM_STATE_DIM*param.CAM_COUNT + param.TOOL_STATE_DIM*param.TOOL_COUNT
    assert tool_idx >= 0 and  tool_idx < param.TOOL_COUNT
    offset = param.CAM_STATE_DIM*param.CAM_COUNT
    return obs[offset+param.TOOL_STATE_DIM*tool_idx:offset+param.TOOL_STATE_DIM*(tool_idx+1)]

def get_tool_poses(obs,norm_flag=True):
	# get the poses of all tools from observations
	toolposes = np.zeros((param.TOOL_COUNT,param.TOOL_STATE_DIM))
	for i in range(param.TOOL_COUNT):
		if norm_flag:
			toolposes[i,:] = np.array(get_tool_pose(obs,i))
		else:
			toolposes[i,:] = np.array(revert_normalize_tool(get_tool_pose(obs,i)))
	return toolposes

def get_cam_pose(obs, cam_idx):
	# get the pose of the ith camera from observations
    assert obs.shape[0] == param.CAM_STATE_DIM*param.CAM_COUNT + param.TOOL_STATE_DIM*param.TOOL_COUNT
    assert cam_idx >= 0 and  cam_idx < param.CAM_COUNT
    return obs[param.CAM_STATE_DIM*cam_idx:param.CAM_STATE_DIM*(cam_idx+1)] 

def get_cam_poses(obs,norm_flag=True):
	# get the poses of all tools from observations
	camposes = np.zeros((param.CAM_COUNT,param.CAM_STATE_DIM))
	for i in range(param.CAM_COUNT):
		if norm_flag:
			camposes[i,:] = np.array(get_cam_pose(obs,i))
		else:
			camposes[i,:] = np.array(revert_normalize_cam(get_cam_pose(obs,i)))
	return camposes

def surface_normal_newell(points):
	# calculate surface normal from three points using the newell method
	n = np.array([0.0, 0.0, 0.0])

	for i, v_curr in enumerate(points):
		v_next = points[(i+1) % len(points), :]
		n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] - v_next[2])
		n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] - v_next[0])
		n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] - v_next[1])

	normalized = unit_vector(n)
	if normalized[2] < 0:
		normalized = - normalized # makes sure it points upward

	return normalized

def surface_normal_cross(points):
	# calculate surface normal from three points using cross product
	n = np.cross(points[1,:]-points[0,:],points[2,:]-points[0,:])

	normalized = unit_vector(n)
	if normalized[2] < 0:
		normalized = - normalized # makes sure it points upward
		
	return normalized

def closest_point(points, idx, n_neighbors):
	# find k closest points to idx point
	assert n_neighbors >= 0
	assert idx < points.shape[0]

	l = points.tolist()
	l.sort(key=lambda coord: (coord[0]-points[idx,0])**2 + (coord[1]-points[idx,1])**2 + (coord[2]-points[idx,2]))
	
	for i in range(points.shape[0]):
		if l[0] != l[i]:
			break
	neighborhood = l[i:i+n_neighbors]
	neighborhood.insert(0,l[0])
	neighborhood = np.array(neighborhood)
	assert neighborhood.shape[0] == n_neighbors+1
	return neighborhood

def cam_vector_from_pose(campose, multiplier):
	# return camera look at vector from campose
	assert multiplier != 0
	tanx = np.tan(campose[3])
	tany = np.tan(campose[4])
	v = np.array([multiplier*tanx, multiplier*tany, multiplier])
	# unit vector in direction of axis
	v = unit_vector(v)
	return v

def cam_angle_constraints(points, camposes, tools, pt_normdir = None):
	# returns the angle of reflection (beta) and projection angle (gamma)
	gamma = np.zeros((points.shape[0],param.CAM_COUNT))
	beta = np.zeros((points.shape[0],param.CAM_COUNT))

	n_neighbors = param.N_NEIGHBORS
	pt_camdir = np.zeros((param.CAM_COUNT,points.shape[0],3))
	cam_normdir = np.zeros((param.CAM_COUNT,3))

	for i in range(points.shape[0]):
		# calculate normal direction for that point
		if pt_normdir.any() == None:
			if i == 0:
				pt_normdir = np.zeros(points.shape)
			neighborhood = closest_point(points.copy(), i, n_neighbors)
			for j in range(param.RANSAC_TRIALS):
				r = random.sample(range(1,n_neighbors), 2)
				neighbors = neighborhood[[0,r[0],r[1]],:]
				pt_normdir[i,:] += surface_normal_cross(neighbors)
			pt_normdir[i,:] = unit_vector(pt_normdir[i,:])

		for j in range(param.CAM_COUNT):
			# calculate campose to point vector
			pt_camdir[j,i,:] = unit_vector(camposes[j,:3] - points[i,:])
			if i == 0:
				cam_normdir[j,:] = cam_vector_from_pose(camposes[j,:], -1)
			# calculate soft angles beta and gamma
			gamma[i,j] = angle_between(-pt_camdir[j,i,:], cam_normdir[j,:])
			beta[i,j] = angle_between(pt_camdir[j,i,:], pt_normdir[i,:])
			
	return gamma, beta

def W(points, tools):
	# calculate the importance of viewing each point
	assert points.shape[1] == 3
	assert tools.shape[1] == 3
	assert tools.shape[0] == param.TOOL_COUNT

	importance_W = param.RECONST_EPSILON*np.ones(points.shape[0])
	for i in range(param.TOOL_COUNT):
		tmp = la.norm(points - tools[i,:], 2, axis=1)
		importance_W = np.maximum(importance_W,tmp)
	importance_W = 1.0 / importance_W

	assert importance_W.shape[0] == points.shape[0]
	return importance_W

def R(gamma, beta):
	# calculate reconstructability score based on soft angle constraints
	score_R = np.zeros(beta.shape)
	for i in range(score_R.shape[0]):
		for j in range(score_R.shape[1]):
			if beta[i,j] < 0.1:
				score_R[i,j] = 0
			elif beta[i,j] > 1.0 or gamma[i,j] > 1.0:
				score_R[i,j] = min(np.cos(beta[i,j]),np.cos(gamma[i,j]))
			else:
				score_R[i,j] = 1.0

	return score_R

def V(gamma):
	# returns the visibility score (hard angle constraints: cam field of view)
	score_V = np.array(gamma < param.CAM_FOV)
	return score_V

def calculate_reconst_reward(surgicaldata,state,timestep):
	# calculate part of the reward value based on reconstructability
	loopstep, flipped = get_loopstep(timestep)

	ptLoc = np.array(surgicaldata.get('ptcloud_loc'))
	ptLoc = np.squeeze(ptLoc[loopstep,:,:])
	ptNorm = np.array(surgicaldata.get('ptcloud_norms'))
	ptNorm = np.squeeze(ptNorm[loopstep,:,:])
	toolposes = get_tool_poses(state,False)
	camposes = get_cam_poses(state,False)
	Np = ptLoc.shape[0]

	gamma, beta = cam_angle_constraints(ptLoc,camposes,toolposes[:,:3],ptNorm) 	# angle constraints
	score_VR = np.multiply(V(gamma), R(gamma, beta))         					# element_wise multiplication
	score_VR = np.sum(score_VR,axis=1) > 2.0  									# check if at least two cameras can see it well
	score_W = W(ptLoc,toolposes[:,:3])              							# calculate importance score
	assert score_VR.shape[0] == score_W.shape[0]

	reconst_reward = np.squeeze(np.dot(score_W,score_VR))
	reconst_reward = reconst_reward / np.sum(score_W)             				# normalize the reconst reward

	reconst_reward = .0

	return reconst_reward


def calculate_action_reward(response):
	# calculate part of the reward value based on action consequences
	if response == param.ActionResult.END_GAME:
		action_reward = 10.0
	elif response == param.ActionResult.ILLEGAL_MOVE:
		action_reward = -2.0
	else:
		action_reward =  0.3
	return action_reward


def truncated_cam_cone(campose, conesize, coneR1):
    # plot the truncated cone representing the camera view
    p_base = campose[:3]
    # vector in direction of axis
    v = cam_vector_from_pose(campose, -conesize)
    # make some vector not in the same direction as v
    not_v = np.array([1, 1, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # normalize n1
    n1 = unit_vector(n1)
    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    n = 20
    t = np.linspace(0, conesize, n)
    theta = np.linspace(0, 2 * np.pi, n)
    # use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(param.CONE_R0, coneR1, n)
    # generate coordinates for surface
    X, Y, Z = [p_base[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    return X, Y, Z

    
def mkdir_p(mypath):
    # Creates a directory. equivalent to using mkdir -p on the command line
    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
