import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import param


def campose_1Dto2D(index1D):

	if index1D < 0 or index1D > (param.GRIDS_PER_EDGE*param.GRIDS_PER_EDGE) - 1:
		print("Error: input index1D is out of bound.")
		return None
	else:
		index2D = [index1D / param.GRIDS_PER_EDGE, index1D % param.GRIDS_PER_EDGE]
		index2D -= [param.GRIDS_PER_EDGE // 2 , param.GRIDS_PER_EDGE // 2]
		index2D *= param.GRID_LENGTH
		return index2D

def campose_2Dto1D(index2D):

	max_dist = param.GRID_LENGTH*param.GRIDS_PER_EDGE

	if min(index2D) < -max_dist or max(index2D) > max_dist:
		print("Error: input index2D is out of bound.")
		return None
	elif index2D[0]%param.GRID_LENGTH != 0 or index2D[1]%param.GRID_LENGTH != 0:
		print("Error: input index2D must be a multiple of " + str(param.GRID_LENGTH) +".")
		return None
	else:
		index2D /= param.GRID_LENGTH
		index2D += [param.GRIDS_PER_EDGE // 2 , param.GRIDS_PER_EDGE // 2]
		index1D = int(index2D[0] * param.GRIDS_PER_EDGE + index2D[1])
		assert index1D >= 0 and index1D < param.GRIDS_PER_EDGE*param.GRIDS_PER_EDGE
		return index1D


