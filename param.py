## Parameter List
from enum import Enum
from enum import IntEnum

TRAIN_ITER = 1500000
EVAL_MAX_ITER = 1000
VIZ_FLAG = True
EVAL_EPISODE = 1
FC_LAYERS = [32,64]

ADAM_LR = 1e-3
ADAM_EPSILON = 0.001

DECAY_UPDATE_PERIOD = 2
DECAY_STEPS = 250000
DECAY_LR_INIT = 1.0
DECAY_LR_END = 0.05

AGENT_UPDATE_PERIOD = 2000
AGENT_GAMMA = 0.99

BUFFER_LENGTH = 1000000
DRIVER_STEPS = 7
DATASET_STEPS = 350000
DATASET_PARALLEL = 3
DATASET_PREFETCH = 3
DATASET_BATCH = 200
DATASET_BUFFER_STEP = 2

CAM_COUNT = 6                       					# number of cameras in the system
GRIDS_PER_EDGE = 201 									# grids in each dimension
CAM_ACT_OPTIONS = 9                                     # number of actions per camera
CAM_ROT_OPTIONS = 5                 					# number of camera rotation options given a cam location
CAM_INIT_DIST = 40                  					# initial camera distance from origin
CAM_EDGE_LENGTH = 200              					    # the abdominal border edge length (-200 ~ 200)
GRIDS_IN_SPACE =  GRIDS_PER_EDGE*GRIDS_PER_EDGE 		# total number of grids
GRID_LENGTH = CAM_EDGE_LENGTH/(GRIDS_PER_EDGE//2)       # how big each grid is

# possible consequences from action
class ActionResult(Enum):
    VALID_MOVE = 1
    ILLEGAL_MOVE = 2
    END_GAME = 3

# possible rotation option given a cam location
class RotFlag(IntEnum):
    CENTER = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SAME = 5