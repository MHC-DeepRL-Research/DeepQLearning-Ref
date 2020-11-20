## Parameter List

TRAIN_ITER = 1500000
EVAL_MAX_ITER = 100
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
DATASET_STEPS = 3500000
DATASET_PARALLEL = 3
DATASET_PREFETCH = 3
DATASET_BATCH = 200
DATASET_BUFFER_STEP = 2


