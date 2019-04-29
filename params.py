# Size of batches to optimize
BATCH_SIZE = 512
# Rate at which to discount rewards in the future
GAMMA = 0.99
# Rate at which to discount rewards over the multistep horizon
# Probably want this to be the same as GAMMA
MULISTEP_GAMMA = 0.99
# Number of frames to look ahead for the result of actions
MULTISTEP_PARAM = 5
# Initial value of epsilon (for random action selection)
EPS_START = 0.9
# Final value of epsilon (for random action selection)
EPS_END = 0.05
# Rate at which epsilon decays from start to end
EPS_DECAY = 500000
# How many games occur between policy and target net updates
TARGET_UPDATE = 25
# Cost of doing a move that isn't a drop or a no-action
MOVEMENT_COST = 0 # 0.01
LAYER_HISTORY = 4
TRAIN_RATE = 4
LEARNING_RATE = 5 * 10 ** (-4)
MEMORY_SIZE = 1000000
