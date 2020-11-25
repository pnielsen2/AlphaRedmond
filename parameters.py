# display parameters

display = True
windowwidth = 1030
windowheight = 830
boardsize = 810

# game parameters
dimension = 9 # leave this as 9 - otherwise won't be compatible with network architechture

# gpu parameters
device = 'gpu' #'cpu'

# network parameters
num_residual_blocks = 9

# training parameters
batch_size = 64
learning_rate = .000001
batches_per_train_loop = 10
games_generated_at_a_time = 1

# data generating parameters
rollouts = 82 #362
replay_buffer_size = 1024
mcts_display = False
