PAST_HISTORY = 30  # how many days in the past we are looking
FORWARD_LOOK = 1  # how many days into the future we are looking
STACK_DEPTH = 2  # the number of stacked LSTM units
LAYER_UNITS = 20  # the hidden depth in the LSTM
DROPOUT = 0.15 # dropout probability
NUM_HEADS = PAST_HISTORY // 4 # number of heads
HIDDEN_SIZE = 2*PAST_HISTORY # dimension of hidden vector
BIAS = True # if we have bias or not

TEST_SIZE = 0.2  # the ratio of data used for testing
N_EPOCHS = 100  # number of epochs our model is training on
LR = 0.01 # Learning rate
BATCH_SIZE = 64 # the size of our batches
SAVE_PATH = "./model.pt"
