import numpy
import tensorflow as tf

#### Parameters related to the execution and hardware

global NUM_CHANNELS
global PIXEL_DEPTH
global NUM_LABELS
global TRAINING_SIZE
global SEED
global BATCH_SIZE
global RECORDING_STEP
global NUM_THREADS
global DATA_AUGMENTATION
global EXTRA_IMAGE_IDS
global FOREGROUND_THRESHOULD

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
INPUT_SIZE = 32 # 100 + 48 augmented
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16
RECORDING_STEP = 1000
NUM_THREADS = 2
DATA_AUGMENTATION = False
EXTRA_IMAGE_IDS = [23,26,27,28,30,32,33,38,42,69,72,73,75,83,88,91]
FOREGROUND_THRESHOULD = 0.25

##### Parameters to be set by phase1 and phase2 before calling

global ADD_INTERCALATED_PATCHES
global NEIGHBORHOOD_ANALYSIS
global IMG_PATCH_SIZE
global CONV_LAYERS
global CONV_FILTER_SIZES #sizes of conv_weights[i]
global CONV_FILTER_DEPTHS #depths of conv_weights[i]
global POOL_FILTER_STRIDES #strides for pooling
global FC1_WEIGHTS_DEPTH #depths of weights in fully connected 1 (before out)
global RANDOMIZE_INPUT_PATCHES
global DROPOUT_RATE #amount of nodes we drop during training (0 for 'no dropout')
global LEARNING_RATE 
global DECAY_RATE #decay of step size of gradient descent
global NUM_EPOCHS

##### 

global PREDICTIONS_PATH
global INPUT_TRAIN_PATH
global INPUT_TEST_PATH 
global SUBMISSION_FILE_PATH
global SUMMARY_MODEL_PATH

PREDICTIONS_PATH = "./predictions/"
GROUNDTRUTH_PATH = "./training/groundtruth/"
INPUT_TRAIN_PATH = "./training/images/"
INPUT_TEST_PATH  = "./test_set_images/" 
SUMMARY_MODEL_PATH = "./tmp/"
SUBMISSION_FILE_PATH = "./submission.csv"
