import config
import mask_to_submission
from CNN import *


if config.DATA_AUGMENTATION == True:  # adds rotated images containing diagonal roads in the data set
    print("Data Augmentation running...")
    config.INPUT_SIZE = create_extra_input()
    print("-- data augmentation done: added", len(config.EXTRA_IMAGE_IDS),"x3 new images, training size=",config.INPUT_SIZE," --")
else:
    print("-- data augmentation disabled, training size=",config.INPUT_SIZE,"--")

cnn = CNN()

"""

TRAINING

""""

""" 

Train Phase 1 

"""

config.ADD_INTERCALATED_PATCHES = False  # creates approx 4x more data by adding intercalated patches during traning
config.NEIGHBORHOOD_ANALYSIS = False  # if false: patch size of single patch with 'same' passing;
                              # if true: patch size addes extra pixels and 'valid' padding
                              # and adds (config.CONV_FILTER_SIZES[0]-1)/2 pixels on each size

config.RANDOMIZE_INPUT_PATCHES = False  # decide, once reduced the dataset in order to have an equal number of patches
                                        # representing th two labels, to eliminate randomly the patches instead of the last ones
config.CONV_LAYERS=2

if config.NEIGHBORHOOD_ANALYSIS == True:
    NEIGHBOR_PIXELS = 10
    # analyses area of config.IMG_PATCH_SIZE and NEIGHBOR_PIXELS pixels on all sides,
    # but learns from the classification given by the config.IMG_PATCH_SIZE area only
    # (ie classifies a patch of pixels by also taking into account neighbor pixels)
    config.CONV_FILTER_SIZES = [NEIGHBOR_PIXELS*2+1, 5, 5, 5]
else:
    config.CONV_FILTER_SIZES = [5, 5, 5, 5]

config.IMG_PATCH_SIZE = 8  # size in px of the patch
config.CONV_FILTER_DEPTHS = [32, 64, 128, 256]  # depth of conv_weights[i]
config.POOL_FILTER_STRIDES = [2, 2, 2, 2]  # stride for pooling
config.FC1_WEIGHTS_DEPTH = 512  # depth of weights in fully connected 1 (before out)
config.DROPOUT_RATE = 0  # amount of nodes we drop during training (0 for 'no dropout')
config.LEARNING_RATE = 0.08
config.DECAY_RATE = 0.95  # exponential decay of step size of gradient descent
config.NUM_EPOCHS = 10


# uncomment this line below to train phase 1

# cnn.run(phase=1, train=True, conv_layers=config.CONV_LAYERS)



""" 

Train phase 2 

"""

config.ADD_INTERCALATED_PATCHES = False
config.NEIGHBORHOOD_ANALYSIS = False

if config.NEIGHBORHOOD_ANALYSIS == True:
    NEIGHBOR_PIXELS = 16
    config.CONV_FILTER_SIZES = [NEIGHBOR_PIXELS*2+1, 5, 5, 5]
else:
    config.CONV_FILTER_SIZES = [5, 5, 5, 5]

config.IMG_PATCH_SIZE = 8
config.CONV_LAYERS=2
config.CONV_FILTER_DEPTHS = [32, 64, 128, 256]  # depth of conv_weights[i]
config.POOL_FILTER_STRIDES = [2, 2, 2, 2]  # stride for pooling
config.FC1_WEIGHTS_DEPTH = 512  # depth of weights in fully connected 1 (before out)
config.RANDOMIZE_INPUT_PATCHES = False
config.DROPOUT_RATE = 0.0  # amount of nodes we drop during training (0 for 'no dropout')
config.LEARNING_RATE = 0.08
config.DECAY_RATE = 0.95  # exponential decay of step size of gradient descent
config.NUM_EPOCHS = 5

# uncomment this line below to train phase 2

#cnn.run(phase=2, train=True, conv_layers=config.CONV_LAYERS)


""" 

To produce the csv file all cnn.run() are not used, because it is 
sufficient to import the models (or pretrained networks), 
present in the ./tmp folder, so cnn.run() for phase 1 and 2 can be commented
and the TESTING (here below) can be run.

"""



""" 

TESTING 

"""

config.IMG_PATCH_SIZE = 8 
config.RESTORE_MODEL=True

config.INPUT_SIZE=50 #Test data set size

""" Test phase 1 """
cnn.run(phase=1, train=False)

""" Test phase 2 """
cnn.run(phase=2, train=False)


config.IMG_PATCH_SIZE = 16
## (specify if we use train or test [default] dataset, and output of phase 1 or 2 [default]3)
#mask_to_submission.generate_submission_csv_file(train=True, phase=2)"""
""" generate submission CSV """
mask_to_submission.generate_submission_csv_file(train=False, phase=2)
