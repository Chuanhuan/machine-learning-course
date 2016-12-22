import config
from mask_to_submission import *
from CNN import *

#To be run only once, generates a larger dataset by rotating existing pictures 
if config.DATA_AUGMENTATION == True:
    print("Data Augmentation running...")
    config.INPUT_SIZE = create_extra_input()
    print("0 : data augmentation done: added", len(config.EXTRA_IMAGE_IDS),"x3 new images, training size=",config.INPUT_SIZE," --")
else:
    print("0 : data augmentation disabled, training size=",config.INPUT_SIZE,"--")

#creates Convolutional Neural Network class
cnn = CNN()

#CNN training phase 1:

config.ADD_INTERCALATED_PATCHES = False #creates approx 4x more data by adding intercalated patches during traning
config.NEIGHBORHOOD_ANALYSIS = False#if false: patch size of single patch with 'same' passing;
                              #if true: patch size addes extra pixels and 'valid' padding
                              #and adds (config.CONV_FILTER_SIZES[0]-1)/2 pixels on each size

config.RANDOMIZE_INPUT_PATCHES = False
config.CONV_LAYERS=2

if config.NEIGHBORHOOD_ANALYSIS == True:
    NEIGHBOR_PIXELS = 10
    #analyses area of config.IMG_PATCH_SIZE and NEIGHBOR_PIXELS pixels on all sides,
    #but learns from the classification given by the config.IMG_PATCH_SIZE area only
    #(ie classifies a patch of pixels by also taking into account neighbor pixels)
    config.CONV_FILTER_SIZES = [NEIGHBOR_PIXELS*2+1, 5, 5, 5]
else:
    config.CONV_FILTER_SIZES = [5, 5, 5, 5]

config.IMG_PATCH_SIZE = 8  #4,8,12,16
config.CONV_FILTER_DEPTHS = [32, 64, 128, 256] #depth of conv_weights[i]
config.POOL_FILTER_STRIDES = [2, 2, 2, 2] #stride for pooling
config.FC1_WEIGHTS_DEPTH = 512 #depth of weights in fully connected 1 (before out)
config.DROPOUT_RATE = 0 #amount of nodes we drop during training (0 for 'no dropout')
config.LEARNING_RATE = 0.08
config.DECAY_RATE = 0.95 #decay of step size of gradient descent
config.NUM_EPOCHS = 10

cnn.run(phase=1, train=True)

#CNN TRAINING phase 2:

config.ADD_INTERCALATED_PATCHES = False
config.NEIGHBORHOOD_ANALYSIS = False

if config.NEIGHBORHOOD_ANALYSIS == True:
    NEIGHBOR_PIXELS = 16
    config.CONV_FILTER_SIZES = [NEIGHBOR_PIXELS*2+1, 5, 5, 5]
else:
    config.CONV_FILTER_SIZES = [5, 5, 5, 5]

config.IMG_PATCH_SIZE = 8
config.CONV_LAYERS=2
config.CONV_FILTER_DEPTHS = [32, 64, 128, 256] #depth of conv_weights[i]
config.POOL_FILTER_STRIDES = [2, 2, 2, 2] #stride for pooling
config.FC1_WEIGHTS_DEPTH = 512 #depth of weights in fully connected 1 (before out)
config.RANDOMIZE_INPUT_PATCHES = False
config.DROPOUT_RATE = 0.0 #amount of nodes we drop during training (0 for 'no dropout')
config.LEARNING_RATE = 0.08
config.DECAY_RATE = 0.95 #decay of step size of gradient descent
config.NUM_EPOCHS = 5

cnn.run(phase=2, train=True)

#CNN Testing

config.RESTORE_MODEL=True
config.INPUT_SIZE=50 #Test data set size

#cnn.run(phase=1, train=False)
#cnn.run(phase=2, train=False)

## generate submission CSV
## (specify if we use train or test [default] dataset, and output of phase 1 or 2 [default]3)
config.IMG_PATCH_SIZE = 16
#generate_submission_csv_file(train=False, phase=2)
