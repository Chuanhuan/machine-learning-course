#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import config

foreground_threshold = config.FOREGROUND_THRESHOULD # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size =  config.IMG_PATCH_SIZE
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

def generate_submission_csv_file(train=False, phase = 2):
    submission_filename = config.SUBMISSION_FILE_PATH
    image_filenames = []
    for i in range(1, config.INPUT_SIZE+1):
        if train == True:
            if phase == 1:
                image_filename = config.PREDICTIONS_PATH + "prediction_raw_train_" + str(i) + ".png"
            else:
                image_filename = config.PREDICTIONS_PATH + "prediction_2_train_" + str(i) + ".png"
        else:
            if phase == 1:
                image_filename = config.PREDICTIONS_PATH + "prediction_raw_test_" + str(i) + ".png"
            else:
                image_filename = config.PREDICTIONS_PATH + "prediction_2_test_" + str(i) + ".png"
        print ("Classifying",image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
