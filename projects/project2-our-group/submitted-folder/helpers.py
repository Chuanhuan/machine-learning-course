import config
import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf
import datetime

def get_path_for_input(phase,train,i):
    if phase ==1:
        if train==True:  #path of input datatypes
            return config.INPUT_TRAIN_PATH + "satImage_%.3d" % i + ".png"
        else:
            return config.INPUT_TEST_PATH + "test_" + str(i) + "/test_" + str(i) + ".png"
    if phase == 2:
        if train==True:
            return config.PREDICTIONS_PATH + "prediction_raw_train_" + str(i) + ".png"
        else:
            return config.PREDICTIONS_PATH + "prediction_raw_test_" + str(i) + ".png"
    return "unknown-phase-train-wrong-path"

def create_extra_input():
        img_new_id = 100
        for i in range(0, len(config.EXTRA_IMAGE_IDS)):
            img_id = config.EXTRA_IMAGE_IDS[i]
            img_filename = get_path_for_input(1,True,img_id)
            if i == 1:
                print("   ", img_filename)
            gt_filename = config.GROUNDTRUTH_PATH + "satImage_" + ("%.3d" % img_id) + ".png"
            if os.path.isfile(img_filename):
                im = Image.open(img_filename)
                gt = Image.open(gt_filename)
                for r in range(1,4): #3 rotations
                    new_id = 100 + i*3 + r
                    img_new_name = config.INPUT_TRAIN_PATH + "satImage_" + ("%.3d" % new_id) + ".png"
                    gt_new_name =  config.GROUNDTRUTH_PATH + "satImage_" + ("%.3d" % new_id) + ".png"
                    im.rotate(90*r).save(img_new_name)
                    gt.rotate(90*r).save(gt_new_name)
            else:
                print ('Oops! File ' + img_filename + ' does not exist')
        return 100+len(config.EXTRA_IMAGE_IDS)*3 

# Extract patches from a given image
def img_crop(im, w, h, add_intercalated_patches, add_neighboorhood):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3 #for ground_truth images test
    
    #stepping w/2 and h/2 so that we add some more patches from interleaved intervals
    step_h = h
    step_w = w
    if add_intercalated_patches == True:
        step_h = int(h/2)
        step_w = int(w/2)
            
    #standard method, non-overlapping patches
    if add_neighboorhood == False:
        #the -h+1 and -w+1 avoid the h from copying incomplete patches
        for i in range(0,imgheight-h+1,step_h):
            for j in range(0,imgwidth-w+1,step_w):
                if is_2d:
                    im_patch = im[j:j+w, i:i+h]
                else:
                    im_patch = im[j:j+w, i:i+h, :]
                list_patches.append(im_patch)        
             
    #patches overlapping by a margin, for neighbors analysis
    else:
        assert config.CONV_FILTER_SIZES[0] % 2 == 1, "config.CONV_FILTER_SIZES[0] must be an ODD number"
        margin = int((config.CONV_FILTER_SIZES[0]-1)/2) #neighbor pixels per side
        new_imgwidth  = int(margin*2+imgwidth)
        new_imgheight = int(margin*2+imgheight)
        
        #create new image with zero-margins and copy original image
        if is_2d == True:
            im2 = numpy.zeros((new_imgwidth,new_imgheight), dtype=type(im[0][0]))
        else:
            im2 = numpy.zeros((new_imgwidth,new_imgheight,im.shape[2]), dtype=type(im[0][0][0]))
    
        for i in range(0,imgheight):
            for j in range(0,imgwidth):
                im2[margin+j,margin+i] = im[j,i]
            
        #copy patches, leave margins on both sides
        for i in range(margin,new_imgheight-margin-h+1,step_h):
            for j in range(margin,new_imgwidth-margin-w+1,step_w):        
                if is_2d:
                    im_patch = im2[(j-margin):(j+w+margin), (i-margin):(i+h+margin)]
                else:
                    im_patch = im2[(j-margin):(j+w+margin), (i-margin):(i+h+margin), :]
                list_patches.append(im_patch)
    return list_patches

#return matrix of image patches
def extract_data(num_images, phase, train):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        image_filename = get_path_for_input(phase,train,i)
        if i == 1:
            print("   ", image_filename)
        if os.path.isfile(image_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')
                
    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    img_patches = [img_crop(imgs[i], config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE, config.ADD_INTERCALATED_PATCHES, config.NEIGHBORHOOD_ANALYSIS) for i in range(num_images)]
    N_PATCHES_PER_IMAGE = len(img_patches)
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return numpy.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = config.FOREGROUND_THRESHOULD 
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1] #non-road (black) ?
    else:
        return [1, 0] #road (white) ?

# Extract label images
def extract_labels(num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        gt_filename = config.GROUNDTRUTH_PATH + "satImage_" + ("%.3d" % i) + ".png"
        if i == 1:
            print("   ", gt_filename)
        if os.path.isfile(gt_filename):
            #print ('Loading ' + image_filename)
            img = mpimg.imread(gt_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + gt_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE, config.ADD_INTERCALATED_PATCHES, False) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


#returns percentage of WRONG labels (right ones stored in predictions)
def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][1] > 0.5:
                l = 1 #black:  non-road
            else:
                l = 0 #white: non-road
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * config.PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*config.PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
