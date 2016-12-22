import config
import tensorflow as tf
from helpers import *

### Convolutional Neural Network class definition

class CNN:

        def __init__(self):
            self.cdata = []

        def run(self, phase, train, conv_layers=2):

            # Make an image summary for 4d tensor image with index idx
            def get_image_summary(img, idx = 0):
                #Take img BATCHx16x16x3 --> slice 1x16x16x1 (-1 means "to all")
                #ie a single patch, all HxV pixels, single column
                V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
                img_w = img.get_shape().as_list()[1] #16: data was config.BATCH_SIZEx16x16x3
                img_h = img.get_shape().as_list()[2]
                min_value = tf.reduce_min(V) #gives min number across all dimensions 
                V = V - min_value  #TRANSLATION: we translate all data (start from 0)
                max_value = tf.reduce_max(V)
                V = V / (max_value*config.PIXEL_DEPTH)  #NORMALIZATION: values in 0 to 1
                V = tf.reshape(V, (img_w, img_h, 1))
                V = tf.transpose(V, (2, 0, 1))
                V = tf.reshape(V, (-1, img_w, img_h, 1))
                return V

            # Make an image summary for 3d tensor image with index idx
            def get_image_summary_3d(img):
                V = tf.slice(img, (0, 0, 0), (1, -1, -1))
                img_w = img.get_shape().as_list()[1]
                img_h = img.get_shape().as_list()[2]
                V = tf.reshape(V, (img_w, img_h, 1))
                V = tf.transpose(V, (2, 0, 1))
                V = tf.reshape(V, (-1, img_w, img_h, 1))
                return V         
            
            # Get prediction for given input image 
            def get_prediction(img,phase, conv_layers):
                data = numpy.asarray(img_crop(img, config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE, False, config.NEIGHBORHOOD_ANALYSIS))
                data_node = tf.constant(data)
                output = tf.nn.softmax(model(data_node,phase,conv_layers))
                output_prediction = s.run(output)
                img_prediction = label_to_img(img.shape[0], img.shape[1], config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE, output_prediction)
                return img_prediction

            # Get a concatenation of the prediction and groundtruth for given input file
            def get_prediction_with_groundtruth(image_idx,phase,conv_layers):
                image_filename = get_path_for_input(phase, True,image_idx)
                if image_idx == 1:
                    print("   ", image_filename)
                img = mpimg.imread(image_filename)
                img_prediction = get_prediction(img,phase,conv_layers)
                return concatenate_images(img, img_prediction)
                
            # Get prediction overlaid on the original image for given input file
            def get_prediction_with_overlay(image_idx,phase,conv_layers):
                image_filename = get_path_for_input(phase, True,image_idx)
                if image_idx == 1:
                    print("   ", image_filename)
                img = mpimg.imread(image_filename)
                img_prediction = get_prediction(img,phase, conv_layers)
                oimg = make_img_overlay(img, img_prediction)
                return oimg

            # We will replicate the model structure for the training subgraph, as well
            # as the evaluation subgraphs, while sharing the trainable parameters.
            def model(data, phase, conv_layers, train=False):
                """The Model definition."""
                convs = [None] * conv_layers
                relus = [None] * conv_layers
                pools = [None] * conv_layers
            
                #define all convolational networks layers
                for i in range (0, conv_layers):
                    if i==0:
                        if config.NEIGHBORHOOD_ANALYSIS == True:
                            padding_string = 'VALID'
                        else:
                            padding_string = 'SAME'
                        convs[i] = tf.nn.conv2d(data,    ###input is data : config.BATCH_SIZEx16x16x3
                                        conv_weights[i], #### 5x5x3x32
                                        strides=[1, 1, 1, 1],
                                        padding=padding_string)
                    else:
                        convs[i] = tf.nn.conv2d(pools[i-1], #input is previous layers output
                                    conv_weights[i],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')
                
                    # activity funtion: bias and rectified linear non-linearity. relu(w.T*x+b)
                    relus[i] = tf.nn.relu(tf.nn.bias_add(convs[i], conv_biases[i]))

                    #pooling: best of CONV_POOLING_STRIDE results from every X and Y output
                    pools[i] = tf.nn.max_pool(relus[i],
                                      ksize  =[1, config.POOL_FILTER_STRIDES[i], config.POOL_FILTER_STRIDES[i], 1],
                                      strides=[1, config.POOL_FILTER_STRIDES[i], config.POOL_FILTER_STRIDES[i], 1],
                                      padding='SAME') 
                    
                # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
                last_pool = pools[conv_layers-1];
                pool_shape = last_pool.get_shape().as_list()
                reshape = tf.reshape(
                    last_pool, #16x4x4x64
                    [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]) #[16, 16*16*64]

                # During training, output data types and sizes
                if train==True :
                    print ("    -- data: ", str(data.get_shape()))
                    for i in range(0,conv_layers):
                        print ("    -- convs["+str(i)+"]:", str(convs[i].get_shape()))
                        print ("    -- conv_biases["+str(i)+"]:", str(conv_biases[i].get_shape()))
                        print ("    -- conv_weights["+str(i)+"]:", str(conv_weights[i].get_shape()))
                        print ("    -- relus["+str(i)+"]:", str(relus[i].get_shape()))
                        print ("    -- relus["+str(i)+"]:", str(relus[i].get_shape()))
                        print ("    -- pools["+str(i)+"]:", str(pools[i].get_shape()))
                    print ("    -- reshape:", str(reshape.get_shape()))
                    print ("    -- fc1_weights:", str(fc1_weights.get_shape()))
                    print ("    -- fc2_weights:", str(fc2_weights.get_shape()))
                
                # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
                hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

                # Add a 50% dropout during training only. Dropout also scales
                # activations such that no rescaling is needed at evaluation time.
                if train and config.DROPOUT_RATE>0:
                    print("    -- Performing", config.DROPOUT_RATE, "dropout during training.")
                    hidden = tf.nn.dropout(hidden, config.DROPOUT_RATE, seed=config.SEED)
                out = tf.matmul(hidden, fc2_weights) + fc2_biases

                if train==True:
                    print ("    -- hidden:", str(hidden.get_shape()))
                    print ("    -- out:", str(out.get_shape()))
 
                if train == True:
                    summary_id = '_0'
                    s_data = get_image_summary(data) #from docs: 3 channels so it's interpreted as RGB
                    filter_summary0 = tf.image_summary('summary_data' + summary_id, s_data)
                    s_convs = [None] * conv_layers
                    filter_summaries = [None] * conv_layers *2
                    s_pools = [None] * conv_layers
                    for i in range (0, conv_layers):
                        s_convs[i] = get_image_summary(convs[i])
                        filter_summaries[i]   = tf.image_summary('summary_conv' + str(i) + summary_id, s_convs[i])
                        s_pools[i] = get_image_summary(pools[i])
                        filter_summaries[i+1] = tf.image_summary('summary_pool' + str(i) + summary_id, s_pools[i])
                return out

            #reset the graph (summaries information) from previous executions
            tf.reset_default_graph()

            #create all convolutional network layers
            conv_weights = [None] * conv_layers
            conv_biases  = [None] * conv_layers
            for i in range (0, conv_layers):
                assert config.CONV_FILTER_SIZES[i] % 2 == 1, "config.CONV_FILTER_SIZES must only contain ODD sizes numbers"
                if i == 0 :
                    conv_weights[i] = tf.Variable(
                        tf.truncated_normal([config.CONV_FILTER_SIZES[i], config.CONV_FILTER_SIZES[i], config.NUM_CHANNELS, config.CONV_FILTER_DEPTHS[i]],
                                stddev=0.1,
                                seed=config.SEED)) #NOTE: this randomness allows the weights not to be started as zero (so that we can start training.. otherwise derivative is 0)
                    conv_biases[i] = tf.Variable(tf.zeros([config.CONV_FILTER_DEPTHS[i]]))  #the +b in the equation above

                else:
                    conv_weights[i] = tf.Variable(
                        tf.truncated_normal([config.CONV_FILTER_SIZES[i], config.CONV_FILTER_SIZES[i], config.CONV_FILTER_DEPTHS[i-1], config.CONV_FILTER_DEPTHS[i]],
                                stddev=0.1,
                                seed=config.SEED))  #each of 64 outputs of conv2 will be connected to 64 nodes in upper layer
                    conv_biases[i] = tf.Variable(tf.constant(0.1, shape=[config.CONV_FILTER_DEPTHS[i]]))  #TODO why is it a constant?
            
            #create the two fully connected layers

            #calculate the total pixels, taking into account all pixels discarded by strides in all layers
            fc1_pixel_size = config.IMG_PATCH_SIZE
            for i in range(0, conv_layers):
                #make sure strides and patches size are divisible
                assert config.IMG_PATCH_SIZE / config.POOL_FILTER_STRIDES[i] % 1 == 0, "config.IMG_PATCH_SIZE / config.POOL_FILTER_STRIDES[%r] is not an integer!" % i
                fc1_pixel_size /= config.POOL_FILTER_STRIDES[i]
                
            fc1_weights = tf.Variable( 
                tf.truncated_normal([int(fc1_pixel_size*fc1_pixel_size*config.CONV_FILTER_DEPTHS[conv_layers-1]), config.FC1_WEIGHTS_DEPTH],
                                    stddev=0.1,
                                    seed=config.SEED))
            fc1_biases = tf.Variable(tf.constant(0.1, shape=[config.FC1_WEIGHTS_DEPTH]))
            fc2_weights = tf.Variable(
                tf.truncated_normal([config.FC1_WEIGHTS_DEPTH, config.NUM_LABELS],
                                    stddev=0.1,
                                    seed=config.SEED))
            fc2_biases = tf.Variable(tf.constant(0.1, shape=[config.NUM_LABELS]))

            num_epochs = config.NUM_EPOCHS #iterations count

            print(phase, ": extract_data...")
            train_data = extract_data(config.INPUT_SIZE,phase,train)

            if train==True:
                print(phase, ": extract_labels...")
                train_labels = extract_labels(config.INPUT_SIZE)


                c0 = 0 #count of tiles labelled as 0
                c1 = 0 #... as 1
                for i in range(len(train_labels)):
                    if train_labels[i][0] == 1:
                        c0 = c0 + 1
                    else:
                        c1 = c1 + 1

                #We are training on the same number of 1s and 0s, to avoid training data being biased!
                print (phase,': before balancing: number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1) + ', random==' + str(config.RANDOMIZE_INPUT_PATCHES))
                min_c = min(c0, c1)
                idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
                idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
            
                #We try to randomize the picking of patches (its discarding "non-road" of last pics only...)
                if config.RANDOMIZE_INPUT_PATCHES == True:
                    if (config.SEED != None):
                        numpy.random.seed(config.SEED)
                    numpy.random.shuffle(idx0)
                    numpy.random.shuffle(idx1)
                new_indices = idx0[0:min_c] + idx1[0:min_c]
                train_data = train_data[new_indices,:,:,:]
                train_labels = train_labels[new_indices]
                train_size = train_labels.shape[0]

                #counts number of c0 and c1
                c0 = 0
                c1 = 0
                for i in range(len(train_labels)):
                    if train_labels[i][0] == 1:
                        c0 = c0 + 1
                    else:
                        c1 = c1 + 1
                print (phase, ': after balancing: Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1) + ', random==' + str(config.RANDOMIZE_INPUT_PATCHES))

                # This is where training samples and labels are fed to the graph.
                # These placeholder nodes will be fed a batch of training data at each
                # training step using the {feed_dict} argument to the Run() call below.

                patch_size = config.IMG_PATCH_SIZE
                if config.NEIGHBORHOOD_ANALYSIS==True:
                    patch_size += config.CONV_FILTER_SIZES[0]-1 #add the margin pixels for both sides
    
                train_data_node = tf.placeholder(
                    tf.float32,
                    shape=(config.BATCH_SIZE, patch_size, patch_size, config.NUM_CHANNELS))
                train_labels_node = tf.placeholder(tf.float32, shape=(config.BATCH_SIZE, config.NUM_LABELS))
                train_all_data_node = tf.constant(train_data) #converting train_data to tensorflow variable
                print("    -- train_all_data_node:", str(train_all_data_node.get_shape()))

                # Training computation: logits + cross-entropy loss.
                print("    -- train_data_node:", train_data_node.get_shape())
                logits = model(train_data_node, phase, conv_layers, True) # config.BATCH_SIZE*16x16x3
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(   #softmax cross entropy is the loss function
                    logits, train_labels_node))
                tf.scalar_summary('loss', loss)

                #set up parameters for nodes
                all_params_node  = []
                for i in range (0, conv_layers):
                    all_params_node.append(conv_weights[i])
                    all_params_node.append(conv_biases[i])
                all_params_node.append(fc1_weights)
                all_params_node.append(fc1_biases)
                all_params_node.append(fc2_weights)
                all_params_node.append(fc2_biases)

                all_params_names = []
                for i in range (0, conv_layers):
                    all_params_names.append('conv_weights['+str(i)+']')
                    all_params_names.append('conv_biases['+str(i)+']')
                all_params_names.append('fc1_weights')
                all_params_names.append('fc1_biases')
                all_params_names.append('fc2_weights')
                all_params_names.append('fc2_biases')
                all_grads_node = tf.gradients(loss, all_params_node)
                all_grad_norms_node = [None] * conv_layers
                for i in range(0, len(all_grads_node)):
                    norm_grad_i = tf.global_norm([all_grads_node[i]])
                    all_grad_norms_node.append(norm_grad_i)
                    tf.scalar_summary(all_params_names[i], norm_grad_i)

                # L2 regularization for the fully connected parameters.
                #### avoid extrploding weights ("it only makes changes to the weights if they will really make a difference")
                regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

                # Add the regularization term to the loss.
                loss += 5e-4 * regularizers

                # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
                batch = tf.Variable(0)
                # Decay once per epoch, using an exponential schedule starting at 0.01.
                learning_rate = tf.train.exponential_decay(
                    config.LEARNING_RATE,       # Base learning rate.
                    batch * config.BATCH_SIZE,  # Current index into the dataset.
                    train_size,                 # Decay step.
                    config.DECAY_RATE,          # Decay of the step size
                    staircase=True)
                tf.scalar_summary('learning_rate', learning_rate)

                # Use simple momentum for the optimization.
                optimizer = tf.train.MomentumOptimizer(learning_rate,0.0).minimize(loss, global_step=batch)

                # Predictions for the minibatch, validation set and test set.
                train_prediction = tf.nn.softmax(logits)
            
                # We'll compute them only once in a while by calling their {eval()} method.
                train_all_prediction = tf.nn.softmax(model(train_all_data_node,phase, conv_layers))


            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Create a local session to run this computation.
            with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=config.NUM_THREADS)) as s:

                if train == False: #Running on TEST data
                    print (str(phase),': Loading existing network model...')
                    saver.restore(s, config.SUMMARY_MODEL_PATH + "model_phase_"+str(phase)+".ckpt")
                    print(phase, ": Model restored.")

                else: #Train data
                    # Run all the initializers to prepare the trainable parameters.
                    tf.global_variables_initializer().run()

                    # Build the summary operation based on the TF collection of Summaries.
                    summary_op = tf.summary.merge_all()
                    summary_writer = tf.train.SummaryWriter(config.SUMMARY_MODEL_PATH,
                                                            graph=s.graph)
                                                            #graph_def=s.graph_def)
                    # Loop through training steps.
                    print (str(phase),': Initializing training, total number of iterations = ' + str(int(num_epochs * train_size / config.BATCH_SIZE)))

                    training_indices = range(train_size)

                    for iepoch in range(num_epochs):

                        # Permute training indices
                        perm_indices = numpy.random.permutation(training_indices)

                        for step in range (int(train_size / config.BATCH_SIZE)):

                            offset = (step * config.BATCH_SIZE) % (train_size - config.BATCH_SIZE)
                            batch_indices = perm_indices[offset:(offset + config.BATCH_SIZE)]

                            # Compute the offset of the current minibatch in the data.
                            # Note that we could use better randomization across epochs.
                            batch_data = train_data[batch_indices, :, :, :]
                            batch_labels = train_labels[batch_indices]
                            # This dictionary maps the batch data (as a numpy array) to the
                            # node in the graph is should be fed to.
                            feed_dict = {train_data_node: batch_data,
                                         train_labels_node: batch_labels}

                            if step % config.RECORDING_STEP == 0:
                                summary_str, _, l, lr, predictions = s.run(
                                    [summary_op, optimizer, loss, learning_rate, train_prediction],
                                    feed_dict=feed_dict)
                                #summary_str = s.run(summary_op, feed_dict=feed_dict) #TODO uncomment this? what does it do?
                                summary_writer.add_summary(summary_str, step)
                                summary_writer.flush()

                                # print_predictions(predictions, batch_labels)

                                print (datetime.datetime.now().strftime("%H:%M:%S"), 'Epoch: ', iepoch,'.',step,', minibatch loss: %.3f' % (l), ', Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))

                                sys.stdout.flush()
                            else:
                                # Run the graph and fetch some of the nodes.
                                _, l, lr, predictions = s.run(
                                    [optimizer, loss, learning_rate, train_prediction],
                                    feed_dict=feed_dict)

                        # Save the variables to disk.
                        save_path = saver.save(s, config.SUMMARY_MODEL_PATH + "model_phase_"+str(phase)+".ckpt")
                        print(phase, ": Model saved in file: %s" % save_path)

                if train == True:
                    print (phase, ": Running prediction on training set, from", config.INPUT_SIZE,"files")
                else: 
                    print (phase, ": Running prediction on test set, classifying", config.INPUT_SIZE,"files")
                prediction_dir = config.PREDICTIONS_PATH

                if not os.path.isdir(prediction_dir):
                    os.mkdir(prediction_dir)

                for i in range(1, config.INPUT_SIZE+1):
                    image_filename = get_path_for_input(phase,train,i)
                    if i == 1:
                        print("   ", image_filename)
                    rimg = mpimg.imread(image_filename)
                    rimg_prediction = get_prediction(rimg, phase, conv_layers)
                    #convert from 2D array 1/0 to RGB
                    w = rimg_prediction.shape[0]
                    h = rimg_prediction.shape[1]
                    rimg_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
                    rimg_mask[:,:,0] = rimg_prediction*config.PIXEL_DEPTH
                    rimg_mask[:,:,1] = rimg_prediction*config.PIXEL_DEPTH
                    rimg_mask[:,:,2] = rimg_prediction*config.PIXEL_DEPTH
                    rimg_final = Image.fromarray(rimg_mask, 'RGB')    
                    
                    if train==True:
                        pimg = get_prediction_with_groundtruth(i,phase,conv_layers)
                        oimg = get_prediction_with_overlay(i,phase,conv_layers)
                    
                    if phase == 1:
                        if train==True:
                            Image.fromarray(pimg).save(prediction_dir + "prediction_" + str(i) + ".png")
                            oimg.save(prediction_dir + "overlay_" + str(i) + ".png")
                            rimg_final.save(prediction_dir + "prediction_raw_train_" + str(i) + ".png")
                        else:
                            rimg_final.save(prediction_dir + "prediction_raw_test_" + str(i) + ".png")
                    if phase == 2:
                        if train==True:
                            Image.fromarray(pimg).save(prediction_dir + "prediction_2_" + str(i) + ".png")
                            rimg_final.save(prediction_dir + "prediction_2_train_" + str(i) + ".png")
                            oimg.save(prediction_dir + "overlay_2_" + str(i) + ".png")
                        else:
                            rimg_final.save(prediction_dir + "prediction_2_test_" + str(i) + ".png")

            print(phase,": -- job done --")

