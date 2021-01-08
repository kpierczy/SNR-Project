# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-09 18:16:36
# @ Modified time: 2020-12-20 16:55:11
# @ Description:
#
#     Script implements an json-config-driven envirenment used to fit a VGG19 model. Before using the
#     script, the 'sourceMe.bash' script should be sourced to prepare appropriate environment variables.
#     Alternatively, required 'PROJECT_HOME' variable can be set manually to the absolute path of the
#     project's home folder.
#
#     The script was prepared so that learning workflow could be run without code's modifications.
#     All parameters configuring the learning process can be adjusted from config/params/*.json config files.
#     Only modifications of the model's structure require direct interference into the script's code.
#
#     The script's workflow:
#        - look for and open configuration files
#        - configure tesnorflow's options
#        - prepare training and validation datasets basing on configured
#          directories and augmentation options
#        - build a basic model (load saved weight, if configured)
#        - prepare training environment (training callbacks)
#        - run the training
#        - save training history to the file
#
#     As the learning rate scheduler the tf.keras.callbacks.ReduceLROnPlateau object is used. It's parameters
#     can be tuned from the fit.json file.
#
# @ Note: This script should be always run from the main project's directory as it relies on the relative
#     paths to and from the config files.
#
# @ Requirements: All required python packages was listed in config/env/requirements*.py files
# ================================================================================================================

import tensorflow as tf
# Pretrained VGG-19
from tensorflow.keras.applications import vgg19
# Model API
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
# Dedicated datapipe
from tools.ImageAugmentation       import ImageAugmentation
from tools.DataPipe                import DataPipe
from tools.ConfusionMatrixCallback import ConfusionMatrixCallback
from tools.LRTensorBoard           import LRTensorBoard
# Images manipulation
from PIL import Image
# Utilities
from datetime import datetime
from glob     import glob
import numpy as np
import pickle
import json
import os

# Directory containing configuration files (relative to PROJECT_HOME)
CONFIG_DIR = 'config/params'

# ------------------------------------------- Environment configuration ------------------------------------------

PROJECT_HOME = os.environ.get('PROJECT_HOME')
CONFIG_DIR = os.path.join(PROJECT_HOME, CONFIG_DIR)

# Load paths to required directories
with open(os.path.join(CONFIG_DIR, 'dirs.json')) as dir_file:
    dirs = json.load(dir_file)

# Load pipeline parameters
with open(os.path.join(CONFIG_DIR, 'pipeline.json')) as pipeline_file:
    pipeline_param = json.load(pipeline_file)

# Load training parameters
with open(os.path.join(CONFIG_DIR, 'fit.json')) as fit_file:
    fit_param = json.load(fit_file)

# Load logging parameters
with open(os.path.join(CONFIG_DIR, 'logging.json')) as logging_file:
    logging_param = json.load(logging_file)

# ------------------------------------------- Tensorflow configuration -------------------------------------------

# Limit GPU's memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(
            memory_limit=fit_param['environment']['gpu_memory_cap_mb']
        )
    ])
  except RuntimeError as e:
    print(e)

# Verbose data placement info
tf.debugging.set_log_device_placement(fit_param['environment']['tf_device_verbosity'])

# -------------------------------------------------- Prepare data ------------------------------------------------

# Get number of classes
num_classes = len(glob(os.path.join(dirs['training'], '*')))

# Load an example image from training dataset to establish input_shape
input_shape = \
    list( Image.open(os.path.join(PROJECT_HOME, glob(os.path.join(dirs['training'], '*/*.jp*g'))[0])).size ) + [3]

# Create data pipe (contains training and validation sets)
pipe = DataPipe()
pipe.initialize(
    dirs['training'],
    dirs['validation'],
    val_split=pipeline_param['valid_split'],
    test_split=pipeline_param['test_split'],
    dtype='float32',
    batch_size=fit_param['batch_size'],
    shuffle_buffer_size=pipeline_param['shuffle_buffer_size'],
    prefetch_buffer_size=pipeline_param['prefetch_buffer_size']
)

# Augment the data pipe
pipe.training_set = ImageAugmentation(
    rotation_range=pipeline_param['augmentation']['rotation_range'],
    brightness_range=pipeline_param['augmentation']['brightness_range'],
    contrast_range=pipeline_param['augmentation']['contrast_range'],
    shear_x_range=pipeline_param['augmentation']['shear_x_range'],
    shear_y_range=pipeline_param['augmentation']['shear_y_range'],
    shear_fill=pipeline_param['augmentation']['shear_fill'],
    vertical_flip=pipeline_param['augmentation']['vertical_flip'],
    horizontal_flip=pipeline_param['augmentation']['horizontal_flip'],
    dtype='float32'
)(pipe.training_set)

# Apply batching to the data sets
pipe.apply_batch()

# -------------------------------------------------- Build model -------------------------------------------------

# Prepare initializers
kernel_initializer = tf.keras.initializers.get(fit_param['kernel_initializer'])
bias_initializer = tf.keras.initializers.get(fit_param['bias_initializer'])

# Create preprocessing layer
model_input = tf.keras.layers.Input(input_shape, dtype='float32')
preprocessing = vgg19.preprocess_input(model_input)

# Construct base model
vgg = vgg19.VGG19(
    weights='imagenet',
    include_top=False
)

# Compute number of original layers not to be reinitialized
to_keep = len(vgg.layers) - fit_param['vgg_layers_to_reinitialize']
if fit_param['vgg_layers_to_reinitialize'] == -1:
    to_keep = 0

# Turn-off original VGG19 layers' trainability
for layer in vgg.layers[:to_keep]:
    layer.trainable = False

# Reinitialize original layers that will be trained
for layer in vgg.layers[to_keep:]:
    
    # Check if layer has wights to reinitialize
    if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):

        # Save old weights and baises
        weights, biases = layer.get_weights()

        # Reinitialize
        weights = kernel_initializer(shape=weights.shape)
        biases = bias_initializer(shape=biases.shape)

        # Set new weights to the layer
        layer.set_weights([weights, biases])

# Concatenate model and the preprocessing layer
model = vgg(preprocessing)

# Add Dense layers
model = Flatten()(model)
model = Dense(
    4096,
    activation='relu',
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer)(model)
model = Dense(
    4096,
    activation='relu',
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer)(model)
model = Dense(
    num_classes,
    activation='softmax',
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer)(model)

# Initialize optimizer
optimizer = tf.keras.optimizers.get({
    "class_name": fit_param['optimization']['optimizer'],
    "config": {"learning_rate": fit_param['optimization']['learning_rate']['init']}}
)

# Compile model
model = Model(inputs=[model_input], outputs=[model])
model.compile(
    loss=fit_param['optimization']['loss'],
    optimizer=optimizer,
    metrics=logging_param['metrics']
)

# Load base model's weights
if fit_param['base_model'] is not None:
    model.load_weights(os.path.join(dirs['models'] ,fit_param['base_model']))

# ----------------------------------------------- Prepare callbacks ----------------------------------------------

callbacks = []

# Create a logging callback (Tensorboard)
if logging_param['log_name'] is not None:

    # Create full logdir name
    logdir = os.path.join(PROJECT_HOME, dirs['logs'])
    logdir = os.path.join(logdir, logging_param['log_name'])

    # Create callback
    tensorboard_callback = LRTensorBoard(
        log_dir=logdir, 
        histogram_freq=logging_param['tensorboard']['histogram_freq'],
        write_graph=logging_param['tensorboard']['write_graph'],
        write_images=logging_param['tensorboard']['write_images'],
        update_freq=logging_param['tensorboard']['update_freq'],
        profile_batch=logging_param['tensorboard']['profile_batch']
    )
    callbacks.append(tensorboard_callback)

# Create a confusion matrix callback
if logging_param['log_name'] is not None:
    class_folders = glob(os.path.join(dirs['validation'], '*'))
    class_names = [os.path.basename(folder) for folder in class_folders]
    class_names.sort()
    cm_callback = ConfusionMatrixCallback(
        logdir=os.path.join(logdir, 'validation/cm'),
        validation_set=pipe.validation_set,
        class_names=class_names,
        freq=logging_param['confusion_matrix']['freq'],
        fig_size=logging_param['confusion_matrix']['size'],
        raw_fig_type=logging_param['confusion_matrix']['raw_ext'],
        to_save=logging_param['confusion_matrix']['to_save']
    )
    callbacks.append(cm_callback)

# Create a checkpoint callback
modeldir  = os.path.join(PROJECT_HOME, dirs['models'])
if logging_param['log_name'] is not None:
    modeldir = os.path.join(modeldir, logging_param['log_name'])
    os.makedirs(modeldir, exist_ok=True)
modelname = os.path.join(modeldir, 'weights-epoch_{epoch:02d}-val_loss_{val_loss:.2f}.hdf5')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=modelname,
    save_weights_only=True,
    verbose=True,
    save_freq='epoch',
    save_best_only=logging_param['save_best_only']
)
callbacks.append(checkpoint_callback)

# Create learning rate scheduler callback
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=fit_param['optimization']['learning_rate']['indicator'],
    factor=fit_param['optimization']['learning_rate']['reduce_factor'],
    patience=fit_param['optimization']['learning_rate']['patience'],
    verbose=fit_param['optimization']['learning_rate']['verbosity'],
    min_delta=fit_param['optimization']['learning_rate']['min_delta'],
    cooldown=fit_param['optimization']['learning_rate']['cooldown'],
    min_lr=fit_param['optimization']['learning_rate']['min']
)
callbacks.append(lr_callback)

# -------------------------------------------------- Train model -------------------------------------------------

# Start training
history = model.fit(
    x=pipe.training_set,
    validation_data=pipe.validation_set,
    epochs=fit_param['epochs'],
    initial_epoch=fit_param['initial_epoch'],
    steps_per_epoch=fit_param['steps_per_epoch'],
    callbacks=callbacks,
    verbose=fit_param['environment']['verbosity'],
    workers=fit_param['environment']['workers'],
    use_multiprocessing=True if fit_param['environment']['workers'] != 1 else False,
    shuffle=False
)

# Save training history
if logging_param['log_name'] is not None:

    # Create full name of the output file
    historydir = os.path.join(PROJECT_HOME, dirs['history'])
    historyname = os.path.join(historydir, logging_param['log_name'])

    # If the name exist it means that the training is continued - add number of the subrun
    if os.path.exists(historyname + '.pickle'):

        # Establish nuber of the subrun and modify the name
        subruns_already = len(glob(historyname + '*'))
        historyname += '_continue_{:d}'.format(subruns_already)

    with open(historyname + '.pickle', 'wb') as history_file:
        pickle.dump(history.history, history_file)

# --------------------------------------------------- Test model -------------------------------------------------

if pipe.test_set is not None and logging_param['log_name'] is not None and logging_param['test'] is not None:

    testdir = os.path.join(PROJECT_HOME, dirs['test'])

    # If the best models hould be evaluated, load appropriate weights
    if logging_param['test'] == 'best':

        # Find epoch's index of the best score
        best_epoch = 21#np.nanargmin(np.array(history.history['val_loss'])) + 1

        # Find the weights file
        weights_file = glob(os.path.join(modeldir, '*epoch_{:d}*'.format(best_epoch)))[0]

        # Load weights
        model.load_weights(weights_file)

    # Establish testlog files names; mark whether best or last model test is performed
    testfile_name = logging_param['log_name'] + '_' + logging_param['test']

    # If the test log already exists it means that the training is continued - add number of the subrun
    testfile_path = os.path.join(testdir, testfile_name + '.pickle')
    if os.path.exists(testfile_path + '.' + logging_param['confusion_matrix']['raw_ext']):

        # Establish nuber of the subrun and modify the name
        subruns_already = len(glob(os.path.join(testdir, testfile_name + '*')))
        testfile_name += '_continue_{:d}'.format(subruns_already)

    # Prepare a new Confusion Matrix callback for the test set
    cm_callback = ConfusionMatrixCallback(
        logdir=os.path.join(testdir, 'cm'),
        validation_set=pipe.test_set,
        class_names=class_names,
        freq=logging_param['confusion_matrix']['freq'],
        fig_size=logging_param['confusion_matrix']['size'],
        raw_fig_type=logging_param['confusion_matrix']['raw_ext'],
        to_save=logging_param['confusion_matrix']['to_save'],
        basename=testfile_name
    )

    # Wrap Confusion Matrix callback to be usable with tf.keras.Model.evaluate() method
    cm_callback.set_model(model)
    cm_callback_decorator = \
        tf.keras.callbacks.LambdaCallback(on_test_end=lambda logs: cm_callback.on_epoch_end('', logs))

    # Evaluate test score
    test_dict = model.evaluate(
        x=pipe.test_set,
        verbose=fit_param['environment']['verbosity'],
        workers=fit_param['environment']['workers'],
        use_multiprocessing=True if fit_param['environment']['workers'] != 1 else False,
        return_dict=True,
        callbacks=[cm_callback_decorator]
    )

    # Save test score
    if logging_param['log_name'] is not None:
        testname = os.path.join(testdir, testfile_name + '.pickle')
        with open(testname, 'wb') as test_file:
            pickle.dump(test_dict, test_file)
