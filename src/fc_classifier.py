# =================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2020-12-09 18:16:36
# @ Modified time: 2020-12-11 12:02:21
# @ Description:
# =================================================================================================================

# Pretrained VGG-19
from tensorflow.keras.applications import vgg19
# Model API
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
# Dedicated datapipe
from ImageAugmentation import ImageAugmentation
from DataPipe import DataPipe
# Images manipulation
from PIL import Image
# Utilities
from tensorflow.keras import callbacks
from datetime import datetime
import tensorflow as tf
from glob import glob
import json
import os

# Directory containing configuration files (relative to PROJECT_HOME)
CONFIG_DIR = 'config/params'


# ---------------------------------------- Prepare configuration ----------------------------------------

# Limit GPU's memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2560)])
  except RuntimeError as e:
    print(e)

# tf.debugging.set_log_device_placement(True)

PROJECT_HOME = os.environ.get('PROJECT_HOME')
CONFIG_DIR = os.path.join(PROJECT_HOME, CONFIG_DIR)

# Load directories info
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

# Get number of classes
folders = glob(os.path.join(dirs['training'], '*'))
num_classes = len(folders)

# Establish input_shape
images = glob(os.path.join(dirs['training'], '*/*.jp*g'))
im = Image.open(os.path.join(PROJECT_HOME, images[0]))
input_shape = list(im.size) + [3]

# Establish number of batches per epoch (the whole dataset if 0)
if fit_param['steps_per_epoch'] != 0:
    steps_per_epoch = fit_param['steps_per_epoch']
else:
    steps_per_epoch = None


# --------------------------------------------- Prepare data --------------------------------------------

# Create data pipe (contains training and validation sets)
pipe = DataPipe()
pipe.initialize(
    dirs['training'],
    dirs['validation'],
    dtype='float32',
    batch_size=fit_param['batch_size'],
    shuffle_buffer_size=pipeline_param['shuffle_buffer_size'],
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE
)

# Augment the data pipe
pipe.training_set = ImageAugmentation(
    rotation_range=pipeline_param['augmentation']['rotation_range'],
    brightness_range=pipeline_param['augmentation']['brightness_range'],
    contrast_range=pipeline_param['augmentation']['contrast_range'],
    shear_x=pipeline_param['augmentation']['shear_x'],
    shear_y=pipeline_param['augmentation']['shear_y'],
    shear_fill=pipeline_param['augmentation']['shear_fill'],
    vertical_flip=pipeline_param['augmentation']['vertical_flip'],
    horizontal_flip=pipeline_param['augmentation']['horizontal_flip'],
    dtype='float32'
)(pipe.training_set)

# Apply batching to the data sets
pipe.apply_batch()


# --------------------------------------------- Build model ---------------------------------------------

# Construct base model
vgg = vgg19.VGG19(
    weights='imagenet',
    include_top=False
)

# Turn-off original VGG19 layers training
for layer in vgg.layers:
    layer.trainable = False

# Create preprocessing layer
model_input = tf.keras.layers.Input(input_shape, dtype='float32')
preprocessing = vgg19.preprocess_input(model_input)

# Concatenate model and the preprocessing layer
model = vgg(preprocessing)

# Add Dense layers
model = Flatten()(model)
model = Dense(4096, activation='relu')(model)
model = Dense(4096, activation='relu')(model)
model = Dense(num_classes, activation='softmax')(model)

# Compile model
model = Model(inputs=[model_input], outputs=[model])
model.compile(
    loss=fit_param['loss'],
    optimizer=fit_param['optimizer'],
    metrics=fit_param['metrics']
)


# --------------------------------------------- Train model ---------------------------------------------

# Set logging system
glob_logdir = os.path.join(PROJECT_HOME, 'logs')
logdir = os.path.join(glob_logdir, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = callbacks.TensorBoard(
    log_dir=logdir, 
    histogram_freq=logging_param['histogram_freq'],
    write_graph=logging_param['write_graph'],
    write_images=logging_param['write_images'],
    update_freq=logging_param['update_freq'],
    profile_batch=logging_param['profile_batch']
)

# Start training
history = model.fit(
    x=pipe.training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=fit_param['epochs'],
    validation_data=pipe.validation_set,
    verbose=fit_param['environment']['verbosity'],
    callbacks=[tensorboard_callback],
    workers=fit_param['environment']['workers'],
    use_multiprocessing=True
)