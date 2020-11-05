# Pretrained VGG-19
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import VGG19
# Model API
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Model
# Dataset manipulation
from keras.preprocessing.image import ImageDataGenerator
# Images manipulation
from PIL import Image
# Utilities
from keras.callbacks import TensorBoard
from datetime import datetime
import tensorflow as tf
from glob import glob
import json
import os

# Directory containing configuration files (relative to PROJECT_HOME)
CONFIG_DIR = 'config/learning'

# ---------------------------------------- Prepare configuration ----------------------------------------

PROJECT_HOME = os.environ.get('PROJECT_HOME')
CONFIG_DIR = os.path.join(PROJECT_HOME, CONFIG_DIR)

# Load directories info
with open(os.path.join(CONFIG_DIR, 'dirs.json')) as dir_file:
    dirs = json.load(dir_file)

# Get number of classes
folders = glob(os.path.join(dirs['training'], '*'))
num_classes = len(folders)

# Establish input_shape
images = glob(os.path.join(dirs['training'], '*/*.jp*g'))
im = Image.open(os.path.join(PROJECT_HOME, images[0]))
input_shape = list(im.size) + [3]

# Load parameters
with open(os.path.join(CONFIG_DIR, 'param.json')) as param_file:
    param = json.load(param_file)

# --------------------------------------------- Prepare data --------------------------------------------

# Create training data generator
training_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

training_gen = training_gen.flow_from_directory(
    directory=dirs['training'],
    target_size=input_shape[0:2],
    batch_size=param['training']['batch_size'],
)

# Create Validation data generator
validation_gen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
)

validation_gen = validation_gen.flow_from_directory(
    dirs['validation'],
    target_size=input_shape[0:2],
    batch_size=200,
    class_mode='categorical'
)

# --------------------------------------------- Build model ---------------------------------------------

# Construct model
vgg = VGG19(
    input_shape=input_shape,
    weights = 'imagenet',
    include_top = False
)

# Turn-off original VGG19 layers training
for layer in vgg.layers:
    layer.trainable = False

model = Flatten()(vgg.output)
model = Dense(4096, activation='relu')(model)
model = Dense(4096, activation='relu')(model)
model = Dense(num_classes, activation='softmax')(model)

# Compile model
model = Model(inputs=vgg.input, outputs=model)
model.compile(
    loss=param['training']['loss'],
    optimizer=param['training']['optimizer'],
    metrics=param['training']['metrics']
)

# --------------------------------------------- Train model ---------------------------------------------

# Set logging system
glob_logdir = os.path.join(PROJECT_HOME, "logs")
logdir = os.path.join(glob_logdir, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=0)

# Start training
model.fit(
    x=training_gen,
    steps_per_epoch=1000,
    epochs=param['training']['epochs'],
    validation_data=validation_gen,
    verbose=True,
    callbacks=[tensorboard_callback]
)