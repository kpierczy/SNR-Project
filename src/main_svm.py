import tensorflow as tf
# Keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
# Configuration files
from config.dirs import dirs
from config.devices import gpu_params
from config.logs import logging_params
from config.model import model_params
from config.pipe import pipeline_params
from config.training import training_params
# training API
from src.api.SVMTrainer import SVMTrainer
# Images manipulation
from PIL import Image
# Utilities
from glob import glob
import os

# ------------------------------------------- Environment configuration ------------------------------------------

# Get project's path

PROJECT_HOME = '/home/krzysztof/PycharmProjects/SNR-Project'

# Resolve all relative paths to absolute paths
dirs["training"] = os.path.join(PROJECT_HOME, dirs["training"])
dirs["validation"] = os.path.join(PROJECT_HOME, dirs["validation"])
dirs["output"] = os.path.join(PROJECT_HOME, dirs["output"])
dirs['test_network_output'] = os.path.join(PROJECT_HOME, dirs['test_network_output'])
dirs['train_network_output'] = os.path.join(PROJECT_HOME, dirs['train_network_output'])

# Get number of classes
num_classes = len(glob(os.path.join(PROJECT_HOME, os.path.join(dirs['training'], '*'))))

# Load an example image from training dataset to establish input_shape
input_shape = \
    list(Image.open(glob(os.path.join(PROJECT_HOME, os.path.join(dirs['training'], '*/*.jp*g')))[0]).size) + [3]

# Limit GPU's memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=gpu_params['memory_cap_mb']
            )
        ])
    except RuntimeError as e:
        print(e)

# Verbose data placement info
tf.debugging.set_log_device_placement(gpu_params['tf_device_verbosity'])

# -------------------------------------------------- Build model -------------------------------------------------

# Prepare initializers
kernel_initializer = tf.keras.initializers.get(model_params['initializer']['kernel'])
bias_initializer = tf.keras.initializers.get(model_params['initializer']['bias'])

# Create preprocessing layer
model_input = tf.keras.layers.Input(input_shape, dtype='float32')
preprocessing = tf.keras.applications.vgg19.preprocess_input(model_input)

# Construct base model
vgg_imagenet = tf.keras.applications.vgg19.VGG19(
    weights='imagenet',
    include_top=False
)

# Remove required number of final layers from original vgg
vgg = tf.keras.Model(
    inputs=vgg_imagenet.input,
    outputs=vgg_imagenet.layers[-(model_params['vgg_to_remove'] + 1)].output,
    name='vgg19'
)

# Reinitialize original convolutional layers that will be trained
reinitialized = 0
for layer in reversed(vgg.layers):
    # # Reinitialize required convolutional layers
    # if model_params['vgg_conv_to_train'] is None or reinitialized < model_params['vgg_conv_to_train']:
    #
    #     # Check if layer has wights to reinitialize
    #     if isinstance(layer, tf.keras.layers.Conv2D):
    #         # Save old weights and baises
    #         weights, biases = layer.get_weights()
    #
    #         # Reinitialize
    #         weights = kernel_initializer(shape=weights.shape)
    #         biases = bias_initializer(shape=biases.shape)
    #
    #         # Set new weights to the layer
    #         layer.set_weights([weights, biases])
    #
    #         # Increment counter of reinitialized layers
    #         reinitialized += 1
    #
    #         continue

    # Froze rest of layers
    layer.trainable = False

# Concatenate model and the preprocessing layer
model = vgg(preprocessing)

# Add Dense layers
model = Flatten()(model)
# model = Dense(
#     4096,
#     activation='relu',
#     kernel_initializer=kernel_initializer,
#     bias_initializer=bias_initializer)(model)
# model = Dense(
#     4096,
#     activation='relu',
#     kernel_initializer=kernel_initializer,
#     bias_initializer=bias_initializer)(model)
# model = Dense(
#     num_classes,
#     activation='softmax',
#     kernel_initializer=kernel_initializer,
#     bias_initializer=bias_initializer)(model)

# Create model
model = Model(inputs=[model_input], outputs=[model])

# Load base model's weights
if model_params['base_model'] is not None:
    model.load_weights(os.path.join(PROJECT_HOME, model_params['base_model']))

# ------------------------------------------------- Run training -------------------------------------------------

# Initialize training API
trainer = SVMTrainer(
    model=model,
    dirs=dirs,
    logging_params=logging_params,
    pipeline_params=pipeline_params,
    training_params=training_params
)

# Run training
trainer.initialize().run()
# import pickle
# with open('/home/erucolindo/Dokumenty/Projekty/Python/SNR-Project/models/perceptron/run_3/history/subrun_1.pickle', mode='rb') as f:
#     o = pickle.load(f)
#     print(o)
