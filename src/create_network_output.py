# import tensorflow as tf
# from tensorflow.keras.applications import vgg19
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.models import Model
# from tools.ImageAugmentation import ImageAugmentation
# from tools.DataPipe import DataPipe
# from PIL import Image
# from glob import glob
# import json
# import os
# import numpy as np
# import tempfile
import numpy as np
from src.config.dirs import dirs

# CONFIG_DIR = 'config/params'
#
# # ------------------------------------------- Environment configuration ------------------------------------------
#
# PROJECT_HOME = '/home/erucolindo/Dokumenty/Projekty/Python/SNR-Project'
# CONFIG_DIR = os.path.join(PROJECT_HOME, CONFIG_DIR)
#
# # Load paths to required directories
# with open(os.path.join(CONFIG_DIR, 'dirs.json')) as dir_file:
#     dirs = json.load(dir_file)
#
# # Load pipeline parameters
# with open(os.path.join(CONFIG_DIR, 'pipeline.json')) as pipeline_file:
#     pipeline_param = json.load(pipeline_file)
#
# # Load training parameters
# with open(os.path.join(CONFIG_DIR, 'fit.json')) as fit_file:
#     fit_param = json.load(fit_file)
#
# # Load logging parameters
# with open(os.path.join(CONFIG_DIR, 'logging.json')) as logging_file:
#     logging_param = json.load(logging_file)
#
# # -------------------------------------------------- Prepare data ------------------------------------------------
#
# # Get number of classes
# num_classes = len(glob(os.path.join(dirs['training'], '*')))
# #
# # Load an example image from training dataset to establish input_shape
# input_shape = list(Image.open(os.path.join(PROJECT_HOME, glob(os.path.join(dirs['training'], '*/*.jp*g'))[0])).size) + [
#     3]
# #
# # # Create data pipe (contains training and validation sets)
# pipe = DataPipe()
# pipe.initialize(
#     dirs['training'],
#     dirs['validation'],
#     dtype='float32',
#     batch_size=fit_param['batch_size'],
#     shuffle_buffer_size=pipeline_param['shuffle_buffer_size'],
#     prefetch_buffer_size=pipeline_param['prefetch_buffer_size'] if pipeline_param[
#         'prefetch_buffer_size'] else tf.data.experimental.AUTOTUNE
# )
#
# # # Augment the data pipe
# pipe.training_set = ImageAugmentation(
#     rotation_range=pipeline_param['augmentation']['rotation_range'],
#     brightness_range=pipeline_param['augmentation']['brightness_range'],
#     contrast_range=pipeline_param['augmentation']['contrast_range'],
#     shear_x_range=pipeline_param['augmentation']['shear_x_range'],
#     shear_y_range=pipeline_param['augmentation']['shear_y_range'],
#     shear_fill=pipeline_param['augmentation']['shear_fill'],
#     vertical_flip=pipeline_param['augmentation']['vertical_flip'],
#     horizontal_flip=pipeline_param['augmentation']['horizontal_flip'],
#     dtype='float32'
# )(pipe.training_set)
# #
# # # Apply batching to the data sets
# pipe.apply_batch()
# #
# # # -------------------------------------------------- Build model -------------------------------------------------
# #
# # # Construct base model
# vgg = vgg19.VGG19(
#     weights='imagenet',
#     include_top=False
# )
# #
# # # Turn-off original VGG19 layers training
# for layer in vgg.layers:
#     layer.trainable = False
# #
# # # Create preprocessing layer
# model_input = tf.keras.layers.Input(input_shape, dtype='float32')
# preprocessing = vgg19.preprocess_input(model_input)
#
# # Concatenate model and the preprocessing layer
# model = vgg(preprocessing)
#
# # Add Dense layers
# model = Flatten()(model)
#
# # Compile model
# model = Model(inputs=[model_input], outputs=[model])
# model.compile(
#     loss=fit_param['loss'],
#     optimizer=fit_param['optimizer'],
#     metrics=fit_param['metrics']
# )
#
# # Load base model's weights
# if fit_param['base_model'] is not None:
#     model.load_weights(fit_param['base_model'])
#
# for image, result in pipe.training_set:
#     # i = 0
#     network_out = model.predict(image)
#     result = result.numpy()
#     file_name = tempfile.NamedTemporaryFile(delete=False, dir=dirs['test_network_output'], suffix='.npz')
#     np.savez(file_name, out=network_out, result=result)
#     # i += fit_param['batch_size']
#     # print('network output number {}'.format(i))
#
#
# print("preprocessing done")

data_ = np.load(dirs['test_network_output'] + '/' + 'tmpzzq6oyzc.npz')
out = data_['out']
result = data_['result']
data2_ = np.load(dirs['test_network_output'] + '/' + 'tmpzruiy7qt.npz')
out2 = data2_['out']
result2 = data2_['result']

out3 = np.concatenate((out, out2), axis=0)
result3 = np.concatenate((result, result2), axis=0)

print(data_)