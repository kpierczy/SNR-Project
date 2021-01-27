from typing import Dict

from tensorflow.python.keras import Model

import tensorflow as tf

from tools.DataPipe          import DataPipe

class SVMPipe:
    def __init__(self, data_pipe: DataPipe, vgg_model: Model):
        self.data_pipe = data_pipe
        self.vgg_model = vgg_model
        self.training_data = {}
        self.validation_data = {}

    def initialize(self):
        self.training_data = SVMPipe.preprocess_data(self.data_pipe.training_set, self.vgg_model)
        self.validation_data = SVMPipe.preprocess_data(self.data_pipe.validation_set, self.vgg_model)

    @staticmethod
    def preprocess_data(images_data_set: Dict[str, tf.data.Dataset], model: Model):
        preprocessed_data = {}

        for clazz in images_data_set.keys():
            clazz_images = images_data_set[clazz]
            preprocessed_images = model.predict(clazz_images)
            # preprocessed_images = [model.predict(x) for x in clazz_images]
            preprocessed_data[clazz] = preprocessed_images

        return preprocessed_data
