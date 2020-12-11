import os
import sys
import keras
import numpy as np
import tensorflow as tf
from glob import glob
from functools import partial


class DataPipe:

    """
    Data Pipe class implements basic set of methods used to load and preprocess image
    data as well as to produce data batches in the tensorflow-generator style. It also
    contains some static methods for inspecting and manipulating datasets.
    """

    def __init__(self):
        pass

    @staticmethod
    def dataset_size_from_dir(directory, dtype='float32', l_dtype=None):

        """
        Computes size of the dataset as it was if converted to the particular dtype.
        The directory should contai a set of folders named with the data's labels.
        Every subdirectory should hold set of images of the given class.

        Params
        ------
        directory : str
            path to the directory to be searched
        dtype : str or np.dtype
            data type that image will be casted to when loaded to the memory
        l_dtype : None or str or np.dtype
            data type that image's label will be casted to when loaded to the memory

        Returns
        -------
        tuple
            tuple holding (images_size, labels_size)
            
        """

        folders = glob(os.path.join(directory, '*'))

        # Load images and labels from directories to RAM
        with tf.device('/CPU:0'):

            img_size = 0
            label_size = 0

            # Load the exampels
            for f in folders:
                images = glob(os.path.join(f, '*.jp*g'))
                for i in images:
                    
                    # Compute image's size
                    img = keras.preprocessing.image.img_to_array(    
                            keras.preprocessing.image.load_img(i),
                            dtype=dtype
                        ) 
                    img_size += img.nbytes

                    # Compute label's size
                    if(l_dtype == None):
                        label_size += sys.getsizeof(os.path.basename(f))
                    else:
                        label_size += np.dtype(l_dtype).itemsize

        return img_size, label_size 


    @staticmethod
    def dataset_from_directory(directories, size=None, dtype='uint8', channels=3):

        """
        Preares a tf.data.Dataset from image data in the directories. Every directory should contain
        a set of subfolders named with data labels. Subfolders should hold images for the given label. 

        Note
        ----
        Only JPG-decode files are possible to use

        Params
        ------
        directories : list of str
            list of paths to the directories holding datasets
        size : tuple of int or None
            if None, files' sizes are preserved when loading
            if tuple, every loaded image is resized to the size[0] x size[1]
        dtype : str or np.dtype
            type that the image data will be casted to when loaded
        channels : 
            number of image's channel

        Note
        ----
        If size=None, sizes of all images should be equal

        Returns
        -------
        if.data.Dataset
            dataset of (image, label) tuples

        """

        def file_to_training_example(path):

            """
            Loads image with the given path. Inspect's the image's label (name of the folder
            that image is hold in). Transforms the image to a tf.Tensor.

            Params
            ------
            path : str
                path to the image file

            Returns
            -------
            tuple
                (image, label) pair

            """ 
            
            # Split the path to the list components
            path_split = tf.strings.split(path, os.path.sep)
            # The second to last part of the path is the class label
            label = path_split[-2]

            # Load the raw image from the file
            img = tf.io.read_file(path)
            # Decode the JPEG file
            img = tf.image.decode_jpeg(img, channels=channels)
            # Convert to the required data type
            img = tf.image.convert_image_dtype(img, tf.dtypes.as_dtype(dtype))
            # Resize the image
            if size is not None:
                img = tf.image.resize(img, [size[0], size[1]])

            return img, label


        # Map of {'directory': tf.data.Dataset} pairs
        datasets = {}

        for d in directories:
        
            # Get list of image files in the directory
            ds = tf.data.Dataset.list_files(os.path.join(d, '*/*.jp*g'), shuffle=False)

            # Convert filenames of images to the dataset
            ds = ds.map(file_to_training_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            datasets[d] = ds

        return datasets
