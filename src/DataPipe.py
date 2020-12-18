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

        """
        Initializes a new DataPipe. The self.initialize() call is required before the
        pipe can be used.
        """

        # Internal datasets
        self.training_set = None
        self.validation_set = None

        # Values hold to be applied on self.apply_batch() call
        self.batch_size = None
        self.prefetch_buffer_size = None

        # Object status
        self.initialized = False
        self.batched = False


    def initialize(self,
        training_dir, 
        validation_dir,
        dtype='uint8',
        ldtype='uint8',
        batch_size=64, 
        shuffle_buffer_size=None,
        prefetch_buffer_size=None
    ):
        """
        Initializes new training and validation datasets hold by the object.

        Params
        ------
        training_dir : string
            directory of the training data (@see DataPipe.dataset_from_directory())
        validation_dir : string
            directory of the validation data (@see DataPipe.dataset_from_directory())
        dtype : string or np.dtype
            type of the images' representation
        ldtype : string or np.dtype
            type of the images labels' representation
        batch_size : int 
            size of the batch
        shuffle_buffer_size : int or None
            size of the buffer used to shuffle the dataset (@see tf.data.Dataset.shuffle())
            if None, shuffling is not performed
        prefetch_buffer_size : int or None
            size of the buffer used to prefetch the data (@see tf.data.Dataset.prefetch())
            if None, prefetching is not performed

        """

        # Create and shuffle a training set
        self.training_set = self.dataset_from_directory([training_dir], dtype=dtype, ldtype=ldtype)[training_dir]
        if shuffle_buffer_size is not None:
            self.training_set = self.training_set.shuffle(shuffle_buffer_size)
        # Create a validation set
        self.validation_set = self.dataset_from_directory([validation_dir], dtype=dtype, ldtype=ldtype)[validation_dir]

        # Holding batching informations for self.apply_batch() call
        self.batch_size = batch_size
        self.prefetch_buffer_size = prefetch_buffer_size

        # Update object's state
        self.initialized = True
        self.batched = False
        
    
    def apply_batch(self):

        """
        Applies batching and sets prefetching buffers for both training and validation
        dataset in an initialized object.
        """

        if self.initialized and not self.batched:

            # Establish batch size and set prefetch buffer for training set
            self.training_set = self.training_set.batch(self.batch_size)
            if self.prefetch_buffer_size is not None:
                self.training_set = self.training_set.prefetch(self.prefetch_buffer_size)

            # Establish batch size and set prefetch buffer for validation set
            self.validation_set = self.validation_set.batch(self.batch_size)
            if self.prefetch_buffer_size is not None:
                self.validation_set = self.validation_set.prefetch(self.prefetch_buffer_size)
        

    @staticmethod
    def dataset_size_from_dir(directory, dtype='float32', ldtype=None):

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
                    if ldtype is None:
                        label_size += sys.getsizeof(os.path.basename(f))
                    else:
                        label_size += np.dtype(ldtype).itemsize

        return img_size, label_size 


    @staticmethod
    def dataset_from_directory(directories, size=None, dtype='uint8', ldtype='uint8', channels=3):

        """
        Prepares a tf.data.Dataset from images data in the directories. Every directory should contain
        a set of subfolders named with data labels. Subfolders should hold images for the given label. 
        Labels are encoded as one-hot vectors.

        Note
        ----
        Only JPG-encoded files are possible to use

        Params
        ------
        directories : list of str
            list of paths to the directories holding datasets
        size : tuple of int or None
            if None, files' sizes are preserved when loading
            if tuple, every loaded image is resized to the size[0] x size[1]
        dtype : str or np.dtype or tf.dtypes
            type that the image data will be casted to when loaded
        ldtype : str or np.dtype or tf.dtypes
            type of the elements of the categorical vectors that labels will be casted to
        channels : int 
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
            Loads image with the given path and tranforms it to a tf.Tensor.

            Params
            ------
            path : str
                path to the image file

            Returns
            -------
            tuple
                (image, label) pair

            """

            # Load the raw image from the file
            img = tf.io.read_file(path)
            # Decode the JPEG file
            img = tf.image.decode_jpeg(img, channels=channels)
            # Convert to the required data type
            img = tf.cast(img, tf.dtypes.as_dtype(dtype))
            # Resize the image
            if size is not None:
                img = tf.image.resize(img, [size[0], size[1]])

            return img

        # Map of {'directory': tf.data.Dataset} pairs
        datasets = {}

        for d in directories:

            # Get list of image files in the directory
            files = glob(os.path.join(d, '*/*.jp*g'))
            files.sort()

            # Associate number values with the classes' subdirectories
            labels_dirs = glob(os.path.join(d, '*'))
            labels_dirs.sort()
            labels_dict = {}
            for i, l in enumerate(labels_dirs):
                labels_dict[l] = i

            # Create list of one-hot labels for the files
            labels = []
            for f in files:
                # Get a file's directory
                label_dir = os.path.dirname(f)
                # Create the one-hot label
                label = tf.one_hot(labels_dict[label_dir], depth=len(labels_dict), dtype=ldtype)
                # Add the pair to the dataset
                labels.append(label)
                
            # Transform list dataset to the tf.data.Dataset
            ds = tf.data.Dataset.from_tensor_slices((files, labels))

            # Convert filenames of images to the dataset
            ds = ds.map(
                lambda file, label: (file_to_training_example(file), label),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

            datasets[d] = ds

        return datasets
