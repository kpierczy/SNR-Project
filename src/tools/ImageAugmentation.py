import tensorflow as tf
import tensorflow_addons as tfa
import math

class ImageAugmentation:

    """
    ImageAugmentation class represents set of augmentation operations that can be
    applied to the tf.data.Dataset object.
    """

    def __init__(self,
        rotation_range=(0, 0),
        brightness_range=(0, 0),
        contrast_range=(0, 0),
        shear_x_range=(0,0),
        shear_y_range=(0,0),
        shear_fill=(0, 0, 0),
        vertical_flip=False,
        horizontal_flip=False,
        dtype='uint8'
    ):

        """
        Initializes object representing set of the augmentation operations applied
        to the tf.data.Dataset at the self.__call__() call.

        Params | Attributes
        -------------------
        rotation_range : tuple or list of two ints
            [min; max) range for random rotations 
        brightness_range : tuple or list of two floats
            [min; max) range for random brightness shift
        contrast_range : tuple or list of two floats
            [min; max) range for random contrast adjustement
        shear_x_range : tuple or list of two floats
            [min; max) range for random x-shear adjustement (@see tfa.image.shear_x())
        shear_y_range : tuple or list of two floats
            [min; max) range for random y-shear adjustement (@see tfa.image.shear_y())
        shear_fill : int or tuple/list of three ints
            values of the pixels needed to be filled after the shear
        vertical_flip : bool
            if True, 50% of images are flipped up-down
        horizontal_flip : bool
            if True, 50% of images are flipped left-right
        dtype : string or tf.dtype
            type of the images' representation
        
        [To Do]
        -------
        Parameters verification

        """

        self.rotation_range   = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range   = contrast_range
        self.shear_x_range    = shear_x_range
        self.shear_y_range    = shear_y_range
        self.shear_fill       = shear_fill
        self.vertical_flip    = vertical_flip
        self.horizontal_flip  = horizontal_flip
        self.dtype            = dtype


    def __call__(self, dataset):

        """
        Applies augmentation operations to the dataset

        Params
        ------
        dataset : tf.data.Dataset
            the dataset to be augmented

        Returns
        -------
        aug_dataset : dataset: tf.data.Dataset
            the augmented dataset

        Note
        ----
        For shearing operation images need to be converted into 'uin8' format, so excessive
        partial values will be discarded.

        """

        def rand_range(lower, upper):
            
            """
            Wrapper around tf.random.uniform() for shorter calls
            """

            return tf.random.uniform([], minval=lower, maxval=upper, dtype=self.dtype)


        # Random shear-x
        if self.shear_x_range[0] != 0 or self.shear_x_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.cast(tfa.image.shear_x(tf.cast(x, 'uint8'), rand_range(self.shear_x_range[0], self.shear_x_range[1]), 0), self.dtype), y )
            )

        # Random shear-y
        if self.shear_y_range[0] != 0 or self.shear_y_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.cast(tfa.image.shear_y(tf.cast(x, 'uint8'), rand_range(self.shear_y_range[0], self.shear_y_range[1]), 0), self.dtype), y )
            )

        # Random rotations
        if self.rotation_range[0] != 0 or self.rotation_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tfa.image.rotate(x, math.pi / 180 * rand_range(self.rotation_range[0], self.rotation_range[1])), y )
            )

        # Random brightness disturbance
        if self.brightness_range[0] != 0 or self.brightness_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.adjust_brightness(x, rand_range(self.brightness_range[0], self.brightness_range[1])), y )
            )

        # Random contrast disturbance
        if self.contrast_range[0] != 0 or self.contrast_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.adjust_contrast(x, rand_range(self.brightness_range[0], self.brightness_range[1])), y )
            )
        
        # Random x-wise flip
        if self.vertical_flip:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.random_flip_left_right(x), y )
            )

        # Random x-wise flip
        if self.horizontal_flip:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.random_flip_up_down(x), y )
            )
         
        return dataset