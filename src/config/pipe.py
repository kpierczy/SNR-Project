# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2021-01-09 17:15:23
# @ Modified time: 2021-01-09 17:18:01
# @ Description:
#
#     Configuration file containing settings of the input data pipeline
#
# ================================================================================================================

pipeline_params = {

    # Ratio of the validation:training datasets' sizes (like valid_test_split[0]:valid_test_split[1])
    'valid_test_split' : [1, 1],

    # Size of the shuffle buffer for training pipeline
    'shuffle_buffer_size' : 5000,

    # Size of the prefetch buffers (None for autotune)
    'prefetch_buffer_size' : None,

    # Pipeline's augmentation 
    'augmentation' : {

        # Random rotations' range (in degrees)
        'rotation_range' : [0, 0],

        # Random brightness variations' range (in [0,255])
        'brightness_range' : [0, 0],

        # Random contrast variations' range (in [0,255])
        'contrast_range' : [0, 0],

        # Random x-wise shear variations' range
        'shear_x_range' : [0, 0],

        # Random y-wise shear variations' range
        'shear_y_range' : [0, 0],

        # Value of the filling pixels for shear operations
        'shear_fill' : [0, 0, 0],

        # Random vertical flips
        'vertical_flip' : False,

        # Random horizontal flips
        'horizontal_flip' : False
    }

}