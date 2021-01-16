# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2021-01-09 17:15:23
# @ Modified time: 2021-01-09 17:16:01
# @ Description:
#
#     Configuration file containing information about logging parameters.
#
# ================================================================================================================

logging_params = {

    # Turn on/off logging
    'on': True,
    
    # @brief:
    #    Indicator of the model that should be evaluated on test dataset after training 
    #
    # @arg 'best': 
    #    the best model will be evaluated
    # @arg 'last': 
    #    the last model will be evaluated
    # @arg None: 
    #    no test evaluation will be run
    'test_model' : 'best',

    # Metrics to be monitored during training
    'metrics' : ['accuracy', 'mse'],

    # Tensorboard settings
    'tensorboard' : {

        # Frequency (in epochs) of histograms drawing
        'histogram_freq' : 1,

        # Frequency of metrics' output ('batch' or 'epoch')
        'update_freq' : 'epoch',

        # Computational graph drawing
        'write_graph' : True,

        # Writting model's weights as an image
        'write_images' : False,

        # Index of the batch to be a base fot profiling (0 to disable profiling)
        'profile_batch' : 0
    },

    # Confusion matrix log
    'confusion_matrix' : {

        # Frequency (in epochs) of matrix' drawing during training [0 to disable]
        'freq' : 10,

        # Extension of the raw matrix image
        'raw_ext' : 'png',

        # Saving mode: 'tf', 'raw' or 'both' (@see ConfusionMatrixCallback.__init__())
        'to_save' : 'both',

        # Size of the matrix image in cm [wight, height]
        'size' : [180, 180]
    },

    # Whether to save only models with improved validation loss
    'save_best_only' : True
}