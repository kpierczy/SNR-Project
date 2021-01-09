# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2021-01-09 17:58:12
# @ Modified time: 2021-01-09 17:59:01
# @ Description:
#
#     Configuration file containing settings of the training's parameters
#
# ================================================================================================================

training_params = {
    
    # Batch size
    'batch_size' : 64,

    # Optimization algorithm settings
    'optimization' : {

        # TF Optimizer's identifier
        'optimizer' : 'adam',

        # TF Loss function's identifier
        'loss' : 'categorical_crossentropy',

        # Learning rate settings (@see tf.keras.callbacks.ReduceLROnPlateau)
        'learning_rate' : {

            # Reduction's indicator 
            'indicator' : 'val_loss',

            # Initial value
            'init': 1e-3,

            # Minimal value
            'min' : 1e-7,

            # Minimal change
            'min_delta' : 1e-7,

            # Reduction factor
            'reduce_factor': 2e-1,

            # Patience (in epochs)
            'patience' : 4,

            # Cooldown
            'cooldown' : 0,

            # Changes' verbosity
            'verbosity' : 1
        }
    },

    # Training's length
    'epochs' : 4,

    # Index of the initial epoch (handy for training's continuation)
    'initial_epoch' : 2,

    # Number of batches per epoch (None if the whole dataset shou7ld be fed)
    'steps_per_epoch' : 50,

    # Training workers
    'workers' : 4,

    # Training's verbosity
    'verbosity' : 1,
}