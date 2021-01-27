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
            'init': 1e-4,

            # Minimal value
            'min' : 1e-7,

            # Minimal indicator change to be noticed
            'min_delta' : 5e-2,

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

    'svm': {
        'create_network_output': False,
        'train': True,
        'test': True,
        'load_model': False,
        'model_name': 'model.svm'
    },
    # Training's length
    'epochs' : 40,

    # Index of the initial epoch (handy for training's continuation)
    'initial_epoch' : 0,

    # Number of batches per epoch (None if the whole dataset should be proceeded)
    'steps_per_epoch' : None,

    # Training workers
    'workers' : 4,

    # Training's verbosity
    'verbosity' : 1,
}