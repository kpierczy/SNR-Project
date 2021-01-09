# Directories config (dirs.json)

1. "training" : 
    - Path to the directory holding training data. (@see `src/tools/DataPipe` class' description)
2. "validation" : 
    - Path to the directory holding validation data. (@see `src/tools/DataPipe` class' description)
3. "models" : 
    - Path to the directory that model's weights will be written into during the training (@note : If logging.json/log_name is set, weights for the given run will be saved in separet subfolder)
4. "logs" : 
    - Path to the directory that tensorboard logs will be written into (@note If logging.json/log_name is `null` the log will not be created).
5. "history" : 
    - Path to the directory that training history will be written into after the training (@note If logging.json/log_name is `null` the training history will not be created)
6. "test" : 
    - Path to the directory that the result of the test dataset evaluation will be written into (@note If logging.json/log_name is `null` the test validation will be not performed)


# Logging config (logging.json)

1. "log_name" : 
    - Basename for files and folders with training-related data generated during learning (`null` to deactivate all logging outputs, i.e. tensorboard metric, confusion matrices and test set evaluation)
2. "test_model" : 
    - either 'best', 'last' or None. If None, no evaluation of the test set is performed after training. Otherwise the final model (for 'last') or the best model, with respect to validation loss (for 'best') is evaluated.
3. "metrics" : 
    - List of names of metrics to be calculated during training (@see tf.keras.Model.compile)
4. "tensorboard/histogram_freq" : 
    - Frequency of the histogram saving (@see tf.keras.callbacks.TensorBoard)
5. "tensorboard/update_freq" : 
    - Frequency of the tensorboard update (@see tf.keras.callbacks.TensorBoard)
6. "tensorboard/write_graph" : 
    - If `true`, the tf computational graph will be save to the tensorboard log
7. "tensorboard/write_images" : 
    - If `true`, the training images will be saved to the tensorboard log
8. "tensorboard/profile_batch" : 
    - (@see tf.keras.callbacks.TensorBoard)
9. "confusion_matrix/freq" : 
    - Frequencu (in epochs) of confusion matrix printin (0: turn off cm generation) (@note If log_name is `null` the confusion maps will be not created)
10. "confusion_matrix/raw_ext" : 
    - Filetype of the raw confusion matrix saved
11. "confusion_matrix/to_save" : 
    - Either "tf" (Tensorboard), "raw" (Raw type) or "both" - type of the matrix image log that will be saved
12. "confusion_matrix/size" : 
    - Size of the confusion matrix plot in cm [width, height]
13. "save_best_only :
    - if `true`, only the best model's weigths will be saved (massively reduces disk usage)


# Pipeline config (pipeline.json)

1. "valid_split" : 
    - ratio of validation:test (like valid_split:test_split) datasets' sizes
1. "test_split" : 
    - ratio of validation:test (like valid_split:test_split) datasets' sizes
2. "shuffle_buffer_size" : 
    - Size of the buffer used to shuffle the dataset (@see tf.data.Dataset.shuffle)
3. "prefetch_buffer_size" : 
    - Number of batches that will be buffered by the CPU during training (multiple workers need to be used to make advantage of this buffering). If `null` tf.data.experimental.AUTOTUNE will be used. If 0, no prefetching is applied.
4. "augmentation/rotation_range" : 
    - range (in degrees) of the random rotations applied to the dataset
5. "augmentation/brightness_range" : 
    - range of the random brightness corrections applied to the dataset (@see tf.image.adjust_brightness())
6. "augmentation/contrast_range" : 
    - range of the random contrast corrections applied to the dataset (@see tf.image.adjust_contrast())
7. "augmentation/shear_x_range" : 
    - range of the random x-wise shear applied to the dataset (@see tfa.image.shear_x())
8. "augmentation/shear_y_range" : 
    - range of the random y-wise shear applied to the dataset (@see tfa.image.shear_y())
9. "augmentation/shear_fill" : 
    - vector holding values that empty pixels will be filled with after shear operations
10. "augmentation/vertical_flip" : 
    - If `true`, the random vertical flips will be applied to the dataset
11. "augmentation/horizontal_flip" : 
    - If `true`, the random horizontal flips will be applied to the dataset


# Training config (fit.json)

1. "base_model" : 
    - Path to the file (relative to dirs.json/models) containing saved weights that will be loaded before the training. If `null`, no weights will be loaded
2. "batch_size" : 
    - Batch size (for both training and validation datasets)
3. "kernal_initializator" :
    - initialization method of weights (@see tf.keras.initialilzers.get())
4. "bias_initializator" :
    - initialization method of biases (@see tf.keras.initialilzers.get())
5. vgg_layers_to_remove :
    - list of original VGG's layers to remove `not implemented yet`
5. "vgg_layers_to_train" :
    - number of last VGG's convolutional layers to be trained (after layers' removing) (-1 for all)
6. "optimization/optimizer" : 
    - Name of the optimiser to be used (@see tf.keras.Model.compile)
7. "optimization/loss" : 
    - Name of the loss function to be used (@see tf.keras.Model.compile)
8. "optimization/learning_rate/indicator" : 
    - Metric that when plateaued determines learning rate's reduction (@see tf.keras.Model.compile)
9. "optimization/learning_rate/init" : 
    - Initial learning rate
10. "optimization/learning_rate/min" : 
    - Minimal learning rate
11. "optimization/learning_rate/min_delta" : 
    - Minimal reduction of the learning rate on the plateau
12. "optimization/learning_rate/reduce_factor" : 
    - Reduction factor of the learning rate on the plateau
13. "optimization/learning_rate/patience" : 
    - Number of epochs on the plateau before learning rate's reduction
14. "optimization/learning_rate/cooldown" : 
    - Number of epochs after reduction when patience counter is not incrementerd (@see: tf.keras.callbacksReduceLROnPlateau)
15. "optimization/learning_rate/verbosity" : 
    - Verbosity of the automatic learning rate's adaptation (0: quite, 1: verbose)
16. "epochs" : 
    - Number of training's epochs
17. "initial_epoch" : 
    - Number of the epoch that training should begin with (handy for re-running the interrupted training)
18. "steps_per_epoch" : 
    - Number of batches of data proceeded during the epoch. If `null` the whole dataset will be proceeeded at the each epoch.
19. "environment/gpu_memory_cap_mb" : 
    - Size of the memory allocated on the GPU by tensorflow
20. "environment/tf_device_verbosity" : 
    - Verbosity of the tensorflow data placement
21. "environment/verbosity" : 
    - Training verbosity level (@see tf.keras.Model.fit())
22. "environment/workers" : 
    - Number of workers used to prefetch data during training (@see tf.keras.Model.fit())
