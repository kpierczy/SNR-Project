# Directories config (dirs.json)

1. "training" : Path to the directory holding training data. (@see `src/tools/DataPipe` class' description)
2. "validation" : Path to the directory holding validation data. (@see `src/tools/DataPipe` class' description)
3. "test" : Path to the directory holding data data. (@see `src/tools/DataPipe` class' description) [Not-used]
4. "logs" : Path to the directory that tensorboard logs will be written into (@note If logging.json/log_name is 
`null` the log will not be created).
5. "models" : Path to the directory that model's weights will be written into during the training
6. "history" : Path to the directory that training history will be written into after the training (@note If 
logging.json/log_name is `null` the training history will not be created).


# Training config (fit.json)

1. "base_model" : Path to the file containing saved weights that will be loaded before the training. If `null`,
no weights will be loaded
2. "batch_size" : Batch size (for both training and validation datasets)
3. "optimizer" : Name of the optimiser to be used (@see tf.keras.Model.compile)
4. "loss" : Name of the loss function to be used (@see tf.keras.Model.compile)
5. "metrics" : List of names of metrics to be calculated during training (@see tf.keras.Model.compile),
6. "epochs" : Number of training's epochs
7. "initial_epoch" : Number of the epoch that training should begin with (handy for re-running the interrupted
training)
8. "steps_per_epoch" : Number of batches of data proceeded during the epoch. If `null` the whole dataset will
be proceeeded at the each epoch.
9. "environment/gpu_memory_cap_mb" : Size of the memory allocated on the GPU by tensorflow
10. "environment/tf_device_verbosity" : Verbosity of the tensorflow data placement
11. "environment/verbosity" : Training verbosity level (@see tf.keras.Model.fit())
12. "environment/workers" : Number of workers used to prefetch data during training (@see tf.keras.Model.fit())


# Logging config (logging.json)

1. "log_name" : Name of the tensorboard log's folder and training's history save file
2. "histogram_freq" : Frequency of the histogram saving (@see tf.keras.callbacks.TensorBoard)
3. "write_graph" : If `true`, the tf computational graph will be save to the tensorboard log
4. "write_images" : If `true`, the training images will be saved to the tensorboard log
5. "update_freq" : Frequency of the tensorboard update (@see tf.keras.callbacks.TensorBoard)
6. "profile_batch" : (@see tf.keras.callbacks.TensorBoard)


# Pipeline config (pipeline.json)

1. "shuffle_buffer_size" : Size of the buffer used to shuffle the dataset (@see tf.data.Dataset.shuffle)
2. "prefetch_buffer_size" : Number of batches that will be buffered by the CPU during training (multiple workers
need to be used to make advantage of this buffering)
3. "augmentation/rotation_range" : range (in degrees) of the random rotations applied to the dataset
4. "augmentation/brightness_range" : range of the random brightness corrections applied to the dataset
(@see tf.image.adjust_brightness())
5. "augmentation/contrast_range" : range of the random contrast corrections applied to the dataset
(@see tf.image.adjust_contrast())
6. "augmentation/shear_x_range" : range of the random x-wise shear applied to the dataset (@see tfa.image.shear_x())
7. "augmentation/shear_y_range" : range of the random y-wise shear applied to the dataset (@see tfa.image.shear_y())
8. "augmentation/shear_fill" : vector holding values that empty pixels will be filled with after shear operations
9. "augmentation/vertical_flip" : If `true`, the random vertical flips will be applied to the dataset
10. "augmentation/horizontal_flip" : If `true`, the random horizontal flips will be applied to the dataset
