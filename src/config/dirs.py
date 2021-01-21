# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2021-01-09 17:15:23
# @ Modified time: 2021-01-09 17:18:01
# @ Description:
#
#     Configuration file containing input and output directories for the training process. Paths are given
#     relative to the PROJECT_HOME environment variable.
#
# ================================================================================================================

dirs = {

    # Path to the Training data directory
    'training' : 'data/fruits-360/Training',

    # Path to the Validation/Test data directory
    'validation' : 'data/fruits-360/Test',

    # Path to the directory that output files (model's weights, logs, etc.) will be saved to (created as needed)
    'output' : 'models/last_two_conv/run_2',
}