# ================================================================================================================
# @ Author: Krzysztof Pierczyk
# @ Create Time: 2021-01-09 17:15:23
# @ Modified time: 2021-01-09 17:18:01
# @ Description:
#
#     Configuration of the model's parameters
#
# ================================================================================================================

model_params = {
    
    # Path to the initial model's weights (relative to PROJECT_HOME environment variable)
    'base_model' : None,

    # TF Kernel and bias intiializers' identifier
    'initializer' : {
        'kernel' : 'glorot_normal',
        'bias' : 'glorot_normal'
    },
    
    # Indices of the original VGG layers that should be removed [Not Implemented Yet]
    'vgg_layers_to_remove' : [],

    # Number of original VGG last convolution layers that should be retrained
    'vgg_layers_to_train' : 2,
}