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
    
    # Number of _last_ original VGG layers that should be removed
    'vgg_to_remove' : 5,

    # Number of _last_ original VGG _conv_ layers to be retrained [None to train all layers]
    'vgg_conv_to_train' : None,
}