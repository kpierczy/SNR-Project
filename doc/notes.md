# The project's workflow

1. Create the data pipe:                                 [<<+>>]
    - Measure dataset's size                             [ <+> ]
    - Choose data loading scheme depending on the size:  [ <+> ]
        - dataset (data fits into memory)                [  +  ]
        - generator (data does not fit into memory)      [  +  ]
    - Choose the training batch'es size                  [ <+> ]
    - Perform data preprocessing                         [ <+> ]
        - scaling                                        [  +  ]
        - NN-specific preparation                        [  +  ]
        - augmentation                                   [  +  ]

2. Create the model                                      [<<+>>]
    - Load the base model                                [ <+> ]
    - remove excessive layers                            [ <+> ]
    - turn off choosen layers                            [ <+> ]
    - add required layers                                [ <+> ]
    - choose th loss function                            [ <+> ]
    - choose the optimiser                               [ <+> ]
    - choose the metrics to be observed                  [ <+> ]

3. Train the model                                       
    - save the model each X epochs                       [ <+> ]
    - log metrics every epoch                            [ <+> ]
    - observe metrics:                                   [ <+> ]
        - stop the training when metrics plateau out     [  +  ]
        - lower the learning rate                        [  +  ]
        - resume the training                            [  +  ]
    - save training history at the end of the training   [ <+> ]

4. Visualize the results
    - visualise the training and test datasets
        - histograms of classes
        - histograms of actual classification at subsequent epochs
        - dimensions reduction (T-SNE, UMAP, grand tour)
    - visualise NN's structure
        - structural graph
        - NN's example filters
        - inputs that maximise choosen neuron's activation
        - histograms of weights' and biases' of some layers at subsequent training steps
        - histograms of neuron's activation in some layers at subsequent training steps
        - histograms of weights' and biases' gradients in some layers at subsequent batches
    - visualise learning metrics over time
    - visualise classification accuracy 
        - two classes: well- and wrong-classified (UMAP, t-SNE)
        - original classes: actual vs predicted
        - dimensions reduction (T-SNE, UMAP, grand tour)


# Handy tools

1. Keras-vis: [https://raghakot.github.io/keras-vis/]
2. AMD WattMan equivalent for Linux: [https://gitlab.com/corectrl/corectrl/-/wikis/home]


# Resources

1. NN visualisation article: [https://www.analyticsvidhya.com/blog/2019/05/understanding-visualizing-neural-networks/]
2. Data & NN visualisation article: [https://jonathan-hui.medium.com/visualize-deep-network-models-and-metrics-part-4-9500fe06e3d0]
