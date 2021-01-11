# The neural nets' workflow

1. Create the data pipe: [:white_check_mark:]
    - Measure dataset's size [:ballot_box_with_check:]
    - Choose data loading scheme depending on the size: [:ballot_box_with_check:]
        - dataset (data fits into memory) [:heavy_check_mark:]
        - generator (data does not fit into memory) [:heavy_check_mark:]
    - Choose the training batch'es size [:ballot_box_with_check:]
    - Perform data preprocessing [:ballot_box_with_check:]
        - scaling [:heavy_check_mark:]
        - NN-specific preparation [:heavy_check_mark:]
        - augmentation [:heavy_check_mark:]
2. Create the model [:white_check_mark:]
    - Load the base model [:ballot_box_with_check:]
    - remove excessive layers [:ballot_box_with_check:]
    - turn off choosen layers [:ballot_box_with_check:]
    - add required layers [:ballot_box_with_check:]
    - choose th loss function [:ballot_box_with_check:]
    - choose the optimiser [:ballot_box_with_check:]
    - choose the metrics to be observed [:ballot_box_with_check:]

3. Train the model [:white_check_mark:]
    - save the model each X epochs [:ballot_box_with_check:]
    - log metrics every epoch [:ballot_box_with_check:]
    - observe metrics [:ballot_box_with_check:]
        - stop the training when metrics plateau out [:heavy_check_mark:]
        - lower the learning rate [:heavy_check_mark:]
        - resume the training [:heavy_check_mark:]
    - save training history at the end of the training [:ballot_box_with_check:]

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


# Report plan

0. Abstract

1. Exercise analysis

2. VGG19 architectue analysis

3. Data set analysis
    - original dataset
    - pipeline description
    - augmentation methods

4. Classifier learning
    - perceptron results' analysis
    - SVM results' analysis
    - comparative analysis

4. Deep net learning
    - results' analysis: the last convolution layer + perceptron
    - results' analysis: the two last convolution layers + perceptron
    - results' analysis: entire net
    - results' analysis: simplified net
    - comparative analysis

5. Visualisation of contribution's areas with Class Activation Map technique
    - perceptron classifier visualisation
    - SVM visualisation
    - the last convolution layer + perceptron visualisation
    - the two last convolution layers + perceptron visualisation
    - entire net visualisation
    - simplified net visualisation
    - comparative analysis

6. Summary and conclusions


# Training parameters adjusting

1. Weights & biases initialization: He initialization (Globrot 2010, He 2015)

2. Validation-Test datasets split ratio: (1:1) because of lack of reliable soruces
 
3. Augmentation: flips, rotations and shifts, random noise (?)

4. Batch size: 64

5. Loss function: categorical corssentropy

6. Optimizer: Adam

7. Learning rate strategy: Relatively 'small'


# Data visualisation

1. Box plots: [https://en.wikipedia.org/wiki/Box_plot]
