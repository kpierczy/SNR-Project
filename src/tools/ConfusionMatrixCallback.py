import io
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    """
    Custom keras callback class writing a confussion matrix at the end of each X epochs.
    The matrix picture is saved to the raw PNG form in the logdir/img folder as well as
    the tensorboard summary file in the logdir.

    Params
    ------
    logdir : string
        at this directory the /cm folder will be created that will hold all created
        matrices
    validation_set : tf.data.Dataset
        validation dataset
    class_names : list of strings
        list of classes' names; index of the name defines the class numerical identifier
    freq : Int, optional (default: 1)
        frequency (in epochs) that the confusion matrix is created at
    size : tuple or list of two Ints
        size of the plot in cm [width, height]

    Credits
    -------
    Original code: [https://www.tensorflow.org/tensorboard/image_summaries]
    """

    def __init__(self,
        logdir,
        validation_set,
        class_names,
        freq=1,
        fig_size=(8, 8),
        raw_fig_type='pdf'
    ):

        self.logdir = logdir
        self.validation_set = validation_set
        self.class_names = class_names
        self.freq = freq
        self.fig_size = np.array(fig_size) / 2.54
        self.raw_fig_type = raw_fig_type

        # Counter of the epochs passed used to print callback's effect every 'freq' epochs
        self.epochs_since_callback = freq - 1

        # Create folder for raw images
        self.imgdir = os.path.join(logdir, 'cm/img')
        os.makedirs(self.imgdir, exist_ok=True)
        


    def on_epoch_end(self, epoch, logs):

        """
        Writes the confusion matrix image to the log file at the end of the each 'self.freq' epochs

        Params
        ------
        epoch : Int
            epoch's index
        logs : Dict
            metric results for this training epoch
        """

        if self.freq <= 0:
            return

        # Update frequency counter
        self.epochs_since_callback += 1
        if self.epochs_since_callback < self.freq:
            return

        self.epochs_since_callback = 0

        # Use the model to predict the values from the validation dataset.
        predictions_softmax = self.model.predict(self.validation_set)
        predictions = tf.argmax(predictions_softmax, axis=1)

        # Calculate the confusion matrix.
        actual_categories_softmax = tf.concat([y for x, y in self.validation_set], axis=0)
        actual_categories = tf.argmax(actual_categories_softmax, axis=1)
        con_matrix = sklearn.metrics.confusion_matrix(actual_categories, predictions)

        # Log the confusion matrix as an image summary.
        figure = self.__plot_confusion_matrix(con_matrix, class_names=self.class_names, size=self.fig_size)

        # Save raw figure image
        figure.savefig(os.path.join(self.imgdir, 'confusion_matrx_{:d}.'.format(epoch) + self.raw_fig_type), bbox_inches='tight')

        # Log the confusion matrix as an image summary.
        cm_image_tf = self.__plot_to_image(figure)
        plt.close(figure)
        file_writer_cm = tf.summary.create_file_writer(os.path.join(self.logdir, 'cm'))
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image_tf, step=epoch)

        return


    @staticmethod
    def __plot_confusion_matrix(con_matrix, class_names, size):

        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Params
        ------
        con_matrix : np.array of shape [n, n]
            a confusion matrix of integer classes
        class_names : np.array of shape [n]
            string names of the integer classes
        size : tuple or list of two Ints
            size of the plot in cm [width, height]
        """

        # Create the figure
        figure = plt.figure(figsize=size)

        # Print confusion matrix to the plot
        plt.imshow(con_matrix, interpolation='nearest', cmap=plt.cm.Blues)

        # Setup plto's environment
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = con_matrix.max() / 2.
        for i, j in itertools.product(range(con_matrix.shape[0]), range(con_matrix.shape[1])):
            color = "white" if con_matrix[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        # Format the figure
        plt.tight_layout()

        # Assign axes
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure


    @staticmethod
    def __plot_to_image(figure):

        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.

        Params
        ------
        figure : plt.figure
            figure to be printed

        Returns
        -------
        image : tf.Tensor
            printed image
        """

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        # Close the figure
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        return image

