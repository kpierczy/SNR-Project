import os
import pickle
import tempfile
from datetime import datetime
from glob import glob
from os import listdir
from os.path import isfile, join

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import scale, normalize
from sklearn.svm import SVC, LinearSVC
from src.api.NeuralTrainer import NeuralTrainer
import numpy as np
from src.api.tools.NetworkOutputMapper import NetworkOutputMapper


class SVMTrainer(NeuralTrainer):
    def __init__(self,
                 model,
                 dirs,
                 logging_params,
                 pipeline_params,
                 training_params,
                 # kernel
                 ):
        super().__init__(model, dirs, logging_params, pipeline_params, training_params)
        # self.kernel = kernel
        self.svm: SVC = None

    def initialize(self):
        super().initialize()
        if self.training_params['svm']['kernel'] == 'linear':
            self.svm = LinearSVC(max_iter=self.training_params['svm']['max_iter'],
                                 verbose=self.training_params['svm']['verbosity'],
                                 loss='hinge')
        else:
            self.svm = SVC(kernel=self.training_params['svm']['kernel'],
                           degree=self.training_params['svm']['poly_degree'],
                           max_iter=self.training_params['svm']['max_iter'],
                           verbose=True)

        return self

    def run(self):
        self.__train()

        self.__test()

        return self

    def __model_path(self) -> str:
        return os.path.join(self.dirs['output'], self.training_params['svm']['kernel'] + '.svm')
        # return self.dirs['output'] + '/' + self.training_params['svm']['model_name']

    def __create_network_output(self, dataset, dir_path: str):
        for image, result in dataset:
            network_out = self.model.predict(image)
            network_out = normalize(network_out)
            result = result.numpy()
            file_name = tempfile.NamedTemporaryFile(delete=False, dir=dir_path,
                                                    suffix='.npz')
            # save net out and prediction results as np array in file
            np.savez(file_name, out=network_out, result=result)

    def __read_network_output(self, dir_path: str) -> (np.ndarray, np.ndarray):
        network_output, result = None, None
        filepaths = [os.path.join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]

        # read net outs and prediction results
        for f in filepaths:
            result_batch = np.load(f)
            out_batch = result_batch['out']
            result_batch = result_batch['result']

            # join out and results
            if network_output is None:
                network_output = out_batch
                result = result_batch
            else:
                network_output = np.concatenate((network_output, out_batch), axis=0)
                result = np.concatenate((result, result_batch), axis=0)
        return network_output, result

    def __train(self):
        if self.training_params['svm']['create_network_output']:
            print('Create network output for train set')
            self.__create_network_output(self.pipe.training_set, self.dirs['train_network_output'])

        print("Learning begin time: {}".format(datetime.now()))
        if self.training_params['svm']['train']:
            print('read train features from file')
            network_out, prediction_results = self.__read_network_output(self.dirs['train_network_output'])
            print('features readed')

            # change network prediction to number of class
            prediction_results = [NetworkOutputMapper.to_class_num(result) for result in prediction_results.tolist()]

            print('try to fit SVM model')
            self.svm.fit(network_out, prediction_results)

            print("Learning end time: {}".format(datetime.now()))
            # save trained model
            if self.training_params['svm']['save_model']:
                with open(self.__model_path(), mode='wb') as f:
                    pickle.dump(self.svm, f)
                    print('SVM model saved in file ' + self.__model_path())

    def __test(self):
        print('Try to test SVM model')

        # wczytaj svm z pliku
        if self.training_params['svm']['load_model']:
            with open(self.__model_path(), mode='rb') as f:
                self.svm = pickle.load(f)
            print("SVM model loaded from file")

        if self.training_params['svm']['create_network_output']:
            print("Create network output for test set")
            self.__create_network_output(self.pipe.test_set, self.dirs['test_network_output'])

        if self.training_params['svm']['test']:
            print('Read train features from file')
            network_out, prediction_results = self.__read_network_output(self.dirs['test_network_output'])
            print('Features readed')

            # change network prediction to number of class
            prediction_results = [NetworkOutputMapper.to_class_num(result) for result in prediction_results.tolist()]

            # count statistics
            print('Testing model')
            decision = self.svm.predict(network_out)
            accuracy = accuracy_score(prediction_results, decision)
            print('accuracy: {}'.format(accuracy))
            cm = confusion_matrix(prediction_results, decision)
            print('confusion matrix: ')
            print(cm)
            self.__history = {'accuracy': accuracy, 'cm': cm}

            # Create path to the output folder
            historydir = os.path.join(self.dirs['output'], 'history')
            os.makedirs(historydir, exist_ok=True)
            subrun = len(glob(os.path.join(historydir, '*.pickle'))) + 1

            # save results
            historyname = os.path.join(historydir, 'subrun_{:d}'.format(subrun))
            with open(historyname + '.pickle', 'wb') as history_file:
                pickle.dump(self.__history, history_file)
                print('Test results saved in ' + historyname)
