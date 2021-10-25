"""The class for testing models and reporting results

A collection of methods to simplify your code.

# license MIT
# author Alessandra Bilardi <alessandra.bilardi@gmail.com>
# see https://github.com/bilardi/smltk for details
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
#%pip install mlxtend --upgrade
from mlxtend.evaluate import bias_variance_decomp
import seaborn as sns
import numpy as np
import pickle

class Metrics():
    def split_tuples(self, tuples):
        """
        splits tuples of sample and its target in the relative lists
            Arguments:
                tuples (list[tuple]): list of tuples with sample and its target
            Returns:
                tuple of list of samples and list of targets
        """
        X_test = []
        y_test = []
        for sample, target in tuples:
            X_test.append(sample)
            y_test.append(target)
        return X_test, y_test

    def prediction(self, model, method, X_test, y_test = []):
        """
        predicts with your model
            Arguments:
                model (obj): object of your model
                X_test (list[]|list[tuple]): list of samples or list of tuples with sample and its target
                y_test (list[]): list of targets
            Returns:
                tuple of list of targets and list of predictions
        """
        y_pred = []
        if type(X_test[0]) is tuple:
            X_test, y_test = self.split_tuples(X_test)
        predictor = getattr(model, method)
        for sample in X_test:
            y_pred.append(predictor(sample))
        return y_test, y_pred

    def create_confusion_matrix(self, y_test, y_pred):
        """
        creates and prints confusion matrix
            Arguments:
                y_test (list[]): list of targets
                y_pred (list[]): list of predictions
            Returns:
                confusion matrix
        """
        matrix = confusion_matrix(y_test, y_pred)
        tick_labels = np.unique(np.array(y_test + y_pred))
        sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False, xticklabels = tick_labels, yticklabels = tick_labels)
        return matrix

    def get_classification_metrics(self, params = []):
        """
        gets classification metrics
            Arguments:
                params (dict):
                    model (obj): object of your model
                    X_train (list[]|list[tuple]): list of samples or list of tuples with sample and its target
                    y_train (list[]): list of targets
                    X_test (list[]|list[tuple]): list of samples or list of tuples with sample and its target
                    y_test (list[]): list of targets
                    y_pred (list[]): list of predictions
                    loss (str): parameter of bias_variance_decomp, default mse
                    num_rounds (int): parameter of bias_variance_decomp, default 200
                    random_seed (int): parameter of bias_variance_decomp, default 3
            Returns:
                dictionary with MSE, Bias, Variance, Accuracy, Precision, Recall, Fscore
        """
        if 'loss' not in params:
            params['loss'] = 'mse'
        if 'num_rounds' not in params:
            params['num_rounds'] = 200
        if 'random_seed' not in params:
            params['random_seed'] = 3
        fit_exists = hasattr(params['model'], 'fit') and callable(getattr(params['model'], 'fit'))
        predict_exists = hasattr(params['model'], 'predict') and callable(getattr(params['model'], 'predict'))
        if fit_exists == True or predict_exists == True:
            mse, bias, variance = bias_variance_decomp(params['model'], params['X_train'], params['y_train'], params['X_test'], params['y_test'], loss=params['loss'], num_rounds=params['num_rounds'], random_seed=params['random_seed'])
        else:
            mse, bias, variance = (0, 0, 0)
        accuracy = accuracy_score(params['y_test'], params['y_pred'])
        report = precision_recall_fscore_support(params['y_test'], params['y_pred'])
        return {'MSE': mse, 'Bias': bias, 'Variance': variance, 'Accuracy': accuracy, 'Precision': report[0], 'Recall': report[1], 'Fscore': report[2]}

    def print_metrics(self, metrics):
        """
        prints metrics
            Arguments:
                metrics (dict): dictionary of metrics with their value
            Returns:
                only the print of metrics
        """
        for metric in metrics.keys():
            print(metric, metrics[metric])

    def save_model(self, model, filename):
        """
        saves model
            Arguments:
                model (obj): object of your model
                filename (str): pathname and filename where you want to save your model
        """
        with open(filename, 'wb') as pickle_file:
            pickle.dump(model, pickle_file)

    def resume_model(self, filename):
        """
        resumes model
            Arguments:
                filename (str): pathname and filename where you want to save your model
            Returns:
                object of your model
        """
        with open(filename, 'rb') as pickle_file:
          return pickle.load(pickle_file)