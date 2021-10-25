import unittest
from smltk.metrics import Metrics
from smltk.preprocessing import Ntk
import nltk
nltk.download('punkt')
import numpy as np
import os

class TestMetrics(unittest.TestCase, Metrics):
    mtr = None
    ntk = None
    doc = 'Good case, Excellent value. I am agree. There is a mistake. Item Does Not Match Picture.'
    docs = []
    target = []
    tuples = []
    def __init__(self, *args, **kwargs):
        self.mtr = Metrics()
        self.ntk = Ntk()
        self.docs = nltk.sent_tokenize(self.doc)
        self.target = [1, 1, 0, 0]
        self.tuples = self.ntk.create_tuples(self.docs, self.target)
        unittest.TestCase.__init__(self, *args, **kwargs)

    def training(self):
        features_lemma = self.ntk.create_features(self.tuples, True)
        classifier = nltk.NaiveBayesClassifier.train(features_lemma)
        return classifier, features_lemma

    def prediction(self):
        classifier, features_lemma = self.training()
        y_test, y_pred = self.mtr.prediction(classifier, 'classify', features_lemma)
        return y_test, y_pred

    def test_split_tuples(self):
        classifier, features_lemma = self.training()
        X_test, y_test = self.split_tuples(features_lemma)
        self.assertEqual(y_test, [1, 1, 0, 0])

    def test_prediction(self):
        y_test, y_pred = self.prediction()
        self.assertEqual(y_test, y_pred)

    def test_create_confusion_matrix(self):
        y_test, y_pred = self.prediction()
        np.testing.assert_array_equal(self.mtr.create_confusion_matrix(y_test, y_pred), [[2, 0], [0, 2]])

    def test_get_classification_metrics(self):
        classifier, features_lemma = self.training()
        y_test, y_pred = self.mtr.prediction(classifier, 'classify', features_lemma)
        X_test, y_test = self.split_tuples(features_lemma)
        params = {
            "model": classifier,
            "X_train": np.array(X_test),
            "y_train": np.array(y_test),
            "X_test": np.array(X_test),
            "y_test": np.array(y_test),
            "y_pred": y_pred
        }
        metrics = self.mtr.get_classification_metrics(params)
        self.assertEqual(metrics['MSE'], 0)
        self.assertEqual(metrics['Accuracy'], 1.0)
        np.testing.assert_array_equal(metrics['Precision'], [1., 1.])

    def test_manage_model(self):
        filename = '/tmp/save.model'
        classifier, features_lemma = self.training()
        y_test1, y_pred1 = self.mtr.prediction(classifier, 'classify', features_lemma)
        self.mtr.save_model(classifier, filename)
        self.assertFalse(os.stat(filename).st_size == 0)
        model = self.mtr.resume_model(filename)
        y_test2, y_pred2 = self.mtr.prediction(model, 'classify', features_lemma)
        self.assertEqual(y_test1, y_test2)
        self.assertEqual(y_pred1, y_pred2)

if __name__ == '__main__':
    unittest.main()