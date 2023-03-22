import unittest
from smltk.datavisualization import DataVisualization

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# import tensorflow_datasets as tfds

class TestDataVisualization(unittest.TestCase, DataVisualization):
    dv = None
    def __init__(self, *args, **kwargs):
        self.dv = DataVisualization()
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_get_df_sklearn(self):
        iris = load_iris()
        df = self.dv.get_df(iris)
        # self.assertEqual(df.count().sum(), 0)
        self.assertEqual(len(sorted(df['target_name'].unique().tolist())), 3)
        self.assertListEqual(sorted(df['target_name'].unique().tolist()), sorted(iris.target_names.tolist()))
        self.assertListEqual(df['target'].tolist(), iris.target.tolist())

    def test_get_inference_df_sklearn(self):
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=3)
        model = DecisionTreeClassifier(random_state=0)
        _ = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        df = self.dv.get_inference_df(iris, x_test, y_test, y_pred)
        self.assertListEqual(df['prediction'].tolist(), y_pred.tolist())
        self.assertListEqual(sorted(df['target_name'].unique().tolist()), sorted(iris.target_names.tolist()))
        self.assertListEqual(df['target'].tolist(), y_test.tolist())

    def test_get_df_fake(self):
        fake = []
        df = self.dv.get_df(fake)
        self.assertEqual(df.count().sum(), 0)

    # def test_get_df_tensorflow(self):
    #     mnist = tfds.load('mnist', batch_size=10)
    #     df = self.dv.get_df(mnist)
    #     self.assertEqual(df.count().sum(), 0)

if __name__ == '__main__':
    unittest.main()