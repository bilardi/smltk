import unittest
from smltk.datavisualization import DataVisualization

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# import tensorflow_datasets as tfds

import requests
from PIL import Image
import torchvision.transforms as transforms
import torch

class TestDataVisualization(unittest.TestCase, DataVisualization):
    dv = None
    def __init__(self, *args, **kwargs):
        self.dv = DataVisualization()
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_get_df_sklearn(self):
        iris = load_iris()
        df = self.dv.get_df(iris)
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

    def test_get_inference_df_fake(self):
        fake = []
        df = self.dv.get_inference_df(fake, [], [], [])
        self.assertEqual(df.count().sum(), 0)

    # def test_get_df_tensorflow(self):
    #     mnist = tfds.load('mnist', batch_size=10)
    #     df = self.dv.get_df(mnist)
    #     self.assertEqual(df.count().sum(), 0)

    def test_get_inference_objects_torch(self):
        url = 'https://www.projectinvictus.it/wp-content/uploads/2022/08/junk-food-scaled.jpg'
        im = Image.open(requests.get(url, stream=True).raw)
        transform = transforms.Compose([ transforms.Resize(800), transforms.ToTensor() ])
        img = transform(im).unsqueeze(0)
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        model.eval();
        prediction = model(img)
        probability, boxes = self.dv.get_inference_objects(im, prediction, 0.7)
        df = self.dv.get_inference_objects_df(probability, boxes)
        self.assertListEqual(df.columns.to_list(), ['class', 'probability', 'xmin', 'ymin', 'xmax', 'ymax'])
        self.assertListEqual(df.groupby('class').count().index.tolist(), ['bowl', 'cup', 'dining table', 'donut'])
        self.assertListEqual(df.groupby('class').count()['probability'].tolist(), [5, 3, 1, 4])

if __name__ == '__main__':
    unittest.main()