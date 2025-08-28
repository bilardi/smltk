from PIL import Image
import requests

# import tensorflow_datasets as tfds
import torch
from torchvision import transforms

# from torchvision.models import ResNet50_Weights
import unittest
from smltk.modeling import ObjectDetection


class TestObjectDetection(unittest.TestCase, ObjectDetection):
    od = None

    def __init__(self, *args, **kwargs):
        self.od = ObjectDetection()
        unittest.TestCase.__init__(self, *args, **kwargs)

    # def test_get_df_tensorflow(self):
    #     mnist = tfds.load('mnist', batch_size=10)
    #     df = self.od.get_df(mnist)
    #     self.assertEqual(df.count().sum(), 0)

    def test_get_inference_objects_torch(self):
        url = "https://www.projectinvictus.it/wp-content/uploads/2022/08/junk-food-scaled.jpg"
        im = Image.open(requests.get(url, stream=True).raw)
        transform = transforms.Compose(
            [transforms.Resize(800), transforms.ToTensor()]
        )
        img = transform(im).unsqueeze(0)
        model = torch.hub.load(
            "facebookresearch/detr",
            "detr_resnet50",
            pretrained=True,
            # "facebookresearch/detr", "detr_resnet50", weights=ResNet50_Weights.DEFAULT (IMAGENET1K_V2)
            # "facebookresearch/detr", "detr_resnet50", weights=ResNet50_Weights.IMAGENET1K_V1
        )
        model.eval()
        prediction = model(img)
        probability, boxes = self.od.get_inference_objects(im, prediction, 0.7)
        df = self.od.get_inference_objects_df(probability, boxes)
        self.assertListEqual(
            df.columns.to_list(),
            ["class", "probability", "xmin", "ymin", "xmax", "ymax"],
        )
        self.assertListEqual(
            df.groupby("class").count().index.tolist(),
            ["bowl", "cup", "dining table", "donut"],
        )
        self.assertListEqual(
            df.groupby("class").count()["probability"].tolist(), [5, 3, 1, 4]
        )


if __name__ == "__main__":
    unittest.main()
