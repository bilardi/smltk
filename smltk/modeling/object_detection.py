"""The class for managing the image

A collection of methods to simplify your code.
"""

try:
    import torch
except ImportError as error:
    message = (
        "The class ObjectDetection requirements are not installed.\n\n"
        "Please use one fo the commands below:\n\n"
        '  pip install "smltk[object_detection]" --upgrade'
        '  python -m pip install "smltk[object_detection]" --upgrade'
    )
    raise ImportError(str(error) + "\n\n" + message) from error

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import smltk.modeling.classes_map as classes


class ObjectDetection:
    sklearn = None

    def __init__(self):
        self.sklearn = load_iris()

    ### images ###
    def bboxes_cxcywh_to_xyxy(self, bboxes):
        """
        Concatenate a sequence of tensors along a new dimension

        Arguments:
            :bboxes (sequence of Tensors): list of boxes Tensors
        Returns:
            sequence of Tensors
        """
        x_c, y_c, w, h = bboxes.unbind(1)
        bbs = [
            (x_c - 0.5 * w),
            (y_c - 0.5 * h),
            (x_c + 0.5 * w),
            (y_c + 0.5 * h),
        ]
        return torch.stack(bbs, dim=1)

    def rescale_bboxes(self, bboxes, size):
        """
        Rescale boxes on image size

        Arguments:
            :bboxes (sequence of Tensors): list of boxes Tensors
            :size (tuple): width and height of image
        Returns:
            sequence of Tensors
        """
        img_w, img_h = size
        bbs = self.bboxes_cxcywh_to_xyxy(bboxes)
        bbs = bbs * torch.tensor(
            [img_w, img_h, img_w, img_h], dtype=torch.float32
        )
        return bbs

    def get_inference_objects(self, image, prediction, threshold=0.7):
        """
        Rescale boxes with probability greater than the threshold

        Arguments:
            :image (PIL Image): object of type PIL Image
            :prediction (dict): prediction of the model with pred_logits and pred_boxes
            :threshold (float): probability value used like threshold, default 0.7
        Returns:
            tuple of sequences of Tensors about probabilities and boxes
        """
        # keep only predictions with threshold confidence
        probability = prediction["pred_logits"].softmax(-1)[0, :, :-1]
        keep = probability.max(-1).values > threshold

        # convert boxes from [0; 1] to image scales
        boxes = self.rescale_bboxes(
            prediction["pred_boxes"][0, keep], image.size
        )
        return probability[keep], boxes

    def get_inference_objects_df(self, probability, boxes):
        """
        Create a DataFrame from the prediction of object detection

        Arguments:
            :probability (sequence of Tensors): list of probabilities Tensors
            :boxes (sequence of Tensors): list of boxes Tensors
        Returns:
            Pandas DataFrame
        """
        df = pd.DataFrame()
        for pb, (xmin, ymin, xmax, ymax) in zip(probability, boxes.tolist()):
            cl = pb.argmax()
            piece = pd.DataFrame(
                {
                    "class": [classes.torch_classes[cl]],
                    "probability": [f"{pb[cl]:0.2f}"],
                    "xmin": [xmin],
                    "ymin": [ymin],
                    "xmax": [xmax],
                    "ymax": [ymax],
                }
            )
            df = pd.concat([df, piece])
        return df

    def plot_inference_objects(self, image, probability, boxes):
        """
        Plot image with boxes

        Arguments:
            :image (PIL Image): object of type PIL Image
            :probability (sequence of Tensors): list of probabilities Tensors
            :boxes (sequence of Tensors): list of boxes Tensors
        Returns:
            plot
        """
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()
        colors = classes.torch_colors * 100
        for pb, (xmin, ymin, xmax, ymax), color in zip(
            probability, boxes.tolist(), colors
        ):
            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    color=color,
                    linewidth=3,
                )
            )
            cl = pb.argmax()
            text = f"{classes.torch_classes[cl]}: {pb[cl]:0.2f}"
            ax.text(
                xmin,
                ymin,
                text,
                fontsize=15,
                bbox={"facecolor": "yellow", "alpha": 0.5},
            )
        plt.axis("off")
        plt.show()
