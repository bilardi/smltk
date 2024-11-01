"""The class for managing the data of the main repositories

    A collection of methods to simplify your code.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import torch
import smltk.classes_map as classes


class DataVisualization:
    sklearn = None

    def __init__(self):
        self.sklearn = load_iris()

    ### features ###
    def get_df(self, data):
        """
        Create a DataFrame from the data of the main repositories

        Arguments:
            :data (mixed): data loaded from one of the main repositories
        Returns:
            Pandas DataFrame
        """
        if isinstance(data, type(self.sklearn)):
            df = pd.DataFrame(data.target, columns=["target"])
            df["target_name"] = df["target"].apply(
                lambda x: data.target_names[x]
            )
            return pd.concat(
                [df, pd.DataFrame(data.data, columns=data.feature_names)],
                axis=1,
            )
        return pd.DataFrame()

    def get_inference_df(self, data, x_test, y_test, y_pred):
        """
        Create a DataFrame from the data of the main repositories

        Arguments:
            :x_test (Pandas DataFrame): features used for the prediction
            :y_test (list of str): list of the targets
            :y_pred (list of str): list of the predictions
        Returns:
            Pandas DataFrame
        """
        if isinstance(data, type(self.sklearn)):
            df = pd.DataFrame(y_test, columns=["target"])
            df["target_name"] = df["target"].apply(
                lambda x: data.target_names[x]
            )
            return pd.concat(
                [
                    pd.DataFrame(y_pred, columns=["prediction"]),
                    pd.DataFrame(df),
                    pd.DataFrame(x_test, columns=data.feature_names),
                ],
                axis=1,
            )
        return pd.DataFrame()

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

    def plot_dc(self, params: dict = {}, return_ax=False):
        """
        Plot image with directional changes

        Arguments: params (dict) with the keys below
            :dc_colors (dict): key-value about each event-color of directional change
            :events (list[str]): list of events names for time point
            :timeseries (list[float]): list of values
            :timestamp (list[int|datetime]): time point list
            :figsize (tuple): default (10, 5)
            :title (str): title of plot
            :x_axis_label (str): label of x axis
            :y_axis_label (str): label of y axis

        Returns:
            plot or its object
        """
        if "dc_colors" not in params:
            params["dc_colors"] = {
                "upward dc": "green",
                "upward overshoot": "lime",
                "downward dc": "red",
                "downward overshoot": "lightcoral",
            }
        if "figsize" not in params:
            params["figsize"] = (10, 5)
        if "events" in params:
            colors = params["events"].copy()
            for event in params["dc_colors"]:
                colors = list(
                    map(
                        lambda x: x.replace(event, params["dc_colors"][event]),
                        colors,
                    )
                )

        handles = []
        for event in params["dc_colors"]:
            handles.append(
                plt.Line2D(
                    [0], [0], color=params["dc_colors"][event], label=event
                )
            )

        if "timeseries" in params and "timestamp" in params:
            _, ax1 = plt.subplots(figsize=params["figsize"])
            ax1.ticklabel_format(style="plain", axis="y", useOffset=False)
            for i, color in enumerate(colors):
                ax1.plot(
                    params["timestamp"][i : i + 2],
                    params["timeseries"][i : i + 2],
                    color=color,
                )
            ax1.set_xlim(0, len(params["timeseries"]) - 1)
            ax1.set_ylim(
                params["timeseries"].min() * 0.9999,
                params["timeseries"].max() * 1.0001,
            )
            if "title" in params:
                ax1.set_title(params["title"])
            if "x_axis_label" in params:
                ax1.set_xlabel(params["x_axis_label"])
            if "y_axis_label" in params:
                ax1.set_ylabel(params["y_axis_label"])
            ax1.legend(handles=handles, loc="upper right", fontsize="small")
            if return_ax is True:
                return ax1
            plt.show()
