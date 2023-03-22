"""The class for managing the data of the main repositories

    A collection of methods to simplify your code.
"""
import pandas as pd
from sklearn.datasets import load_iris

class DataVisualization():
    sklearn = None
    def __init__(self, *args, **kwargs):
        self.sklearn = load_iris()

    def get_df(self, data):
        """
            Create a DataFrame from the data of the main repositories

            Arguments:
                :data (mixed): data loaded from one of the main repositories
            Returns:
                Pandas DataFrame
        """
        if type(data) == type(self.sklearn):
            df = pd.DataFrame(data.target, columns=['target'])
            df['target_name'] = df['target'].apply(lambda x: data.target_names[x])
            return pd.concat([df, pd.DataFrame(data.data, columns=data.feature_names)], axis=1)
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
        if type(data) == type(self.sklearn):
            df = pd.DataFrame(y_test, columns=['target'])
            df['target_name'] = df['target'].apply(lambda x: data.target_names[x])
            return pd.concat([pd.DataFrame(y_pred, columns=['prediction']), pd.DataFrame(df), pd.DataFrame(x_test, columns=data.feature_names)], axis=1)
        return pd.DataFrame()
