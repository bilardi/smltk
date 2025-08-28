from sklearn.datasets import load_iris
import unittest
from smltk.data_analysis import DataAnalysis
from smltk.data_processing import DataProcessing


class TestDataAnalysis(unittest.TestCase, DataAnalysis):
    da = None
    dp = None

    def __init__(self, *args, **kwargs):
        self.da = DataAnalysis()
        self.dp = DataProcessing()
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_get_eda(self):
        iris = load_iris()
        df = self.dp.get_df(iris)
        features = self.da.get_eda(
            "target", df, {"columns_to_filter": ["target"]}
        )
        correlations = features.pop("correlations")
        self.assertDictEqual(
            features,
            {
                "data_amount": 150,
                "hue_order": [0, 1, 2],
                "categorical_features": ["target_name"],
                "numerical_features": [
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ],
                "data_missing": {
                    "target": 0.0,
                    "target_name": 0.0,
                    "sepal length (cm)": 0.0,
                    "sepal width (cm)": 0.0,
                    "petal length (cm)": 0.0,
                    "petal width (cm)": 0.0,
                },
            },
        )
        self.assertListEqual(
            list(correlations.keys()), ["pearson", "spearman", "kendall"]
        )
        categorical_features, df = self.dp.transform_categories(df)
        self.assertListEqual(
            list(categorical_features.keys()), ["target_name"]
        )
        self.assertListEqual(
            list(categorical_features["target_name"]),
            ["setosa", "versicolor", "virginica"],
        )
        features = self.da.get_eda("target", df, {"sample.frac": 1})


if __name__ == "__main__":
    unittest.main()
