import os
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

    def _maybe_skip_plots(self, params: dict) -> dict:
        """Add all blocks to plots.skip unless PLOTS_SKIP=false."""
        if os.environ.get("PLOTS_SKIP", "true").lower() != "false":
            existing = set(params.get("plots.skip", []))
            existing |= set(DataAnalysis.VALID_BLOCK_NAMES)
            params["plots.skip"] = sorted(existing)
        return params

    def test_valid_block_names_constant(self):
        """The class exposes the canonical set of skippable block names."""
        expected = {
            "cat_countplot",
            "num_pairplot",
            "feat_violinplots",
            "feat_barplots",
            "relations_heatmaps",
            "missingval_plot",
            "cat_plots",
        }
        self.assertEqual(set(DataAnalysis.VALID_BLOCK_NAMES), expected)

    def test_is_skipped_default(self):
        """No skip lists -> both flags are False."""
        skip_all, skip_plot = self.da._is_skipped("cat_countplot", {})
        self.assertFalse(skip_all)
        self.assertFalse(skip_plot)

    def test_is_skipped_analyses(self):
        """Block listed in analyses.skip -> skip_all True."""
        params = {"analyses.skip": ["cat_countplot"]}
        skip_all, skip_plot = self.da._is_skipped("cat_countplot", params)
        self.assertTrue(skip_all)
        self.assertFalse(skip_plot)

    def test_is_skipped_plots(self):
        """Block listed in plots.skip -> skip_plot True, skip_all False."""
        params = {"plots.skip": ["relations_heatmaps"]}
        skip_all, skip_plot = self.da._is_skipped("relations_heatmaps", params)
        self.assertFalse(skip_all)
        self.assertTrue(skip_plot)

    def test_is_skipped_other_block_unaffected(self):
        """Skipping one block does not affect other blocks."""
        params = {"analyses.skip": ["num_pairplot"]}
        skip_all, skip_plot = self.da._is_skipped("cat_countplot", params)
        self.assertFalse(skip_all)
        self.assertFalse(skip_plot)

    def test_get_eda(self):
        iris = load_iris()
        df = self.dp.get_df(iris)
        params = self._maybe_skip_plots({"columns_to_filter": ["target"]})
        features = self.da.get_eda("target", df, params)
        relations = features.pop("relations")
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
            list(relations.keys()), ["pearson", "spearman", "kendall"]
        )
        categorical_features, df = self.dp.transform_categories(df)
        self.assertListEqual(
            list(categorical_features.keys()), ["target_name"]
        )
        self.assertListEqual(
            list(categorical_features["target_name"]),
            ["setosa", "versicolor", "virginica"],
        )
        # features = self.da.get_eda("target", df, {"sample.frac": 1})
        # print(features["relations"])

    def test_get_eda_unknown_block_raises(self):
        """Unknown block name in skip lists -> ValueError."""
        iris = load_iris()
        df = self.dp.get_df(iris)
        with self.assertRaises(ValueError) as ctx:
            self.da.get_eda(
                "target",
                df,
                {
                    "columns_to_filter": ["target"],
                    "analyses.skip": ["pariplot"],
                },
            )
        self.assertIn("pariplot", str(ctx.exception))
        self.assertIn("Valid names", str(ctx.exception))

    def test_get_eda_unknown_in_plots_skip_raises(self):
        """Unknown name in plots.skip is also caught."""
        iris = load_iris()
        df = self.dp.get_df(iris)
        with self.assertRaises(ValueError) as ctx:
            self.da.get_eda(
                "target",
                df,
                {
                    "columns_to_filter": ["target"],
                    "plots.skip": ["typo"],
                },
            )
        self.assertIn("typo", str(ctx.exception))

    def test_get_eda_skip_all_analyses(self):
        """Skipping all blocks returns only the 5 base keys."""
        iris = load_iris()
        df = self.dp.get_df(iris)
        params = {
            "columns_to_filter": ["target"],
            "analyses.skip": list(DataAnalysis.VALID_BLOCK_NAMES),
        }
        features = self.da.get_eda("target", df, params)
        self.assertEqual(
            set(features.keys()),
            {
                "data_amount",
                "hue_order",
                "categorical_features",
                "numerical_features",
                "data_missing",
            },
        )
        self.assertNotIn("relations", features)

    def test_get_eda_skip_plot_keeps_relations(self):
        """plots.skip on relations_heatmaps keeps the data, hides the plot."""
        iris = load_iris()
        df = self.dp.get_df(iris)
        params = {
            "columns_to_filter": ["target"],
            "plots.skip": ["relations_heatmaps"],
            "analyses.skip": [
                "cat_countplot",
                "num_pairplot",
                "feat_violinplots",
                "feat_barplots",
                "missingval_plot",
                "cat_plots",
            ],
        }
        features = self.da.get_eda("target", df, params)
        self.assertIn("relations", features)
        self.assertListEqual(
            list(features["relations"].keys()),
            ["pearson", "spearman", "kendall"],
        )

    def test_get_eda_skip_analyses_drops_relations(self):
        """analyses.skip on relations_heatmaps drops the key from return."""
        iris = load_iris()
        df = self.dp.get_df(iris)
        params = {
            "columns_to_filter": ["target"],
            "analyses.skip": ["relations_heatmaps"],
            "plots.skip": [
                "cat_countplot",
                "num_pairplot",
                "feat_violinplots",
                "feat_barplots",
                "missingval_plot",
                "cat_plots",
            ],
        }
        features = self.da.get_eda("target", df, params)
        self.assertNotIn("relations", features)


if __name__ == "__main__":
    unittest.main()
