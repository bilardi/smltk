import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import unittest
from smltk.data_processing import DataProcessing


class TestDataProcessing(unittest.TestCase, DataProcessing):
    dp = None
    doc = "Good case, Excellent value. I am agree. There is a mistake. Item Does Not Match Picture. The processing of this data needs to be studied thoroughly, 3 times"
    docs = []
    default_doc_filtered = "good case excellent value i be agree there be a mistake item do not match picture the process of this datum need to be study thoroughly time"

    def __init__(self, *args, **kwargs):
        self.dp = DataProcessing()
        self.docs = self.doc.split(".")
        unittest.TestCase.__init__(self, *args, **kwargs)

    # tabular data
    def test_get_df_sklearn(self):
        iris = load_iris()
        df = self.dp.get_df(iris)
        self.assertEqual(len(sorted(df["target_name"].unique().tolist())), 3)
        self.assertListEqual(
            sorted(df["target_name"].unique().tolist()),
            sorted(iris.target_names.tolist()),
        )
        self.assertListEqual(df["target"].tolist(), iris.target.tolist())

    def test_get_inference_df_sklearn(self):
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=3
        )
        model = DecisionTreeClassifier(random_state=0)
        _ = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        df = self.dp.get_inference_df(iris, x_test, y_test, y_pred)
        self.assertListEqual(df["prediction"].tolist(), y_pred.tolist())
        self.assertListEqual(
            sorted(df["target_name"].unique().tolist()),
            sorted(iris.target_names.tolist()),
        )
        self.assertListEqual(df["target"].tolist(), y_test.tolist())

    def test_get_df_sklearn_regression(self):
        diabetes = load_diabetes()
        df = self.dp.get_df(diabetes)
        self.assertNotIn("target_name", df.columns)
        self.assertListEqual(df["target"].tolist(), diabetes.target.tolist())
        self.assertListEqual(
            sorted(c for c in df.columns if c != "target"),
            sorted(diabetes.feature_names),
        )

    def test_get_inference_df_sklearn_regression(self):
        diabetes = load_diabetes()
        x_train, x_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target, test_size=0.2, random_state=3
        )
        model = DecisionTreeRegressor(random_state=0)
        _ = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        df = self.dp.get_inference_df(diabetes, x_test, y_test, y_pred)
        self.assertListEqual(df["prediction"].tolist(), y_pred.tolist())
        self.assertNotIn("target_name", df.columns)
        self.assertListEqual(df["target"].tolist(), y_test.tolist())

    def test_get_df_sklearn_regression_with_target_names(self):
        ch = fetch_california_housing()
        df = self.dp.get_df(ch)
        self.assertNotIn("target_name", df.columns)
        self.assertListEqual(df["target"].tolist(), ch.target.tolist())
        self.assertListEqual(
            sorted(c for c in df.columns if c != "target"),
            sorted(ch.feature_names),
        )

    def test_get_inference_df_sklearn_regression_with_target_names(self):
        ch = fetch_california_housing()
        x_train, x_test, y_train, y_test = train_test_split(
            ch.data, ch.target, test_size=0.2, random_state=3
        )
        model = DecisionTreeRegressor(random_state=0)
        _ = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        df = self.dp.get_inference_df(ch, x_test, y_test, y_pred)
        self.assertListEqual(df["prediction"].tolist(), y_pred.tolist())
        self.assertNotIn("target_name", df.columns)
        self.assertListEqual(df["target"].tolist(), y_test.tolist())

    def test_get_df_fake(self):
        fake = []
        df = self.dp.get_df(fake)
        self.assertEqual(df.count().sum(), 0)

    def test_get_inference_df_fake(self):
        fake = []
        df = self.dp.get_inference_df(fake, [], [], [])
        self.assertEqual(df.count().sum(), 0)

    def is_there_nan(self, iris, original_df, numeric_list):
        categorical_features, numeric_df = self.dp.transform_categories(
            original_df.copy()
        )
        self.assertEqual(len(categorical_features.keys()), 1)
        self.assertEqual(
            len(categorical_features["target_name"]), len(iris["target_names"])
        )
        self.assertTrue(
            is_integer_dtype(numeric_df["target_name"])
            or is_float_dtype(numeric_df["target_name"])
        )
        self.assertEqual(
            len(list(numeric_df["target_name"].unique())), len(numeric_list)
        )
        self.assertTrue(
            np.allclose(
                list(numeric_df["target_name"].unique()),
                list(numeric_list),
                equal_nan=True,
            )
        )

    def test_transformation_categories(self):
        iris = load_iris()
        original_df = self.dp.get_df(iris)
        self.is_there_nan(iris, original_df, [0, 1, 2])
        original_df.loc[10, "target_name"] = np.nan
        self.is_there_nan(
            iris, original_df, np.array([0, np.nan, 1, 2], np.dtype("float64"))
        )
        original_df.loc[10, "target_name"] = None
        self.is_there_nan(
            iris, original_df, np.array([0, np.nan, 1, 2], np.dtype("float64"))
        )
        original_df.loc[10, "target_name"] = pd.NA
        self.is_there_nan(iris, original_df, [0, np.nan, 1, 2])

    # textual data
    def test_get_tokens_cleaned(self):
        tokens = self.dp.get_tokens_cleaned(self.docs[0])
        self.assertEqual(tokens, ["good", "case", "excellent", "value"])
        tokens = self.dp.get_tokens_cleaned(self.docs[1], is_lemma=True)
        self.assertEqual(tokens, ["i", "be", "agree"])
        tokens = self.dp.get_tokens_cleaned(self.docs[1], is_lemma=False)
        self.assertEqual(tokens, ["i", "am", "agree"])

        tokens = self.dp.get_tokens_cleaned(
            self.docs[4], is_lemma=False, is_stem=True
        )
        self.assertEqual(
            tokens,
            [
                "the",
                "process",
                "of",
                "this",
                "data",
                "need",
                "to",
                "be",
                "studi",
                "thorough",
                "time",
            ],
        )

        tokens = self.dp.get_tokens_cleaned(self.docs[4], is_alpha=False)
        self.assertEqual(
            tokens,
            [
                "the",
                "process",
                "of",
                "this",
                "datum",
                "need",
                "to",
                "be",
                "study",
                "thoroughly",
                "3",
                "time",
            ],
        )

        tokens = self.dp.get_tokens_cleaned(self.docs[4], is_punctuation=False)
        self.assertEqual(
            tokens,
            [
                "the",
                "process",
                "of",
                "this",
                "datum",
                "need",
                "to",
                "be",
                "study",
                "time",
            ],
        )

        tokens = self.dp.get_tokens_cleaned(
            self.docs[4],
            is_lemma=False,
            is_stem=False,
            is_alpha=False,
            is_punctuation=False,
        )
        self.assertEqual(
            tokens,
            [
                "the",
                "processing",
                "of",
                "this",
                "data",
                "needs",
                "to",
                "be",
                "studied",
                "thoroughly,",
                "3",
                "times",
            ],
        )

    def test_get_doc_cleaned(self):
        self.assertEqual(
            self.dp.get_doc_cleaned(self.doc),
            self.default_doc_filtered,
        )

    def test_build_variant_lookup_empty(self):
        """Empty or None synonyms returns empty dict."""
        self.assertEqual(self.dp._build_variant_lookup({}), {})
        self.assertEqual(self.dp._build_variant_lookup(None), {})

    def test_build_variant_lookup_basic(self):
        """Inverts canonical->variants into variant->canonical."""
        synonyms = {
            "explore": ["exploration", "data exploration"],
            "read": ["reading", "import or load dataset"],
        }
        result = self.dp._build_variant_lookup(synonyms)
        self.assertEqual(
            result,
            {
                "exploration": "explore",
                "data exploration": "explore",
                "reading": "read",
                "import or load dataset": "read",
            },
        )

    def test_build_variant_lookup_normalizes_variants(self):
        """Variants are normalized to lowercase + strip."""
        synonyms = {"explore": [" Exploration ", "DATA exploration"]}
        result = self.dp._build_variant_lookup(synonyms)
        self.assertEqual(
            result,
            {"exploration": "explore", "data exploration": "explore"},
        )

    def test_build_variant_lookup_overlapping_raises(self):
        """Same variant in two groups raises ValueError."""
        synonyms = {"a": ["x"], "b": ["x"]}
        with self.assertRaises(ValueError) as ctx:
            self.dp._build_variant_lookup(synonyms)
        self.assertIn("x", str(ctx.exception))

    def test_build_variant_lookup_canonical_as_variant_raises(self):
        """A canonical of one group used as variant of another raises."""
        synonyms = {"a": ["x"], "b": ["a"]}
        with self.assertRaises(ValueError) as ctx:
            self.dp._build_variant_lookup(synonyms)
        self.assertIn("a", str(ctx.exception))

    def test_harmonize_words_lowercase_and_lemma(self):
        """Without synonyms: lowercase + lemma normalize variants."""
        words = ["Cleaning", "cleaning", "Clean", "Statistics"]
        result = self.dp.harmonize_words(words)
        self.assertEqual(result, ["clean", "clean", "clean", "statistic"])

    def test_harmonize_words_unknown_word(self):
        """Word not recognized by lemmatizer survives lowercased."""
        self.assertEqual(self.dp.harmonize_words(["Bho"]), ["bho"])

    def test_harmonize_words_multi_token_without_synonyms(self):
        """Multi-token entries are not lemmatized; only lowercased."""
        self.assertEqual(
            self.dp.harmonize_words(["Data exploration"]),
            ["data exploration"],
        )

    def test_harmonize_words_none_and_empty(self):
        """None passes through; empty/whitespace becomes empty string."""
        result = self.dp.harmonize_words([None, "", "   "])
        self.assertEqual(result, [None, "", ""])

    def test_harmonize_words_with_synonyms(self):
        """Synonyms map variants to canonical, both before and after lemma."""
        words = [
            "cleaning",
            "Read",
            "reading",
            "Data exploration",
            "Exploration",
            "explore",
        ]
        synonyms = {
            "explore": ["exploration", "data exploration"],
            "read": ["reading", "import or load dataset"],
        }
        result = self.dp.harmonize_words(words, synonyms)
        self.assertEqual(
            result,
            ["clean", "read", "read", "explore", "explore", "explore"],
        )

    def test_harmonize_words_overlapping_variants_raises(self):
        """A variant mapped to two canonicals raises ValueError."""
        synonyms = {"a": ["x"], "b": ["x"]}
        with self.assertRaises(ValueError):
            self.dp.harmonize_words(["x"], synonyms)

    def test_harmonize_words_multilang(self):
        """The lang parameter is accepted and applied."""
        result = self.dp.harmonize_words(["Cleaning"], lang=["it", "en"])
        self.assertEqual(result, ["clean"])


if __name__ == "__main__":
    unittest.main()
