from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

    def test_get_df_fake(self):
        fake = []
        df = self.dp.get_df(fake)
        self.assertEqual(df.count().sum(), 0)

    def test_get_inference_df_fake(self):
        fake = []
        df = self.dp.get_inference_df(fake, [], [], [])
        self.assertEqual(df.count().sum(), 0)

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


if __name__ == "__main__":
    unittest.main()
