"""The class for managing the data of the main repositories

A collection of methods to simplify your code.
"""

import pandas as pd
import re
from sklearn.datasets import load_iris
import simplemma
import snowballstemmer
import string
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import CharDelimiterSplit


class DataProcessing:
    """
    The class DataProcessing contains the simple methods to manage the data, both tabular and textual.

    Arguments: params (dict) with the key below
        :language (str): default is english

    Here's an example:

        >>> from smltk.data_processing import DataProcessing
        >>> doc = 'Good case, Excellent value.'
        >>> dp = DataProcessing()
        >>> get_doc_cleaned = dp.get_doc_cleaned(doc)
        >>> print(get_doc_cleaned)
        good case excellent value
    """

    language = "english"
    sklearn = None

    def __init__(self, params: dict = {}):
        self.sklearn = load_iris()
        if "language" in params:
            self.language = params["language"]

    # tabular data
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

    def transform_categories(
        self,
        data: pd.DataFrame,
        categorical_features: dict = None,
        features: dict = None,
    ) -> list:
        """
        Transform categorical features in discrete features

        Arguments:
            :data (Pandas DataFrame): features to elaborate
            :categorical_features (dict): with key the name of the categorical features and value the ordered list of values
            :features (dict): with key categorical_features and value the list of the categorical features
        Returns:
            list of categorical_features and data transformed
        """
        use_categorical_features = True
        if categorical_features is None:
            use_categorical_features = False
            categorical_features = {}
        if features is None:
            features = {"categorical_features": []}
            for feature in data.keys():
                if data[feature].dtype == type(object):
                    features["categorical_features"].append(feature)
        for feature in features["categorical_features"]:
            if feature in data.keys():
                if use_categorical_features is False:
                    categorical_features[feature] = pd.Categorical(
                        data[feature]
                    ).categories
                    data[feature] = pd.Categorical(data[feature]).codes
                else:
                    data[feature] = pd.Categorical(
                        data[feature], categories=categorical_features[feature]
                    ).codes
        return categorical_features, data

    # textual data
    def tokenize(self, doc: str) -> list:
        """
        Tokenizes doc

        Arguments:
            :doc (str): text
        Returns:
            list of words filtered
        """
        # split into tokens by white space
        char_splitter = CharDelimiterSplit(" ")
        return [token for token, _ in char_splitter.pre_tokenize_str(doc)]

    def clean_doc(
        self,
        tokens: list,
        is_alpha: bool = True,
        is_punctuation: bool = True,
    ) -> list:
        """
        Filters tokens from punctuation, changes upper characters in lower characters
        if is_alpha is True, remove no alphabetic characters
        and if is_punctuation is True, remove punctuation

        Arguments:
            :doc (str): text
            :is_alpha (bool): default is True
            :is_punctuation (bool): default is True
        Returns:
            list of words filtered
        """
        if is_punctuation is True:
            # prepare regex for char filtering
            punctuation = re.compile(f"[{re.escape(string.punctuation)}]")
            # remove punctuation from each word
            tokens = [punctuation.sub("", token) for token in tokens]
        # remove remaining tokens with no alphabetic characters
        if is_alpha is True:
            tokens = [token for token in tokens if token.isalpha()]
        # convert in lower case
        tokens = [token.lower() for token in tokens]
        return tokens

    def tokenize_and_clean_doc(
        self, doc: str, is_alpha: bool = True, is_punctuation: bool = True
    ) -> list:
        """
        Tokenizes doc
        and filters tokens from punctuation, changes upper characters in lower characters
        if is_alpha is True, remove no alphabetic characters
        and if is_punctuation is True, remove punctuation

        Arguments:
            :doc (str): text
            :is_alpha (bool): default is True
            :is_punctuation (bool): default is True
        Returns:
            list of words filtered
        """
        tokens = self.tokenize(doc)
        return self.clean_doc(tokens, is_alpha, is_punctuation)

    def lemmatize(self, tokens: list) -> list:
        """
        Lemmatizes tokens

        Arguments:
            :tokens (list[str]): list of words
        Returns:
            list of tokens lemmatized
        """
        # lemmatization
        lemmas = [
            simplemma.lemmatize(token, self.language[:2]).lower()
            for token in tokens
        ]
        return lemmas

    def stem(self, tokens: list) -> list:
        """
        Stems tokens

        Arguments:
            :tokens (list[str]): list of words
        Returns:
            list of tokens stemmed
        """
        # stemming
        stemmer = snowballstemmer.stemmer(self.language)
        return stemmer.stemWords(tokens)

    def get_tokens_cleaned(
        self,
        doc: str,
        is_lemma: bool = True,
        is_stem: bool = False,
        is_alpha: bool = True,
        is_punctuation: bool = True,
    ) -> list:
        """
        Tokenizes doc
        and filters tokens from punctuation, changes upper characters in lower characters
        and if is_lemma is True, also it lemmatizes
        or if is_stem is True, also it stems

        Arguments:
            :doc (str): text
            :is_lemma (bool): default is True
            :is_stem (bool): default is False
            :is_alpha (bool): default is True
            :is_punctuation (bool): default is True
        Returns:
            list of tokens cleaned
        """
        tokens = self.tokenize_and_clean_doc(doc, is_alpha, is_punctuation)
        if is_lemma is True:
            tokens = self.lemmatize(tokens)
        if is_stem is True:
            tokens = self.stem(tokens)
        return tokens

    def get_doc_cleaned(
        self,
        doc: str,
        is_lemma: bool = True,
        is_stem: bool = False,
        is_alpha: bool = True,
        is_punctuation: bool = True,
    ) -> str:
        """
        Filters doc from punctuation, changes upper characters in lower characters
        and if is_lemma is True, also it lemmatizes
        or if is_stem is True, also it stems

        Arguments:
            :doc (str): text
            :is_lemma (bool): default is True
            :is_stem (bool): default is False
            :is_alpha (bool): default is True
            :is_punctuation (bool): default is True
        Returns:
            string cleaned
        """
        tokens = self.get_tokens_cleaned(
            doc, is_lemma, is_stem, is_alpha, is_punctuation
        )
        return " ".join(map(str, tokens))
