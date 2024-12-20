"""The classes for data preparation with nltk, sklearn and more

    A collection of methods to simplify your code.
"""

import re
import string
from collections import defaultdict
from collections import Counter
import numpy as np
import nltk

nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("vader_lexicon")
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams as ng
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# %matplotlib inline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Ntk:
    """
    The class Ntk contains the Natural Language Processing tool kit.

    Arguments: params (dict) with the keys below
        :language (str): default is english
        :lemmatizer (obj): obj with method lemmatize() like WordNetLemmatizer()
        :min_length (int): default is 1, so all words will be used
        :stop_words (list[str]): default is stopwords.words()
        :tag_map (dict): default contains J, V, R

    Here's an example:

        >>> from smltk.preprocessing import Ntk
        >>> doc = 'Good case, Excellent value.'
        >>> ntk = Ntk()
        >>> get_doc_cleaned = ntk.get_doc_cleaned(doc)
        >>> print(get_doc_cleaned)
        good case excellent value
    """

    language = "english"
    lemmatizer = None
    min_length = 1
    stop_words = []
    tag_map = {}
    sia = None
    vectorizer = None

    def __init__(self, params=[]):
        if "language" in params:
            self.language = params["language"]
        if "lemmatizer" in params:
            self.lemmatizer = params["lemmatizer"]
        else:
            self.lemmatizer = WordNetLemmatizer()
        if "min_length" in params:
            self.min_length = params["min_length"]
        if "stop_words" in params:
            self.stop_words = set(params["stop_words"])
        else:
            self.stop_words = set(stopwords.words(self.language))
        if "tag_map" in params:
            self.tag_map = params["tag_map"]
        else:
            self.tag_map = defaultdict(lambda: wn.NOUN)
            self.tag_map["J"] = wn.ADJ
            self.tag_map["V"] = wn.VERB
            self.tag_map["R"] = wn.ADV
        if len(self.tag_map) == 0:
            for pos in ["E", "D", "J", "N", "V", "R"]:
                self.tag_map[pos] = wn.NOUN
        self.sia = SentimentIntensityAnalyzer()

    def word_tokenize(self, doc):
        """
        Splits document in each word

        Arguments:
            :doc (str): text
        Returns:
            list of words
        """
        return nltk.word_tokenize(doc)

    def tokenize_and_clean_doc(self, doc):
        """
        Tokenizes doc
        and filters tokens from punctuation, numbers, stop words, and words <= min_length
        and changes upper characters in lower characters

        Arguments:
            :doc (str): text
        Returns:
            list of words filtered
        """
        # split into tokens by white space
        tokens = self.word_tokenize(doc)
        # prepare regex for char filtering
        # punctuation = re.compile("[%s]" % re.escape(string.punctuation))
        punctuation = re.compile(f"[{re.escape(string.punctuation)}]")
        # remove punctuation from each word
        tokens = [punctuation.sub("", token) for token in tokens]
        # remove remaining tokens with no alphabetic characters
        tokens = [token for token in tokens if token.isalpha()]
        # convert in lower case
        tokens = [token.lower() for token in tokens]
        # filter out stop words
        tokens = [token for token in tokens if not token in self.stop_words]
        # filter out short tokens
        tokens = [token for token in tokens if len(token) >= self.min_length]
        return tokens

    def lemmatize(self, tokens):
        """
        Lemmatizes tokens

        Arguments:
            :tokens (list[str]): list of words
        Returns:
            list of tokens lemmatized
        """
        # lemmatization
        lemmas = [
            self.lemmatizer.lemmatize(token, self.tag_map[tag[0]])
            for token, tag in nltk.pos_tag(tokens)
        ]
        return lemmas

    def get_tokens_cleaned(self, doc, is_lemma=True):
        """
        Tokenizes doc
        and filters tokens from punctuation, numbers, stop words, and words <= min_length
        and changes upper characters in lower characters
        and if is_lemma == True, also it lemmatizes

        Arguments:
            :doc (str): text
            :is_lemma (bool): default is True
        Returns:
            list of tokens cleaned
        """
        tokens = self.tokenize_and_clean_doc(doc)
        if is_lemma is True:
            tokens = self.lemmatize(tokens)
        return tokens

    def __call__(self, doc):
        """
        Alias of get_tokens_cleaned for using in models

        Arguments:
            :doc (str): text
        Returns:
            list of tokens cleaned
        """
        return self.get_tokens_cleaned(doc)

    def get_doc_cleaned(self, doc, is_lemma=True):
        """
        Filters doc from punctuation, numbers, stop words, and words <= min_length
        and changes upper characters in lower characters
        and if is_lemma == True, also it lemmatizes

        Arguments:
            :doc (str): text
            :is_lemma (bool): default is True
        Returns:
            string cleaned
        """
        tokens = self.get_tokens_cleaned(doc, is_lemma)
        return " ".join(map(str, tokens))

    def add_doc_to_vocab(self, doc, vocab, is_lemma=True):
        """
        Adds tokens of that doc to vocabulary and updates vocabulary

        Arguments:
            :doc (str): text
            :vocab (collections.Counter): dictionary of tokens with its count
            :is_lemma (bool): default is True
        Returns:
            list of tokens of that doc and vocab updated
        """
        tokens = self.get_tokens_cleaned(doc, is_lemma)
        vocab.update(tokens)
        return tokens

    def create_vocab_from_docs(self, docs, is_lemma=True):
        """
        Creates vocabulary from list of docs

        Arguments:
            :docs (list[str]): list of texts
            :is_lemma (bool): default is True
        Returns:
            dictionary of tokens with its count in an object collections.Counter
        """
        vocab = Counter()
        [self.add_doc_to_vocab(doc, vocab, is_lemma) for doc in docs]
        return vocab

    def get_stats_vocab(self, vocab, min_occurance=1):
        """
        Gets statistics of vocabulary

        Arguments:
            :vocab (collections.Counter): dictionary of tokens with its count
            :min_occurance (int): minimum occurance considered
        Returns:
            tuple of tokens number with >= min_occurance and total tokens number
        """
        tokens = [
            key for key, count in vocab.items() if count >= min_occurance
        ]
        partial = len(tokens)
        total = len(vocab)
        return partial, total

    def create_tuples(self, docs=[], target=[]):
        """
        Creates tuples with sample and its target

        Arguments:
            :docs (list[str]): list of texts
            :target (list[str]): list of targets
        Returns:
            list of tuples with sample and its target
        """
        return list(zip(docs, target))

    def create_vocab_from_tuples(self, tuples, is_lemma=True):
        """
        Creates vocabulary from list of tuples

        Arguments:
            :tuples (list[tuples]): list of tuples with sample and its target
            :is_lemma (bool): default is True
        Returns:
            dictionary of tokens with its count in an object collections.Counter
        """
        vocab = Counter()
        [self.add_doc_to_vocab(doc, vocab, is_lemma) for doc, target in tuples]
        return vocab

    def get_words_top(self, vocab, how_many):
        """
        Gets words top for each target

        Arguments:
            :vocab (collections.Counter): dictionary of tokens with its count
            :how_many (int): how many words in your top how_many list
        Returns:
            dictionary of the top how_many list
        """
        return {word for word, count in vocab.most_common(how_many)}

    def get_vocabs_cleaned(self, vocabs):
        """
        Cleans vocabs from common words among targets

        Arguments:
            :vocabs (dict): keys are targets, values are vocabularies for that target
        Returns:
            vocabs cleaned from common words among targets
        """
        targets = vocabs.keys()
        for target1 in targets:
            for target2 in targets:
                if target1 != target2:
                    common_set = set(vocabs[target1]).intersection(
                        vocabs[target2]
                    )
                    for word in common_set:
                        del vocabs[target1][word]
                        del vocabs[target2][word]
        return vocabs

    def get_ngrams(
        self, degree=2, doc="", tokens=[], is_tuple=True, is_lemma=False
    ):
        """
        Gets ngrams from doc or tokens

        Arguments:
            :degree (int): degree of ngrams, default is 2
            :doc (str): text, option if you pass tokens
            :tokens (list[str]): list of tokens, option if you pass doc
            :is_tuple (bool): default is True
            :is_lemma (bool): default is False
        Returns:
            list of tuples (n_grams) for that degree, or list of string (token)
        """
        ngrams = []
        if doc and not tokens:
            tokens = self.get_tokens_cleaned(doc, is_lemma)
        if tokens:
            ngrams = list(ng(tokens, degree))
        if is_tuple is False:
            n_grams = []
            for n_gram in ngrams:
                ngram = " ".join(n_gram)
                n_grams.append(ngram)
            ngrams = n_grams
        return ngrams

    def get_ngrams_features(self, degree=2, doc="", tokens=[], is_lemma=False):
        """
        Gets ngrams features from doc or tokens

        Arguments:
            :degree (int): degree of ngrams, default is 2
            :doc (str): text, option if you pass tokens
            :tokens (list[str]): list of tokens, option if you pass doc
            :is_lemma (bool): default is False
        Returns:
            dictionary of ngrams extracted
        """
        features = {}
        ngrams = self.get_ngrams(
            degree=degree,
            doc=doc,
            tokens=tokens,
            is_tuple=False,
            is_lemma=is_lemma,
        )
        for token in ngrams:
            features[token] = 1
        return features

    def create_ngrams_features_from_docs(
        self, docs, target, is_lemma=True, degree=2
    ):
        """
        Creates ngrams features from docs

        Arguments:
            :docs (list[str]): list of text
            :target (str): target name of the docs
            :is_lemma (bool): default is True
            :degree (int): degree of ngrams, default is 2
        Returns:
            list of tuples with features and relative target
        """
        return [
            (
                self.get_ngrams_features(
                    degree=degree, doc=doc, is_lemma=is_lemma
                ),
                target,
            )
            for doc in docs
        ]

    def create_ngrams_features_from_tuples(
        self, tuples, is_lemma=True, degree=2
    ):
        """
        Creates ngrams features from tuples

        Arguments:
            :tuples (list[tuples]): list of tuples with sample and its target
            :is_lemma (bool): default is True
            :degree (int): degree of ngrams, default is 2
        Returns:
            list of tuples with features and relative target
        """
        return [
            (
                self.get_ngrams_features(
                    degree=degree, doc=doc, is_lemma=is_lemma
                ),
                target,
            )
            for (doc, target) in tuples
        ]

    def get_features(self, doc, is_lemma=True, words_top={}, degree=0):
        """
        Gets features

        Arguments:
            :doc (str): text
            :is_lemma (bool): default is True
            :words_top (dict): dictionary of the words top
            :degree (int): degree of ngrams, default is 0
        Returns:
            dictionary of features extracted
        """
        tokens = self.get_tokens_cleaned(doc, is_lemma)
        features = {"words_top": 0}
        if degree > 0:
            features.update(
                self.get_ngrams_features(
                    degree=degree, tokens=tokens, is_lemma=is_lemma
                )
            )
        for token in tokens:
            if token in words_top:
                features["words_top"] += 1
        features.update(self.sia.polarity_scores(doc))
        return features

    def get_features_from_docs(
        self, docs, is_lemma=True, words_top={}, degree=0
    ):
        """
        Gets features

        Arguments:
            :docs (list[str]): list of text
            :is_lemma (bool): default is True
            :words_top (dict): dictionary of the words top
            :degree (int): degree of ngrams, default is 0
        Returns:
            dictionary of features extracted
        """
        return [
            self.get_features(doc, is_lemma, words_top, degree) for doc in docs
        ]

    def create_features_from_docs(
        self, docs, target, is_lemma=True, words_top={}, degree=0
    ):
        """
        Creates features from docs

        Arguments:
            :docs (list[str]): list of text
            :target (str): target name of the docs
            :is_lemma (bool): default is True
            :words_top (dict): dictionary of the words top
            :degree (int): degree of ngrams, default is 0
        Returns:
            list of tuples with features and relative target
        """
        return [
            (self.get_features(doc, is_lemma, words_top, degree), target)
            for doc in docs
        ]

    def create_features_from_tuples(
        self, tuples, is_lemma=True, words_top={}, degree=0
    ):
        """
        Creates features from tuples

        Arguments:
            :tuples (list[tuples]): list of tuples with sample and its target
            :is_lemma (bool): default is True
            :words_top (dict): dictionary of the words top
            :degree (int): degree of ngrams, default is 0
        Returns:
            list of tuples with features and relative target
        """
        return [
            (self.get_features(doc, is_lemma, words_top, degree), target)
            for (doc, target) in tuples
        ]

    def create_words_map(self, words):
        """
        Creates the map of words

        Arguments:
            :words (list[str]): words list
        Returns:
            string of all words
        """
        lower = [sentence.lower() for sentence in words]
        return " ".join(map(str, lower))

    def create_words_cloud(self, words, is_test=False):
        """
        Creates the cloud of words

        Arguments:
            :words (str): words
            :is_test (bool): default is False
        Returns:
            only words cloud plot
        """
        words_cloud = WordCloud(
            width=1200, height=800, background_color="white"
        ).generate(words)
        if is_test is False:
            plt.imshow(words_cloud)
            plt.axis("off")
            plt.show()
        return words_cloud

    def vectorize_docs(
        self, docs, is_count=True, is_lemma=False, is_test=False
    ):
        """
        Vectorizes docs

        Arguments:
            :docs (list[str]): list of texts
            :is_count (bool): default is True
            :is_lemma (bool): default is True
            :is_test (bool): default is False
        Returns:
            list of scipy.sparse.csr.csr_matrix, one for each doc
        """
        if isinstance(docs, list) and isinstance(docs[0], dict):
            vectorizer = DictVectorizer()
        elif is_count is True:
            vectorizer = CountVectorizer()
        else:
            vectorizer = TfidfVectorizer()
        if is_lemma is True:
            docs = [self.get_doc_cleaned(doc) for doc in docs]
        if is_test is True:
            return self.vectorizer.transform(docs)
        self.vectorizer = vectorizer
        return self.vectorizer.fit_transform(docs)


class Indicator:
    """
    The class Indicator contains the tool kit to calculate the principal indicators.

    Arguments: params (dict) with the keys below
        :events (list[str]): list of directional change events
        :timeseries (list[int|float]): list of values, default None

    Here's an example:

        >>> from smltk.preprocessing import Indicator
        >>> timeseries = numpy.array()
        >>> indicator = Indicator()
        >>> dc_events = indicator.get_dc_events(timeseries)
        >>> print(dc_events)
        array['upward dc', 'downward dc', ..]
    """

    events = None
    timeseries = None

    def __init__(self, params={}):
        if "events" in params:
            self.events = params["events"]
        if "timeseries" in params:
            self.timeseries = params["timeseries"]

    def get_dc_events(
        self, timeseries: np.array = None, threshold: float = 0.0001
    ) -> list:
        """
        Compute all relevant Directional Change parameters

        Arguments:
            :timeseries (list[int|float]): list of values
            :threshold (float): default is 0.0001
        Returns:
            list of directional change events
        """
        if timeseries is None and self.timeseries is None:
            raise ValueError(
                "Timeseries data has to be a no empty numpy.array()"
            )
        if timeseries is None and self.timeseries is not None:
            timeseries = self.timeseries

        time_value_list = []
        time_point_list = []
        events = []

        ext_point_n = timeseries[0]
        curr_event_max = timeseries[0]
        curr_event_min = timeseries[0]
        time_point_max = 0
        time_point_min = 0
        trend_status = "up"
        time_point = 0
        for i, ts_value in enumerate(timeseries):
            time_value = (ts_value - ext_point_n) / (ext_point_n * threshold)
            time_value_list.append(time_value)
            time_point_list.append(time_point)
            time_point += 1
            if trend_status == "up":
                events.append("upward overshoot")
                if ts_value < ((1 - threshold) * curr_event_max):
                    trend_status = "down"
                    curr_event_min = ts_value
                    ext_point_n = curr_event_max
                    time_point = i - time_point_max
                    num_points_change = i - time_point_max
                    for j in range(1, num_points_change + 1):
                        events[-j] = "downward dc"
                else:
                    if ts_value > curr_event_max:
                        curr_event_max = ts_value
                        time_point_max = i
            else:
                events.append("downward overshoot")
                if ts_value > ((1 + threshold) * curr_event_min):
                    trend_status = "up"
                    curr_event_max = ts_value
                    ext_point_n = curr_event_min
                    time_point = i - time_point_min
                    num_points_change = i - time_point_min
                    for j in range(1, num_points_change + 1):
                        events[-j] = "upward dc"
                else:
                    if ts_value < curr_event_min:
                        curr_event_min = ts_value
                        time_point_min = i
        self.events = events
        return events

    def get_dc_events_starts(
        self, events: list = None, timeseries: list = None
    ) -> dict:
        """
        Get only Directional Changes starts

        Arguments:
            :events (list[str]): list of directional change events
            :timeseries (list[int|float]): list of values
        Returns:
            dictionary of boolean lists when each directional change events starts
        """
        starts = {}
        previous_change = None
        if events is None and self.events is None:
            raise ValueError("Events data has to be a no empty numpy.array()")
        if events is None and self.events is not None:
            events = self.events
        if timeseries is None and self.timeseries is not None:
            timeseries = self.timeseries
        directional_changes = set(events)
        for directional_change in directional_changes:
            if directional_change not in starts:
                starts[directional_change] = []
        for index, current_change in enumerate(events):
            for directional_change in directional_changes:
                starts[directional_change].append(0)
            if previous_change != current_change:
                starts[current_change][-1] = (
                    1 if timeseries is None else timeseries[index]
                )
            previous_change = current_change
        return starts

    def get_dc_events_ends(
        self, events: list = None, timeseries: list = None
    ) -> dict:
        """
        Get only Directional Changes ends

        Arguments:
            :events (list[str]): list of directional change events
            :timeseries (list[int|float]): list of values
        Returns:
            dictionary of boolean lists when each directional change events ends
        """
        ends = {}
        previous_change = None
        if events is None and self.events is None:
            raise ValueError("Events data has to be a no empty numpy.array()")
        if events is None and self.events is not None:
            events = self.events
        if timeseries is None and self.timeseries is not None:
            timeseries = self.timeseries
        directional_changes = set(events)
        for directional_change in directional_changes:
            if directional_change not in ends:
                ends[directional_change] = []
        for index, current_change in enumerate(events):
            for directional_change in directional_changes:
                ends[directional_change].append(0)
            if previous_change != current_change:
                if previous_change is not None:
                    ends[previous_change][-2] = (
                        1 if timeseries is None else timeseries[index]
                    )
            previous_change = current_change
        ends[previous_change][-1] = 1 if timeseries is None else timeseries[-1]
        return ends
