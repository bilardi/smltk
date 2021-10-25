"""The classes for data preparation with nltk, sklearn and more

    A collection of methods to simplify your code.

"""
import re
import string
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Ntk():
    """
    The class Ntk contains the Natural Language Processing tool kit.
    It needs the property named **params** (dict)
        :param language (str): default is english
        :param lemmatizer (obj): obj with method lemmatize() like WordNetLemmatizer()
        :param min_length (int): default is 1, so all words will be used
        :param stop_words (list[str]): default is stopwords.words()
        :param tag_map (dict): default contains J, V, R

    Here's an example:

        >>> from smltk.preprocessing import Ntk
        >>> doc = 'Good case, Excellent value.'
        >>> ntk = Ntk()
        >>> get_doc_cleaned = ntk.get_doc_cleaned(doc)
        >>> print(get_doc_cleaned)
        good case excellent value
    """
    language = 'english'
    lemmatizer = None
    min_length = 1
    stop_words = []
    tag_map = {}

    def __init__(self, params = []):
        if 'language' in params:
            self.language = params['language']
        if 'lemmatizer' in params:
            self.lemmatizer = params['lemmatizer']
        else:
            self.lemmatizer = WordNetLemmatizer()
        if 'min_length' in params:
            self.min_length = params['min_length']
        if 'stop_words' in params:
            self.stop_words = set(params['stop_words'])
        else:
            self.stop_words = set(stopwords.words(self.language))
        if 'tag_map' in params:
            self.tag_map = params['tag_map']
        else:
            self.tag_map = defaultdict(lambda : wn.NOUN)
            self.tag_map['J'] = wn.ADJ
            self.tag_map['V'] = wn.VERB
            self.tag_map['R'] = wn.ADV
        if not len(self.tag_map):
            for pos in ['E', 'D', 'J', 'N', 'V', 'R']:
                self.tag_map[pos] = wn.NOUN

    def word_tokenize(self, doc):
        """
        splits document in each word
            Arguments:
                doc (str): text
            Returns:
                list of words
        """
        return nltk.word_tokenize(doc)

    def tokenize_and_clean_doc(self, doc):
        """
        tokenizes doc
        and filters tokens from punctuation, numbers, stop words, and words <= min_length
        and changes upper characters in lower characters
            Arguments:
                doc (str): text
            Returns:
                list of words filtered
        """
        # split into tokens by white space
        tokens = self.word_tokenize(doc)
        # prepare regex for char filtering
        punctuation = re.compile('[%s]' % re.escape(string.punctuation))
        # remove punctuation from each word
        tokens = [punctuation.sub('', token) for token in tokens]
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
        lemmatizes tokens
            Arguments:
                tokens (list[str]): list of words
            Returns:
                list of tokens lemmatized
        """
        # lemmatization
        lemmas = [self.lemmatizer.lemmatize(token, self.tag_map[tag[0]]) for token, tag in nltk.pos_tag(tokens)]
        return lemmas

    def get_tokens_cleaned(self, doc, is_lemma = True):
        """
        tokenizes doc
        and filters tokens from punctuation, numbers, stop words, and words <= min_length
        and changes upper characters in lower characters
        and if is_lemma == True, also it lemmatizes
            Arguments:
                doc (str): text
                is_lemma (bool): default is True
            Returns:
                list of tokens cleaned
        """
        tokens = self.tokenize_and_clean_doc(doc)
        if is_lemma == True:
            tokens = self.lemmatize(tokens)
        return tokens

    def __call__(self, doc):
        """
        alias of get_tokens_cleaned for using in models
            Arguments:
                doc (str): text
            Returns:
                list of tokens cleaned
        """
        return self.get_tokens_cleaned(doc)

    def get_doc_cleaned(self, doc, is_lemma = True):
        """
        filters doc from punctuation, numbers, stop words, and words <= min_length
        and changes upper characters in lower characters
        and if is_lemma == True, also it lemmatizes
            Arguments:
                doc (str): text
                is_lemma (bool): default is True
            Returns:
                string cleaned
        """
        tokens = self.get_tokens_cleaned(doc, is_lemma)
        return ' '.join(map(str, tokens))

    def add_doc_to_vocab(self, doc, vocab, is_lemma = True):
        """
        adds tokens of that doc to vocabulary and updates vocabulary
            Arguments:
                doc (str): text
                vocab (list[str]): list of tokens
                is_lemma (bool): default is True
            Returns:
                list of tokens of that doc and vocab updated
        """
        tokens = self.get_tokens_cleaned(doc, is_lemma)
        vocab.update(tokens)
        return tokens

    def create_vocab_from_docs(self, docs, is_lemma = True):
        """
        creates vocabulary from list of docs
            Arguments:
                docs (list[str]): list of texts
                is_lemma (bool): default is True
            Returns:
                list of tokens
        """
        vocab = Counter()
        [self.add_doc_to_vocab(doc, vocab, is_lemma) for doc in docs]
        return vocab

    def get_stats_vocab(self, vocab, min_occurance = 1):
        """
        gets statistics of vocabulary
            Arguments:
                vocab (list[str]): list of tokens
                min_occurance (int): minimum occurance considered
            Returns:
                tuple of tokens number with >= min_occurance and total tokens number
        """
        tokens = [ key for key, count in vocab.items() if count >= min_occurance]
        partial = len(tokens)
        total = len(vocab)
        return partial, total

    def create_tuples(self, docs = [], target = []):
        """
        creates tuples with sample and its target
            Arguments:
                docs (list[str]): list of texts
                target (list[str]): list of targets
            Returns:
                list of tuples with sample and its target
        """
        return list(zip(docs, target))

    def create_vocab_from_tuples(self, tuples, is_lemma = True):
        """
        creates vocabulary from list of tuples
            Arguments:
                tuples (list[tuples]): list of tuples with sample and its target
                is_lemma (bool): default is True
            Returns:
                list of tokens
        """
        vocab = Counter()
        [self.add_doc_to_vocab(doc, vocab, is_lemma) for doc, target in tuples]
        return vocab

    def find_features(self, doc, is_lemma = True):
        """
        finds features
            Arguments:
                doc (str): text
                is_lemma (bool): default is True
            Returns:
                dictionary of tokens cleaned
        """
        tokens = self.get_tokens_cleaned(doc, is_lemma)
        features = {}
        for token in tokens:
            features[token] = True
        return features

    def create_features(self, tuples, is_lemma = True):
        """
        creates features
            Arguments:
                tuples (list[tuples]): list of tuples with sample and its target
                is_lemma (bool): default is True
            Returns:
                list of tuples with features and relative target
        """
        return [(self.find_features(doc, is_lemma), target) for (doc, target) in tuples]

    def vectorize_docs(self, docs, is_count = True, is_lemma = True):
        """
        vectorizes docs
            Arguments:
                docs (list[str]): list of texts
                is_count (bool): default is True
                is_lemma (bool): default is True
            Returns:
                list of scipy.sparse.csr.csr_matrix, one for each doc
        """
        if is_count == True:
            vectorizer = CountVectorizer()
        else:
            vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(docs)
