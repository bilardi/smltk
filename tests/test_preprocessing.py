import unittest
from smltk.preprocessing import Ntk
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import scipy

class TestNlp(unittest.TestCase, Ntk):
    ntk = None
    doc = 'Good case, Excellent value. I am agree. There is a mistake. Item Does Not Match Picture.'
    docs = []
    target = []
    tuples = []
    default_doc_filtered = 'good case excellent value agree mistake item match picture'
    without_stop_words_doc_filtered = 'good case excellent value i be agree there be a mistake item do not match picture'
    without_stop_words_with_min_length_doc_filtered = 'good case excellent value be agree there be mistake item do not match picture'
    without_stop_words_without_tag_map_doc_filtered = 'good case excellent value i am agree there is a mistake item doe not match picture'
    def __init__(self, *args, **kwargs):
        self.ntk = Ntk()
        self.docs = nltk.sent_tokenize(self.doc)
        self.target = [1, 1, 0, 0]
        self.tuples = self.ntk.create_tuples(self.docs, self.target)
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_doc_cleaned(self):
        self.assertEqual(self.ntk.get_doc_cleaned(self.doc), self.default_doc_filtered)

        vocab_token = self.ntk.create_vocab_from_docs(self.docs, False)
        self.assertEqual(len(vocab_token), 9)
        self.assertEqual(self.ntk.get_stats_vocab(vocab_token), (9, 9))
        self.assertEqual(self.ntk.get_stats_vocab(vocab_token, 2), (0, 9))

        vocab_lemma = self.ntk.create_vocab_from_docs(self.docs, True)
        self.assertEqual(len(vocab_lemma), 9)
        self.assertEqual(self.ntk.get_stats_vocab(vocab_lemma), (9, 9))
        self.assertEqual(self.ntk.get_stats_vocab(vocab_lemma, 2), (0, 9))

        features_token = self.ntk.create_features(self.tuples, False)
        self.assertEqual(features_token[1], ({'agree': True}, 1))

        features_lemma = self.ntk.create_features(self.tuples, True)
        self.assertEqual(features_lemma[1], ({'agree': True}, 1))

        docs_cleaned = []
        for doc in self.docs:
            docs_cleaned.append(self.ntk.get_doc_cleaned(doc, False))
        tuples = self.ntk.create_tuples(docs_cleaned, self.target)
        vocab_token = self.ntk.create_vocab_from_tuples(tuples, False)
        self.assertEqual(self.ntk.get_stats_vocab(vocab_token), (9, 9))
        self.assertEqual(self.ntk.get_stats_vocab(vocab_token, 2), (0, 9))

    def test_doc_cleaned_with_lemmatizer(self):
        ntk = Ntk({'lemmatizer': WordNetLemmatizer()})
        self.assertEqual(ntk.get_doc_cleaned(self.doc), self.default_doc_filtered)

    def test_doc_cleaned_without_stop_words(self):
        ntk = Ntk({'stop_words': []})
        self.assertEqual(ntk.get_doc_cleaned(self.doc), self.without_stop_words_doc_filtered)

        vocab_token = ntk.create_vocab_from_docs(self.docs, False)
        self.assertEqual(len(vocab_token), 16)
        self.assertEqual(vocab_token['am'], 1)
        self.assertEqual(vocab_token['is'], 1)
        self.assertEqual(vocab_token['does'], 1)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (16, 16))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (0, 16))

        vocab_lemma = ntk.create_vocab_from_docs(self.docs, True)
        self.assertEqual(len(vocab_lemma), 15)
        self.assertEqual(vocab_lemma['be'], 2)
        self.assertEqual(vocab_lemma['do'], 1)
        self.assertEqual(ntk.get_stats_vocab(vocab_lemma), (15, 15))
        self.assertEqual(ntk.get_stats_vocab(vocab_lemma, 2), (1, 15))

        features_token = ntk.create_features(self.tuples, False)
        self.assertEqual(features_token[1], ({'agree': True, 'am': True, 'i': True}, 1))

        features_lemma = ntk.create_features(self.tuples, True)
        self.assertEqual(features_lemma[1], ({'agree': True, 'be': True, 'i': True}, 1))

        docs_cleaned = []
        for doc in self.docs:
            docs_cleaned.append(ntk.get_doc_cleaned(doc, False))
        tuples = ntk.create_tuples(docs_cleaned, self.target)
        vocab_token = ntk.create_vocab_from_tuples(tuples, False)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (16, 16))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (0, 16))

        docs_cleaned = []
        for doc in self.docs:
            docs_cleaned.append(ntk.get_doc_cleaned(doc, True))
        tuples = ntk.create_tuples(docs_cleaned, self.target)
        vocab_token = ntk.create_vocab_from_tuples(tuples)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (15, 15))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (1, 15))

    def test_doc_cleaned_without_stop_words_with_min_length(self):
        ntk = Ntk({'stop_words': [], 'min_length': 2})
        self.assertEqual(ntk.get_doc_cleaned(self.doc), self.without_stop_words_with_min_length_doc_filtered)

        vocab_token = ntk.create_vocab_from_docs(self.docs, False)
        self.assertEqual(len(vocab_token), 14)
        self.assertEqual(vocab_token['am'], 1)
        self.assertEqual(vocab_token['is'], 1)
        self.assertEqual(vocab_token['does'], 1)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (14, 14))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (0, 14))

        vocab_lemma = ntk.create_vocab_from_docs(self.docs, True)
        self.assertEqual(len(vocab_lemma), 13)
        self.assertEqual(vocab_lemma['be'], 2)
        self.assertEqual(vocab_lemma['do'], 1)
        self.assertEqual(ntk.get_stats_vocab(vocab_lemma), (13, 13))
        self.assertEqual(ntk.get_stats_vocab(vocab_lemma, 2), (1, 13))

        features_token = ntk.create_features(self.tuples, False)
        self.assertEqual(features_token[1], ({'agree': True, 'am': True}, 1))

        features_lemma = ntk.create_features(self.tuples, True)
        self.assertEqual(features_lemma[1], ({'agree': True, 'be': True}, 1))

        docs_cleaned = []
        for doc in self.docs:
            docs_cleaned.append(ntk.get_doc_cleaned(doc, False))
        tuples = ntk.create_tuples(docs_cleaned, self.target)
        vocab_token = ntk.create_vocab_from_tuples(tuples, False)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (14, 14))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (0, 14))

        docs_cleaned = []
        for doc in self.docs:
            docs_cleaned.append(ntk.get_doc_cleaned(doc, True))
        tuples = ntk.create_tuples(docs_cleaned, self.target)
        vocab_token = ntk.create_vocab_from_tuples(tuples)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (13, 13))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (1, 13))

    def test_doc_cleaned_without_stop_words_without_tag_map(self):
        ntk = Ntk({'stop_words': [], 'tag_map': {}})
        self.assertEqual(ntk.get_doc_cleaned(self.doc), self.without_stop_words_without_tag_map_doc_filtered)

        vocab_token = ntk.create_vocab_from_docs(self.docs, False)
        self.assertEqual(len(vocab_token), 16)
        self.assertEqual(vocab_token['does'], 1)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (16, 16))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (0, 16))

        vocab_lemma = ntk.create_vocab_from_docs(self.docs, True)
        self.assertEqual(len(vocab_lemma), 16)
        self.assertEqual(vocab_lemma['doe'], 1)
        self.assertEqual(ntk.get_stats_vocab(vocab_lemma), (16, 16))
        self.assertEqual(ntk.get_stats_vocab(vocab_lemma, 2), (0, 16))

        features_token = ntk.create_features(self.tuples, False)
        self.assertEqual(features_token[1], ({'agree': True, 'am': True, 'i': True}, 1))

        features_lemma = ntk.create_features(self.tuples, True)
        self.assertEqual(features_lemma[1], ({'agree': True, 'am': True, 'i': True}, 1))

        docs_cleaned = []
        for doc in self.docs:
            docs_cleaned.append(ntk.get_doc_cleaned(doc, False))
        tuples = ntk.create_tuples(docs_cleaned, self.target)
        vocab_token = ntk.create_vocab_from_tuples(tuples, False)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (16, 16))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (0, 16))

        docs_cleaned = []
        for doc in self.docs:
            docs_cleaned.append(ntk.get_doc_cleaned(doc, True))
        tuples = ntk.create_tuples(docs_cleaned, self.target)
        vocab_token = ntk.create_vocab_from_tuples(tuples)
        self.assertEqual(ntk.get_stats_vocab(vocab_token), (16, 16))
        self.assertEqual(ntk.get_stats_vocab(vocab_token, 2), (0, 16))

    def test_vectorize_docs(self):
        X_test = self.ntk.vectorize_docs(self.docs)
        self.assertEqual(type(X_test[0]), scipy.sparse.csr.csr_matrix)
        X_test = self.ntk.vectorize_docs(self.docs, False)
        self.assertEqual(type(X_test[0]), scipy.sparse.csr.csr_matrix)

if __name__ == '__main__':
    unittest.main()