"""Lsi Model."""
import os

from sklearn.base import BaseEstimator, TransformerMixin


class GensimLsi(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for LSI.

    Wrapped by sklearn.
    """

    def __init__(self, lsi_path=None):
        """
        Instantiate GensimLsi object.

        Parameters
        ----------
        lsi_path : str
            Path to location of saved gensim LsiModel.
            If specified, the model will load and use this object
            as its LsiModel.
        """
        self.model = None
        self.id2word = None
        self.num_topics = None
        self.corpus = None
        self.index = None
        if lsi_path is not None:
            self.load(lsi_path)

    @staticmethod
    def load(lsi_path):
        """
        Load previously saved LSI model.

        Parameters
        ----------
        lsi_path: str
            File-path designating where self.model should be saved.
        """
        import pickle
        if not os.path.exists(lsi_path):
            raise IOError('File path to the LsiModel was not found.'
                          'Please insert correct path to LsiModel.')
        with file(lsi_path, 'rb') as f:
            return pickle.load(f)

    def save(self, lsi_path):
        """
        Save model to disc.

        Parameters
        ----------
        lsi_path: str
            File-path designating where self.model should be saved.
        """
        import pickle
        if not (self.model):
            raise AttributeError('Nothing to save yet, please run .fit first.')
        f = file(lsi_path, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def fit(self, documents, id2word, num_topics=10, labels=None):
        """
        Fit LsiModel to documents.

        Parameters
        ----------
        documents: iterable
            List of documents, each itself a list of preprocessed tokens.
        labels: iterable (default=None)
            Optional list specifying document labels.

        """
        from gensim.models import LsiModel
        if self.id2word is None:
            self.id2word = id2word
        self.num_topics = num_topics
        self.model = LsiModel(corpus=documents,
                              id2word=id2word,
                              num_topics=num_topics)
        self.corpus = self.model[documents]
        return self

    def transform(self, documents):
        """
        Return ectorized embedding for each document in documents.

        Parameters
        -----------
        documents: iterable
            List of documents. Each document must be a list of tokens.

        Returns
        -------
            iterable: list of vectorized documents.
        """
        if self.model is None:
            raise AttributeError('Must have a fit model in order'
                                 ' to call transform.')
        return self.model[documents]

    def similarity(self, org, n=10):
        """
        Return n most similar orgs to org.

        Parameters
        ----------
        org:
            A document. embedded in same tfidf space as model.
        n: int (default=10)
            Number of most similar items to return.

        Returns
        -------
            sims: dictionary of (item, distance) key, value pairs sorted by
                similarity in descending order.
        """
        if self.model is None:
            raise AttributeError('Must have a fit model in order'
                                 ' to call similarity.')
        if self.index is None:
            print "no index, creating our own"
            print
            from gensim.similarities import MatrixSimilarity
            self.index = MatrixSimilarity(self.corpus)
        # if n is larger than the number of documents, return 1 result
        _n = n * (n < len(self.index)) + (1 - (n < len(self.index))) * 1

        if _n == 1:
            print("{n} too large a number, returning 1 result.").format(n=n)
        doc_lsi = self.model[org]
        sims = self.index[doc_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[0:_n]
        return sims
