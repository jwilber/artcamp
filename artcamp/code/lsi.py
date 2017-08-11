ass GensimLsi(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer to convert tokenized,
    preprocessed data to tf-idf representation.
    """
    def __init__(self, lsi_path=None, dictionary_path=None):
        """
        Instantiate GensimTfidf object. If loading previously fit Dictionary and
        LsiModel, you must specify a path to both the Dictionary and the LsiModel.

        Parameters
        ----------
        lsi_path : str
            Path to location of saved gensim LsiModel.
            If specified, the model will load and use this object
            as its LsiModel.
        dictionary_path : str
            Path to location of saved gensim Dictionary.
            If specified, the model will load and use this object
            as its Dictionary.
        """
        self.dictionary = None
        self.model = None
        self.corpus = None
        self.index = None
        # if both paths specified, load object
        if lsi_path and dictionary_path:
            self._load(lsi_path=lsi_path, dictionary_path=dictionary_path)
        elif lsi_path or dictionary_path:
            raise AttributeError('If loading pre-fit Dictionary and LsiModel,'
                                 ' both must be specified, not just one.')

    def _load(self, lsi_path, dictionary_path):
        """
        If specified, attempts to load gensim LsiModel from `lsi_path`
        and gensim Dictionary from `dictionary_path`.

        Parameters
        ----------
        lsi_path: str
            File-path designating where self.model should be saved.
        dictionary_path: str
            File-path designating where self.dictionary should be saved.
        """
        from gensim.models import LsiModel
        from gensim.corpora.dictionary import Dictionary
        if not os.path.exists(lsi_path):
            raise IOError('The provided file path to the LsiModel was not found.'
                          'Please ensure that the argument is the correct path.')
        if not os.path.exists(dictionary_path):
            raise IOError('The provided file path to the Dictionary was not found.'
                          'Please ensure that the argument is the correct path.')
        self.model = LsiModel().load(lsi_path)
        self.dictionary = Dictionary().load(dictionary_path)

    def save(self, lsi_path, dictionary_path):
        """
        Saves objects from fit process: gensim.LsiModel to `lsi_path`
        and gensim.Dictionary to `dictionary_path`.
        If either self.model or self.dictionary does not exist, an
        AttributeError is raised.

        Parameters
        ----------
        lsi_path: str
            File-path designating where self.model should be saved.
        dictionary_path: str
            File-path designating where self.dictionary should be saved.
        """
        if not (self.model and self.dictionary):
            raise AttributeError('Nothing to save yet, please run .fit first.')
        self.model.save(lsi_path)
        self.dictionary.save(dictionary_path)

    def fit(self, documents, id2word, num_topics=10, labels=None):
        """
        Fits a gensim LsiModel to documents.

        Parameters
        ----------
        documents: iterable
            List of documents. Each document must be a list of preprocessed tokens.
        labels: iterable
            Optional list of same size as documents, specifying label for each document.

        """
        from gensim.models import LsiModel
        from gensim.corpora.dictionary import Dictionary
        if self.dictionary is None:
            self.dictionary = id2word
        self.model = LsiModel(documents, id2word=self.dictionary, num_topics=num_topics)
        self.corpus = self.model[documents]
        
        return self

    def transform(self, documents):
        """
        Returns a vectorized embedding of each document in documents.

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
    
    def similarity(self, doc, n=10):
        """
        Returns the `n` most similar items in `self.corpus`
        to `doc`.
        
        Parameters
        ----------
        doc:
            A document
        n: int (default=10)
            Number of most similar items to return.
            
        Returns
        -------
            sims: dictionary of (item, distance) key, value pairs sorted by
                similarity in descending order.
        """
        _n = n
        if self.model is None:
            raise AttributeError('Must have a fit model in order'
                                 ' to call similarity.')
        if self.index is None:
            from gensim.similarities import MatrixSimilarity
            self.index = MatrixSimilarity(self.corpus)
            
        # if n is too large, return 1 result
        _n = n * (n < len(index)) + (1 - (n < len(index))) * 1
                                     
        if _n == 1:
            print("{n} too large a number, return 1 result instead.").format(n=n)
        doc_lsi = self.model[doc]
        sims = self.index[doc_lsi[0]]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[0:_n]
        return sims
