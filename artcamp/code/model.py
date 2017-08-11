""""""

from sklearn.pipeline import Pipeline

from cleaner import TextProcessor
from lsi import GensimLsi
from utils import fetch_url, name_list
from vectorizers import GensimTfidf



class Org2Org():
    """
    Implements Org-to-Org Queries.
    """

    def __init__(self, lsi_path, dict_path, tfidf_path):
    	"""
        Initialize class.

        Parameters
        ----------
        lsi_path : str
            Path to location of saved gensim LsiModel.
            If specified, the model will load and use this object
            as its LsiModel.
        dict_path : str
            Path to location of saved gensim Dictionary.
            If specified, the model will load and use this object
            as its Dictionary.
        tfidf_path: str
            File-path designating where self.tfidf should be saved.
        """
        self.model = Pipeline([
                ('norm', TextProcessor()),
                ('tfidf', GensimTfidf(tfidf_path=tfidf_path, dictionary_path=dict_path)),
                ('model', GensimLsi(lsi_path=lsi_path, dictionary_path=dict_path))
            ])

    @staticmethod
    def closest_match(string1, strings):
    	""""""
    	pass

    @classmethod
    def resolve_query(cls, org):
    	"""
        Finds most similar org to 'org'.
        This is used to ensure that a match occurs.
        Maybe implement as edit distance for every doc
        """
        if org in name_list:
            return org
        return closest_match(org, name_list)




    def similarity(self, org, n=10):
        """"""
        doc = self.resolve_query(org)
        result= self.model.similarity(doc=doc, n=n)


class Art2Org():
    pass


class Org2Art():
    pass
