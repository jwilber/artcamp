"""ocstring."""
from difflib import get_close_matches

from sklearn.pipeline import Pipeline

from cleaner import TextProcessor
from lsi import GensimLsi
from data import name_list
from vectorizers import GensimTfidf


class Org2Org():
    """Implements Org-to-Org Queries."""

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
            ('tfidf', GensimTfidf(tfidf_path=tfidf_path,
                                  dictionary_path=dict_path)),
            ('model', GensimLsi(lsi_path=lsi_path,
                                dictionary_path=dict_path))])

    @staticmethod
    def closest_match(string1, strings):
        """
        Return the most similar org in name_list.

        Parameters
        ----------
        string1: str
            String being queried.
        strings: list
            List of org names.
        """
        result = get_close_matches('american cancer society', name_list)
        try:
            return result[0]
        except IndexError:
            return "Not Found"

    @classmethod
    def resolve_query(cls, org):
        """
        Find most similar org to 'org'.

        Parameters
        ----------
        org: str
            Name of organizatin to query.
        """
        if org in set(name_list):
            correct_org = org
        else:
            correct_org = cls.closest_match(org)
        # return associated data
        return correct_org

    def similarity(self, org, n=10):
        """A docstring."""
        doc, idx = Org2Org.resolve_query(org)
        if doc == "Not Found":
            return "Org not found, please search for another name."
        # Find data associated with doc before returning anything
        # i.e., doc = self.dictionary[idx]
        # solution, load corpus and name_list together.
        result = self.model.similarity(doc=doc, n=n)
        return result


class Art2Org():
    """init."""

    pass


class Org2Art():
    """init."""

    pass
