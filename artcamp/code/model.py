"""docstring."""


from difflib import get_close_matches

from sklearn.pipeline import Pipeline

from cleaner import TextProcessor
from lsi import GensimLsi
from vectorizers import GensimTfidf
from utils import load_json

org_data = load_json('../data/org_data.txt')
name_list = load_json('../data/name_list.txt')


# lsi = LsiModel(data_tfidf, id2word=tfidf_dict, num_topics=600)


class Org2Org():
    """Implements Org-to-Org Queries."""

    def __init__(self, lsi_path, dict_path, tfidf_path, org_data, name_list):
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
        org_data: list
            List of data.
        name_list: list
            List of names.
        """
        self.org_data = org_data
        self.name_list = name_list
        self.processor = TextProcessor()
        self.tfidf = GensimTfidf(tfidf_path=tfidf_path,
                                 dictionary_path=dict_path,
                                 use_sparse_representation=True)
        self.lsi = GensimLsi.load(lsi_path)
        self.transformer = Pipeline([
            ('norm', self.processor),
            ('tfidf', self.tfidf)])

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
        result = get_close_matches(string1, strings)
        try:
            return result[0]
        except IndexError:
            return "Not Found"

    def resolve_query(self, org):
        """
        Find most similar org to 'org'.

        Parameters
        ----------
        org: str
            Name of organizatin to query.
        """
        if org in set(self.name_list):
            correct_org = org
        else:
            correct_org = self.closest_match(org, self.name_list)
        # return associated data
        return correct_org, self.name_list.index(correct_org)

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
        doc, idx = self.resolve_query(org)
        if doc == "Not Found":
            return "Org not found, please search for another name."
        # Find data associated with doc before returning anything
        doc_data = self.org_data[idx]
        if isinstance(doc_data, list):
            doc_data = doc_data
        else:
            doc_data = self.processor.transform([doc_data])
        tfidf_data = self.tfidf.transform([doc_data])
        result = self.lsi.similarity(org=tfidf_data[0], n=n)
        return result


class Art2Org():
    """init."""

    pass


class Org2Art():
    """init."""

    pass


def main():
    """Main func."""
    # print org_data[1]
    o2o = Org2Org(lsi_path='test_gensimlsisave.pkl',
                  dict_path='tfidf_dict.pkl',
                  tfidf_path='tfidf.pkl',
                  org_data=org_data,
                  name_list=name_list)

    results = o2o.similarity('SurfAid International USA, Inc.')
    print "Obtained results:"
    print results


if __name__ == '__main__':
    main()
