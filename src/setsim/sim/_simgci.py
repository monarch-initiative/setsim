import abc
import typing
import hpotk
from setsim.model import Phenotyped
from . import OntoSetSimilarityKernel, SetSimilaritiesKernel, WeightedSimilarity


class SimGciSimilarity(WeightedSimilarity, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """

    def _normalization_method(self):
        return "union"


class SimGciSimilarityKernel(OntoSetSimilarityKernel, SimGciSimilarity):
    def __init__(self, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)
        SimGciSimilarity.__init__(self, delta_ic_dict)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class SimGciSimilaritiesKernel(SetSimilaritiesKernel, SimGciSimilarity, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)
        ic_dict = {term.value: ic for term, ic in delta_ic_dict.items()}
        WeightedSimilarity.__init__(self, ic_dict)

