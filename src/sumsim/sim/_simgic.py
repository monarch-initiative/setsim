import abc
import typing
from abc import ABC
import hpotk
from typing import Set
from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult, OntoSetSimilarityKernel, SetSimilarityKernel, \
    SetSimilaritiesKernel, WeightedSimilarity


class SimGicSimilarity(WeightedSimilarity, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """

    def _normalization_method(self):
        return "union"


class SimGicSimilarityKernel(OntoSetSimilarityKernel, SimGicSimilarity):
    def __init__(self, hpo: hpotk.GraphAware, ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)
        SimGicSimilarity.__init__(self, ic_dict)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class SimGicSimilaritiesKernel(SetSimilaritiesKernel, SimGicSimilarity, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)
        WeightedSimilarity.__init__(self, ic_dict)

