import abc
import typing
from abc import ABC

import hpotk

from typing import Set

from setsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult, OntoSetSimilarityKernel, SetSimilarityKernel, \
    SetSimilaritiesKernel, WeightedSimilarity


class SimIciSimilarity(WeightedSimilarity, metaclass=abc.ABCMeta):
    def _normalization_method(self):
        return "none"


class SimIciSimilarityKernel(OntoSetSimilarityKernel, SimIciSimilarity, metaclass=abc.ABCMeta):
    def __init__(self, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)
        WeightedSimilarity.__init__(self, delta_ic_dict)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class SimIciSimilaritiesKernel(SetSimilaritiesKernel, SimIciSimilarity, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)
        ic_dict = {term.value: ic for term, ic in delta_ic_dict.items()}
        WeightedSimilarity.__init__(self, ic_dict)
