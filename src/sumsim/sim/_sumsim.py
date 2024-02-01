import abc
import typing
from abc import ABC

import hpotk

from typing import Set

from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult, OntoSetSimilarityKernel, SetSimilarityKernel, \
    SetSimilaritiesKernel


class SumSimilarity(SetSimilarityKernel, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """
    def __init__(self, delta_ic_dict):
        self._delta_ic_dict = delta_ic_dict

    def _normalize_by_union(self) -> bool:
        return False

    def _score_feature_set(self, feature_set: typing.Set[hpotk.TermId]) -> float:
        try:
            sim = sum(self._delta_ic_dict.get(term, None) for term in feature_set)
        except KeyError:
            features = [feature for feature in feature_set if self._delta_ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return sim


class SumSimSimilarityKernel(OntoSetSimilarityKernel, SumSimilarity, metaclass=abc.ABCMeta):
    def __init__(self, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)
        SumSimilarity.__init__(self, delta_ic_dict)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class SumSimSimilaritiesKernel(SetSimilaritiesKernel, SumSimilarity, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)
        SumSimilarity.__init__(self, delta_ic_dict)

