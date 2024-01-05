import abc
import typing
from abc import ABC

import hpotk

from typing import Set

from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult, OntoSetSimilarityKernel, SetSimilarityKernel


class SumSimilarity(SetSimilarityKernel, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """

    def __init__(self, delta_ic_dict: typing.Mapping[hpotk.TermId, float]):
        self._delta_ic_dict = delta_ic_dict

    def _score_shared_features(self, all_features: typing.Set[hpotk.TermId]) -> float:
        try:
            shared_terms = sum(self._delta_ic_dict.get(term, None) for term in all_features)
        except KeyError:
            features = [feature for feature in all_features if self._delta_ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return shared_terms


class SumSimSimilarityKernel(OntoSetSimilarityKernel, SumSimilarity):
    def __init__(self, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)
        SumSimilarity.__init__(self, delta_ic_dict)

