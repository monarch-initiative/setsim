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

    def __init__(self, delta_ic_dict: typing.Mapping[hpotk.TermId, float]):
        self._delta_ic_dict = delta_ic_dict

    def _score_feature_sets(self, feature_sets: (typing.Set[hpotk.TermId], typing.Set[hpotk.TermId])) -> float:
        all_features = feature_sets[0].intersection(feature_sets[1])
        try:
            sim = sum(self._delta_ic_dict.get(term, None) for term in all_features)
        except KeyError:
            features = [feature for feature in all_features if self._delta_ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return sim


class SumSimSimilarityKernel(OntoSetSimilarityKernel, SumSimilarity):
    def __init__(self, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)
        SumSimilarity.__init__(self, delta_ic_dict)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class SumSimSimilaritiesKernel(SetSimilaritiesKernel, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, delta_ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)
        self._delta_ic_dict = delta_ic_dict

    def _score_addition(self, new_feature_set: Set[hpotk.TermId], disease_leftovers: Set[hpotk.TermId]) \
            -> (float, Set[hpotk.TermId]):
        all_features = new_feature_set.intersection(disease_leftovers)
        try:
            sim = sum(self._delta_ic_dict.get(term, None) for term in all_features)
        except KeyError:
            features = [feature for feature in all_features if self._delta_ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return sim, disease_leftovers.difference(all_features)

    def compute(self, sample: Phenotyped) -> typing.Iterable[float]:
        disease_leftovers = self._disease_features.copy()
        sim = 0.0
        results = []
        for next_set in self._sample_iterator(sample):
            sim_addition, disease_leftovers = self._score_addition(next_set, disease_leftovers)
            sim += sim_addition
            results.append(sim)
        return results

