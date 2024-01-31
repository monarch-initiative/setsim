import abc
import typing
from abc import ABC
import hpotk
from typing import Set
from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult, OntoSetSimilarityKernel, SetSimilarityKernel, \
    SetSimilaritiesKernel


class SimGicSimilarity(SetSimilarityKernel, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """

    def __init__(self, delta_ic_dict: typing.Mapping[hpotk.TermId, float]):
        self._delta_ic_dict = delta_ic_dict

    def _score_feature_sets(self, feature_sets: (typing.Set[hpotk.TermId], typing.Set[hpotk.TermId])) -> float:
        intersection_features = feature_sets[0].intersection(feature_sets[1])
        union_features = feature_sets[0].union(feature_sets[1])
        try:
            ic_intersection = sum(self._delta_ic_dict.get(term, None) for term in intersection_features)
            ic_union = sum(self._delta_ic_dict.get(term, None) for term in union_features)
        except KeyError:
            features = [feature for feature in intersection_features if self._delta_ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return ic_intersection / ic_union


class SimGicSimilarityKernel(OntoSetSimilarityKernel, SimGicSimilarity):
    def __init__(self, hpo: hpotk.GraphAware, ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)
        SimGicSimilarity.__init__(self, ic_dict)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class SimGicSimilaritiesKernel(SetSimilaritiesKernel, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, ic_dict: typing.Mapping[hpotk.TermId, float],
                 root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)
        self._ic_dict = ic_dict

    def _intersection_addition(self, new_feature_set: Set[hpotk.TermId], disease_leftovers: Set[hpotk.TermId]) \
            -> (float, Set[hpotk.TermId]):
        additional_intersection_set = new_feature_set.intersection(disease_leftovers)
        try:
            ic_intersection = sum(self._ic_dict.get(term, None) for term in additional_intersection_set)
        except KeyError:
            features = [feature for feature in additional_intersection_set if self._ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return ic_intersection, disease_leftovers.difference(additional_intersection_set)

    def _next_union(self, new_feature_set: Set[hpotk.TermId], current_union_set: Set[hpotk.TermId]) \
            -> (float, Set[hpotk.TermId]):
        next_union = new_feature_set.union(current_union_set).intersection(self._features_under_root)
        try:
            ic_union = sum(self._ic_dict.get(term, None) for term in next_union)
        except KeyError:
            features = [feature for feature in next_union if self._ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return ic_union, next_union

    def compute(self, sample: Phenotyped, return_last_result: bool = False) \
            -> typing.Union[typing.Sequence[float], float]:
        disease_leftovers = self._disease_features.copy()
        union_set = disease_leftovers.copy()
        intersection = 0.0
        results = []
        for next_set in self._sample_iterator(sample):
            intersection_addition, disease_leftovers = self._intersection_addition(next_set, disease_leftovers)
            intersection += intersection_addition
            union, union_set = self._next_union(next_set, union_set)
            results.append(intersection / union)
        if return_last_result:
            return results[-1]
        return results
