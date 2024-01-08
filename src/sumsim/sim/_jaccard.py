import abc
import re
import typing

from collections import namedtuple
from typing import Set

import hpotk

from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult, SetSimilarityKernel, OntoSetSimilarityKernel, \
    SetSimilaritiesKernel

HPO_PATTERN = re.compile(r"HP:(?P<ID>\d{7})")

SimpleFeature = namedtuple('SimplePhenotypicFeature', field_names=('identifier', 'is_present'))
"""
An implementation detail for Jaccard kernels. 
"""


class JaccardSimilarity(SetSimilarityKernel, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """

    def _score_feature_sets(self, feature_sets: (typing.Set[hpotk.TermId], typing.Set[hpotk.TermId])) -> float:
        union_len = len(feature_sets[0].union(feature_sets[1]))
        if union_len == 0:
            return 0.0
        intersection_len = len(feature_sets[0].intersection(feature_sets[1]))
        return intersection_len / union_len


class JaccardSimilarityKernel(OntoSetSimilarityKernel, JaccardSimilarity):
    """
    `JaccardSimilarityKernel` uses *both* present and excluded phenotypic features to calculate the similarity.

    The kernel prepares induced graphs for each sample by adding ancestors implied by the present terms
    and descendants of the excluded terms and proceeds with applying Jaccard coefficient - the ratio of intersection
    over union.

    If ``exact=True`` then the implied annotations are ignored. In result, the kernel performs exact matching.

    Note that no special penalization is applied if a feature is present in one and excluded in the other.
    The observation status mismatch is accounted as a simple mismatch.

    :param hpo: hpo-toolkit's representation of Human Phenotype Ontology.
    :param exact: `True` if the exact matching should be performed.
    """

    def __init__(self, hpo: hpotk.GraphAware, root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class JaccardSimilaritiesKernel(SetSimilaritiesKernel, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)

    @staticmethod
    def _intersection_addition(new_feature_set: Set[hpotk.TermId], disease_leftovers: Set[hpotk.TermId]) \
            -> (float, Set[hpotk.TermId]):
        additional_intersection_set = new_feature_set.intersection(disease_leftovers)
        return len(additional_intersection_set), disease_leftovers.difference(additional_intersection_set)

    def _next_union(self, new_feature_set: Set[hpotk.TermId], current_union_set: Set[hpotk.TermId]) \
            -> (float, Set[hpotk.TermId]):
        next_union = new_feature_set.union(current_union_set).intersection(self._features_under_root)
        return len(next_union), next_union

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
