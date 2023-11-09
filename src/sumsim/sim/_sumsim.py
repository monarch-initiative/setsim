import typing
import hpotk

from typing import Set

from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult


class SumSimSimilarityKernel(SimilarityKernel):

    def __init__(self, hpo: hpotk.GraphAware,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float], root: str = "HP:0000118"):
        self._hpo = hpo.graph
        self._root = root
        self._delta_ic = delta_ic_dict

    def compute(self, a: Phenotyped, b: Phenotyped) -> SimilarityResult:
        shared_features = self._get_all_shared_features(a, b)
        return SimilarityResult(self._calculate_total_ic(shared_features))

    def _get_all_shared_features(self, a: Phenotyped, b: Phenotyped):
        a_features = set(ancestor for pf in a.phenotypic_features for ancestor in
                         self._hpo.get_ancestors(pf, include_source=True))
        b_features = set(ancestor for pf in b.phenotypic_features for ancestor in
                         self._hpo.get_ancestors(pf, include_source=True))
        features_under_root = set(self._hpo.get_descendants(self._root, include_source=True))
        return a_features.intersection(b_features).intersection(features_under_root)

    def _calculate_total_ic(self, all_features: Set[hpotk.TermId]) -> float:
        try:
            shared_terms = sum(self._delta_ic.get(term, None) for term in all_features)
        except:
            print("Samples share features that are not included in the provided dictionary.")
            raise
        return shared_terms
