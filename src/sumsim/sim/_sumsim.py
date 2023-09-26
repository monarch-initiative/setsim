import typing
import hpotk

from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult


class SumSimSimilarityKernel(SimilarityKernel):

    def __init__(self, hpo: hpotk.GraphAware,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float]):
        self._hpo = hpo.graph
        self._delta_ic = delta_ic_dict

    def compute(self, a: Phenotyped, b: Phenotyped) -> SimilarityResult:
        a_features = set(ancestor for pf in a.phenotypic_features for ancestor in
                         self._hpo.get_ancestors(pf, include_source=True))
        b_features = set(ancestor for pf in b.phenotypic_features for ancestor in
                         self._hpo.get_ancestors(pf, include_source=True))
        ab_similarity = sum(self._delta_ic.get(pf, None) for pf in a_features.intersection(b_features))
        if None in ab_similarity:
            raise KeyError("One or more term IDs were not found in the IC dictionary")
        return SimilarityResult(ab_similarity)
