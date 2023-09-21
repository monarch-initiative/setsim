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
        # TODO - implement the similarity calculation using the delta IC dict and the ontology graph
        pass
