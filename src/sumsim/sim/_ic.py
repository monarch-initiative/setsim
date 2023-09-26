import typing
from statistics import mean

import hpotk
import hpotk.algorithm


class IcTransformer:
    """
    Transform information contents to delta information contents.
    """

    def __init__(self, hpo: hpotk.MinimalOntology,
                 strategy: str = 'mean'):
        self._hpo = hpo
        self._strategy = strategy

    def transform(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        if self._strategy == 'mean':
            return self._use_mean(ic_dict)
        elif self._strategy == 'max':
            return self._use_max(ic_dict)
        else:
            raise ValueError(f'Unknown strategy {self._strategy}')

    def _use_mean(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        delta_ic_dict = {}
        for TermID in ic_dict:
            parents = self._hpo.graph.get_parents(TermID)
            if len(parents) > 0:
                parent_ic = mean([ic_dict[i.value] for i in parents])
            else:
                parent_ic = 0
            delta_ic_dict[TermID] = ic_dict[TermID] - parent_ic
        return delta_ic_dict

    def _use_max(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        delta_ic_dict = {}
        for TermID in ic_dict:
            parent_ic = max([ic_dict[i.value] for i in self._hpo.graph.get_parents(TermID)], default=0)
            delta_ic_dict[TermID] = ic_dict[TermID] - parent_ic
        return delta_ic_dict
