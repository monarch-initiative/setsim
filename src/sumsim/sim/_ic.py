import typing

import hpotk


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
        # TODO - implement
        return {}

    def _use_max(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        # TODO - implement
        return {}


