import typing
from statistics import mean
from sumsim.model._base import Sample
import multiprocessing
from math import log
import numpy as np

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


class IcCalculator:
    """
    Create a dictionary providing the information content of terms.
    """

    def __init__(self, hpo: hpotk.MinimalOntology):
        self._hpo = hpo.graph
        self.samples = None
        self.used_terms = set()
        self.sample_array = None

    def calculate_ic_from_samples(self, samples: typing.Sequence[Sample], root: str = "HP:0000001") -> typing.Mapping[hpotk.TermId, float]:
        self.samples = samples
        self.used_terms = set(pf.value for sample in samples for pf in sample.phenotypic_features)
        self.sample_array = self._get_sample_array()
        # Define the number of processes to use
        num_processes = multiprocessing.cpu_count() - 2  # Use all but 2 available CPU cores
        # Create a multiprocessing pool
        pool = multiprocessing.Pool(processes=num_processes)
        results = list(pool.imap(self._get_term_freq, self._hpo.get_descendants(root, include_source=True)))
        pool.close()
        ic_dict = {key: value for key, value in results}
        return ic_dict

    def _get_sample_array(self) -> np.array:
        dtype = [(col, bool) for col in self.used_terms]
        array = np.zeros(len(self.samples), dtype=dtype)
        i = 0
        for sample in self.samples:
            for term in sample.phenotypic_features:
                array[term.value][i] = True
            i = i + 1
        return array


    def _get_term_freq(self, term: hpotk.TermId) -> (str, float):
        term_descendants = set(i.value for i in self._hpo.get_descendants(term.value, include_source=True))
        relevant_descendants = list(term_descendants & self.used_terms)
        if len(relevant_descendants) > 0:
            freq = sum(1 if any(row[relevant_descendants]) else 0 for row in self.sample_array)
        else:
            freq = 1
        ic = log(len(self.sample_array)/freq)
        return (term.value, ic)

