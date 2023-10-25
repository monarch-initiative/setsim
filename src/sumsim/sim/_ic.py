import typing
import warnings

from statistics import mean
from sumsim.model import Sample
import multiprocessing

from math import log
import numpy as np

import hpotk


class IcTransformer:
    """
    Transform information contents to delta information contents.
    """

    def __init__(self, hpo: hpotk.MinimalOntology, root: str = "HP:0000118"):
        self._hpo = hpo
        self._root = hpo.get_term(root).identifier
        self._strategy = str

    def transform(self, ic_dict: typing.Mapping[hpotk.TermId, float],
                  strategy: str = 'mean') -> typing.Mapping[hpotk.TermId, float]:
        self._strategy = strategy
        pheno_abn = {i for i in self._hpo.graph.get_descendants(self._root, include_source=True)}
        dict_keys = set(ic_dict.keys())
        incompatible_terms = dict_keys.difference(dict_keys & pheno_abn)
        if incompatible_terms:
            raise ValueError(f'Original dictionary contains the following terms which are not descendants of the  root '
                             f'({self._root.value}):\n{incompatible_terms}')
        if self._strategy == 'mean':
            return self._use_mean(ic_dict)
        elif self._strategy == 'max':
            return self._use_max(ic_dict)
        else:
            raise ValueError(f'Unknown strategy {self._strategy}')

    def _use_mean(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        delta_ic_dict = {}
        for term, term_ic in ic_dict.items():
            if term != self._root:
                parents = self._hpo.graph.get_parents(term)
                # Get mean IC of parents of term ignoring those not in the dictionary.
                parent_ic = mean(ic_dict[i] for i in parents if i in ic_dict)
            else:
                parent_ic = 0
            delta_ic_dict[term] = term_ic - parent_ic
        return delta_ic_dict

    def _use_max(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        delta_ic_dict = {}
        for term, term_ic in ic_dict.items():
            if term != self._root:
                parents = self._hpo.graph.get_parents(term)
                parent_ic = max([ic_dict[i] for i in parents], default=0)
            else:
                parent_ic = 0
            delta_ic_dict[term] = term_ic - parent_ic
        return delta_ic_dict


class IcCalculator:
    """
    Create a dictionary providing the information content of terms.
    """

    def __init__(self, hpo: hpotk.MinimalOntology, root: str = "HP:0000118"):
        self._hpo = hpo.graph
        self._root = hpo.get_term(root)
        self.samples = None
        self.used_terms = set()
        self.used_pheno_abn = set()
        self.sample_array = None

    def calculate_ic_from_samples(self, samples: typing.Sequence[Sample]) -> typing.Mapping[hpotk.TermId, float]:
        self.samples = samples
        self.used_terms = set(pf for sample in samples for pf in sample.phenotypic_features)
        self.used_pheno_abn = self.used_terms & {i for i in self._hpo.get_descendants(self._root, include_source=True)}
        if len(self.used_terms) != len(self.used_pheno_abn):
            warnings.warn("Your samples include terms that are not included as a Phenotypic abnormality (HP:0000118) "
                          "in your ontology! These terms will be ignored.")
        self.sample_array = self._get_sample_array()
        # Define the number of processes to use
        num_processes = max(1, multiprocessing.cpu_count() - 2)  # Use all but 2 available CPU cores
        # Create a multiprocessing pool
        pool = multiprocessing.Pool(processes=num_processes)
        results = list(pool.imap(self._get_term_ic, self._hpo.get_descendants(self._root, include_source=True)))
        pool.close()
        ic_dict = {key: value for key, value in results}
        return ic_dict

    def _get_sample_array(self) -> np.array:
        # Convert hpotk.TermID to string for array index
        array_type = [(col.value, bool) for col in self.used_pheno_abn]
        array = np.zeros(len(self.samples), dtype=array_type)
        i = 0
        for sample in self.samples:
            for term in sample.phenotypic_features:
                array[term.value][i] = True
            i = i + 1
        return array

    def _get_term_ic(self, term: hpotk.TermId) -> (hpotk.TermId, float):
        term_descendants = set(i for i in self._hpo.get_descendants(term, include_source=True))
        relevant_descendants = [i.value for i in list(term_descendants & self.used_pheno_abn)]
        if len(relevant_descendants) > 0:
            freq = sum(1 if any(row[relevant_descendants]) else 0 for row in self.sample_array)
        else:
            freq = 1
        ic = log(len(self.sample_array) / freq)
        return term, ic
