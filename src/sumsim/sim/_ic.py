import math
import multiprocessing
import typing
import warnings

import hpotk
import numpy as np

from statistics import mean

from sumsim.model import Sample


class IcTransformer:
    """
    Transform information contents to delta information contents.

    :param hpo: a representation of HPO
    :param root: a `str` or :class:`hpotk.TermId` of the term that should be used as the root
     for the purpose of IC transformation. Defaults to `Phenotypic abnormality`.
    """

    def __init__(self, hpo: hpotk.MinimalOntology,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self._hpo = hpotk.util.validate_instance(hpo, hpotk.MinimalOntology, 'hpo')
        # As a side effect of getting the term and using its identifier,
        # we ensure `self._root` corresponds to an ID of current (non-obsolete) term for given HPO version.
        root_term = hpo.get_term(root)
        if root_term is None:
            raise ValueError(f'Root {root} is not in provided HPO!')
        self._root = root_term.identifier

    def transform(self, ic_dict: typing.Mapping[hpotk.TermId, float],
                  strategy: str = 'mean') -> typing.Mapping[hpotk.TermId, float]:
        pheno_abn = set(self._hpo.graph.get_descendants(self._root, include_source=True))
        dict_keys = set(ic_dict.keys())
        incompatible_terms = dict_keys.difference(dict_keys.intersection(pheno_abn))
        if incompatible_terms:
            root_term = self._hpo.get_term(self._root)
            raise ValueError(f'Original dictionary contains the following terms which are not descendants of the root '
                             f'{root_term.name} ({self._root.value}):\n{incompatible_terms}')

        if strategy == 'mean':
            return self._use_mean(ic_dict)
        elif strategy == 'max':
            return self._use_max(ic_dict)
        else:
            raise ValueError(f'Unknown strategy {strategy}')

    def _use_mean(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        delta_ic_dict = {}
        for term_id, term_ic in ic_dict.items():
            if term_id != self._root:
                parents = self._hpo.graph.get_parents(term_id)
                # Get mean IC of parents of term ignoring those not in the dictionary.
                parent_ic = mean(ic_dict[parent] for parent in parents if parent in ic_dict)
            else:
                parent_ic = 0
            delta_ic_dict[term_id] = term_ic - parent_ic
        return delta_ic_dict

    def _use_max(self, ic_dict: typing.Mapping[hpotk.TermId, float]) -> typing.Mapping[hpotk.TermId, float]:
        delta_ic_dict = {}
        for term_id, term_ic in ic_dict.items():
            if term_id != self._root:
                parent_ic = max((ic_dict[parent] for parent in self._hpo.graph.get_parents(term_id)), default=0)
            else:
                parent_ic = 0
            delta_ic_dict[term_id] = term_ic - parent_ic
        return delta_ic_dict


class IcCalculator:
    """
    Create a dictionary providing the information content of terms.
    """

    def __init__(self, hpo: hpotk.MinimalOntology,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self._hpo = hpo.graph
        # As a side effect of getting the term and using its identifier,
        # we ensure `self._root` corresponds to an ID of current (non-obsolete) term for given HPO version.
        root_term = hpo.get_term(root)
        if root_term is None:
            raise ValueError(f'Root {root} is not in provided HPO!')
        self._root = root_term.identifier

        self._samples = None
        self._sample_terms = set()
        self._sample_array = None

    def calculate_ic_from_samples(self, samples: typing.Sequence[Sample]) -> typing.Mapping[hpotk.TermId, float]:
        self._samples = samples
        all_terms_in_samples = set(pf for sample in samples for pf in sample.phenotypic_features)
        self._sample_terms = all_terms_in_samples & {i for i in self._hpo.get_descendants(self._root, include_source=True)}
        if len(all_terms_in_samples) != len(self._sample_terms):
            warnings.warn("Your samples include terms that are not included as a Phenotypic abnormality (HP:0000118) "
                          "in your ontology! These terms will be ignored.")
        self._sample_array = self._get_sample_array(self._samples, self._sample_terms)
        # Define the number of processes to use
        num_processes = max(1, multiprocessing.cpu_count() - 2)  # Use all but 2 available CPU cores
        # Create a multiprocessing pool
        pool = multiprocessing.Pool(processes=num_processes)
        results = list(pool.imap(self._get_term_ic, self._hpo.get_descendants(self._root, include_source=True),
                                 chunksize=10))
        pool.close()
        ic_dict = {key: value for key, value in results}
        return ic_dict

    def calculate_ic_from_diseases(self):
        return None

    @staticmethod
    def _get_sample_array(samples: typing.Sequence[Sample],
                          used_pheno_abn: typing.Iterable[hpotk.TermId]) -> np.array:
        # Convert hpotk.TermID to string for array index
        array_type = [(col.value, bool) for col in used_pheno_abn]
        array = np.zeros(len(samples), dtype=array_type)
        i = 0
        for sample in samples:
            for term in sample.phenotypic_features:
                array[term.value][i] = True
            i = i + 1
        return array

    def _get_term_ic(self, term: hpotk.TermId) -> (hpotk.TermId, float):
        term_descendants = set(i for i in self._hpo.get_descendants(term, include_source=True))
        relevant_descendants = [i.value for i in list(term_descendants.intersection(self._sample_terms))]
        if len(relevant_descendants) > 0:
            freq = sum(1 if any(row[relevant_descendants]) else 0 for row in self._sample_array)
        else:
            freq = 1
        ic = math.log(len(self._sample_array) / freq)
        return term, ic
