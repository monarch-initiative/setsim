import csv
import itertools
import math
import multiprocessing
import typing
import warnings

import hpotk
import numpy as np

from datetime import datetime
from statistics import mean
from tqdm import tqdm

from sumsim.model import Sample, Phenotyped, DiseaseModel
from sumsim.sim.phenomizer import TermPair


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
        self._hpo_version = hpo.version
        # As a side effect of getting the term and using its identifier,
        # we ensure `self._root` corresponds to an ID of current (non-obsolete) term for given HPO version.
        root_term = hpo.get_term(root)
        if root_term is None:
            raise ValueError(f'Root {root} is not in provided HPO!')
        self._root = root_term.identifier

        self._phenotypes = None
        self._phenotyped_terms = set()
        self._phenotyped_array = None
        self.ic_dict = None
        self._anc_dict = None  # used for calculating mica but could be calculated here

    def calculate_ic_from_samples(self, samples: typing.Sequence[Sample]) -> typing.Mapping[hpotk.TermId, float]:
        return self._calculate_ic_from_phenotyped(samples)

    def calculate_ic_from_diseases(self, diseases: typing.Sequence[DiseaseModel]):
        return self._calculate_ic_from_phenotyped(diseases)

    def _calculate_ic_from_phenotyped(self, phenotypes: typing.Sequence[Phenotyped]) \
            -> typing.Mapping[hpotk.TermId, float]:
        self._phenotypes = phenotypes
        all_terms_in_samples = set(pf for phenotyped in phenotypes for pf in phenotyped.phenotypic_features)
        self._phenotyped_terms = all_terms_in_samples.intersection(
            {i for i in self._hpo.get_descendants(self._root, include_source=True)})
        if len(all_terms_in_samples) != len(self._phenotyped_terms):
            warnings.warn("Your samples include terms that are not included as a Phenotypic abnormality (HP:0000118) "
                          "in your ontology! These terms will be ignored.")
        self._phenotyped_array = self._get_phenotyped_array(self._phenotypes, self._phenotyped_terms)
        # Define the number of processes to use
        num_processes = max(1, multiprocessing.cpu_count() - 2)  # Use all but 2 available CPU cores
        # Create a multiprocessing pool
        pool = multiprocessing.Pool(processes=num_processes)

        results = []
        for result in tqdm(pool.imap(self._get_term_ic, self._hpo.get_descendants(self._root, include_source=True),
                                     chunksize=100),
                           total=len(list(self._hpo.get_descendants(self._root, include_source=True)))):
            results.append(result)

        pool.close()
        self.ic_dict = {key: value for key, value in results}
        return self.ic_dict

    def _get_term_ic(self, term: hpotk.TermId) -> (hpotk.TermId, float):
        term_descendants = set(i for i in self._hpo.get_descendants(term, include_source=True))
        relevant_descendants = [i.value for i in term_descendants.intersection(self._phenotyped_terms)]
        if len(relevant_descendants) > 0:
            freq = sum(1 if any(row[relevant_descendants]) else 0 for row in self._phenotyped_array)
        else:
            freq = 1
        ic = math.log(len(self._phenotyped_array) / freq)
        return term, ic

    def create_mica_ic_dict(self, terms_in_samples: typing.Set[hpotk.TermId], ic_dict=None)\
            -> typing.Mapping[TermPair, float]:
        """

        @param terms_in_samples: This is the set of terms that are included in the sample to be analyzed. Only the
                                 terminal terms in each sample are necessary to include.
        @param ic_dict:
        @return:
        """
        used_terms = set(self._hpo.get_descendants(self._root, include_source=True))
        self._anc_dict = {term: used_terms.intersection(set(self._hpo.get_ancestors(term, include_source=True))) for
                          term in used_terms}

        if self.ic_dict is None:
            if ic_dict is None:
                raise ValueError("No IC dictionary was provided or exists in the class object.")
            else:
                self.ic_dict = ic_dict
        elif ic_dict != self.ic_dict and ic_dict is not None:
            raise ValueError("An IC dictionary was provided when there is already one in the class object.")

        # Use generator expression for term pairs
        used_terms_list = list(terms_in_samples.intersection(set(self._hpo.get_descendants(self._root, include_source=True))))
        term_pairs = itertools.combinations(used_terms_list, 2)
        total = len(used_terms_list) * (len(used_terms_list) - 1) // 2
        ic_list = self._create_mica_ic_list(term_pairs, total)

        # Combine the dictionaries into a single dictionary
        term_pairs = itertools.combinations(used_terms_list, 2)
        return {TermPair.of(term[0], term[1]): ic for term, ic in zip(term_pairs, ic_list)}

    def create_mica_ic_dict_file(self, file_path: str, ic_dict=None, hpoa_version: str = "N/A") -> None:
        used_terms = set(self._hpo.get_descendants(self._root, include_source=True))
        self._anc_dict = {term: used_terms.intersection(set(self._hpo.get_ancestors(term, include_source=True))) for
                          term in used_terms}

        if self.ic_dict is None:
            if ic_dict is None:
                raise ValueError("No IC dictionary was provided or exists in the class object.")
            else:
                self.ic_dict = ic_dict
        elif ic_dict != self.ic_dict and ic_dict is not None:
            raise ValueError("An IC dictionary was provided when there is already one in the class object.")

        # Use generator expression for term pairs
        term_pairs = itertools.combinations(self.ic_dict.keys(), 2)
        total = len(self.ic_dict) * (len(self.ic_dict) - 1) // 2
        ic_list = self._create_mica_ic_list(term_pairs, total)

        # Combine the dictionaries into a single dictionary
        term_pairs = itertools.combinations(self.ic_dict.keys(), 2)
        self._create_mica_dict_file(ic_list, term_pairs, file_path, hpoa_version, total)
        return None

    def _create_mica_ic_list(self, term_pairs, total) -> typing.Sequence[str]:
        # Define the number of processes to use
        num_processes = max(1, multiprocessing.cpu_count() - 2)  # Use all but 2 available CPU cores

        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use list comprehension with imap to get the results
            ic_list = [ic for ic in
                       tqdm(pool.imap(self._get_mica_ic, term_pairs, chunksize=10 ** 6), total=total,
                            desc="Calculating IC of MICA for term pairs")]
        return ic_list

    def _get_mica_ic(self, term_pair: typing.Sequence[hpotk.TermId]) -> float:
        shared_ancestors = self._anc_dict[term_pair[0]].intersection(self._anc_dict[term_pair[1]])
        mica_ic = max(self.ic_dict.get(ancestor, 0.0) for ancestor in shared_ancestors)
        return mica_ic

    def _create_mica_dict_file(self, ic_list, term_pairs, file_path: str, hpoa_version: str, total: int):
        # Get today's date
        today = datetime.now().strftime("%Y_%m_%d")

        # Define the text above the header
        header_text = "# Information content of the most informative common ancestor for term pairs\n" \
                      "# HPO=" + self._hpo_version + ";HPOA=" + hpoa_version + ";CREATED=" + today + "\n"

        # Write the data to the CSV file
        with open(file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")

            # Write the text above the header
            csv_file.write(header_text)

            # Write the header
            writer.writerow(["term_a", "term_b", "ic_mica"])

            # Write the term pairs and IC values with tqdm
            for term, ic in tqdm(zip(term_pairs, ic_list), total=total, desc="Writing to CSV"):
                writer.writerow([term[0], term[1], ic])

        print(f"CSV file '{file_path}' has been created.")
        return None

    @staticmethod
    def _get_phenotyped_array(phenotypes: typing.Sequence[Phenotyped],
                              used_pheno_abn: typing.Iterable[hpotk.TermId]) -> np.array:
        # Convert hpotk.TermID to string for array index
        array_type = [(col.value, bool) for col in used_pheno_abn]
        array = np.zeros(len(phenotypes), dtype=array_type)
        i = 0
        for sample in phenotypes:
            for term in sample.phenotypic_features:
                if term in used_pheno_abn:
                    array[term.value][i] = True
            i = i + 1
        return array

