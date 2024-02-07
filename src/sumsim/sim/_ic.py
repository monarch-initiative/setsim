import csv
import itertools
import math
import multiprocessing
import numpy as np
import typing
import warnings

from datetime import datetime
from statistics import mean

import pandas as pd
from tqdm import tqdm

import hpotk

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
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118", samples: typing.Sequence[Phenotyped] = None):
        self._hpo = hpotk.util.validate_instance(hpo, hpotk.MinimalOntology, 'hpo')
        # As a side effect of getting the term and using its identifier,
        # we ensure `self._root` corresponds to an ID of current (non-obsolete) term for given HPO version.
        root_term = hpo.get_term(root)
        if root_term is None:
            raise ValueError(f'Root {root} is not in provided HPO!')
        self._root = root_term.identifier
        if samples is not None:
            self._samples = [sample for sample in samples if len(sample.phenotypic_features) > 0]
        else:
            self._samples = None

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
        elif strategy == 'bayesian':
            return self._use_bayesian(ic_dict)
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

    def _use_bayesian(self, ic_dict: typing.Mapping[hpotk.TermId, float]):
        if self._samples is None:
            raise ValueError("Bayesian calculation requires samples or diseases used for IC dictionary creation.")
        phenotyped_array = get_phenotyped_array(self._samples, self._hpo.graph)
        delta_ic_dict = {}
        for term_id, term_ic in ic_dict.items():
            if term_id != self._root:
                parents = list(self._hpo.graph.get_parents(term_id))
                if len(parents) > 1:
                    relevant_parents = [i.value for i in set(parents).intersection(ic_dict.keys())]
                    parent_freq = sum(1 if any(row[relevant_parents]) else 0 for row in phenotyped_array)
                    term_freq = sum(row[term_id.value] for row in phenotyped_array)
                    if term_freq == 0:
                        term_freq = 1
                    if parent_freq == 0:
                        parent_freq = 1
                    delta_ic_dict[term_id] = math.log(parent_freq / term_freq)
                else:
                    delta_ic_dict[term_id] = term_ic - ic_dict.get(parents[0], 0)
            else:
                delta_ic_dict[term_id] = term_ic
        return delta_ic_dict


def import_mica_ic_dict(file_path: str) -> typing.Mapping[TermPair, float]:
    """
    Import a dictionary that goes from TermPairs to MICA IC to be used for a specific instance of phenomizer.

    @param file_path: Path and name for the MICA IC dictionary to be imported.
    @return: Return a dictionary of TermPairs to MICA IC of that pair.
    """
    # Read the CSV file
    with open(file_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        next(reader)
        next(reader)
        header = next(reader)
        if header[0] != "term_a" or header[1] != "term_b" or header[2] != "ic_mica":
            raise ValueError("The header of the CSV file does not match the expected format.")

        # Create the dictionary
        mica_ic_dict = {TermPair.of(row[0], row[1]): float(row[2]) for row in reader}
    return mica_ic_dict


def import_one_sided_mica_ic_dict(file_path: str, disease: DiseaseModel) -> typing.Mapping[TermPair, float]:
    """
    Import a dictionary that goes from TermPairs to MICA IC to be used for a specific instance of phenomizer.

    @param disease: The disease that will be matched to. Only pairs that include one of the features in this disease
    will be included in the mica dictionary.
    @param file_path: Path and name for the MICA IC dictionary to be imported.
    @return: Return a dictionary of TermPairs to MICA IC of that pair.
    """
    # Read the CSV file
    with open(file_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        next(reader)
        next(reader)
        header = next(reader)
        if header[0] != "term_a" or header[1] != "term_b" or header[2] != "ic_mica":
            raise ValueError("The header of the CSV file does not match the expected format.")

        # Create the dictionary
        mica_ic_dict = {TermPair.of(row[0], row[1]): float(row[2]) for row in reader
                        if (hpotk.TermId.from_curie(row[0]) in disease.phenotypic_features or
                            hpotk.TermId.from_curie(row[1]) in disease.phenotypic_features)}
    return mica_ic_dict


class IcCalculator:
    """
    Create a dictionary providing the information content of terms.
    """

    def __init__(self, hpo: hpotk.MinimalOntology,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118",
                 multiprocess: bool = True,
                 num_processes: int = None,
                 progress_bar: bool = False):
        self._hpo = hpo.graph
        self._hpo_version = hpo.version
        # As a side effect of getting the term and using its identifier,
        # we ensure `self._root` corresponds to an ID of current (non-obsolete) term for given HPO version.
        root_term = hpo.get_term(root)
        if root_term is None:
            raise ValueError(f'Root {root} is not in provided HPO!')
        self._root = root_term.identifier
        self._features_under_root = set(self._hpo.get_descendants(self._root, include_source=True))

        self._phenotypes = None
        self._phenotyped_terms = set()
        self._phenotyped_array = None
        self.ic_dict = None
        self._anc_dict = None  # used for calculating mica_ic_dict
        self.multiprocess = multiprocess
        if num_processes is None:
            num_processes = max(multiprocessing.cpu_count() - 2, 1)
        self._num_processes = num_processes
        self._progress_bar = progress_bar

    def calculate_ic_from_samples(self, samples: typing.Sequence[Sample]) -> typing.Mapping[hpotk.TermId, float]:
        return self._calculate_ic_from_phenotyped(samples)

    def calculate_ic_from_diseases(self, diseases: typing.Sequence[DiseaseModel]):
        return self._calculate_ic_from_phenotyped(diseases)

    def _calculate_ic_from_phenotyped(self, phenotypes: typing.Sequence[Phenotyped]) \
            -> typing.Mapping[hpotk.TermId, float]:

        all_terms_in_samples = set(pf for phenotyped in phenotypes for pf in phenotyped.phenotypic_features)
        self._phenotyped_terms = all_terms_in_samples.intersection(self._hpo.get_descendants(self._root,
                                                                                             include_source=True))
        if len(all_terms_in_samples) > len(self._phenotyped_terms):
            excluded_features = [term for term in all_terms_in_samples if term not in self._phenotyped_terms]
            warnings.warn(f"The terms {excluded_features} are not included under the chosen root ({self._root})"
                          " in your ontology! These terms will be ignored.")
        self._phenotypes = [phenotype for phenotype in phenotypes if self._phenotyped_terms.intersection(
            phenotype.phenotypic_features) != set()]
        if len(phenotypes) > len(self._phenotypes):
            excluded_phenotypes = [phenotype for phenotype in phenotypes if phenotype not in self._phenotypes]
            warnings.warn(f"The sample(s) {excluded_phenotypes} were removed because they have no features.")

        phenotyped_array = pd.DataFrame(columns=range(len(self._phenotypes)),
                                        index=[term.value for term in self._features_under_root], dtype=bool)
        phenotyped_array.iloc[:, :] = False
        i = 0
        for sample in phenotypes:
            full_term_list = [term.value for term in
                              (set(anc for term in sample.phenotypic_features
                                   for anc in
                                   self._hpo.get_ancestors(term, include_source=True)) & self._features_under_root)
                              ]
            phenotyped_array.loc[full_term_list, i] = True
            i += 1
        num_samples = len(self._phenotypes)
        ic_values = phenotyped_array.sum(axis=1).apply(lambda x: math.log(num_samples / max(1, x))).tolist()
        self.ic_dict = dict(zip(self._features_under_root, ic_values))
        return self.ic_dict

    def _get_term_ic(self, term: hpotk.TermId) -> (hpotk.TermId, float):
        freq = max(1, self._phenotyped_array[:][term.value].sum())
        ic = math.log(len(self._phenotyped_array) / freq)
        return term, ic

    def create_mica_ic_dict(self, terms_in_samples: typing.Set[hpotk.TermId] = None,
                            samples: typing.Sequence[Phenotyped] = None, ic_dict=None, one_sided: bool = False,
                            fragile_dict=False) \
            -> typing.Union[typing.Mapping[TermPair, float], typing.Mapping[typing.Tuple[int, int], float]]:
        """
        Create a dictionary that goes from TermPairs to MICA IC to be used for a specific instance of phenomizer. The
        dictionary requires only the terms that are annotated in the samples being analyzed. (It is not necessary to
        include the parents of terms that are themselves not explicitly annotated in a sample.)

        @param one_sided: If True, only pairs that include one of the features in the disease will be included in the
        dictionary.
        @param terms_in_samples: This is the set of terms that are included in the sample to be analyzed. Only the
        terminal terms in each sample are necessary to include.
        @param samples: Allows the user to supply a list of Phenotyped sample/diseases that are being analyzed as an
        alternative to providing the set of annotated terms.
        @param ic_dict: A dictionary that goes from hpotk.TermId's to their respective IC's. Not needed for class
        instances that already have an ic_dict stored.
        @return: Return a dictionary of TermPairs to MICA IC of that pair.
        """
        if fragile_dict and not one_sided:
            raise ValueError("Fragile dictionary can only be used with one_sided=True.")

        used_terms = self._features_under_root
        if terms_in_samples is None and samples is None:
            terms_in_samples = used_terms
        self._anc_dict = {term: used_terms.intersection(self._hpo.get_ancestors(term, include_source=True)) for term in
                          used_terms}
        if self.ic_dict is None:
            if ic_dict is None:
                raise ValueError("No IC dictionary was provided or exists in the class object.")
            else:
                self.ic_dict = ic_dict
        elif ic_dict != self.ic_dict and ic_dict is not None:
            raise ValueError("An IC dictionary was provided when there is already one in the class object.")

        # Use generator expression for term pairs
        if terms_in_samples is None:
            terms_in_samples = set(feature for sample in samples for feature in sample.phenotypic_features)
        used_terms_list = list(terms_in_samples.intersection(self._hpo.get_descendants(self._root,
                                                                                       include_source=True)))
        if one_sided:
            term_pairs = itertools.product(used_terms_list, self._features_under_root)
            term_pairs_2 = itertools.product(used_terms_list, self._features_under_root)
            total = len(used_terms_list) * len(self._features_under_root)
        else:
            term_pairs = itertools.combinations(used_terms_list, 2)
            term_pairs_2 = itertools.combinations(used_terms_list, 2)
            total = len(used_terms_list) * (len(used_terms_list) - 1) // 2
        ic_list = self._create_mica_ic_list(term_pairs, total)
        if fragile_dict:
            # Create matched set
            matched_dict = {(int(term.id), int(term.id)): self.ic_dict[term] for term in used_terms_list}
            # Combine the dictionaries into a single dictionary
            mica_dict = {(int(term[0].id), int(term[1].id)): ic for term, ic in zip(term_pairs_2, ic_list) if ic > 0}
            return {**matched_dict, **mica_dict}
        else:
            # Create matched set
            matched_dict = {TermPair.of(term, term): self.ic_dict[term] for term in used_terms_list}
            # Combine the dictionaries into a single dictionary
            mica_dict = {TermPair.of(term[0], term[1]): ic for term, ic in zip(term_pairs_2, ic_list) if ic > 0}
        return {**matched_dict, **mica_dict}

    def create_mica_ic_dict_file(self, file_path: str, ic_dict=None, hpoa_version: str = "N/A") -> None:
        """
        Create a dictionary that goes from TermPairs to MICA IC to be used for all possible pairs of terms under the
        chosen root under the chosen ontology.

        @param file_path: Path and name for the MICA IC dictionary to be saved.
        @param ic_dict: A dictionary that goes from hpotk.TermId's to their respective IC's. Not needed for class
        instances that already have an ic_dict stored.
        @param hpoa_version:
        @return: Return a dictionary of TermPairs to MICA IC of that pair.
        """
        used_terms = self._features_under_root
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
        if self.multiprocess:
            # Create a multiprocessing pool
            with multiprocessing.Pool(processes=self._num_processes) as pool:
                if self._progress_bar:
                    # Use list comprehension with imap to get the results
                    ic_list = [ic for ic in
                               tqdm(pool.imap(self._get_mica_ic, term_pairs, chunksize=10 ** 6), total=total,
                                    desc="Calculating IC of MICA for term pairs")]
                else:
                    # Use list comprehension with imap to get the results
                    ic_list = [ic for ic in pool.imap(self._get_mica_ic, term_pairs, chunksize=10 ** 6)]
        else:
            ic_list = [self._get_mica_ic(term_pair) for term_pair in term_pairs]
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

            # Write matched set
            [writer.writerow([term, term, ic]) for term, ic in self.ic_dict.items()]

            # Write the term pairs and IC values with tqdm
            for term, ic in tqdm(zip(term_pairs, ic_list), total=total, desc="Writing to CSV"):
                if ic > 0:
                    writer.writerow([term[0], term[1], ic])

        print(f"CSV file '{file_path}' has been created.")
        return None


def get_phenotyped_array(phenotypes: typing.Sequence[Phenotyped],
                         hpo: hpotk.MinimalOntology.graph) -> np.array:
    # Convert hpotk.TermID to string for array index
    used_pheno_abn = set(hpo.get_descendants("HP:0000118", include_source=True))
    array_type = [(col.value, bool) for col in used_pheno_abn]
    array = np.zeros(len(phenotypes), dtype=array_type)
    i = 0
    for sample in phenotypes:
        for term in sample.phenotypic_features:
            for ancestor in hpo.get_ancestors(term, include_source=True):
                if ancestor in used_pheno_abn:
                    array[ancestor.value][i] = True
        i = i + 1
    return array
