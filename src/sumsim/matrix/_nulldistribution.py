import abc
import multiprocessing
import typing
import random
import warnings
from typing import Sequence

import hpotk
import numpy as np
from tqdm import tqdm

from sumsim.model import DiseaseModel, Phenotyped
from sumsim.model._base import FastPhenotyped
from sumsim.sim import SimilarityKernel, IcCalculator
from sumsim.sim._base import SimilaritiesKernel
from sumsim.sim import CountSimilaritiesKernel
from sumsim.sim import JaccardSimilaritiesKernel
from sumsim.sim import PhrankSimilaritiesKernel
from sumsim.sim import RoxasSimilaritiesKernel
from sumsim.sim import SimGciSimilaritiesKernel
from sumsim.sim import SimGicSimilaritiesKernel
from sumsim.sim import SimIciSimilaritiesKernel
from sumsim.sim.phenomizer import TermPair
from sumsim.sim.phenomizer._algo import PhenomizerSimilaritiesKernel


class PatientGenerator:
    """
    Generate patients with random phenotypes.
    """

    def __init__(self, hpo: hpotk.MinimalOntology, num_patients: int, num_features_per_patient: int,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118", ic_dict=None):
        self.root = root
        self.hpo = hpo
        self.num_patients = num_patients
        if num_features_per_patient < 1:
            raise ValueError("Number of features must be greater than 0.")
        self.num_features_per_patient = num_features_per_patient
        # If ic_dict is provided, remove most informative term from available terms because they are likely uncommon
        # for samples
        if ic_dict is not None:
            max_ic = max(ic_dict.values())
            self.available_terms = [key for key in ic_dict.keys() if ic_dict[key] < max_ic]
            if len(self.available_terms) < 1000:
                warnings.warn("Using ic_dict to avoid terms with max ic in null distribution patients requires that at "
                              "least 1,000 terms have an ic less than the max ic.")
                self.available_terms = list(self.hpo.graph.get_descendants(root, include_source=True))
        else:
            self.available_terms = list(self.hpo.graph.get_descendants(root, include_source=True))

    def generate(self):
        for patient_num in range(self.num_patients):
            yield self._generate_patient()

    def _generate_patient(self) -> Phenotyped:
        # Generate random phenotypic features for greatest number of features
        features = []
        while len(features) < self.num_features_per_patient:
            features = random.sample(self.available_terms, self.num_features_per_patient + 1)
            ancestors = set(ancestor for feature in features for ancestor in set(self.hpo.graph.get_ancestors(feature)))
            for feature in features:
                if feature in ancestors:
                    features.remove(feature)
        return FastPhenotyped(phenotypic_features=features[:self.num_features_per_patient])


class KernelIterator:
    def __init__(self, hpo: hpotk.MinimalOntology,
                 ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 bayes_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 mica_dict: typing.Mapping[TermPair, float] = None,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self.hpo = hpo
        self.root = root
        self.ic_dict = ic_dict
        self.delta_ic_dict = delta_ic_dict
        self.bayes_ic_dict = bayes_ic_dict
        self.mica_dict = mica_dict

    def _define_kernel(self, disease, method) \
            -> typing.Union[SimilaritiesKernel, SimilarityKernel]:
        if method in ["simici", "sumsim", "simgci", "roxas"]:
            if self.delta_ic_dict is None:
                raise ValueError("delta_ic_dict must be provided for sumsim method.")
            if method == "simici" or method == "sumsim":
                if method == "sumsim":
                    DeprecationWarning("The name of SumSim method has been changed to SimICI.")
                kernel = SimIciSimilaritiesKernel(disease, self.hpo, self.delta_ic_dict, self.root)
            elif method == "simgci":
                kernel = SimGciSimilaritiesKernel(disease, self.hpo, self.delta_ic_dict, self.root)
            elif method == "roxas":
                kernel = RoxasSimilaritiesKernel(disease, self.hpo, self.delta_ic_dict, self.root)
        elif method == "phrank":
            if self.bayes_ic_dict is None:
                raise ValueError("bayes_ic_dict must be provided for phrank method.")
            kernel = PhrankSimilaritiesKernel(disease, self.hpo, self.bayes_ic_dict, self.root)
        elif method == "simgic":
            if self.ic_dict is None:
                raise ValueError("ic_dict must be provided for simgic method.")
            kernel = SimGicSimilaritiesKernel(disease, self.hpo, self.ic_dict, self.root)
        elif method == "phenomizer":
            if self.mica_dict is None:
                if self.ic_dict is None:
                    raise ValueError("mica_dict or ic_dict must be provided for phenomizer method.")
                else:
                    # This allows for dynamic calculation of mica dictionary using one-sided method, resulting in
                    # smaller dictionary for multiprocessing.
                    calc = IcCalculator(hpo=self.hpo, root=self.root, multiprocess=False)
                    temp_mica_dict = calc.create_mica_ic_dict(samples=[disease], ic_dict=self.ic_dict,
                                                              one_sided=True, fragile_dict=True)
            else:
                temp_mica_dict = self._make_fragile_mica_ic(disease, self.mica_dict)
            kernel = PhenomizerSimilaritiesKernel(disease, temp_mica_dict, use_fragile_mica_dict=True)
        elif method == "jaccard" or method == "simui":
            kernel = JaccardSimilaritiesKernel(disease, self.hpo, self.root)
        elif method == "count":
            kernel = CountSimilaritiesKernel(disease, self.hpo, self.root)
        else:
            raise ValueError("Invalid method.")
        return kernel

    def _make_fragile_mica_ic(self, disease, mica_dict):
        features_under_root = set(self.hpo.graph.get_descendants(self.root, include_source=True))
        disease_features = set(int(feature.id) for feature in disease.phenotypic_features)
        # Create fragile dictionary that avoids creating TermPair objects by using integers and always putting integers
        # for the disease on the left of the tuple
        fragile_dict = {(d_f, int(a.id)): mica_dict.get(TermPair(d_f, int(a.id)), 0.0)
                        for a in features_under_root for d_f in disease_features}
        return fragile_dict


class GetNullDistribution(KernelIterator, metaclass=abc.ABCMeta):

    def __init__(self, disease: DiseaseModel, hpo: hpotk.MinimalOntology, num_patients: int,
                 num_features_per_patient: int,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118",
                 kernel: SimilarityKernel = None,
                 method: str = None, mica_dict: typing.Mapping[TermPair, float] = None,
                 ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 bayes_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 precomputed_patients: Sequence[Phenotyped] = None):
        KernelIterator.__init__(self, hpo, ic_dict, delta_ic_dict, bayes_ic_dict, mica_dict, root)
        self.disease = disease
        self.method = method
        self.num_patients = num_patients
        self.num_features_per_patient = num_features_per_patient
        self.kernel = kernel
        self.precomputed_patients = precomputed_patients
        self.patient_similarity_array = self._get_null_distribution()

    def get_patient_similarity_array(self):
        return self.patient_similarity_array

    def _get_null_distribution(self, ):
        """Get null distribution.

        Returns
        -------
        null_distribution : array-like
            Null distribution.
        """

        # Define similarity kernel
        if self.kernel is not None:
            kernel = self.kernel
        else:
            kernel = self._define_kernel(self.disease, self.method)
        if self.precomputed_patients is None:
            p_gen = PatientGenerator(self.hpo, self.num_patients, self.num_features_per_patient, self.root)
            patient_similarity_array = np.array([kernel.compute(patient) for patient in p_gen.generate()])
        else:
            patient_similarity_array = np.array([kernel.compute(patient) for patient in self.precomputed_patients])
        return patient_similarity_array

    def get_pval(self, similarity: float, num_features: int):
        """Get p-value.

        Parameters
        ----------
        @param similarity : float
            Observed similarity.
        @param num_features : int

        Returns
        -------
        pval : float
            p-value.
        """
        if num_features <= self.num_features_per_patient:
            col = num_features - 1
        elif num_features < 1:
            raise ValueError("Samples must have at least 1 phenotypic feature.")
        else:
            col = self.num_features_per_patient - 1
        pval = (self.patient_similarity_array[:, col] >= similarity).sum() / len(self.patient_similarity_array)
        return pval
