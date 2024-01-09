import abc
import multiprocessing
import typing
import random
from typing import Sequence

import hpotk
import numpy as np
from tqdm import tqdm

from sumsim.model import Sample, DiseaseModel, Phenotyped
from sumsim.model._base import FastPhenotyped
from sumsim.sim import SumSimSimilarityKernel, SimilarityKernel, IcCalculator, JaccardSimilarityKernel
from sumsim.sim._base import SimilaritiesKernel
from sumsim.sim._jaccard import JaccardSimilaritiesKernel
from sumsim.sim._sumsim import SumSimSimilaritiesKernel
from sumsim.sim.phenomizer import OneSidedSemiPhenomizer, PrecomputedIcMicaSimilarityMeasure, TermPair
from sumsim.sim.phenomizer._algo import PhenomizerSimilaritiesKernel


class PatientGenerator:
    """
    Generate patients with random phenotypes.
    """

    def __init__(self, hpo: hpotk.MinimalOntology, num_patients: int, num_features_per_patient: int,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self.root = root
        self.hpo = hpo
        self.num_patients = num_patients
        if num_features_per_patient < 1:
            raise ValueError("Number of features must be greater than 0.")
        self.num_features_per_patient = num_features_per_patient
        self.terms_under_root = list(self.hpo.graph.get_descendants(root, include_source=True))

    def generate(self):
        for patient_num in range(self.num_patients):
            yield self._generate_patient()

    def _generate_patient(self) -> Phenotyped:
        # Generate random phenotypic features for greatest number of features
        features = []
        while len(features) < self.num_features_per_patient:
            features = random.sample(self.terms_under_root, self.num_features_per_patient + 1)
            ancestors = set(ancestor for feature in features for ancestor in set(self.hpo.graph.get_ancestors(feature)))
            for feature in features:
                if feature in ancestors:
                    features.remove(feature)
        return FastPhenotyped(phenotypic_features=features[:self.num_features_per_patient])


class KernelIterator:
    def __init__(self, hpo: hpotk.MinimalOntology,
                 ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 mica_dict: typing.Mapping[TermPair, float] = None,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self.hpo = hpo
        self.root = root
        self.ic_dict = ic_dict
        self.delta_ic_dict = delta_ic_dict
        self.mica_dict = mica_dict

    def _define_kernel(self, disease, method) \
            -> typing.Union[SimilaritiesKernel, SimilarityKernel]:
        if method == "sumsim":
            if self.delta_ic_dict is None:
                raise ValueError("delta_ic_dict must be provided for sumsim method.")
            kernel = SumSimSimilaritiesKernel(disease, self.hpo, self.delta_ic_dict, self.root)
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
        elif method == "jaccard":
            kernel = JaccardSimilaritiesKernel(disease, self.hpo, self.root)
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
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 precomputed_patients: Sequence[Phenotyped] = None):
        KernelIterator.__init__(self, hpo, ic_dict, delta_ic_dict, mica_dict, root)
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
