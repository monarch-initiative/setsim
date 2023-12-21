import multiprocessing
import typing
import random
from typing import Sequence

import hpotk
import numpy as np
from tqdm import tqdm

from sumsim.model import Sample, DiseaseModel, Phenotyped
from sumsim.sim import SumSimSimilarityKernel, SimilarityKernel
from sumsim.sim.phenomizer import OneSidedSemiPhenomizer, PrecomputedIcMicaSimilarityMeasure, TermPair


class PatientGenerator:
    """
    Generate patients with random phenotypes.
    """

    def __init__(self, hpo: hpotk.MinimalOntology, num_patients: int, num_features_per_patient: Sequence[int],
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self.root = root
        self.hpo = hpo.graph
        self.num_patients = num_patients
        if min(num_features_per_patient) < 1:
            raise ValueError("Number of features must be greater than 0.")
        self.num_features_per_patient = num_features_per_patient
        self.max_features = max(num_features_per_patient)
        self.terms_under_root = list(self.hpo.get_descendants(root, include_source=True))

    def generate(self):
        for patient_num in range(self.num_patients):
            yield self._generate_patient(patient_num)

    def _generate_patient(self, patient_num: int) -> Sequence[Sample]:
        # Generate random phenotypic features for greatest number of features
        phenotypic_features = []
        while len(phenotypic_features) < self.max_features:
            feature = random.choice(self.terms_under_root)
            feature_ancestors_in_list = set(phenotypic_features) & set(self.hpo.get_ancestors(feature,
                                                                                              include_source=True))
            # Replace more general ancestor with more specific feature
            if bool(feature_ancestors_in_list):
                for feature_ancestor in feature_ancestors_in_list:
                    phenotypic_features.remove(feature_ancestor)
            phenotypic_features.append(feature)
        samples = []
        # Create samples with for each number of phenotypic features
        for num_features in self.num_features_per_patient:
            if num_features < self.max_features:
                samples.append(Sample(label=f"Patient_{patient_num}_Phe{num_features}",
                                      phenotypic_features=phenotypic_features[:num_features]))
            else:
                samples.append(Sample(label=f"Patient_{patient_num}_Phe{self.max_features}",
                                      phenotypic_features=phenotypic_features))
        return samples


class GetNullDistribution:

    def __init__(self, disease: DiseaseModel, method: str, hpo: hpotk.MinimalOntology, num_patients: int,
                 num_features_per_patient: Sequence[int], mic_dict: typing.Mapping[TermPair, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self.disease = disease
        self.method = method
        self.hpo = hpo
        self.num_patients = num_patients
        self.num_features_per_patient = num_features_per_patient
        self.mica_dict = mic_dict
        self.delta_ic_dict = delta_ic_dict
        self.root = root
        self.column_names = [str(col) for col in self.num_features_per_patient]
        self.patient_similarity_array = self._get_null_distribution()

    def _get_null_distribution(self):
        """Get null distribution.

        Returns
        -------
        null_distribution : array-like
            Null distribution.
        """

        # Define similarity kernel
        if self.method == "sumsim":
            if self.delta_ic_dict is None:
                raise ValueError("delta_ic_dict must be provided for sumsim method.")
            kernel = SumSimSimilarityKernel(self.hpo, self.delta_ic_dict, self.root)
        elif self.method == "phenomizer":
            if self.mica_dict is None:
                raise ValueError("mica_dict must be provided for phenomizer method.")
            kernel = OneSidedSemiPhenomizer(PrecomputedIcMicaSimilarityMeasure(self.mica_dict))
        else:
            raise ValueError("Invalid method.")
        array_type = [(col, float) for col in self.column_names]
        p_gen = PatientGenerator(self.hpo, self.num_patients, self.num_features_per_patient, self.root)
        kernel_wrapper = SimilarityWrapper(kernel, self.disease)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
            similarities = [similarity_list for similarity_list in
                            tqdm(
                                pool.imap(kernel_wrapper.compute, p_gen.generate(), chunksize=100),
                                total=self.num_patients)
                            ]
        patient_similarity_array = np.array(similarities, dtype=array_type)
        for col_name in self.column_names:
            col_values = patient_similarity_array[col_name]
            col_values.sort()
            patient_similarity_array[col_name] = col_values
        return patient_similarity_array

    def get_pval(self, ic: float, num_features: int):
        """Get p-value.

        Parameters
        ----------
        @param ic : float
            Observed similarity.
        @param num_features : int

        Returns
        -------
        pval : float
            p-value.
        """
        if num_features in self.num_features_per_patient:
            col_name = str(num_features)
        elif num_features < 1:
            raise ValueError("Samples must have at least 1 phenotypic feature.")
        else:
            col_name = str(min(self.num_features_per_patient, key=lambda x: abs(x - num_features)))
        pval = (self.patient_similarity_array[col_name] >= ic).sum() / len(self.patient_similarity_array)
        return pval


class SimilarityWrapper:
    def __init__(self, kernel: SimilarityKernel, sample: Phenotyped):
        self.sample = sample
        self.kernel = kernel

    def compute(self, pts: Sequence[Sample]) -> Sequence[float]:
        return tuple(self.kernel.compute(pt, self.sample).similarity for pt in pts)
