import abc
import multiprocessing
import warnings
from warnings import filterwarnings

import pandas as pd
import typing
from typing import Sequence

import hpotk
from tqdm import tqdm

from sumsim.model import DiseaseModel, Sample, Phenotyped
from sumsim.sim.phenomizer import TermPair, OneSidedSemiPhenomizer, PrecomputedIcMicaSimilarityMeasure
from ._nulldistribution import GetNullDistribution, KernelIterator, PatientGenerator
from sumsim.sim import SimIciSimilarityKernel, IcCalculator, JaccardSimilarityKernel


class SimilarityMatrix(KernelIterator, metaclass=abc.ABCMeta):
    def __init__(self, hpo: hpotk.MinimalOntology, patients: Sequence[Sample], n_iter_distribution: int,
                 num_features_distribution: int, similarity_methods: Sequence[str],
                 mica_dict: typing.Mapping[TermPair, float] = None,
                 ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 bayes_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118",
                 multiprocess: bool = True,
                 num_cpus: int = None,
                 chunksize: int = 1, verbose: bool = True,
                 avoid_max_ic_for_null_patients: bool = False):
        KernelIterator.__init__(self, hpo=hpo, mica_dict=mica_dict, ic_dict=ic_dict, bayes_ic_dict=bayes_ic_dict,
                                delta_ic_dict=delta_ic_dict, root=root)
        self.patients = patients
        self.n_iter_distribution = n_iter_distribution
        self.num_features_distribution = num_features_distribution
        self.similarity_methods = similarity_methods
        self.chunksize = chunksize
        self.multiprocess = multiprocess
        if num_cpus is None:
            num_cpus = max(multiprocessing.cpu_count() - 2, 1)
        self.num_cpus = num_cpus
        data = [(None, len(patient.phenotypic_features)) if patient.disease_identifier is None
                else (patient.disease_identifier.identifier.value, len(patient.phenotypic_features))
                for patient in self.patients]
        self.patient_table = pd.DataFrame(index=[patient.label for patient in self.patients],
                                          columns=['disease_id', 'num_features'],
                                          data=data)
        if avoid_max_ic_for_null_patients:
            p_gen = PatientGenerator(self.hpo, self.n_iter_distribution, self.num_features_distribution, self.root,
                                     ic_dict=self.ic_dict)
        else:
            p_gen = PatientGenerator(self.hpo, self.n_iter_distribution, self.num_features_distribution, self.root)
        self.precomputed_patients = [patient for patient in p_gen.generate()]
        if not verbose:
            filterwarnings("ignore")

    def compute_diagnostic_similarities(self, diseases: Sequence[DiseaseModel]):
        contains_disease_zero_features = False
        for disease in diseases:
            if len(disease.phenotypic_features) < 1:
                contains_disease_zero_features = True
                warnings.warn(f"Disease {disease.label} has no phenotypic features.")
        if contains_disease_zero_features:
            diseases = [disease for disease in diseases if len(disease.phenotypic_features) > 0]
        return self._compute_similarities(diseases)

    def compute_person2person_similarities(self, samples: Sequence[Sample]):
        contains_disease_zero_features = False
        for sample in samples:
            if len(sample.phenotypic_features) < 1:
                contains_disease_zero_features = True
                warnings.warn(f"Sample {sample.label} has no phenotypic features.")
        if contains_disease_zero_features:
            samples = [sample for sample in samples if len(sample.phenotypic_features) > 0]
        return self._compute_similarities(samples)

    def _compute_similarities(self, samples: Sequence[typing.Union[DiseaseModel, Sample]]):
        print(
            f"There are {multiprocessing.cpu_count()} CPUs available for multiprocessing. Using {self.num_cpus} CPUs.")
        patient_dict = {}
        if self.multiprocess:
            with multiprocessing.Pool(processes=self.num_cpus) as pool:
                disease_dicts = pool.imap_unordered(self._compute_across_methods, samples, chunksize=self.chunksize)
                for result in tqdm(disease_dicts, total=len(samples), desc="Diseases"):
                    patient_dict = {**patient_dict, **result}
        else:
            for disease in tqdm(samples, total=len(samples), desc="Diseases"):
                patient_dict = {**patient_dict, **self._compute_across_methods(disease)}
        self.patient_table = pd.concat([self.patient_table, pd.DataFrame(patient_dict, index=self.patient_table.index)],
                                       axis=1)
        return self.patient_table

    def _compute_across_methods(self, sample: typing.Union[DiseaseModel, Sample])\
            -> typing.Mapping[str, typing.List[float]]:
        patient_dict = {}
        for method in self.similarity_methods:
            patient_dict = {**patient_dict, **self._compute_across_patients(method, sample)}
        return patient_dict

    def _compute_across_patients(self, similarity_method: str, sample: typing.Union[DiseaseModel, Sample]):
        kernel = self._define_kernel(sample, similarity_method)
        # Get Column Names
        if type(sample) is DiseaseModel:
            if not sample.identifier.value:
                label = sample.label
            else:
                label = sample.identifier.value.replace(':', '_')
        elif type(sample) is Sample:
            label = sample.label
        else:
            raise TypeError("Sample must be either DiseaseModel or Sample.")
        sim = f'{label}_{similarity_method}_sim'
        pval = f'{label}_{similarity_method}_pval'
        # Calculate similarity of each patient to the disease
        patient_dict = {sim: [kernel.compute(patient, return_last_result=True) for patient in self.patients]}
        if self.n_iter_distribution > 0:
            dist_method = GetNullDistribution(sample, hpo=self.hpo, num_patients=self.n_iter_distribution,
                                              num_features_per_patient=self.num_features_distribution, root=self.root,
                                              kernel=kernel, precomputed_patients=self.precomputed_patients)
            patient_dict[pval] = [dist_method.get_pval(similarity, len(patient.phenotypic_features))
                                  for similarity, patient in zip(patient_dict[sim], self.patients)]
        return patient_dict
