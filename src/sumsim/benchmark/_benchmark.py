import abc
import multiprocessing
from warnings import filterwarnings

import pandas as pd
import typing
from typing import Sequence

import hpotk
from tqdm import tqdm

from sumsim.model import DiseaseModel, Sample
from sumsim.sim.phenomizer import TermPair, OneSidedSemiPhenomizer, PrecomputedIcMicaSimilarityMeasure
from ._nulldistribution import GetNullDistribution, KernelIterator
from sumsim.sim import SumSimSimilarityKernel, IcCalculator, JaccardSimilarityKernel


class Benchmark(KernelIterator, metaclass=abc.ABCMeta):
    def __init__(self, hpo: hpotk.MinimalOntology, patients: Sequence[Sample], n_iter_distribution: int,
                 num_features_distribution: int, similarity_methods: Sequence[str],
                 mica_dict: typing.Mapping[TermPair, float] = None,
                 ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118",
                 multiprocess: bool = True,
                 num_cpus: int = None,
                 chunksize: int = 1, verbose: bool = False):
        KernelIterator.__init__(self, hpo=hpo, mica_dict=mica_dict, ic_dict=ic_dict, delta_ic_dict=delta_ic_dict,
                                root=root)
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
        if not verbose:
            filterwarnings("ignore")

    def compute_ranks(self, diseases: Sequence[DiseaseModel]):
        print(
            f"There are {multiprocessing.cpu_count()} CPUs available for multiprocessing. Using {self.num_cpus} CPUs.")
        patient_dict = {}
        if self.multiprocess:
            with multiprocessing.Pool(processes=self.num_cpus) as pool:
                disease_dicts = pool.imap_unordered(self._rank_across_methods, diseases, chunksize=self.chunksize)
                for result in tqdm(disease_dicts, total=len(diseases), desc="Diseases"):
                    patient_dict = {**patient_dict, **result}
        else:
            for disease in tqdm(diseases, total=len(diseases), desc="Diseases"):
                patient_dict = {**patient_dict, **self._rank_across_methods(disease)}
        self.patient_table = pd.concat([self.patient_table, pd.DataFrame(patient_dict, index=self.patient_table.index)],
                                       axis=1)
        return self.patient_table

    def _rank_across_methods(self, disease: DiseaseModel) -> typing.Mapping[str, typing.List[float]]:
        patient_dict = {}
        for method in self.similarity_methods:
            patient_dict = {**patient_dict, **self._rank_patients(method, disease)}
        return patient_dict

    def _rank_patients(self, similarity_method: str, disease: DiseaseModel):
        multiple_features_kernel = self._define_kernel(disease, similarity_method)
        single_kernel = self._define_kernel(disease, similarity_method, single_feature_version=True)
        # Get Column Names
        if not disease.identifier.value:
            label = disease.label
        else:
            label = disease.identifier.value.replace(':', '_')
        sim = f'{label}_{similarity_method}_sim'
        pval = f'{label}_{similarity_method}_pval'

        # Calculate similarity of each patient to the disease
        patient_dict = {sim: [single_kernel.compute(patient, disease).similarity for patient in self.patients]}
        dist_method = GetNullDistribution(disease, hpo=self.hpo, num_patients=self.n_iter_distribution,
                                          num_features_per_patient=self.num_features_distribution, root=self.root,
                                          kernel=multiple_features_kernel)
        patient_dict[pval] = [dist_method.get_pval(similarity, len(patient.phenotypic_features))
                              for similarity, patient in zip(patient_dict[sim], self.patients)]
        return patient_dict
