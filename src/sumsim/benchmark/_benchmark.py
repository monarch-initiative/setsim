import multiprocessing
from warnings import filterwarnings

import pandas as pd
import typing
from typing import Sequence

import hpotk
from tqdm import tqdm

from sumsim.model import DiseaseModel, Sample
from sumsim.sim.phenomizer import TermPair, OneSidedSemiPhenomizer, PrecomputedIcMicaSimilarityMeasure
from ._nulldistribution import GetNullDistribution
from sumsim.sim import SumSimSimilarityKernel, IcCalculator, JaccardSimilarityKernel


class Benchmark:
    def __init__(self, hpo: hpotk.MinimalOntology, patients: Sequence[Sample], n_iter_distribution: int,
                 num_features_distribution: Sequence[int], mica_dict: typing.Mapping[TermPair, float] = None,
                 ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118",
                 num_cpus: int = None,
                 chunksize: int = 100, verbose: bool = False):
        self.hpo = hpo
        self.patients = patients
        self.n_iter_distribution = n_iter_distribution
        self.num_features_distribution = num_features_distribution
        self.mica_dict = mica_dict
        self.ic_dict = ic_dict
        self.delta_ic_dict = delta_ic_dict
        self.root = root
        self.chunksize = chunksize
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

    def compute_ranks(self, similarity_methods: Sequence[str], diseases: Sequence[DiseaseModel]):
        print(f"There are {multiprocessing.cpu_count()} CPUs available for multiprocessing. Using {self.num_cpus} CPUs.")
        num_diseases = len(diseases)
        if num_diseases < 6:
            for disease in diseases:
                for method in similarity_methods:
                    self._rank_patients(method, disease, True)
        else:
            for disease in tqdm(diseases):
                for method in similarity_methods:
                    self._rank_patients(method, disease, False)

        return self.patient_table

    def _rank_patients(self, similarity_method: str, disease: DiseaseModel, progress_bar: bool):
        # Define similarity kernel
        if similarity_method == "sumsim":
            if self.delta_ic_dict is None:
                raise ValueError("delta_ic_dict must be provided for sumsim method.")
            kernel = SumSimSimilarityKernel(self.hpo, self.delta_ic_dict, self.root)
        elif similarity_method == "phenomizer":
            if self.mica_dict is None:
                if self.ic_dict is None:
                    raise ValueError("mica_dict or ic_dict must be provided for phenomizer method.")
                else:
                    # This allows for dynamic calculation of mica dictionary using one-sided method, resulting in
                    # smaller dictionary for multiprocessing.
                    calc = IcCalculator(hpo=self.hpo, root=self.root, num_processes=self.num_cpus)
                    # Avoid assigning to self.mica_dict so that mica_dict is always calculated for each disease
                    mica_dict = calc.create_mica_ic_dict(samples=[disease], ic_dict=self.ic_dict, one_sided=True)
            else:
                mica_dict = self.mica_dict
            kernel = OneSidedSemiPhenomizer(PrecomputedIcMicaSimilarityMeasure(mica_dict))
        elif similarity_method == "jaccard":
            kernel = JaccardSimilarityKernel(self.hpo, self.root)
        else:
            raise ValueError("Invalid method.")

        # Get Column Names
        if not disease.identifier.value:
            label = disease.label
        else:
            label = disease.identifier.value.replace(':', '_')
        sim = f'{label}_{similarity_method}_sim'
        pval = f'{label}_{similarity_method}_pval'

        # Calculate similarity of each patient to the disease
        self.patient_table[sim] = [kernel.compute(patient, disease).similarity for patient in self.patients]
        dist_method = GetNullDistribution(disease, hpo=self.hpo, num_patients=self.n_iter_distribution,
                                          num_features_per_patient=self.num_features_distribution, root=self.root,
                                          kernel=kernel, chunksize=self.chunksize, num_cpus=self.num_cpus,
                                          progress_bar=progress_bar)
        self.patient_table[pval] = [dist_method.get_pval(similarity, len(patient.phenotypic_features))
                                    for similarity, patient in zip(self.patient_table[sim], self.patients)]
        pass
