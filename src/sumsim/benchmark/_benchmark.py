import pandas as pd
import typing
from typing import Sequence

import hpotk

from sumsim.model import DiseaseModel, Sample
from sumsim.sim.phenomizer import TermPair, OneSidedSemiPhenomizer, PrecomputedIcMicaSimilarityMeasure
from ._nulldistribution import GetNullDistribution
from ..sim import SumSimSimilarityKernel


class Benchmark:
    def __init__(self, hpo: hpotk.MinimalOntology, patients: Sequence[Sample], n_iter_distribution: int,
                 num_features_distribution: Sequence[int], mica_dict: typing.Mapping[TermPair, float] = None,
                 delta_ic_dict: typing.Mapping[hpotk.TermId, float] = None,
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self.hpo = hpo
        self.patients = patients
        self.n_iter_distribution = n_iter_distribution
        self.num_features_distribution = num_features_distribution
        self.mica_dict = mica_dict
        self.delta_ic_dict = delta_ic_dict
        self.root = root
        self.patient_table = pd.DataFrame(index = [patient.label for patient in self.patients])

    def compute_ranks(self, similarity_methods: Sequence[str], diseases: Sequence[DiseaseModel]):
        for disease in diseases:
            for method in similarity_methods:
                self._rank_patients(method, disease)
        return self.patient_table

    def _rank_patients(self, similarity_method: str, disease: DiseaseModel):
        # Define similarity kernel
        if similarity_method == "sumsim":
            if self.delta_ic_dict is None:
                raise ValueError("delta_ic_dict must be provided for sumsim method.")
            kernel = SumSimSimilarityKernel(self.hpo, self.delta_ic_dict, self.root)
        elif similarity_method == "phenomizer":
            if self.mica_dict is None:
                raise ValueError("mica_dict must be provided for phenomizer method.")
            kernel = OneSidedSemiPhenomizer(PrecomputedIcMicaSimilarityMeasure(self.mica_dict))
        else:
            raise ValueError("Invalid method.")

        # Get Column Names
        sim = f'{disease.label}_{similarity_method}_sim'
        pval = f'{disease.label}_{similarity_method}_pval'
        rank = f'{disease.label}_{similarity_method}_rank'

        # Calculate similarity of each patient to the disease
        self.patient_table[sim] = [kernel.compute(patient, disease).similarity for patient in self.patients]
        dist_method = GetNullDistribution(disease, similarity_method, self.hpo, self.n_iter_distribution,
                                          self.num_features_distribution, self.mica_dict, self.delta_ic_dict,
                                          self.root)
        self.patient_table[pval] = [dist_method.get_pval(similarity, len(patient.phenotypic_features))
                                    for similarity, patient in zip(self.patient_table[sim], self.patients)]
        self.patient_table[rank] = self.patient_table[pval].rank(method='min', ascending=True)
        pass



