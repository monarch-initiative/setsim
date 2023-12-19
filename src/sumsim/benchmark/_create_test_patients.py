import typing
import random
from typing import Sequence

import hpotk

from sumsim.model import Sample


class PatientGenerator:
    """
    Generate patients with random phenotypes.
    """
    def __init__(self, hpo: hpotk.MinimalOntology, num_patients: int, num_features_per_patient: Sequence[int],
                 root: typing.Union[str, hpotk.TermId] = "HP:0000118"):
        self.root = root
        self.hpo = hpo.graph
        self.num_patients = num_patients
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


    
