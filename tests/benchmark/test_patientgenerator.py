import os
import unittest

import hpotk
from hpotk import MinimalOntology

from pkg_resources import resource_filename

from sumsim.benchmark import PatientGenerator
from sumsim.benchmark._nulldistribution import GetNullDistributions

test_data = resource_filename(__name__, '../data')
fpath_hpo = os.path.join(test_data, 'hp.toy.json')
hpo: MinimalOntology = hpotk.load_minimal_ontology(fpath_hpo)


class TestPatientGenerator(unittest.TestCase):
    def test_generation(self):
        number_of_patients = 10
        p_gen = PatientGenerator(hpo, number_of_patients, [1, 2, 10, 3])
        list1 = []
        list2 = []
        list3 = []
        list10 = []
        for patient in p_gen.generate():
            [p1, p2, p10, p3] = patient
            self.assertEqual(len(p1.phenotypic_features), 1)
            self.assertEqual(len(p2.phenotypic_features), 2)
            self.assertEqual(len(p3.phenotypic_features), 3)
            self.assertEqual(len(p10.phenotypic_features), 10)
            list1.append(p1)
            list2.append(p2)
            list3.append(p3)
            list10.append(p10)

        self.assertEqual(len(list1), number_of_patients)
        self.assertEqual(len(list2), number_of_patients)
        self.assertEqual(len(list3), number_of_patients)
        self.assertEqual(len(list10), number_of_patients)


if __name__ == '__main__':
    unittest.main()
