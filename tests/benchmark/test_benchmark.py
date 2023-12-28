import os
import unittest

import hpotk
from hpotk import MinimalOntology, TermId

from pkg_resources import resource_filename

import sumsim
from sumsim.benchmark import PatientGenerator, Benchmark, GetNullDistribution
from sumsim.model import DiseaseModel
from sumsim.sim import IcCalculator, IcTransformer

test_data = resource_filename(__name__, '../data')
fpath_hpo = os.path.join(test_data, 'hp.toy.json')
hpo: MinimalOntology = hpotk.load_minimal_ontology(fpath_hpo)

# test_phenopackets has five samples with Four Terms
test_samples = sumsim.io.read_folder(os.path.join(test_data, 'test_phenopackets'), hpo)
test_samples = [sample for sample in test_samples if sample.label != "Jed"]

# Generate IC dictionary
calc = IcCalculator(hpo)
ic_dict = calc.calculate_ic_from_samples(samples=test_samples)

# Generate Delta IC Dictionary
transformer = IcTransformer(hpo)
delta_ic_dict = transformer.transform(ic_dict, strategy="max")


# The table below shows which terms each sample has:
#        HP:0004021  HP:0003981  HP:0004026  HP:0032648
# Tom          True       False        True        True
# Matt         True       False       False       False
# Bill        False       False        True       False
# Kayla       False        True       False       False


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


class TestGetNullDistribution(unittest.TestCase):
    def test_get_null_distribution(self):
        disease_id = TermId.from_curie("MONDO:1234567")
        disease_features = [TermId.from_curie(term) for term in ["HP:0004026", "HP:0032648"]]
        disease = DiseaseModel(disease_id, "Test_Disease", disease_features, hpo)
        number_of_patients = 100
        get_dist = GetNullDistribution(disease, "sumsim", hpo, number_of_patients, [1, 2, 10, 3],
                                       delta_ic_dict=delta_ic_dict)
        self.assertEqual(get_dist.get_pval(0, 2), 1.0)
        self.assertEqual(get_dist.get_pval(10, 4), 0.0)
        self.assertEqual(get_dist.get_pval(7, 10), get_dist.get_pval(7, 500))


class TestBenchmark(unittest.TestCase):
    def test_benchmark(self):
        disease_id = TermId.from_curie("MONDO:1234567")
        disease_features = [TermId.from_curie(term) for term in ["HP:0004026", "HP:0032648"]]
        disease = DiseaseModel(disease_id, "Test_Disease", disease_features, hpo)
        benchmark = Benchmark(hpo, test_samples, 100, [1, 2, 10, 3], delta_ic_dict=delta_ic_dict)
        results = benchmark.compute_ranks(["sumsim"], [disease])
        self.assertEqual(results["Test_Disease_sumsim_rank"].loc["Tom"], 1)
        self.assertEqual(results["Test_Disease_sumsim_rank"].loc["Bill"], 2)
        self.assertTrue(results["Test_Disease_sumsim_pval"].loc["Tom"] <
                        results["Test_Disease_sumsim_pval"].loc["Bill"])
        self.assertTrue(results["Test_Disease_sumsim_sim"].loc["Tom"] >
                        results["Test_Disease_sumsim_sim"].loc["Bill"])


if __name__ == '__main__':
    unittest.main()
