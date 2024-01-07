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
mica_dict = calc.create_mica_ic_dict(samples=test_samples, ic_dict=ic_dict)

# Generate Delta IC Dictionary
transformer = IcTransformer(hpo)
delta_ic_dict = transformer.transform(ic_dict, strategy="max")


# The table below shows which terms each sample has:
#        HP:0004021  HP:0003981  HP:0004026  HP:0032648
# Tom          True       False        True        True
# Matt         True       False       False       False
# Bill        False       False        True       False
# Kayla       False        True       False       False


class TestGetNullDistribution(unittest.TestCase):
    def test_get_null_distribution(self):
        disease_id = TermId.from_curie("MONDO:1234567")
        disease_features = [TermId.from_curie(term) for term in ["HP:0004026", "HP:0032648"]]
        disease = DiseaseModel(disease_id, "Test_Disease", disease_features, hpo)
        number_of_patients = 100
        get_dist = GetNullDistribution(disease, method="sumsim", hpo=hpo, num_patients=number_of_patients,
                                       num_features_per_patient=10, delta_ic_dict=delta_ic_dict)
        self.assertEqual(get_dist.get_pval(0, 2), 1.0)
        self.assertEqual(get_dist.get_pval(10, 4), 0.0)
        self.assertEqual(get_dist.get_pval(7, 10), get_dist.get_pval(7, 500))


class TestBenchmark(unittest.TestCase):
    def test_benchmark(self):
        disease_id = TermId.from_curie("MONDO:1234567")
        disease_features = [TermId.from_curie(term) for term in ["HP:0004026", "HP:0032648"]]
        disease = DiseaseModel(disease_id, "Test_Disease", disease_features, hpo)
        benchmark = Benchmark(hpo, test_samples, 100, 10, delta_ic_dict=delta_ic_dict, mica_dict=mica_dict,
                              chunksize=1, similarity_methods=["sumsim", "phenomizer", "jaccard"])
        results = benchmark.compute_ranks([disease])
        self.assertTrue(results["MONDO_1234567_sumsim_pval"].loc["Tom"] <
                        results["MONDO_1234567_sumsim_pval"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_sumsim_sim"].loc["Tom"] >
                        results["MONDO_1234567_sumsim_sim"].loc["Bill"])
        self.assertEqual(results["MONDO_1234567_phenomizer_pval"].loc["Matt"],
                         results["MONDO_1234567_phenomizer_pval"].loc["Kayla"])
        self.assertTrue(results["MONDO_1234567_phenomizer_sim"].loc["Tom"] >
                        results["MONDO_1234567_phenomizer_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_jaccard_sim"].loc["Tom"] >
                        results["MONDO_1234567_jaccard_sim"].loc["Bill"])

        # Repeat with ic_dict
        benchmark = Benchmark(hpo, test_samples, 100, 10, delta_ic_dict=delta_ic_dict, ic_dict=ic_dict,
                              chunksize=1, similarity_methods=["sumsim", "phenomizer", "jaccard"])
        results = benchmark.compute_ranks([disease])
        self.assertTrue(results["MONDO_1234567_sumsim_pval"].loc["Tom"] <
                        results["MONDO_1234567_sumsim_pval"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_sumsim_sim"].loc["Tom"] >
                        results["MONDO_1234567_sumsim_sim"].loc["Bill"])
        self.assertEqual(results["MONDO_1234567_phenomizer_pval"].loc["Matt"],
                         results["MONDO_1234567_phenomizer_pval"].loc["Kayla"])
        self.assertTrue(results["MONDO_1234567_phenomizer_sim"].loc["Tom"] >
                        results["MONDO_1234567_phenomizer_sim"].loc["Bill"])


if __name__ == '__main__':
    unittest.main()
