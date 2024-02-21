import os
import unittest

import hpotk
from hpotk import MinimalOntology, TermId

from pkg_resources import resource_filename

import sumsim
from sumsim.matrix import PatientGenerator, SimilarityMatrix, GetNullDistribution
from sumsim.matrix._nulldistribution import KernelIterator
from sumsim.matrix._rank import Rank
from sumsim.model import DiseaseModel
from sumsim.sim import IcCalculator, IcTransformer
from sumsim.sim.phenomizer._algo import PhenomizerSimilaritiesKernel

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
transformer = IcTransformer(hpo, samples=test_samples)
delta_ic_dict = transformer.transform(ic_dict, strategy="max")
bayes_ic_dict = transformer.transform(ic_dict, strategy="bayesian")


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
        get_dist = GetNullDistribution(disease, method="simici", hpo=hpo, num_patients=number_of_patients,
                                       num_features_per_patient=10, delta_ic_dict=delta_ic_dict)
        self.assertEqual(get_dist.get_pval(0, 2), 1.0)
        self.assertEqual(get_dist.get_pval(10, 4), 0.0)
        self.assertEqual(get_dist.get_pval(7, 10), get_dist.get_pval(7, 500))


class TestFragileMicaKernel(unittest.TestCase):
    def test_fragile_mica_kernel(self):
        disease_id = TermId.from_curie("MONDO:1234567")
        disease_features = [TermId.from_curie(term) for term in ["HP:0004026", "HP:0032648"]]
        disease = DiseaseModel(disease_id, "Test_Disease", disease_features, hpo)
        kernel_generator = KernelIterator(hpo, mica_dict=mica_dict, root="HP:0000118")
        fragile_phenomizer = kernel_generator._define_kernel(disease, "phenomizer")
        phenomizer = PhenomizerSimilaritiesKernel(disease, mica_dict=mica_dict, use_fragile_mica_dict=False)
        for sample in test_samples:
            self.assertEqual(fragile_phenomizer.compute(sample), phenomizer.compute(sample))


class TestBenchmark(unittest.TestCase):


    def test_matrix(self):
        disease_id = TermId.from_curie("MONDO:1234567")
        disease_features = [TermId.from_curie(term) for term in ["HP:0004026", "HP:0032648"]]
        disease = DiseaseModel(disease_id, "Test_Disease", disease_features, hpo)
        benchmark = SimilarityMatrix(hpo, test_samples, 100, 10, ic_dict=ic_dict, bayes_ic_dict=bayes_ic_dict,
                                     delta_ic_dict=delta_ic_dict, mica_dict=mica_dict, chunksize=1,
                                     similarity_methods=["simici", "phenomizer", "jaccard", "simgic", "phrank",
                                                         "simgci",
                                                         "count"])
        results = benchmark.compute_diagnostic_similarities([disease])
        self.assertTrue(results["MONDO_1234567_simici_pval"].loc["Tom"] <
                        results["MONDO_1234567_simici_pval"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_simici_sim"].loc["Tom"] >
                        results["MONDO_1234567_simici_sim"].loc["Bill"])
        self.assertEqual(results["MONDO_1234567_phenomizer_pval"].loc["Matt"],
                         results["MONDO_1234567_phenomizer_pval"].loc["Kayla"])
        self.assertTrue(results["MONDO_1234567_phenomizer_sim"].loc["Tom"] >
                        results["MONDO_1234567_phenomizer_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_jaccard_sim"].loc["Tom"] >
                        results["MONDO_1234567_jaccard_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_simgic_sim"].loc["Tom"] >
                        results["MONDO_1234567_simgic_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_phrank_sim"].loc["Tom"] >
                        results["MONDO_1234567_phrank_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_simgci_sim"].loc["Tom"] >
                        results["MONDO_1234567_simgci_sim"].loc["Bill"])

        # Repeat with ic_dict
        benchmark = SimilarityMatrix(hpo, test_samples, 100, 10, bayes_ic_dict=bayes_ic_dict, delta_ic_dict=delta_ic_dict,
                                     ic_dict=ic_dict,
                                     chunksize=1,
                                     similarity_methods=["simici", "phenomizer", "jaccard", "simgic", "phrank", "simgci"])
        results = benchmark.compute_diagnostic_similarities([disease])
        self.assertTrue(results["MONDO_1234567_simici_pval"].loc["Tom"] <
                        results["MONDO_1234567_simici_pval"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_simici_sim"].loc["Tom"] >
                        results["MONDO_1234567_simici_sim"].loc["Bill"])
        self.assertEqual(results["MONDO_1234567_phenomizer_pval"].loc["Matt"],
                         results["MONDO_1234567_phenomizer_pval"].loc["Kayla"])
        self.assertTrue(results["MONDO_1234567_phenomizer_sim"].loc["Tom"] >
                        results["MONDO_1234567_phenomizer_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_jaccard_sim"].loc["Tom"] >
                        results["MONDO_1234567_jaccard_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_simgic_sim"].loc["Tom"] >
                        results["MONDO_1234567_simgic_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_phrank_sim"].loc["Tom"] >
                        results["MONDO_1234567_phrank_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_simgci_sim"].loc["Tom"] >
                        results["MONDO_1234567_simgci_sim"].loc["Bill"])

        # Repeat without distribution
        benchmark = SimilarityMatrix(hpo, test_samples, 0, 10, delta_ic_dict=delta_ic_dict, ic_dict=ic_dict,
                                     chunksize=1, similarity_methods=["simici", "phenomizer", "jaccard"])
        results = benchmark.compute_diagnostic_similarities([disease])
        self.assertTrue(results["MONDO_1234567_simici_sim"].loc["Tom"] >
                        results["MONDO_1234567_simici_sim"].loc["Bill"])
        self.assertTrue(results["MONDO_1234567_phenomizer_sim"].loc["Tom"] >
                        results["MONDO_1234567_phenomizer_sim"].loc["Bill"])

    def test_patient2patient_similarities(self):
        benchmark = SimilarityMatrix(hpo, test_samples, 100, 10, ic_dict=ic_dict, bayes_ic_dict=bayes_ic_dict,
                                     delta_ic_dict=delta_ic_dict, mica_dict=mica_dict, chunksize=1,
                                     similarity_methods=["simici", "phenomizer", "jaccard", "simgic", "phrank", "simgci"])
        results = benchmark.compute_person2person_similarities(test_samples)
        # assert symmetries
        self.assertEqual(results["Kayla_phrank_sim"].loc["Tom"],
                         results["Tom_phrank_sim"].loc["Kayla"])
        self.assertEqual(results["Bill_simici_sim"].loc["Tom"],
                         results["Tom_simici_sim"].loc["Bill"])
        self.assertEqual(results["Bill_jaccard_sim"].loc["Tom"],
                         results["Tom_jaccard_sim"].loc["Bill"])
        self.assertAlmostEqual(results["Bill_simgic_sim"].loc["Tom"],
                         results["Tom_simgic_sim"].loc["Bill"], 8)
        self.assertEqual(results["Bill_simgci_sim"].loc["Tom"],
                         results["Tom_simgci_sim"].loc["Bill"])

    def test_rank(self):
        disease_id = [TermId.from_curie("MONDO:1234567"), TermId.from_curie("MONDO:2345678")]
        disease_features = [[TermId.from_curie(term) for term in ["HP:0004021", "HP:0004026", "HP:0032648"]],
                            [TermId.from_curie(term) for term in ["HP:0004026"]]]
        diseases = [DiseaseModel(disease_id[i], f"Test_Disease_{i}", disease_features[i], hpo) for i in range(2)]
        benchmark = SimilarityMatrix(hpo, test_samples, 100, 10, ic_dict=ic_dict, bayes_ic_dict=bayes_ic_dict,
                                     delta_ic_dict=delta_ic_dict, mica_dict=mica_dict, chunksize=1,
                                     similarity_methods=["simici", "phenomizer", "jaccard", "simgic", "phrank",
                                                         "simgci",
                                                         "count"])
        results = benchmark.compute_diagnostic_similarities(diseases)
        mrank = Rank(results)
        mrank.rank()
        rankings = mrank.get_rankings()
        self.assertEqual(rankings.loc["Tom", ~rankings.columns.isin(["disease_id", "num_features"])].mean(), 1)
        self.assertEqual(len(rankings.columns), 16)


if __name__ == '__main__':
    unittest.main()
