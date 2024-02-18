import os
import unittest
from math import log

from pkg_resources import resource_filename

import hpotk
from hpotk import MinimalOntology

import sumsim
from sumsim.model import Sample
from sumsim.model._base import FastPhenotyped
from sumsim.sim import SimIciSimilarityKernel, JaccardSimilarityKernel
from sumsim.sim import IcCalculator, IcTransformer
from sumsim.sim._count import CountSimilarityKernel
from sumsim.sim._jaccard import JaccardSimilaritiesKernel
from sumsim.sim._phrank import PhrankSimilarityKernel, PhrankSimilaritiesKernel
from sumsim.sim._simgci import SimGciSimilarityKernel, SimGciSimilaritiesKernel
from sumsim.sim._simgic import SimGicSimilarityKernel, SimGicSimilaritiesKernel
from sumsim.sim._simici import SimIciSimilaritiesKernel

test_data = resource_filename(__name__, '../data')
fpath_hpo = os.path.join(test_data, 'hp.toy.json')
hpo: MinimalOntology = hpotk.load_minimal_ontology(fpath_hpo)

root_sample = Sample(phenotypic_features=[hpo.get_term("HP:0000118").identifier], label="Root", hpo=hpo)
empty_sample = Sample(phenotypic_features=[], label="Empty", hpo=hpo)

# test_phenopackets has five samples with Four Terms
temp_samples = sumsim.io.read_folder(os.path.join(test_data, 'test_phenopackets'), hpo)
test_samples = list(temp_samples)
for sample in temp_samples:
    if sample.label == "Tom":
        test_samples[0] = sample
    elif sample.label == "Matt":
        test_samples[1] = sample
    elif sample.label == "Bill":
        test_samples[2] = sample
    elif sample.label == "Kayla":
        test_samples[3] = sample
    elif sample.label == "Jed":
        test_samples[4] = sample

# Generate IC dictionary
calc = IcCalculator(hpo)
ic_dict = calc.calculate_ic_from_samples(samples=test_samples)

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
# Jed         False       False       False       False


class TestSumsim(unittest.TestCase):

    def test_calculate_total_ic(self):
        kernel = SimIciSimilarityKernel(hpo, delta_ic_dict)
        root_ic = kernel._score_feature_sets(
            ({hpo.get_term("HP:0000118").identifier}, {hpo.get_term("HP:0000118").identifier}))
        self.assertAlmostEqual(root_ic, 0.0, 8)  # add assertion here
        used_terms = ic_dict.keys()
        for term, ic in ic_dict.items():
            term_ancestors = set(hpo.graph.get_ancestors(term, include_source=True)).intersection(used_terms)
            self.assertLessEqual(ic, kernel._score_feature_sets((term_ancestors, term_ancestors)),
                                 f'\n{term} has an IC of {ic} but the sample has a total IC of '
                                 f'{kernel._score_feature_sets((term_ancestors, term_ancestors))}.\n Term ancestors: {term_ancestors}')

    def test_get_all_shared_features(self):
        kernel = SimIciSimilarityKernel(hpo, delta_ic_dict)
        matt_bill_overlap_ids = ['HP:0000118',
                                 'HP:0033127',
                                 'HP:0040064',
                                 'HP:0000924',
                                 'HP:0002817',
                                 'HP:0040068',
                                 'HP:0011842',
                                 'HP:0002973',
                                 'HP:0009809',
                                 'HP:0040070',
                                 'HP:0002813',
                                 'HP:0011844',
                                 'HP:0040072',
                                 'HP:0004015',
                                 'HP:0000944',
                                 'HP:0011314',
                                 'HP:0002818']
        matt_bill_overlap = set(hpo.get_term(term_id).identifier for term_id in matt_bill_overlap_ids)
        feature_sets = kernel._get_feature_sets(test_samples[1], test_samples[2])
        self.assertEqual(feature_sets[0].intersection(feature_sets[1]), matt_bill_overlap)
        # Test that passing terms with zero ic returns zero similarity
        self.assertEqual(kernel.compute(root_sample, root_sample).similarity, 0.0)

    def test_count(self):
        sample_0_length = len(set(anc for term in test_samples[0].phenotypic_features
                                  for anc in hpo.graph.get_ancestors(term, include_source=True))
                              - set(term for term in hpo.graph.get_ancestors("HP:0000118"))
                              )
        kernel = CountSimilarityKernel(hpo)
        self.assertAlmostEqual(kernel.compute(test_samples[0], test_samples[0]).similarity,
                               sample_0_length, 8)
        # Test that passing terms with zero ic returns zero similarity
        self.assertEqual(kernel.compute(test_samples[0], root_sample).similarity, 1)
        self.assertEqual(kernel.compute(root_sample, root_sample).similarity, 1)
        test_sample_1 = Sample(phenotypic_features=hpo.graph.get_children("HP:0000118"), label="test", hpo=hpo)
        test_sample_2 = Sample(phenotypic_features=[hpo.get_term("HP:0000152").identifier], label="test", hpo=hpo)
        self.assertAlmostEqual(kernel.compute(test_sample_1, test_sample_2).similarity, 2, 8)

    def test_phrank(self):
        # Phrank would normally takes the bayes_ic_dict as an argument, but we are using the delta_ic_dict for testing
        # since it should behave exactly like sumsim when given the delta_ic_dict
        kernel_phrank = PhrankSimilarityKernel(hpo, bayes_ic_dict=delta_ic_dict)
        kernel_sumsim = SimIciSimilarityKernel(hpo, delta_ic_dict=delta_ic_dict)
        for i_sample in test_samples:
            self.assertEqual(kernel_phrank.compute(i_sample, test_samples[0]).similarity,
                             kernel_sumsim.compute(i_sample, test_samples[0]).similarity)

    def test_phranksimilarities(self):
        similarity_kernel = PhrankSimilarityKernel(hpo, bayes_ic_dict)
        toms_features = test_samples[0].phenotypic_features
        sample_iteration = [FastPhenotyped(phenotypic_features=toms_features[:i])
                            for i in range(1, len(toms_features) + 1)]
        similarity_results = [round(similarity_kernel.compute(s_iter, test_samples[3]).similarity, 8) for s_iter in
                              sample_iteration]
        similarities_kernel = PhrankSimilaritiesKernel(disease=test_samples[3], hpo=hpo, bayes_ic_dict=bayes_ic_dict)
        similarities_result = [round(ic, 8) for ic in similarities_kernel.compute(test_samples[0])]
        self.assertEqual(similarity_results, similarities_result)
        self.assertEqual(similarities_kernel.compute(root_sample), [0.0])

    def test_simgic(self):
        kernel = SimGicSimilarityKernel(hpo, ic_dict)
        self.assertAlmostEqual(kernel.compute(test_samples[0], test_samples[0]).similarity, 1.0, 8)
        # Test that passing terms with zero ic returns zero similarity
        self.assertEqual(kernel.compute(test_samples[0], root_sample).similarity, 0.0)
        self.assertEqual(kernel.compute(root_sample, root_sample).similarity, 0.0)
        test_sample_1 = Sample(phenotypic_features=hpo.graph.get_children("HP:0000118"), label="test", hpo=hpo)
        test_sample_2 = Sample(phenotypic_features=[hpo.get_term("HP:0000152").identifier], label="test", hpo=hpo)
        self.assertAlmostEqual(kernel.compute(test_sample_1, test_sample_2).similarity, 1 / 11, 8)

    def test_simgicimilarities(self):
        similarity_kernel = SimGicSimilarityKernel(hpo, ic_dict)
        toms_features = test_samples[0].phenotypic_features
        sample_iteration = [FastPhenotyped(phenotypic_features=toms_features[:i])
                            for i in range(1, len(toms_features) + 1)]
        similarity_results = [round(similarity_kernel.compute(s_iter, test_samples[3]).similarity, 8) for s_iter in
                              sample_iteration]
        similarities_kernel = SimGicSimilaritiesKernel(disease=test_samples[3], hpo=hpo, ic_dict=ic_dict)
        similarities_result = [round(ic, 8) for ic in similarities_kernel.compute(test_samples[0])]
        self.assertEqual(similarity_results, similarities_result)
        self.assertEqual(similarities_kernel.compute(root_sample), [0.0])

    def test_simcic(self):
        kernel = SimGicSimilarityKernel(hpo, delta_ic_dict)
        self.assertAlmostEqual(kernel.compute(test_samples[0], test_samples[0]).similarity, 1.0, 8)
        # Test that passing terms with zero ic returns zero similarity
        self.assertEqual(kernel.compute(test_samples[0], root_sample).similarity, 0.0)
        self.assertEqual(kernel.compute(root_sample, root_sample).similarity, 0.0)
        test_sample_1 = Sample(phenotypic_features=hpo.graph.get_children("HP:0000118"), label="test", hpo=hpo)
        test_sample_2 = Sample(phenotypic_features=[hpo.get_term("HP:0000152").identifier], label="test", hpo=hpo)
        self.assertAlmostEqual(kernel.compute(test_sample_1, test_sample_2).similarity, 1 / 11, 8)

    def test_simcicimilarities(self):
        similarity_kernel = SimGciSimilarityKernel(hpo, delta_ic_dict)
        toms_features = test_samples[0].phenotypic_features
        sample_iteration = [FastPhenotyped(phenotypic_features=toms_features[:i])
                            for i in range(1, len(toms_features) + 1)]
        similarity_results = [round(similarity_kernel.compute(s_iter, test_samples[3]).similarity, 8) for s_iter in
                              sample_iteration]
        similarities_kernel = SimGciSimilaritiesKernel(disease=test_samples[3], hpo=hpo, delta_ic_dict=delta_ic_dict)
        similarities_result = [round(ic, 8) for ic in similarities_kernel.compute(test_samples[0])]
        self.assertEqual(similarity_results, similarities_result)
        self.assertEqual(similarities_kernel.compute(root_sample), [0.0])

    def test_sumsimsimilarities(self):
        similarity_kernel = SimIciSimilarityKernel(hpo, delta_ic_dict)
        toms_features = test_samples[0].phenotypic_features
        sample_iteration = [FastPhenotyped(phenotypic_features=toms_features[:i])
                            for i in range(1, len(toms_features) + 1)]
        similarity_results = [round(similarity_kernel.compute(s_iter, test_samples[3]).similarity, 8) for s_iter in
                              sample_iteration]
        similarities_kernel = SimIciSimilaritiesKernel(disease=test_samples[3], hpo=hpo, delta_ic_dict=delta_ic_dict)
        similarities_result = [round(ic, 8) for ic in similarities_kernel.compute(test_samples[0])]
        self.assertEqual(similarity_results, similarities_result)
        self.assertEqual(similarities_kernel.compute(root_sample), [0.0])

    def test_jaccardsimilarities(self):
        similarity_kernel = JaccardSimilarityKernel(hpo)
        toms_features = test_samples[0].phenotypic_features
        sample_iteration = [FastPhenotyped(phenotypic_features=toms_features[:i])
                            for i in range(1, len(toms_features) + 1)]
        similarity_results = [round(similarity_kernel.compute(s_iter, test_samples[3]).similarity, 8) for s_iter in
                              sample_iteration]
        similarities_kernel = JaccardSimilaritiesKernel(disease=test_samples[3], hpo=hpo)
        similarities_result = [round(ic, 8) for ic in similarities_kernel.compute(test_samples[0])]
        self.assertEqual(similarity_results, similarities_result)


if __name__ == '__main__':
    unittest.main()
