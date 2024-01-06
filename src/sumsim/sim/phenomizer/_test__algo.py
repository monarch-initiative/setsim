import os
import typing
import unittest

from pkg_resources import resource_filename

import hpotk

from sumsim.model import Sample

from ._algo import TermPair, PhenomizerSimilarityKernel, PrecomputedIcMicaSimilarityMeasure, OneSidedSemiPhenomizer, \
    PhenomizerSimilaritiesKernel
from ...model._base import FastPhenotyped

test_data = resource_filename(__name__, '../../../../tests/data')
fpath_hpo = os.path.join(test_data, 'hp.toy.json')
hpo: hpotk.MinimalOntology = hpotk.load_minimal_ontology(fpath_hpo)


def map_to_phenotypic_features(term_ids) -> typing.Iterable[hpotk.TermId]:
    return (hpotk.TermId.from_curie(curie) for curie in term_ids)


class PhenomizerTest(unittest.TestCase):

    def setUp(self) -> None:
        mica_dict = PhenomizerTest._create_mica_dict()
        self.phenomizer = PhenomizerSimilarityKernel(PrecomputedIcMicaSimilarityMeasure(mica_dict))

    def test_normal_input(self):
        patient_a = Sample(label='A',
                           #
                           phenotypic_features=map_to_phenotypic_features(('HP:0004021', 'HP:0004026')),
                           hpo=hpo)
        # print(list(patient_a.phenotypic_features))

        patient_b = Sample(label='B',
                           #
                           phenotypic_features=map_to_phenotypic_features(('HP:0004021', 'HP:0003981', 'HP:0003856')),
                           hpo=hpo)
        similarity = self.phenomizer.compute(patient_a, patient_b)
        self.assertAlmostEqual(similarity.similarity, 5.416666666, delta=1E-9)

    def test_empty_returns_zero(self):
        patient_a = Sample(label='A', phenotypic_features=map_to_phenotypic_features([]), hpo=hpo)
        patient_b = Sample(label='B', phenotypic_features=map_to_phenotypic_features([]), hpo=hpo)
        similarity = self.phenomizer.compute(patient_a, patient_b)
        self.assertAlmostEqual(similarity.similarity, 0., delta=1E-9)

    def test_similarities(self):
        mica_dict = PhenomizerTest._create_mica_dict()
        similarity_kernel = OneSidedSemiPhenomizer(PrecomputedIcMicaSimilarityMeasure(mica_dict))
        disease = FastPhenotyped(phenotypic_features=map_to_phenotypic_features(('HP:0004021', 'HP:0004026')))
        sample = Sample(label='test',
                        phenotypic_features=map_to_phenotypic_features(('HP:0004021', 'HP:0003981', 'HP:0003856')),
                        hpo=hpo)
        sample_features = list(sample.phenotypic_features)
        sample_iteration = [FastPhenotyped(phenotypic_features=sample_features[:i])
                            for i in range(1, len(sample_features) + 1)]
        similarity_results = [similarity_kernel.compute(s_iter, disease).similarity for s_iter in
                              sample_iteration]
        similarities_kernel = PhenomizerSimilaritiesKernel(disease=disease, mica_dict=mica_dict)
        similarities_result = [i.similarity for i in similarities_kernel.compute(sample)]
        self.assertEqual(similarity_results, similarities_result)

    @staticmethod
    def _create_mica_dict():
        lytic_defects_rm = 'HP:0004021'
        broad_radius = 'HP:0003981'
        broad_radial_metaph = 'HP:0004026'
        tubul_bowman_cap = 'HP:0032648'
        # intellectual_disability = "HP:0001249"
        return {
            TermPair.of(lytic_defects_rm, lytic_defects_rm): 10.,
            TermPair.of(broad_radius, broad_radius): 3.,
            TermPair.of(lytic_defects_rm, broad_radius): 2.,

            TermPair.of(tubul_bowman_cap, tubul_bowman_cap): 4.,
            TermPair.of(broad_radial_metaph, broad_radial_metaph): 5.,
            TermPair.of(broad_radius, broad_radial_metaph): 3.
        }


class TestTermPair(unittest.TestCase):

    def test_good_input(self):
        tp = TermPair.of("HP:1234567", "HP:9876543")
        self.assertEqual(tp.t1, "HP:1234567")
        self.assertEqual(tp.t2, "HP:9876543")

    def test_input_is_sorted(self):
        tp = TermPair.of("HP:9876543", "HP:1234567")
        self.assertEqual(tp.t1, "HP:1234567")
        self.assertEqual(tp.t2, "HP:9876543")

    def test_invalid_input_raises(self):
        # One or more misformatted inputs
        self.assertRaises(ValueError, TermPair.of, "HP:1234567", "HP:123456")
        self.assertRaises(ValueError, TermPair.of, "HP:123456", "HP:1234567")
        self.assertRaises(ValueError, TermPair.of, "HP:12345", "HP:123456")
