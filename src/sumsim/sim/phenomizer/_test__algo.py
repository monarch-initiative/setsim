import typing
import unittest

import hpotk

from sumsim.model import Sample

from ._algo import TermPair, PhenomizerSimilarityKernel, PrecomputedIcMicaSimilarityMeasure


def map_to_phenotypic_features(term_ids) -> typing.Iterable[hpotk.TermId]:
    return (hpotk.TermId.from_curie(curie) for curie in term_ids)


class PhenomizerTest(unittest.TestCase):

    def setUp(self) -> None:
        mica_dict = PhenomizerTest._create_mica_dict()
        self.phenomizer = PhenomizerSimilarityKernel(PrecomputedIcMicaSimilarityMeasure(mica_dict))

    def test_normal_input(self):
        patient_a = Sample(label='A',
                           # arachnodactyly and portal hypertension
                           phenotypic_features=map_to_phenotypic_features(("HP:0001166", "HP:0001409")))

        patient_b = Sample(label='B',
                           # arachnodactyly, hypertension, and intellectual disability
                           phenotypic_features=map_to_phenotypic_features(("HP:0001166", "HP:0000822", "HP:0001249")))
        similarity = self.phenomizer.compute(patient_a, patient_b)
        self.assertAlmostEqual(similarity.similarity, 5.416666666, delta=1E-9)

    def test_empty_returns_zero(self):
        patient_a = Sample(label='A', phenotypic_features=map_to_phenotypic_features([]))
        patient_b = Sample(label='B', phenotypic_features=map_to_phenotypic_features([]))
        similarity = self.phenomizer.compute(patient_a, patient_b)
        self.assertAlmostEqual(similarity.similarity, 0., delta=1E-9)

    @staticmethod
    def _create_mica_dict():
        arachnodactyly = "HP:0001166"
        abnormality_of_finger = "HP:0001167"
        hypertension = "HP:0000822"
        portal_hypertension = "HP:0001409"
        # intellectual_disability = "HP:0001249"
        return {
            TermPair.of(arachnodactyly, arachnodactyly): 10.,
            TermPair.of(abnormality_of_finger, abnormality_of_finger): 5.,
            TermPair.of(arachnodactyly, abnormality_of_finger): 5.,

            TermPair.of(hypertension, hypertension): 4.,
            TermPair.of(portal_hypertension, portal_hypertension): 5.,
            TermPair.of(hypertension, portal_hypertension): 3.
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
