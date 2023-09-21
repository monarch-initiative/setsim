import unittest

from c2s2.model import Sample, PhenotypicFeature

from ._algo import TermPair, PhenomizerSimilarityKernel, PrecomputedIcMicaSimilarityMeasure


def map_to_phenotypic_features(term_ids):
    return tuple(PhenotypicFeature.from_values(identifier=term_id, is_present=True)
                 for term_id in term_ids)


class PhenomizerTest(unittest.TestCase):

    def setUp(self) -> None:
        mica_dict = PhenomizerTest._create_mica_dict()
        self.phenomizer = PhenomizerSimilarityKernel(PrecomputedIcMicaSimilarityMeasure(mica_dict))

    def test_normal_input(self):
        # arachnodactyly and portal hypertension
        patient_a = Sample.from_values(label='A', phenotypic_features=map_to_phenotypic_features(["HP:0001166", "HP:0001409"]))
        # arachnodactyly, hypertension, and intellectual disability
        patient_b = Sample.from_values(label='B', phenotypic_features=map_to_phenotypic_features(["HP:0001166", "HP:0000822", "HP:0001249"]))
        similarity = self.phenomizer.compute(patient_a, patient_b)
        self.assertAlmostEqual(similarity.similarity, 5.416666666, delta=1E-9)

    def test_empty_returns_zero(self):
        # arachnodactyly and portal hypertension
        patient_a = Sample.from_values(label='A', phenotypic_features=map_to_phenotypic_features([]))
        # arachnodactyly, hypertension, and intellectual disability
        patient_b = Sample.from_values(label='B', phenotypic_features=map_to_phenotypic_features([]))
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
