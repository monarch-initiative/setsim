import os
import unittest
from math import log

from pkg_resources import resource_filename

import hpotk
from hpotk import MinimalOntology

import sumsim
from sumsim.sim import IcCalculator, IcTransformer
from sumsim.sim.phenomizer import TermPair

test_data = resource_filename(__name__, '../data')
fpath_hpo = os.path.join(test_data, 'hp.toy.json')
hpo: MinimalOntology = hpotk.load_minimal_ontology(fpath_hpo)

# test_phenopackets has five samples with Four Terms
test_samples = sumsim.io.read_folder(os.path.join(test_data, 'test_phenopackets'))


# The table below shows which terms each sample has:
#        HP:0004021  HP:0003981  HP:0004026  HP:0032648
# Tom          True       False        True        True
# Matt         True       False       False       False
# Bill        False       False        True       False
# Kayla       False        True       False       False
# Jed         False       False       False       False


class TestIcCalculator(unittest.TestCase):
    def test_calculate_ic_from_samples(self):
        calc = IcCalculator(hpo)
        ic_dict = calc.calculate_ic_from_samples(samples=test_samples)

        # Test that five samples are collected
        self.assertEqual(5, len(test_samples))

        # Check that all descendants of "Phenotypic abnormality" are in dictionary
        phe_abnormalities = [i for i in hpo.graph.get_descendants("HP:0000118", include_source=True)]
        keys = ic_dict.keys()
        onto_size = len(phe_abnormalities)
        dict_size = len(keys)
        self.assertEqual(onto_size, dict_size)
        self.assertEqual(set(phe_abnormalities), set(keys))

        # Test that Phenotypic abnormality is present in 4 out of 5 samples
        phe_abn = hpo.get_term("HP:0000118")
        self.assertAlmostEqual(0.22314355131, ic_dict[phe_abn.identifier], 8)

        # Test various values
        # Dictionary - DefaultTermId : Number of samples with feature
        value_tests = {
            hpo.get_term("HP:0032648").identifier: 1,  # Just Tom
            hpo.get_term("HP:0031264").identifier: 1,  # Just Tom (HP:0031264 is the parent of HP:0032648)
            hpo.get_term("HP:0031263").identifier: 1,  # Just Tom (HP:0031263 is the parent of HP:0031264)
            hpo.get_term("HP:0000119").identifier: 1,  # Just Tom (HP:0000119 is an ancestor of HP:0032648)
            hpo.get_term("HP:0004021").identifier: 2,  # Tom and Matt
            hpo.get_term("HP:0003981").identifier: 3,  # Tom, Bill, and Kayla (HP:0003981 is an ancestor of HP:0004026)
            hpo.get_term("HP:0032599").identifier: 1,  # No one (minimum IC set count to 1 instead of 0)
            hpo.get_term("HP:0004015").identifier: 3  # Tom, Matt, and Bill (HP:0004021 and HP:0004026 are descendants)
        }
        for key, value in value_tests.items():
            self.assertEqual(ic_dict[key], log(5 / value))

        # Test that no term was assigned to all samples
        for ic in set(ic_dict.values()):
            self.assertNotAlmostEqual(ic, 0, 8)

        # Test mica dictionary creation
        sample_term_string = ["HP:0004021", "HP:0003981", "HP:0004026", "HP:0032648"]
        sample_terms = set(hpo.get_term(term).identifier for term in sample_term_string)
        mica_dict = calc.create_mica_ic_dict(sample_terms, ic_dict)
        self.assertEqual(mica_dict, calc.create_mica_ic_dict(samples=test_samples))
        test_pair_1 = TermPair.of(hpo.get_term("HP:0004021").identifier, hpo.get_term("HP:0004026").identifier)
        test_pair_2 = TermPair.of(hpo.get_term("HP:0032648").identifier, hpo.get_term("HP:0004026").identifier)
        test_pair_3 = TermPair.of(hpo.get_term("HP:0032648").identifier, hpo.get_term("HP:0032648").identifier)
        self.assertAlmostEqual(log(5 / 3), mica_dict.get(test_pair_1, 0.0), 8)
        self.assertAlmostEqual(log(5 / 4), mica_dict.get(test_pair_2, 0.0), 8)
        self.assertAlmostEqual(log(5 / 1), mica_dict.get(test_pair_3, 0.0), 8)


class TestIcTransformer(unittest.TestCase):

    def setUp(self):
        # These functions won't work if the above tests fail
        self.root = "HP:0000118"
        self.root_identifier = hpo.get_term(self.root).identifier
        calc = IcCalculator(hpo, root=self.root)
        self.ic_dict = calc.calculate_ic_from_samples(samples=test_samples)
        self.transformer = IcTransformer(hpo, root=self.root)

    def test_use_mean(self):
        delta_ic_dict = self.transformer._use_mean(self.ic_dict)
        key_iterator = iter(delta_ic_dict)
        self.assertIsInstance(next(key_iterator), hpotk.TermId)
        self.assertEqual(set(delta_ic_dict), set(self.ic_dict))
        # IC and delta_ic should be the same for the root since the root has no parents (parent_ic=0).
        self.assertEqual(delta_ic_dict[self.root_identifier], self.ic_dict[self.root_identifier])

        # Test various values
        # Dictionary - DefaultTermId : New IC
        value_tests = {
            hpo.get_term("HP:0032648").identifier: 0,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0031264").identifier: 0,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0031263").identifier: 0,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0000119").identifier: 1.38629436112,  # HP:0000119 is an ancestor of HP:0032648 and the
            # child of HP:0000118
            hpo.get_term("HP:0004021").identifier: 0.20273255405,  # HP:0004021 has 2 parents,
            # one with 2 annotations and the other with 3
            hpo.get_term("HP:0003981").identifier: 0.09589402415,  # HP:0003981 has 3 parents, 1 has 4 annotations
            # and the others have 3
            hpo.get_term("HP:0032599").identifier: 0,  # No one has term or parent
            hpo.get_term("HP:0004015").identifier: 0.14384103622  # HP:0004015 has 2 parents, 1 has 4 annotations
            # and the other has 3
        }
        for key, value in value_tests.items():
            msg = (f'\n   {key.value} has the has a delta_ic of {delta_ic_dict[key]} when it is expected to be {value}.'
                   f'\n   The original IC (from self.ic_dict) of {key.value} is {self.ic_dict[key]}')
            self.assertAlmostEqual(value, delta_ic_dict[key], 8, msg=msg)

    def test_use_max(self):
        delta_ic_dict = self.transformer._use_max(self.ic_dict)
        key_iterator = iter(delta_ic_dict)
        self.assertIsInstance(next(key_iterator), hpotk.TermId)
        self.assertEqual(set(delta_ic_dict), set(self.ic_dict))
        # IC and delta_ic should be the same for the root since the root has no parents (parent_ic=0).
        self.assertEqual(delta_ic_dict[self.root_identifier], self.ic_dict[self.root_identifier])

        # Test various values
        # Dictionary - DefaultTermId : New IC
        value_tests = {
            hpo.get_term("HP:0032648").identifier: 0,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0031264").identifier: 0,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0031263").identifier: 0,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0000119").identifier: 1.38629436112,  # HP:0000119 is an ancestor of HP:0032648 and the
            # child of HP:0000118
            hpo.get_term("HP:0004021").identifier: 0,  # HP:0004021 has 2 parents, one with 2 annotations
            # and the other with 3
            hpo.get_term("HP:0003981").identifier: 0,  # HP:0003981 has 3 parents, 1 has 4 annotations
            # and the others have 3
            hpo.get_term("HP:0032599").identifier: 0,  # No one has term or parent
            hpo.get_term("HP:0004015").identifier: 0  # HP:0004015 has 2 parents, 1 has 4 annotations
            # and the other has 3
        }
        for key, value in value_tests.items():
            msg = (f'\n   {key.value} has the has a delta_ic of {delta_ic_dict[key]} when it is expected to be {value}.'
                   f'\n   The original IC (from self.ic_dict) of {key.value} is {self.ic_dict[key]}')
            self.assertAlmostEqual(value, delta_ic_dict[key], 8, msg=msg)

    def test_transform(self):
        # Make sure error is thrown when illegal value is passed in dictionary
        new_dict = {**self.ic_dict, hpo.get_term("HP:0000001").identifier: log(5 / 4)}
        with self.assertRaises(ValueError):
            self.transformer.transform(new_dict)
        mean_ic_dict = self.transformer.transform(self.ic_dict)
        max_ic_dict = self.transformer.transform(self.ic_dict, strategy='max')
        self.assertEqual(set(mean_ic_dict.keys()), set(max_ic_dict.keys()))
        self.assertAlmostEqual(max_ic_dict[hpo.get_term("HP:0000119").identifier], 1.38629436112, 8)
        # Test various values
        # Dictionary - DefaultTermId : Equal
        value_tests = {
            hpo.get_term("HP:0032648").identifier: True,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0031264").identifier: True,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0031263").identifier: True,  # Term and parents have only 1 annotation
            hpo.get_term("HP:0000119").identifier: True,  # HP:0000119 is an ancestor of HP:0032648 and the
            # child of HP:0000118
            hpo.get_term("HP:0004021").identifier: False,  # HP:0004021 has 2 parents, one with 2 annotations
            # and the other with 3
            hpo.get_term("HP:0003981").identifier: False,  # HP:0003981 has 3 parents, 1 has 4 annotations
            # and the others have 3
            hpo.get_term("HP:0032599").identifier: True,  # No one has term or parent
            hpo.get_term("HP:0004015").identifier: False  # HP:0004015 has 2 parents, 1 has 4 annotations
            # and the other has 3
        }
        for key, equal in value_tests.items():
            if equal:
                self.assertAlmostEqual(max_ic_dict[key], mean_ic_dict[key], 8)
            else:
                self.assertNotAlmostEqual(max_ic_dict[key], mean_ic_dict[key], 8)


if __name__ == '__main__':
    unittest.main()
