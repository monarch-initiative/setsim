from unittest import TestCase
from math import log
import hpotk
from hpotk import MinimalOntology

import sumsim.io
from src.sumsim.sim._ic import IcCalculator

hpo: MinimalOntology = hpotk.ontology.load.obographs.load_minimal_ontology('../data/hp.toy.json')

# test_phenopackets has five samples with Four Terms
test_samples = sumsim.io.read_folder("../data/test_phenopackets")


# The table below shows which terms each sample has:
#        HP:0004021  HP:0003981  HP:0004026  HP:0032648
# Tom          True       False        True        True
# Matt         True       False       False       False
# Bill        False       False        True       False
# Kayla       False        True       False       False
# Jed         False       False       False       False


class TestIcCalculator(TestCase):
    def test_calculate_ic_from_samples(self):
        calc = IcCalculator(hpo)
        ic_dict = calc.calculate_ic_from_samples(samples=test_samples)

        # Test that five samples are collected
        self.assertEqual(len(test_samples), 5)

        # Check that all descendants of "Phenotypic abnormality" are in dictionary
        phe_abnormalities = [i for i in hpo.graph.get_descendants("HP:0000118", include_source=True)]
        keys = ic_dict.keys()
        onto_size = len(phe_abnormalities)
        dict_size = len(keys)
        self.assertEqual(onto_size, dict_size)
        self.assertEqual(set(keys), set(phe_abnormalities))

        # Test that Phenotypic abnormality is present in 4 out of 5 samples
        phe_abn = hpo.get_term("HP:0000118")
        self.assertAlmostEqual(ic_dict[phe_abn.identifier], 0.22314355131, 8)

        # Test various values
        # Dictionary - DefaultTermId : Number of samples with feature
        value_tests = {
            hpo.get_term("HP:0032648").identifier: 1,  # Just Tom
            hpo.get_term("HP:0004021").identifier: 2,  # Tom and Matt
            hpo.get_term("HP:0003981").identifier: 3,  # Tom, Bill, and Kayla (HP:0003981 is an ancestor of HP:0004026)
            hpo.get_term("HP:0032599").identifier: 1,  # No one (minimum IC set count to 1 instead of 0)
            hpo.get_term("HP:0004015").identifier: 3   # Tom, Matt, and Bill (HP:0004021 and HP:0004026 are ancestors)
        }
        for key, value in value_tests.items():
            self.assertEqual(ic_dict[key], log(5 / value))

        # Test that no term was assigned to all samples
        for ic in set(ic_dict.values()):
            self.assertNotAlmostEqual(ic, 0, 8)

