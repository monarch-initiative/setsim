from unittest import TestCase

import hpotk
from hpotk import MinimalOntology

from src.sumsim.sim._ic import IcCalculator

hpo: MinimalOntology = hpotk.ontology.load.obographs.load_minimal_ontology('src/sumsim/tests/data/hp.toy.json')


class TestIcCalculator(TestCase):
    def test_calculate_ic_from_samples(self):
        calc = IcCalculator(hpo)
        ic_dict = calc.calculate_ic_from_samples(samples="src/sumsim/tests/data/example-cohort.json")
        self.assertEquals(1, 1)


    #def test__get_sample_array(self):

    #def test__get_term_freq(self):
