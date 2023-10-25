import unittest
import os

import hpotk
from hpotk.ontology import MinimalOntology

import sumsim

hpo: MinimalOntology = hpotk.ontology.load.obographs.load_minimal_ontology('data/hp.toy.json')


class TestIo(unittest.TestCase):
    def test_read_phenopacket(self):
        path = os.path.join("data/test_phenopackets/Bill.json")
        bill = sumsim.io.read_phenopacket(path)
        print(type(bill.phenotypic_features[0]))
        self.assertIsInstance(bill, sumsim.model._base.Sample)
        self.assertEqual((hpo.get_term("HP:0004026").identifier,), bill.phenotypic_features)

    def test_read_folder(self):
        path = os.path.join("data/test_phenopackets")
        samples = sumsim.io.read_folder(path)
        self.assertEqual(len(samples), 5)
        sample_dict = {sample.label: sample for sample in samples}
        self.assertEqual(len(sample_dict["Tom"].phenotypic_features), 3)
        self.assertEqual(len(sample_dict["Jed"].phenotypic_features), 0)
        self.assertIn(hpo.get_term("HP:0032648").identifier, set(sample_dict["Tom"].phenotypic_features))
        self.assertNotIn(hpo.get_term("HP:0031264").identifier, set(sample_dict["Tom"].phenotypic_features))


if __name__ == '__main__':
    unittest.main()
