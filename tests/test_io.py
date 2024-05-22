import os
import unittest

from pkg_resources import resource_filename

import hpotk

import setsim

data_dir = resource_filename(__name__, 'data')
hpo = hpotk.load_minimal_ontology(os.path.join(data_dir, 'hp.toy.json'))


class TestIo(unittest.TestCase):

    def test_read_phenopacket(self):
        path = os.path.join(data_dir, "test_phenopackets",  "Bill.json")
        bill = setsim.io.read_phenopacket(path, hpo)
        print(type(bill.phenotypic_features[0]))
        self.assertIsInstance(bill, setsim.model._base.Sample)
        self.assertEqual((hpo.get_term("HP:0004026").identifier,), bill.phenotypic_features)

    def test_read_folder(self):
        path = os.path.join(data_dir, "test_phenopackets")
        samples = setsim.io.read_folder(path, hpo)
        self.assertEqual(len(samples), 5)
        sample_dict = {sample.label: sample for sample in samples}
        self.assertEqual(len(sample_dict["Tom"].phenotypic_features), 3)
        self.assertEqual(len(sample_dict["Jed"].phenotypic_features), 0)
        self.assertIn(hpo.get_term("HP:0032648").identifier, set(sample_dict["Tom"].phenotypic_features))
        self.assertNotIn(hpo.get_term("HP:0031264").identifier, set(sample_dict["Tom"].phenotypic_features))

    def test_read_gene_to_phenotype(self):
        path = os.path.join(data_dir, "MiniG2Ph.txt")
        diseases = setsim.io.read_gene_to_phenotype(path, hpo)
        self.assertEqual(5639, len(diseases))
        test_pfs = {hpo.get_term(i).identifier for i in ['HP:0001257',
                                                         'HP:0001263',
                                                         'HP:0001290',
                                                         'HP:0001629',
                                                         'HP:0001638']}
        self.assertEqual(set(diseases[2].phenotypic_features),
                         test_pfs)


if __name__ == '__main__':
    unittest.main()
