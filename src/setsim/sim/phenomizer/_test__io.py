import unittest

from pkg_resources import resource_filename

from ._io import TermPair, read_ic_mica_data


class TestPhenomizerIO(unittest.TestCase):

    def setUp(self) -> None:
        self.fpath = resource_filename(__name__, 'test_data/tps.csv.gz')

    def test_read_ic_mica_data(self):
        mica = read_ic_mica_data(self.fpath)
        # We have 50 lines in total, 3 lines are header + comments, leaving 47 MICA entries
        self.assertEqual(len(mica), 47)
        # first entry
        self.assertAlmostEqual(mica[TermPair.of("HP:0006060", "HP:0001180")], 1.829056832, delta=5E-9)
        # last entry
        self.assertAlmostEqual(mica[TermPair.of("HP:0006077", "HP:0001018")], 1.487333401, delta=5E-9)
