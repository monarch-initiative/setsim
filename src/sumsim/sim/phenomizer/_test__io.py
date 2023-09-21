import unittest
import sqlite3

from pkg_resources import resource_filename

from ._io import TermPair, read_ic_mica_data


class TestPhenomizerIO(unittest.TestCase):

    def setUp(self) -> None:
        self._conn = sqlite3.connect(resource_filename(__name__, 'test_data/mica_db_2022_10_05_subset.sql'))
        self._cursor = self._conn.cursor()
        self.fpath = resource_filename(__name__, 'test_data/tps.csv.gz')

    def test_read_ic_mica_data(self):
        mica = read_ic_mica_data(self.fpath)
        # We have 50 lines in total, 3 lines are header + comments, leaving 47 MICA entries
        self.assertEqual(len(mica), 47)
        # first entry
        self.assertAlmostEqual(mica[TermPair.of("HP:0006060", "HP:0001180")], 1.829056832, delta=5E-9)
        # last entry
        self.assertAlmostEqual(mica[TermPair.of("HP:0006077", "HP:0001018")], 1.487333401, delta=5E-9)

    def test_read_ic_sql_mica_data(self):
        list_of_terms = [['1180', '6060', 1.829056832], ['1018', '6077', 1.487333401]]
        for term_a, term_b, correct_mica in list_of_terms:
            query = 'SELECT * FROM ic_mica WHERE term_a = ? AND term_b = ?'
            self._cursor.execute(query, (term_a, term_b))
            row = self._cursor.fetchone()  # Fetch the first matching row
            if row:
                term_a, term_b, ic_mica = row
            else:
                ic_mica = 0
            self.assertAlmostEqual(ic_mica, correct_mica, delta=5E-9)
        self._conn.close()
