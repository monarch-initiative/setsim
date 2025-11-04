import csv
import gzip
import io
import re
import typing

from urllib.request import urlopen

import hpotk

HPO_PATTERN = re.compile(r"HP:(?P<ID>\d{7})")


class TermPair:
    """
    A class to represent a pair of HPO terms.

    Use :func:`of` to construct an instance.
    """

    @staticmethod
    def of(left: typing.Union[str, hpotk.model.TermId],
           right: typing.Union[str, hpotk.model.TermId]):
        """
        Create a TermPair from given HPO term IDs.

        :param left: a CURIE or a term ID of the first HPO term (e.g. `HP:0001250`)
        :param right: a CURIE or a term ID of the second HPO term (e.g. `HP:0001166`)
        """
        l: typing.Optional[int] = TermPair._decode_term_id(left)
        r: typing.Optional[int] = TermPair._decode_term_id(right)
        if not (l and r):
            raise ValueError(f"Invalid HPO terms a={left}, b={right}")
        return TermPair(l, r)

    @staticmethod
    def _decode_term_id(payload: typing.Union[str, hpotk.model.TermId]) -> typing.Optional[int]:
        if isinstance(payload, hpotk.model.TermId):
            return int(payload.id)
        elif isinstance(payload, str):
            match = HPO_PATTERN.match(payload)
            if match:
                return int(match.group('ID'))
        return None

    @staticmethod
    def _compute_hash(a, b):
        return hash((a, b))

    def __init__(self, a: int, b: int):
        """
        Create a TermPair from given HPO term IDs.

        :param a: ID of the first HPO term (e.g. 1234567 for `HP:1234567`)
        :param b: ID of the second HPO term (e.g. 1234567 for `HP:1234567`)
        """
        if a < b:
            self._t1 = a
            self._t2 = b
        else:
            self._t1 = b
            self._t2 = a
        self._hash = self._compute_hash(self._t1, self._t2)

    @property
    def t1(self) -> str:
        """
        Get the first HPO term.
        """
        return self._id_to_hpo_representation(self._t1)

    @property
    def t2(self) -> str:
        """
        Get the second HPO term.
        """
        return self._id_to_hpo_representation(self._t2)

    def __eq__(self, other):
        return isinstance(other, TermPair) and self._t1 == other._t1 and self._t2 == other._t2

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"TermPair(left={self._id_to_hpo_representation(self._t1)}, right={self._id_to_hpo_representation(self._t2)})"

    @staticmethod
    def _id_to_hpo_representation(term_id: int) -> str:
        """
        Turn the integral ID into an HPO CURIE (e.g. 1234 -> `HP:0001234`).
        """
        return 'HP:' + str(term_id).rjust(7, '0')


def read_ic_mica_data(fpath: str, timeout: float = 30.) -> typing.Mapping[TermPair, float]:
    """
    Read a CSV table with information contents of most informative common ancestors from given `fpath`.

    If samples are provided, the resulting dictionary will only contain TermPairs that are observed in the samples.

    The file is uncompressed on the fly if the file name ends with `.gz`.

    :param fpath: Path to the CSV file.
    :param timeout: Timeout for opening the file.
    :return: Dictionary mapping TermPairs to their IC_MICA values.
     """
    with _open_file_handle(fpath, timeout) as fh:
        comments, header = _parse_header(fh, comment_char='#')
        fieldnames = header.split(",")
        reader = csv.DictReader(fh, fieldnames=fieldnames)

        # Read the lines
        mica = {}
        for row in reader:
            pair = TermPair.of(row['term_a'], row['term_b'])
            mica[pair] = float(row['ic_mica'])

        return mica


def _parse_header(fh, comment_char):
    """Parse header into a list of comments and a header line. As a side effect, the `fh` is set to the position where
    CSV parser can take over and read the records.

    :return: a tuple with a list of comment lines and the header line
    """
    comments = []
    header = None
    for line in fh:
        if line.startswith(comment_char):
            comments.append(line.strip())
        else:
            header = line.strip()
            break
    return comments, header


def _open_file_handle(fpath: str, timeout) -> typing.IO:
    looks_like_url = fpath.startswith('http://') or fpath.startswith('https://')
    looks_compressed = fpath.endswith('.gz')

    fh = urlopen(fpath, timeout=timeout) if looks_like_url else open(fpath, mode='rb')
    return gzip.open(fh, mode='rt', newline='') if looks_compressed else io.TextIOWrapper(fh, encoding='utf-8')
