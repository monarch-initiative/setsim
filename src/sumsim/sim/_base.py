import abc

import hpotk

from sumsim.model import Phenotyped



class SimilarityMeasureResult:
    """
    A container for a result of a :class:`SimilarityMeasure` between a pair of phenotypic features.
    """


    def __init__(self, similarity: float):
        self._similarity = similarity

    @property
    def similarity(self) -> float:
        return self._similarity


class SimilarityMeasure(metaclass=abc.ABCMeta):
    """
    Calculate similarity between a pair of phenotypic features.
    """

    @abc.abstractmethod
    def compute_similarity(self, a: hpotk.TermId, b: hpotk.TermId) -> SimilarityMeasureResult:
        """
        Calculate similarity between two phenotypic features.

        :param a: the first term ID.
        :param b: the other term ID.
        :return: the similarity as a non-negative float.
        """
        pass

    @property
    @abc.abstractmethod
    def is_symmetric(self) -> bool:
        """
        Returns `True` if the similarity measure is symmetric (when `sim(a, b) == sim(b, a)`).
        """
        pass



class SimilarityResult:
    """
    Container to hold similarity between a pair of :class:`sumsim.model.Phenotyped` entities
    calculated by a :class:`SimilarityKernel`.
    """

    def __init__(self, similarity: float):
        self._sim = similarity

    @property
    def similarity(self) -> float:
        """
        Get the `float` with the similarity value.
        """
        return self._sim



class SimilarityKernel(metaclass=abc.ABCMeta):
    """
    `SimilarityKernel` calculates similarity for a pair of :class:`Phenotyped` entities based on
    their observed and excluded phenotypic features. The result is returned as
    an instance of :class:`SimilarityResult`.

    Similarity is a non-negative `float` where greater value represents greater similarity and `0` represents
    no similarity whatsoever.

    The kernel is not necessarily one-sided or two-sided, it depends on the implementation.
    """

    @abc.abstractmethod
    def compute(self, a: Phenotyped, b: Phenotyped) -> SimilarityResult:
        """
        Calculate semantic similarity between a pair of :class:`Phenotyped` entities.

        :param a: the first entity.
        :param b: the second entity.

        :return: the similarity as a non-negative `float`.
        """
        pass
