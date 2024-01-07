import abc
import typing
from abc import ABC

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


class SetSimilarityKernel(SimilarityKernel, metaclass=abc.ABCMeta):
    """
    A base class for similarity kernels that calculate similarity by intersecting sets of terms and ancestors.
    """

    @abc.abstractmethod
    def _score_feature_sets(self, shared_features: typing.Set[hpotk.TermId]) -> float:
        pass


class OntoSetSimilarityKernel(SetSimilarityKernel, metaclass=abc.ABCMeta):
    """
    A base class for similarity kernels that find ancestors of terms.
    """

    def __init__(self, hpo: hpotk.GraphAware, root: str = "HP:0000118", **kwargs):
        self._hpo = hpo.graph
        self._features_under_root = set(self._hpo.get_descendants(root, include_source=True))

    def _get_feature_sets(self, a: Phenotyped, b: Phenotyped) -> (typing.Set[hpotk.TermId], typing.Set[hpotk.TermId]):
        a_features = set(ancestor for pf in a.phenotypic_features for ancestor in
                         self._hpo.get_ancestors(pf, include_source=True) if ancestor in self._features_under_root)
        b_features = set(ancestor for pf in b.phenotypic_features for ancestor in
                         self._hpo.get_ancestors(pf, include_source=True) if ancestor in self._features_under_root)
        return a_features, b_features

    def compute(self, a: Phenotyped, b: Phenotyped) -> SimilarityResult:
        feature_sets = self._get_feature_sets(a, b)
        return SimilarityResult(self._score_feature_sets(feature_sets))


class SimilaritiesKernel(metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped):
        self._disease = disease

    @abc.abstractmethod
    def compute(self, sample: Phenotyped) -> typing.Sequence[float]:
        pass


class SetSimilaritiesKernel(SimilaritiesKernel, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, root: str = "HP:0000118"):
        SimilaritiesKernel.__init__(self, disease)
        self._hpo = hpo.graph
        self._features_under_root = set(self._hpo.get_descendants(root, include_source=True))
        self._disease_features = set(ancestor for pf in disease.phenotypic_features for ancestor in
                                     self._hpo.get_ancestors(pf, include_source=True) if
                                     ancestor in self._features_under_root)

    def _sample_iterator(self, sample: Phenotyped) -> typing.Iterable[typing.Set[hpotk.TermId]]:
        for pf in sample.phenotypic_features:
            # Samples may include terms that are not under root since this set is intersecting with disease features.
            yield set(ancestor for ancestor in self._hpo.get_ancestors(pf, include_source=True))

