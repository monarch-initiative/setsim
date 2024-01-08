import abc
import typing

import numpy as np
import hpotk

from sumsim.model import Phenotyped

from .._base import SimilarityKernel, SimilarityMeasure, SimilarityResult, SimilarityMeasureResult, SimilaritiesKernel
from ._io import TermPair


class PrecomputedIcMicaSimilarityMeasure(SimilarityMeasure):
    """
    `PrecomputedIcMicaSimilarityMeasure` uses the information content of the most informative common ancestor
    (:math:`IC_{MICA}`) as a measure of semantic similarity
    for a pair of present :class:`c2s2.model.PhenotypicFeature` instances.

    Note: the similarity measure assigns a similarity of *zero* to a pair consisting of a present and excluded feature,
    as well as to a pair of two excluded features.

    :param mica_dict: a mapping from :class:`c2s2.semsim.TermPair` to :math:`IC_{MICA}` as a `float`.
    """

    def __init__(self, mica_dict: typing.Mapping[TermPair, float]):
        self._mica_dict = mica_dict

    def compute_similarity(self, a: hpotk.TermId, b: hpotk.TermId) -> SimilarityMeasureResult:
        tp = TermPair.of(a, b)
        ic_mica = self._mica_dict.get(tp, 0.)
        return SimilarityMeasureResult(ic_mica)

    @property
    def is_symmetric(self) -> bool:
        return True


class DynamicIcMicaSimilarityMeasure(SimilarityMeasure):
    """
    `DynamicIcMicaSimilarityMeasure` uses HPO hierarchy to find common ancestors of a phenotypic feature pair,
    then gets the information content (IC) values from provided `ic_dict` and uses the highest IC value as
    a similarity.

    Note, the similarity measure assigns a similarity of *zero* to a pair consisting of a present and excluded feature,
    as well as to a pair of two excluded features.

    :param hpo: HPO graph or an :class:`hpotk.graph.GraphAware` entity.
    :param ic_dict: mapping from present :class:`hpotk.model.TermId` to information content as :class:`float`.
    """

    def __init__(self, hpo: typing.Union[hpotk.OntologyGraph, hpotk.GraphAware],
                 ic_dict: typing.Mapping[hpotk.TermId, float]):
        self._hpo = hpotk.util.validate_instance(hpo, hpotk.GraphAware, 'hpo')
        self._ic_dict = ic_dict

    def compute_similarity(self, a: hpotk.TermId, b: hpotk.TermId) -> SimilarityMeasureResult:
        common_ancestors = self._get_common_ancestors(a, b)
        ic_mica = max(
            map(lambda term_id: self._ic_dict.get(term_id, 0.), common_ancestors),
            default=0.)
        return SimilarityMeasureResult(ic_mica)

    def _get_common_ancestors(self,
                              left: hpotk.TermId,
                              right: hpotk.TermId) -> typing.Set[hpotk.TermId]:
        la = set(self._hpo.graph.get_ancestors(left, include_source=True))
        ra = set(self._hpo.graph.get_ancestors(right, include_source=True))
        return la.intersection(ra)

    @property
    def is_symmetric(self) -> bool:
        return True


class BasePhenomizerSimilarityKernel(SimilarityKernel, metaclass=abc.ABCMeta):
    # Not a member of the public API!

    def __init__(self, similarity_measure: SimilarityMeasure):
        self._psm = hpotk.util.validate_instance(similarity_measure,
                                                 SimilarityMeasure,
                                                 'similarity_measure')


class OneSidedSemiPhenomizer(BasePhenomizerSimilarityKernel):
    """
    One-sided similarity measure (probably not kernel) to measure similarity between a patient and a disease.

    IMPORTANT - ensure patient is `a` and disease is `b` in :func:`compute`.
    """

    def compute(self, a: Phenotyped, b: Phenotyped) -> SimilarityResult:
        a_n_observed = len(a.phenotypic_features)
        b_n_observed = len(b.phenotypic_features)
        if a_n_observed == 0 or b_n_observed == 0:
            return SimilarityResult(0.)

        similarities = np.zeros(shape=(a_n_observed, b_n_observed))
        for i, hpoA in enumerate(a.phenotypic_features):
            for j, hpoB in enumerate(b.phenotypic_features):
                similarities[i, j] = self._psm.compute_similarity(hpoA, hpoB).similarity
        max_a = np.max(similarities, axis=1)
        mean_a = max_a.mean()
        return SimilarityResult(mean_a)


class PhenomizerSimilarityKernel(BasePhenomizerSimilarityKernel):
    """
    `PhenomizerSimilarityKernel` calculates semantic similarity between two phenotyped entities.
    The kernel ignores the excluded features.
    """

    @classmethod
    def precomputed_mica(cls, mica_dict: typing.Mapping[TermPair, float]):
        return cls(PrecomputedIcMicaSimilarityMeasure(mica_dict))

    def compute(self, a: Phenotyped, b: Phenotyped) -> SimilarityResult:
        """
        Calculate semantic similarity between patients using the Phenomizer algorithm.

        :param a: HPO terms of first individual
        :param b: HPO terms of second individual
        """
        a_n_observed = len(a.phenotypic_features)
        b_n_observed = len(b.phenotypic_features)
        if a_n_observed == 0 or b_n_observed == 0:
            return SimilarityResult(0.)

        similarities = np.zeros(shape=(a_n_observed, b_n_observed))
        for i, hpoA in enumerate(a.phenotypic_features):
            for j, hpoB in enumerate(b.phenotypic_features):
                similarities[i, j] = self._psm.compute_similarity(hpoA, hpoB).similarity
        max_a = np.max(similarities, axis=1)
        mean_a = max_a.mean()
        max_b = np.max(similarities, axis=0)
        mean_b = max_b.mean()
        return SimilarityResult((mean_a + mean_b) * .5)


class PhenomizerSimilaritiesKernel(SimilaritiesKernel, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, mica_dict: typing.Mapping[TermPair, float]):
        SimilaritiesKernel.__init__(self, disease)
        self._mica_dict = mica_dict

    @staticmethod
    def _sample_iterator(sample: Phenotyped) -> typing.Iterable[hpotk.TermId]:
        for term in sample.phenotypic_features:
            yield term

    def _term_similarity(self, a: hpotk.TermId) -> float:
        a_as_int = int(a.id)
        return max(self._mica_dict.get(TermPair(int(pf.id), a_as_int), 0) for pf in self._disease.phenotypic_features)

    def compute(self, sample: Phenotyped) -> typing.Sequence[float]:
        sim = 0.0
        i = 0
        results = []
        for next_term in self._sample_iterator(sample):
            sim = (self._term_similarity(next_term) + sim * i) / (i + 1)
            i += 1
            results.append(sim)
        return results
