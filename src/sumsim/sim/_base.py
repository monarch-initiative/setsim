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
    def _normalization_method(self) -> typing.Literal["union", "reciprocal average", "none"]:
        pass

    def _score_feature_sets(self, feature_sets: (typing.Union[typing.Set[str], typing.Set[hpotk.TermId]],
                                                 typing.Union[typing.Set[str], typing.Set[hpotk.TermId]])) -> float:
        intersection_score = self._score_feature_set(feature_sets[0].intersection(feature_sets[1]))
        if self._normalization_method() == "union":
            union_score = self._score_feature_set(feature_sets[0].union(feature_sets[1]))
            if union_score == 0:
                return 0.0
            else:
                return intersection_score / self._score_feature_set(feature_sets[0].union(feature_sets[1]))
        return intersection_score

    @abc.abstractmethod
    def _score_feature_set(self, feature_set: typing.Union[typing.Set[str], typing.Set[hpotk.TermId]]) -> float:
        pass


class WeightedSimilarity(SetSimilarityKernel, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """

    def __init__(self, ic_dict: typing.Mapping[typing.Union[str, hpotk.TermId], float]):
        self._ic_dict = ic_dict

    def _score_feature_set(self, feature_set: typing.Set[typing.Union[str, hpotk.TermId]]) -> float:
        try:
            sim = sum(self._ic_dict.get(term, None) for term in feature_set)
        except TypeError:
            features = [feature for feature in feature_set if self._ic_dict.get(feature, None) is None]
            print(f'Samples share the following features which are not included in the provided dictionary:'
                  f'\n{features}')
            raise
        return sim


class SetSizeSimilarity(SetSimilarityKernel, metaclass=abc.ABCMeta):
    def _score_feature_set(self, feature_set: typing.Union[typing.Set[str], typing.Set[hpotk.TermId]]) -> float:
        return len(feature_set)


class OntoSetSimilarityKernel(SetSimilarityKernel, metaclass=abc.ABCMeta):
    """
    A base class for similarity kernels that find ancestors of terms.
    """

    def __init__(self, hpo: hpotk.GraphAware, root: str = "HP:0000118"):
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

    def compute_from_sets(self, set_a: typing.Set[hpotk.TermId], set_b: typing.Set[hpotk.TermId]) -> SimilarityResult:
        feature_sets = set_a, set_b
        return SimilarityResult(self._score_feature_sets(feature_sets))


class SimilaritiesKernel(metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped):
        self._disease = disease

    @abc.abstractmethod
    def compute(self, sample: Phenotyped, return_last_result: bool = False) -> typing.Union[
        typing.Sequence[float], float]:
        pass


class SetSimilaritiesKernel(SimilaritiesKernel, SetSimilarityKernel, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, root: str = "HP:0000118"):
        SimilaritiesKernel.__init__(self, disease)
        self._hpo = hpo.graph
        self._features_under_root = set(self._hpo.get_descendants(root, include_source=True))
        self._ancestor_dict = {
            feature.value: set(anc.value for anc in self._hpo.get_ancestors(feature, include_source=True)
                               if anc in self._features_under_root)
            for feature in self._features_under_root}
        self._disease_features = set(ancestor for pf in disease.phenotypic_features for ancestor in
                                     self._ancestor_dict[pf.value] if pf in self._features_under_root)

    def _sample_iterator(self, sample: Phenotyped) -> typing.Iterable[typing.Set[hpotk.TermId]]:
        for pf in sample.phenotypic_features:
            yield self._ancestor_dict.get(pf.value, set())

    def compute(self, sample: Phenotyped, return_last_result: bool = False) \
            -> typing.Union[typing.Sequence[float], float]:
        if len(sample.phenotypic_features) == 0:
            raise ValueError("Sample has no phenotypic features.")
        disease_leftovers = self._disease_features.copy()
        union_set = set()
        disease_score = 0.0
        normalization_method = self._normalization_method()
        if normalization_method == "reciprocal average":
            disease_score = self._score_feature_set(self._disease_features)
        elif normalization_method == "union":
            union_set = disease_leftovers.copy()
        intersection_score = 0.0
        union_score = self._score_feature_set(union_set)
        results = []
        for next_set in self._sample_iterator(sample):
            intersection_score_addition = self._score_feature_set(next_set & disease_leftovers)
            disease_leftovers -= next_set
            intersection_score += intersection_score_addition
            if normalization_method == "none":
                results.append(intersection_score)
            else:
                new_union_terms = next_set - union_set
                union_set |= new_union_terms
                union_score += self._score_feature_set(new_union_terms)
                if normalization_method == "union":
                    results.append(intersection_score / union_score)
                elif normalization_method == "reciprocal average":
                    disease_weighted_score = intersection_score / disease_score
                    sample_weighted_score = intersection_score / union_score
                    results.append((disease_weighted_score + sample_weighted_score) / 2)
        if return_last_result:
            return results[-1]
        return results
