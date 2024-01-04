import re
import typing

from collections import namedtuple

import hpotk

from sumsim.model import Phenotyped
from ._base import SimilarityKernel, SimilarityResult

HPO_PATTERN = re.compile(r"HP:(?P<ID>\d{7})")


SimpleFeature = namedtuple('SimplePhenotypicFeature', field_names=('identifier', 'is_present'))
"""
An implementation detail for Jaccard kernels. 
"""


class JaccardSimilarityKernel(SimilarityKernel):
    """
    `JaccardSimilarityKernel` uses *both* present and excluded phenotypic features to calculate the similarity.

    The kernel prepares induced graphs for each sample by adding ancestors implied by the present terms
    and descendants of the excluded terms and proceeds with applying Jaccard coefficient - the ratio of intersection
    over union.

    If ``exact=True`` then the implied annotations are ignored. In result, the kernel performs exact matching.

    Note that no special penalization is applied if a feature is present in one and excluded in the other.
    The observation status mismatch is accounted as a simple mismatch.

    :param hpo: hpo-toolkit's representation of Human Phenotype Ontology.
    :param exact: `True` if the exact matching should be performed.
    """

    def __init__(self, hpo: hpotk.GraphAware, root: str = "HP:0000118", exact: bool = False):
        self._hpo = hpo.graph
        self._features_under_root = set(self._hpo.get_descendants(root, include_source=True))
        self._exact = hpotk.util.validate_instance(exact, bool, 'exact')

    def compute(self, a: Phenotyped, b: Phenotyped) -> SimilarityResult:
        if len(a.phenotypic_features) == 0 or len(b.phenotypic_features) == 0:
            return SimilarityResult(0.)

        ig_a = self._prepare_induced_graph(a)
        ig_b = self._prepare_induced_graph(b)

        intersection = ig_a.intersection(ig_b)
        union = ig_a.union(ig_b)
        # In case no features exist under root
        if len(intersection) == 0:
            return SimilarityResult(0.)
        return SimilarityResult(len(intersection) / len(union))

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True

    def _prepare_induced_graph(self, p: Phenotyped) -> typing.Set[hpotk.TermId]:
        terms = set()

        for term_id in p.phenotypic_features:
            if self._exact:
                terms.add(term_id)
            else:
                for anc in self._hpo.get_ancestors(term_id, include_source=True):
                    terms.add(anc)

        return terms.intersection(self._features_under_root)
