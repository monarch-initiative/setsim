import abc
import re

from collections import namedtuple

import hpotk

from setsim.model import Phenotyped
from ._base import OntoSetSimilarityKernel, SetSimilaritiesKernel, SetSizeSimilarity

HPO_PATTERN = re.compile(r"HP:(?P<ID>\d{7})")

SimpleFeature = namedtuple('SimplePhenotypicFeature', field_names=('identifier', 'is_present'))
"""
An implementation detail for Jaccard kernels. 
"""


class JaccardSimilarity(SetSizeSimilarity, metaclass=abc.ABCMeta):
    """
    `SumSimilarity` is a base class for similarity kernels that calculate similarity by summing the similarity
    of all pairs of phenotypic features.
    """

    def _normalization_method(self):
        return "union"


class JaccardSimilarityKernel(OntoSetSimilarityKernel, JaccardSimilarity, metaclass=abc.ABCMeta):
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

    def __init__(self, hpo: hpotk.GraphAware, root: str = "HP:0000118"):
        OntoSetSimilarityKernel.__init__(self, hpo, root)

    @property
    def is_symmetric(self) -> bool:
        # Yes, it is!
        return True


class JaccardSimilaritiesKernel(SetSimilaritiesKernel, JaccardSimilarity, metaclass=abc.ABCMeta):
    def __init__(self, disease: Phenotyped, hpo: hpotk.GraphAware, root: str = "HP:0000118"):
        SetSimilaritiesKernel.__init__(self, disease, hpo, root)
