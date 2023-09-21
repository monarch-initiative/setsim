import abc
import copy
import typing

import hpotk
from hpotk import TermId


class Phenotyped(metaclass=abc.ABCMeta):
    """
    A mixin class for entities that are annotated with phenotypic features.
    """

    @property
    @abc.abstractmethod
    def phenotypic_features(self) -> typing.Sequence[hpotk.TermId]:
        """
        Get/set a sequence of the phenotypic features.
        """
        pass

    @phenotypic_features.setter
    @abc.abstractmethod
    def phenotypic_features(self, value: typing.Iterable[hpotk.TermId]):
        pass


class Labeled(metaclass=abc.ABCMeta):
    """
    A mixin class for entities that are labeled with identifier designed for human consumption.
    """

    @property
    @abc.abstractmethod
    def label(self) -> str:
        pass


class Sample(Phenotyped, Labeled, metaclass=abc.ABCMeta):
    """
    `Sample` describes the requirements for the subject data, as far as C2S2 is concerned.
    """

    @staticmethod
    def from_values(label: str, phenotypic_features: typing.Iterable[hpotk.TermId]):
        """
        Create a `Sample` from the `label` and `phenotypic_features`.

        :param label: a label `str`.
        :param phenotypic_features: an iterable of phenotypic features.
        :return: the new sample.
        """
        return SimpleSample(label, phenotypic_features)

    @abc.abstractmethod
    def __copy__(self):
        """
        Sample must support (shallow) copy to support modification in the perturb step.
        """
        pass

    def __eq__(self, other):
        return isinstance(other, Sample) \
            and self.label == other.label \
            and self.phenotypic_features == other.phenotypic_features

    def __str__(self):
        return f'Sample(label="{self.label}", n_features={len(self.phenotypic_features)})'


class BaseSample(Sample, metaclass=abc.ABCMeta):
    """
    The bare-bones `Sample` implementation.
    """

    def __init__(self, label: str,
                 phenotypic_features: typing.Iterable[hpotk.TermId]):
        self._label = label
        self._pfs = tuple(phenotypic_features)

    @property
    def label(self) -> str:
        return self._label

    @property
    def phenotypic_features(self) -> typing.Sequence[hpotk.TermId]:
        return self._pfs

    @phenotypic_features.setter
    def phenotypic_features(self, value: typing.Iterable[hpotk.TermId]):
        self._pfs = tuple(value)

    def __copy__(self):
        # We pass along the label ref but we carbon copy each phenotypic feature
        return type(self)(self.label, [copy.copy(feature) for feature in self.phenotypic_features])


class SimpleSample(BaseSample):
    def __repr__(self):
        return f'SimpleSample(label="{self.label}", phenotypic_features={self.phenotypic_features})'

    def __str__(self):
        return f'SimpleSample(label={self.label}, n_phenotypic_features={len(self.phenotypic_features)})'


class DiseaseModel(hpotk.model.Identified, Labeled, Phenotyped):


    def __init__(self, hpo_disease: hpotk.annotations.HpoDisease):
        self._hpo_disease = hpo_disease

    @property
    def identifier(self) -> TermId:
        """
        Get disease identifier, e.g. MONDO:1234567 or OMIM:245000
        :return:
        """
        return self._hpo_disease.identifier

    @property
    def label(self) -> str:
        """"""
        return self._hpo_disease.name


    @property
    def phenotypic_features(self) -> typing.Sequence[hpotk.TermId]:
        return tuple(a.identifier for a in self._hpo_disease.present_annotations())
