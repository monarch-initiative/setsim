import abc
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


class DiseaseIdentifier(hpotk.model.Identified, hpotk.model.Named):
    """
    Disease credentials consisting of an identifier and a name.
    """

    def __init__(self, disease_id: hpotk.model.TermId,
                 name: str):
        self._disease_id = hpotk.util.validate_instance(disease_id, hpotk.TermId, 'disease_id')
        self._name = hpotk.util.validate_instance(name, str, 'name')

    @property
    def identifier(self) -> hpotk.model.TermId:
        return self._disease_id

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other):
        return isinstance(other, DiseaseIdentifier) \
            and self.identifier == other.identifier \
            and self.name == other.name

    def __hash__(self):
        return hash((self.identifier, self.name))

    def __str__(self):
        return f"DiseaseIdentifier(identifier={self.identifier}, name={self.name})"

    def __repr__(self):
        return str(self)



class Sample(Phenotyped, Labeled):
    """
    `Sample` describes the requirements for the subject data, as far as C2S2 is concerned.
    """

    def __init__(self, label: str,
                 phenotypic_features: typing.Iterable[hpotk.TermId],
                 disease_identifier: typing.Optional[DiseaseIdentifier] = None):
        self._label = label
        self._pfs = tuple(phenotypic_features)
        self._di = disease_identifier

    @property
    def disease_identifier(self) -> typing.Optional[DiseaseIdentifier]:
        return self._di


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
        return (f'Sample(label="{self.label}", '
                f'n_features={len(self.phenotypic_features)}, '
                f'disease_identifier={self.disease_identifier})')

    def __repr__(self):
        return (f'SimpleSample(label="{self.label}", '
                f'phenotypic_features={self.phenotypic_features}, '
                f'disease_identifier={self.disease_identifier})')

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
