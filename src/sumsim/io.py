import io
import logging
import os
import typing

from google.protobuf.json_format import Parse
from google.protobuf.message import Message
from hpotk.model import TermId
from phenopackets import Phenopacket, Cohort
from sumsim.model._base import Sample


# A generic type for a Protobuf message
MESSAGE = typing.TypeVar('MESSAGE', bound=Message)

logger = logging.getLogger(__name__)


def read_phenopacket(phenopacket: typing.Union[Phenopacket, typing.IO, str]) -> Sample:
    """
    Read Phenopacket into a `sumsim.model.Sample`.

    :param phenopacket: a Phenopacket object, path to a phenopacket JSON file, or an IO wrapper.
    :return: the parsed `sumsim.model.Sample`.
    :raises: IOError in case of IO issues or a ValueError if the input is not a proper `Phenopacket`
    """
    if not isinstance(phenopacket, Message):
        phenopacket: Phenopacket = read_protobuf_message(phenopacket, Phenopacket())
    return _parse_phenopacket(phenopacket)


def _parse_phenopacket(phenopacket: Phenopacket) -> Sample:
    """
    Extract the relevant parts of a `Phenopacket` into `sumsim.model.Sample`. The function uses `subject.id` for
    the `sample.identifier` and the `type.id` and `excluded` attributes of phenopacket's `PhenotypicFeature`s
    for `sample.phenotypic_features`.

    :raises: a `ValueError` if the input is not a `Phenopacket`.
    """
    if not isinstance(phenopacket, Phenopacket):
        raise ValueError(f'Expected an argument with type {Phenopacket} but got {type(phenopacket)}')
    identifier = phenopacket.subject.id
    phenotypic_features = []
    for feature in phenopacket.phenotypic_features:
        term_id = TermId.from_curie(feature.type.id)
        observed = not feature.excluded
        pf = SimplePhenotypicFeature(term_id, is_present=observed)
        phenotypic_features.append(pf)

    return SimpleSample(identifier, phenotypic_features)


def read_cohort(cohort: typing.Union[Cohort, typing.IO, str]) -> typing.Sequence[Sample]:
    """
    Read Cohort into a `sumsim.model.Sample`.

    :param cohort: a Cohort object, path to a cohort JSON file, or an IO wrapper.
    :return: a sequence of `sumsim.model.Sample`s corresponding to Cohort members.
    :raises: IOError in case of IO issues or a ValueError if the input is not a proper `Cohort`.
    """
    if not isinstance(cohort, Message):
        cohort = read_protobuf_message(cohort, Cohort())
    return _parse_cohort(cohort)

