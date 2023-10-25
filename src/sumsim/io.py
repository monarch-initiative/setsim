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
        if not feature.excluded:
            term_id = TermId.from_curie(feature.type.id)
            phenotypic_features.append(term_id)
    return Sample(identifier, phenotypic_features)


def read_folder(fpath_pp: str) -> typing.Sequence[Sample]:
    samples = []
    for filename in os.listdir(fpath_pp):
        if filename.endswith(".json"):
            file_path = os.path.join(fpath_pp, filename)
            if os.path.isfile(file_path):
                samples.append(read_phenopacket(file_path))
    return samples


def read_protobuf_message(fh: typing.Union[typing.IO, str], message: MESSAGE, encoding: str = 'utf-8') -> MESSAGE:
    """
    Read Protobuf data from ``fh`` and store in the ``message``.

    :param fh: a `str` pointing to a file with JSON representation of the ``message`` or the actual JSON `str`
           or IO wrapper (either text or binary) with the JSON content. File and binary inputs are decoded
           using provided ``encoding``.
    :param message: protobuf object to be filled with the input data
    :param encoding: encoding to parse binary or file inputs.
    :return:
    """
    if isinstance(fh, str):
        if os.path.exists(fh):
            if os.path.isfile(fh):
                logger.debug(f'A file has been provided, decoding using `{encoding}`')
                with open(fh, 'r', encoding=encoding) as handle:
                    return read_protobuf_message(handle, message)
            elif os.path.isdir(fh):
                raise ValueError(f'Cannot read protobuf message from a directory {fh}. Please provide a file instead')
            else:
                raise ValueError(f'Unrecognized `str` input')
        else:
            logger.debug(f'Assuming `fh` is a JSON `str` of a top-level phenopacket schema object, '
                         f'decoding using `{encoding}`')
            return Parse(fh, message)
    elif isinstance(fh, io.TextIOBase):
        return Parse(fh.read(), message)
    elif isinstance(fh, io.BufferedIOBase):
        logger.debug(f'Byte IO handle provided, decoding using `{encoding}`')
        return Parse(fh.read().decode(encoding), message)
    else:
        raise ValueError(f'Expected a path to phenopacket JSON, phenopacket JSON `str`, or an IO wrapper '
                         f'but received {type(fh)}')
