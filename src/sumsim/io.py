import io
import logging
import os
import warnings
from warnings import filterwarnings

import hpotk
import pandas as pd
import typing

from collections import defaultdict
from typing import Sequence, Tuple, List, Any

from google.protobuf.json_format import Parse
from google.protobuf.message import Message
from hpotk import TermId
from hpotk.annotations.load.hpoa import SimpleHpoaDiseaseLoader
from phenopackets import Phenopacket, Cohort

from sumsim.model import Sample, DiseaseModel
from sumsim.model import DiseaseModel
from sumsim.model._base import DiseaseIdentifier

# A generic type for a Protobuf message
MESSAGE = typing.TypeVar('MESSAGE', bound=Message)

logger = logging.getLogger(__name__)


def read_phenopacket(phenopacket: typing.Union[Phenopacket, typing.IO, str], hpo: hpotk.GraphAware) -> Sample:
    """
    Read Phenopacket into a `sumsim.model.Sample`.

    :param phenopacket: a Phenopacket object, path to a phenopacket JSON file, or an IO wrapper.
    :param hpo: Ontology to use to remove ancestors
    :return: the parsed `sumsim.model.Sample`.
    :raises: IOError in case of IO issues or a ValueError if the input is not a proper `Phenopacket`
    """
    if not isinstance(phenopacket, Message):
        phenopacket: Phenopacket = read_protobuf_message(phenopacket, Phenopacket())
    return _parse_phenopacket(phenopacket, hpo)


def _parse_phenopacket(phenopacket: Phenopacket, hpo: hpotk.GraphAware) -> Sample:
    """
    Extract the relevant parts of a `Phenopacket` into `sumsim.model.Sample`. The function uses `subject.id` for
    the `sample.identifier` and the `type.id` and `excluded` attributes of phenopacket's `PhenotypicFeature`s
    for `sample.phenotypic_features`.

    :raises: a `ValueError` if the input is not a `Phenopacket`.
    """
    if not isinstance(phenopacket, Phenopacket):
        raise ValueError(f'Expected an argument with type {Phenopacket} but got {type(phenopacket)}')
    identifier = phenopacket.subject.id
    interpretations = phenopacket.interpretations
    if len(interpretations) < 1:
        diagnosis = None
    else:
        diagnosis = DiseaseIdentifier(disease_id=hpotk.TermId.from_curie(interpretations[0].diagnosis.disease.id),
                                      name=interpretations[0].diagnosis.disease.label)
    phenotypic_features = []
    for feature in phenopacket.phenotypic_features:
        if not feature.excluded:
            term_id = TermId.from_curie(feature.type.id)
            phenotypic_features.append(term_id)
    return Sample(identifier, phenotypic_features, hpo,
                  disease_identifier=diagnosis)


def read_folder(fpath_pp: str, hpo: hpotk.GraphAware, ignore_warnings: bool = False, recursive: bool = False) -> Sequence[Sample]:
    if ignore_warnings:
        filterwarnings("ignore")
    samples = []
    if recursive:
        for root, dirs, files in os.walk(fpath_pp):
            for filename in files:
                if filename.endswith(".json"):
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        try:
                            samples.append(read_phenopacket(file_path, hpo))
                        except Exception:
                            warnings.warn(f"Could not read phenopacket from {file_path}")
    else:
        for filename in os.listdir(fpath_pp):
            if filename.endswith(".json"):
                file_path = os.path.join(fpath_pp, filename)
                if os.path.isfile(file_path):
                    samples.append(read_phenopacket(file_path, hpo))
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


def read_gene_to_phenotype(fpath_g2p: str, hpo: hpotk.GraphAware, root: str = "HP:0000118",
                           return_gene2phe: bool = False,
                           verbose: bool = False) \
        -> typing.Union[typing.Tuple[Sequence[DiseaseModel], typing.Mapping[Any, set]], Sequence[DiseaseModel]]:
    if not verbose:
        filterwarnings("ignore")
    subontology_terms = set(i.value for i in hpo.graph.get_descendants(root, include_source=True))
    df_g2ph = pd.read_csv(fpath_g2p, sep='\t', header=0)
    disease2phe = defaultdict(list)
    gene2phe = defaultdict(set)
    for index, row in df_g2ph.iterrows():
        disease2phe[row['disease_id']].append(row['hpo_id'])
        if return_gene2phe:
            gene2phe[row['gene_symbol']].add(row['hpo_id'])
    diseases = []
    for disease, terms in disease2phe.items():
        list_ids = [TermId.from_curie(term) for term in terms if term in subontology_terms]
        diseases.append(DiseaseModel(TermId.from_curie(disease), "", list_ids, hpo))
    if return_gene2phe:
        return diseases, gene2phe
    return diseases


def read_hpoa(fpath: str, hpo: hpotk.MinimalOntology, root: str = "HP:0000118", include_non_omim=False, include_diseases_without_phenotypes=False) \
        -> Sequence[DiseaseModel]:
    terms_under_root = set(hpo.graph.get_descendants(root, include_source=True))
    annotator = SimpleHpoaDiseaseLoader(hpo)
    disease_annotations = annotator.load(fpath)
    diseases = []
    for annotation in disease_annotations:
        if "OMIM" in annotation.identifier.value or include_non_omim:
            phenotypic_features = [pf.identifier for pf in annotation.annotations if pf.identifier in terms_under_root]
            if len(phenotypic_features) == 0 and not include_diseases_without_phenotypes:
                continue
            new_disease = DiseaseModel(identifier=annotation.identifier, label=annotation.name,
                                       phenotypic_features=phenotypic_features, hpo=hpo)
            diseases.append(new_disease)
    return diseases
