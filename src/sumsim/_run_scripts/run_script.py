# Declare variables
sample_sets = ["ANKH", "ANKRD11", "COL3A1", "ERI1", "EZH1", "FBN1", "FBXL4", "GLI3", "ISCA2", "KDM6B", "MAPK8IP3",
               "MPV17", "OFD1", "POT1", "PPP2R1A", "PTPN11", "RPGRIP1", "SCN2A", "SETD2", "SLC45A2", "SMARCB1",
               "SMARCC2", "SON", "STXBP1", "SUOX", "TRAF7", "WWOX", "ZSWIM6", "LIRICAL"]
methods = ['sumsim', 'phenomizer', 'jaccard']
num_features = 25
null_dist_samples = 100
num_cpus = 10

import sumsim
from hpotk.ontology import MinimalOntology
from hpotk.ontology.load.obographs import load_minimal_ontology

hp: MinimalOntology = load_minimal_ontology('/home/bcoleman/Projects/human-phenotype-ontology/src/ontology/hp.json')

from sumsim.io import read_folder
import os

samples = []
sample_store_path = "/home/bcoleman/Projects/phenopacket-store/notebooks"
for sample in sample_sets:
    if sample == "LIRICAL":
        path = os.path.join(sample_store_path, sample, "v2phenopackets")
    else:
        path = os.path.join(sample_store_path, sample, "phenopackets")
    samples = samples + [sample for sample in read_folder(path, hp) if len(sample.phenotypic_features) > 0]

from sumsim.sim import IcCalculator, IcTransformer

fpath_hpoa = '../data/phenotype.hpoa'
diseases = sumsim.io.read_hpoa(fpath_hpoa, hp)
calc = IcCalculator(hp, multiprocess=True, progress_bar=True)
ic_dict = calc.calculate_ic_from_diseases(diseases)
transformer = IcTransformer(hp)
delta_ic_dict = transformer.transform(ic_dict)

# Create noisy samples
import math
from sumsim.model import Sample
import random
from statistics import mean

common_terms = [term for term in ic_dict.keys() if ic_dict[term] <= math.log(10)]
noisy_samples = []
for sample in samples:
    noisy_features = list(sample.phenotypic_features) + list(random.sample(common_terms, 10))
    noisy_label = sample.label + "_noisy"
    noisy_samples.append(
        Sample(label=noisy_label, phenotypic_features=noisy_features, disease_identifier=sample.disease_identifier,
               hpo=hp))

print(f"There are {len(common_terms)} common terms with IC equal to or less than ln(10) ({math.log(10)}).")
print(
    f'The samples have an average of {mean([len(sample.phenotypic_features) for sample in samples])} terms after removing ancestors and duplicates.')
print(
    f'The noisy samples (with 10 random common terms added) have an average of {mean([len(sample.phenotypic_features) for sample in noisy_samples])} terms after removing ancestors and duplicates.')

# Run Benchmarking
from sumsim.matrix import SimilarityMatrix

b_mark = SimilarityMatrix(hpo=hp,
                          chunksize=10,
                          delta_ic_dict=delta_ic_dict,
                          ic_dict=ic_dict,
                          n_iter_distribution=null_dist_samples,
                          num_cpus=num_cpus,
                          num_features_distribution=num_features,
                          patients=samples + noisy_samples,
                          similarity_methods=methods,
                          multiprocess=True
                          )
results = b_mark.compute_diagnostic_similarities(diseases[:500])  # or whatever slice we want
export_name = f'benchmark_results_m_{methods}_ndist_{null_dist_samples}_nfeatures_{num_features}.csv'
results.to_csv(export_name)
