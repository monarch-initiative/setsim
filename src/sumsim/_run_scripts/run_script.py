import sumsim
from sumsim.io import read_folder
from sumsim.sim import IcCalculator, IcTransformer
from sumsim.matrix import SimilarityMatrix
from sumsim.matrix import Rank
import os
from hpotk.ontology import MinimalOntology
from hpotk.ontology.load.obographs import load_minimal_ontology
import math
from sumsim.model import Sample
import random
from statistics import mean

# Declare variabls
methods = ['phenomizer', 'count', 'simici', 'phrank', 'jaccard', 'simgic', 'simgci']
num_features = 30
null_dist_samples = 0#100000
num_cpus = None

fpath_hpo: str = '/home/bcoleman/Projects/human-phenotype-ontology/src/ontology/hp.json'
fpath_phenopackets = "/home/bcoleman/Projects/phenopacket-store/notebooks"
fpath_hpoa = '/home/bcoleman/Downloads/phenotype.hpoa'
export_result_name = f'benchmark_results_m_{methods}_ndist_{null_dist_samples}_nfeatures_{num_features}.csv'
export_rank_name = f'benchmark_rankings_m_{methods}_ndist_{null_dist_samples}_nfeatures_{num_features}.csv'

hpo: MinimalOntology = load_minimal_ontology(fpath_hpo)

samples = read_folder(fpath_phenopackets, hpo, recursive = True)
samples = [sample for sample in samples if len(sample.phenotypic_features) > 0]

diseases = sumsim.io.read_hpoa(fpath_hpoa, hpo)
calc = IcCalculator(hpo, multiprocess=True, progress_bar=True)
ic_dict = calc.calculate_ic_from_diseases(diseases)
transformer = IcTransformer(hpo, samples=diseases)
delta_ic_dict = transformer.transform(ic_dict)
bayes_ic_dict = transformer.transform(ic_dict, strategy="bayesian")

# Create noisy samples


common_terms = [term for term in ic_dict.keys() if ic_dict[term] <= math.log(10)]
noisy_samples = []
for sample in samples:
    noisy_features = list(sample.phenotypic_features) + list(random.sample(common_terms, 20))
    noisy_label = sample.label + "noisy"
    noisy_samples.append(
        Sample(label=noisy_label, phenotypic_features=noisy_features, disease_identifier=sample.disease_identifier,
               hpo=hpo))

print(f"There are {len(common_terms)} common terms with IC equal to or less than ln(10) ({math.log(10)}).")
print(f'The samples have an average of {mean([len(sample.phenotypic_features) for sample in samples])} terms after removing ancestors and duplicates.')
print(f'The noisy samples (with 20 random common terms added) have an average of {mean([len(sample.phenotypic_features) for sample in noisy_samples])} terms after removing ancestors and duplicates.')

# Run Sim Matrix
sim_matrix = SimilarityMatrix(hpo=hpo,
                              chunksize=10,
                              delta_ic_dict=delta_ic_dict,
                              ic_dict=ic_dict,
                              bayes_ic_dict=bayes_ic_dict,
                              n_iter_distribution=null_dist_samples,
                              num_cpus=num_cpus,
                              num_features_distribution=num_features,
                              patients=samples + noisy_samples,
                              similarity_methods=methods,
                              multiprocess=True
                              )
results = sim_matrix.compute_diagnostic_similarities(diseases)  # or whatever slice we want
results.to_csv(export_result_name)

# Rank the results
rank = Rank(results)
rank.rank()
rankings = rank.get_graphable(suffix_variable="noisy", max_term_num=30)
rankings.to_csv(export_rank_name)
