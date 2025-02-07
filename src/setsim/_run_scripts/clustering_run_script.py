# Record paths for resouces collected above
fpath_hpo: str = '/home/bcoleman/Projects/human-phenotype-ontology/src/ontology/hp.json'  # path for hpo.json file
fpath_phenopackets = '/home/bcoleman/Projects/phenopacket-store/notebooks'  # path for phenopackets-store notebooks
fpath_hpoa = '/home/bcoleman/Downloads/phenotype (3).hpoa'  # path for hpo phenotype.hpoa file

from hpotk.ontology.load.obographs import load_minimal_ontology
from hpotk.ontology import MinimalOntology
from setsim.io import read_folder, read_hpoa
import warnings

# Import hpo.json file
hpo: MinimalOntology = load_minimal_ontology(fpath_hpo)

# Import samples. Recursive import finds phenopackets in subfolders.
# read_folder and read_hpoa will generate warnings from phenopackets with redundant terms (ancestors of other included terms)
# and terms that cannot be read. We will filter warnings for this step to silence these.
warnings.filterwarnings('ignore')
samples = read_folder(fpath_phenopackets, hpo, recursive=True)

# Samples with no features should always have a similarity of 0. We will remove them to avoid problems.
samples = [sample for sample in samples if len(sample.phenotypic_features) > 0]

# Import diseases in a similar format to samples (individuals).
diseases = read_hpoa(fpath_hpoa, hpo)

# Turn warnings back to default
warnings.filterwarnings('default')

from setsim.sim import IcCalculator, IcTransformer

# The IcCalculator class object will be used to create a dictionary with term ids and ic values.
# IC values are calculated as the base 10 log of the number of diseases divided by the number of diseases annotated with that term.
# Diseases are annotated with a term when the disease is annotated with that term or any of its descendants.
calc = IcCalculator(hpo, multiprocess=True, progress_bar=True)
ic_dict = calc.calculate_ic_from_diseases(
    diseases)  # alternatively "calculate_ic_from_samples" can be used to calculate ic using feature
# prevalence in samples.


import math
import random
from statistics import mean
from setsim.model import Sample

common_terms = [term for term in ic_dict.keys() if ic_dict[term] <= math.log(10)]
noisy_samples = []
for sample in samples:
    noisy_features = list(sample.phenotypic_features) + list(random.sample(common_terms, 20))
    noisy_label = sample.label + "noisy"
    noisy_samples.append(
        Sample(label=noisy_label, phenotypic_features=noisy_features, disease_identifier=sample.disease_identifier,
               hpo=hpo))

print(f"There are {len(common_terms)} common terms with IC equal to or less than ln(10) ({math.log(10)}).")
print(
    f'The samples have an average of {mean([len(sample.phenotypic_features) for sample in samples])} terms after removing ancestors and duplicates.')
print(
    f'The noisy samples (with 20 random common terms added) have an average of {mean([len(sample.phenotypic_features) for sample in noisy_samples])} terms after removing ancestors and duplicates.')

# Get disease counts
sample_disease_list = [sample.disease_identifier.identifier for sample in samples]
disease_counts = dict()
for disease in sample_disease_list:
    disease_counts[disease] = disease_counts.get(disease, 0) + 1

# Get diseases with 30 pts
disease30p_identifiers = []
for disease, n in disease_counts.items():
    if n >= 30:
        disease30p_identifiers.append(disease)

# Create sets of five diseases with prevalence >30
subset_size = 5
num_sets = 1000
disease_identifier_sets = random_sets = [set(random.sample(disease30p_identifiers, subset_size)) for _ in
                                         range(num_sets)]


# function for calculating cluster true positive rate
def roc_score(pred, standard, cluster_diseases_list):
    disease_list = list(set(standard))
    total_true_pos = 0
    for d in cluster_diseases_list:
        cluster_assignment = [pred[i] for i in range(len(standard)) if standard[i] == d.value]
        assignment_count = Counter(cluster_assignment)
        disease_cluster = max(assignment_count, key=assignment_count.get)
        true_pos = assignment_count[disease_cluster]
        total_true_pos = total_true_pos + true_pos
    total_true_pos = total_true_pos / len(pred)
    return total_true_pos


from setsim.matrix import SimilarityMatrix
from sklearn.cluster import KMeans
from collections import Counter

# Need to change this to include all methods
methods = ['phenomizer', 'count', 'simici', 'jaccard', 'simgic', 'simgci']

cluster_scores = []

for disease_identifiers_set in disease_identifier_sets[:6]:

    ic_diseases = [disease for disease in diseases if disease.identifier not in disease_identifiers_set]
    # not sure if this can made into an else with the above
    cluster_diseases = [disease for disease in diseases if disease.identifier in disease_identifiers_set]

    # Create ic and delta ic dict
    calc = IcCalculator(hpo, multiprocess=True, progress_bar=True)
    ic_dict = calc.calculate_ic_from_diseases(ic_diseases)
    transformer = IcTransformer(hpo)
    delta_ic_dict = transformer.transform(ic_dict)

    for loop_samples, type in zip([samples, noisy_samples], ["normal", "noisy"]):
        # Get samples for clustering
        cluster_samples = []
        for sample in loop_samples:
            if sample.disease_identifier.identifier in disease_identifiers_set:
                cluster_samples.append(sample)

        print(f'There are {len(cluster_samples)} samples selected for clustering.')

        # Run Sim Matrix
        sim_matrix = SimilarityMatrix(hpo=hpo,
                                      chunksize=13,  # Chunksize refers to the number of diseases per "chunk".
                                      # Smaller chunksizes are often optimal because diseases with more features take longer to run.
                                      delta_ic_dict=delta_ic_dict,  # Used for simici and simgci
                                      ic_dict=ic_dict,  # Used for phenomizer and simgic
                                      # bayes_ic_dict=bayes_ic_dict, # Used for phrank
                                      n_iter_distribution=0,  # This is 0 because we aren't calculating p-values
                                      num_features_distribution=1,
                                      num_cpus=15,
                                      patients=cluster_samples,
                                      similarity_methods=methods,
                                      multiprocess=True
                                      )
        similarity_matrices = sim_matrix.compute_person2person_similarities(cluster_samples)

        for method in methods:
            cols = [col for col in similarity_matrices.columns if col.endswith(f"_{method}_sim")]
            kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto").fit(similarity_matrices[cols])
            score = (roc_score(kmeans.fit_predict(similarity_matrices[cols])[:-20],
                               similarity_matrices["disease_id"].iloc[:-20], disease_identifiers_set))
            cluster_scores.append({"method": method, "type": type, "score": score})

        # Save results
import pandas as pd

score_list_df = pd.DataFrame(cluster_scores)
score_list_df.to_csv('score_list_results.csv', index=False)