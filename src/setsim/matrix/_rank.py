import pandas as pd


class Rank:
    def __init__(self, matrix: pd.DataFrame):
        self.columns = matrix.columns.tolist()
        self.sims = [col for col in self.columns if col.endswith("_sim")]
        self.pvals = [col for col in self.columns if col.endswith("_pval")]
        self.methods = list(set(col.split("_")[-2] for col in self.sims))
        self.diseases = [col.split("_" + self.methods[0])[0] for col in self.sims if self.methods[0] in col]
        if len(self.sims) == 0:
            raise ValueError("Matrix does not contain any similarity scores.")
        if len(self.methods) == 0:
            raise ValueError("Matrix does not contain any similarity methods.")
        if len(self.diseases) < 2:
            raise ValueError("Matrix must contain at least 2 diseases.")
        if len(self.diseases) != len(set(self.diseases)):
            raise ValueError("Matrix contains duplicate diseases.")
        self.matrix = matrix
        self.matrix.iloc[:, 0] = self.matrix.iloc[:, 0].astype(str).str.replace(":", "_").str.replace(" ", "")
        keep_index = [True if diagnosis in self.diseases else False for diagnosis in self.matrix.iloc[:, 0]]
        drop_index = [not idx for idx in keep_index]
        if not all(keep_index):
            drop_patients = self.matrix.index[drop_index].tolist()
            keep_patients = self.matrix.index[keep_index].tolist()
            self.missing_diseases = set(self.matrix.iloc[drop_index, 0].tolist())
            print(f"Removing {len(drop_patients)}  patients with a diagnosis not in the matrix.\n"
                  f"Keeping {len(keep_patients)} patients.\n"
                  f"There are {len(self.missing_diseases)} missing diagnoses.\n")
            self.drop_patients = drop_patients
            self.matrix = self.matrix[keep_index]
            self.rankings = self.matrix.iloc[:, :2]

    def rank(self):
        for method in self.methods:
            r_matrix = self.matrix[[col for col in self.sims if method in col]].rank(axis=1, ascending=False)
            r_matrix["disease_id"] = self.matrix.iloc[:, 0]
            self.rankings[f"{method}_sim_rank"] = r_matrix.apply(lambda row: row[f'{row["disease_id"]}_{method}_sim'],
                                                                 axis=1)
            if len(self.pvals) > 0:
                rp_matrix = self.matrix[[col for col in self.pvals if method in col]].rank(axis=1, ascending=True)

                # Columns need to match the names of the similarity columns to be able to add them together.
                rp_matrix.columns = [col.replace("_pval", "_sim") for col in rp_matrix.columns]

                # Adding similarity rank from similarity to break ties. Similarity rank is divided by the total number
                # of diseases + 1 to prevent the rank from changing more than 1.
                rp_matrix = rp_matrix + r_matrix.loc[:, r_matrix.columns != "disease_id"] / (len(self.diseases) + 1)
                rp_matrix = rp_matrix.rank(axis=1, ascending=True)
                rp_matrix["disease_id"] = self.matrix.iloc[:, 0]
                self.rankings[f"{method}_pval_rank"] = rp_matrix.apply(lambda row:
                                                                       row[f'{row["disease_id"]}_{method}_sim'], axis=1)

    def get_graphable(self, suffix_variable: str = None, max_term_num: int = None):
        if suffix_variable is not None:
            self.rankings[suffix_variable] = [True if suffix_variable in idx else False for idx in self.rankings.index]
        if max_term_num is not None:
            temp_df = self.rankings.loc[self.rankings["num_features"] <= max_term_num].iloc[:, 2:]
            print(f"Removing {len(self.rankings) - len(temp_df)} patients with more than {max_term_num} terms.\n"
                  f"Keeping {len(temp_df)} patients.\n")
            graphable = pd.melt(temp_df, id_vars=suffix_variable)
        else:
            graphable = pd.melt(self.rankings.iloc[:, 2:], id_vars=suffix_variable)
        graphable["Test"] = graphable["variable"].str.split("_", expand=True)[1]
        if suffix_variable is not None:
            self.rankings.drop(columns=[suffix_variable], inplace=True)
            graphable["Test"].loc[graphable[suffix_variable]] = [f"{test} & {suffix_variable}" for test in
                                                                 graphable["Test"].loc[graphable[suffix_variable]]]
        graphable["variable"] = graphable["variable"].str.split("_", expand=True)[0]
        graphable["variable"] = graphable["variable"].str.capitalize()
        graphable["variable"] = [algo[:-3] + algo[-3:].upper()
                                 if algo.startswith("Sim") else algo for algo in graphable["variable"]]
        graphable.columns = [suffix_variable.capitalize(), "Similarity Method", "Rank", "Test"]
        return graphable

    def get_rankings(self):
        return self.rankings

    def get_unknown_diagnoses(self):
        return self.missing_diseases

    def get_patients_with_unknown_diagnosis(self):
        return self.drop_patients
