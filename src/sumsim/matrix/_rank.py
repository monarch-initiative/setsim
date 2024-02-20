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
        self.matrix.iloc[:, 0] = self.matrix.iloc[:, 0].astype(str).str.replace(":", "_")


    def rank(self):
        #print(self.matrix.iloc[:, 0])
        #self.rankings = pd.DataFrame(index=self.matrix.index)
        for method in self.methods:
            r_matrix = self.matrix[[col for col in self.sims if method in col]].rank(axis=1, ascending=False)
            print(r_matrix)


