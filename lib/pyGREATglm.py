import pandas as pd
import numpy as np
import pyranges as pr
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
from statsmodels.genmod.families.family import Binomial, NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from scipy.sparse import coo_matrix, csr_matrix
from .utils import overlap_utils
from gprofiler import GProfiler
from scipy.stats import  t
import matplotlib.pyplot as plt

def permutationFdrCriticalValue(p_exp, p_perm, alpha=0.05):
    sortedPexp = np.sort(p_exp)
    lenRatio = len(p_perm)/len(p_exp)
    for i in range(sortedPexp.shape[0]):
        pExp_i = sortedPexp[i]
        # Correct fp counts with the number of permutations
        fp = np.sum(p_perm <= pExp_i)/ lenRatio
        tp = i + 1
        fdr_i = fp / (tp + fp)
        print(fdr_i, pExp_i)
        if fdr_i > alpha:
            return pExp_i

            
class regLogicGREAT:
    def __init__(self, upstream, downstream, distal):
        self.upstream = upstream
        self.downstream = downstream
        self.distal = distal

    def __call__(self, txDF):
        # Infered regulatory domain logic
        txDF.sort_values(by=["Chromosome", "Start"], inplace=True)
        regPm = txDF["Start"] - self.upstream * np.sign(txDF["End"]-txDF["Start"])
        regPp = txDF["Start"] + self.downstream * np.sign(txDF["End"]-txDF["Start"])
        gb = txDF.groupby("Chromosome")
        perChr = dict([(x,gb.get_group(x)) for x in gb.groups])
        for c in perChr:
            inIdx = txDF["Chromosome"] == c
            previousReg = np.roll(regPp[inIdx], 1)
            previousReg[0] = 0
            previousReg[-1] = int(1e10)
            nextReg = np.roll(regPm[inIdx], -1)
            nextReg[-1] = int(1e10)
            extendedM = np.maximum(txDF["Start"][inIdx] - self.distal, np.minimum(previousReg, regPm[inIdx]))
            extendedP = np.minimum(txDF["Start"][inIdx] + self.distal, np.maximum(nextReg, regPp[inIdx]))
            txDF.loc[txDF["Chromosome"] == c, "Start"] = extendedM
            txDF.loc[txDF["Chromosome"] == c, "End"] = extendedP
        return txDF

class regLogic1MB:
    def __init__(self, upstream, downstream, distal):
        self.upstream = upstream
        self.downstream = downstream
        self.distal = distal

    def __call__(self, txDF):
        # Infered regulatory domain logic
        txDF["Start"] = np.minimum(txDF["Start"] - self.distal, 0)
        txDF["End"] = txDF["End"] + self.distal
        return txDF

class pyGREAT:
    """
    doc
    """
    def __init__(self, oboFile, geneFile, geneGoFile, gtfGeneCol="gene_name"):
        self.gtfGeneCol = "gene_name"
        # Parse GO terms
        allLines = []
        termId = None
        namespace = None
        name = None
        with open(oboFile) as f:
            for l in f.readlines():
                if l.startswith("[Term]"):
                    if (not termId == None) and (not namespace == None) and (not name == None):
                        allLines.append((termId, namespace, name))
                    termId = None
                    namespace = None
                    name = None
                elif l.startswith("id"):
                    termId = l.rstrip("\n").split(": ")[1]
                elif l.startswith("namespace"):
                    namespace = l.rstrip("\n").split(": ")[1]
                elif l.startswith("name"):
                    name = l.rstrip("\n").split(": ")[1]
        self.df = pd.DataFrame(allLines)
        self.df.columns = ["id", "namespace", "name"]
        # self.df.set_index("id", inplace=True)
        # Read organism GO gene annotations
        goAnnotation = pd.read_csv(geneGoFile, sep="\t", skiprows=41, header=None)
        # Remove NOT associations
        goAnnotation = goAnnotation[np.logical_not(goAnnotation[3].str.startswith("NOT"))]
        goAnnotation = goAnnotation[[2, 4]]
        goAnnotation.dropna(inplace=True)
        goFull = goAnnotation.merge(self.df, left_on=4, right_on="id")
        goFull.drop(4, 1, inplace=True)
        goFull.rename({2:self.gtfGeneCol}, axis=1, inplace=True)
        # Subset by GO class
        gb = goFull.groupby("namespace")
        self.goClasses = dict([(x,gb.get_group(x)) for x in gb.groups])
        # Read gtf file
        gencode = pr.read_gtf(geneFile)
        gencode = gencode.as_df()
        transcripts = gencode[gencode["Feature"] == "gene"].copy()
        del gencode
        transcripts = transcripts[["Chromosome", "Start", "End", gtfGeneCol, "Strand"]]
        # Reverse positions on opposite strand for convenience
        geneInList = np.isin(transcripts[self.gtfGeneCol], np.unique(goAnnotation[2]), assume_unique=True)
        reversedTx = transcripts.copy()[["Chromosome", "Start", "End", self.gtfGeneCol]][geneInList]
        reversedTx["Start"] = transcripts["Start"].where(transcripts["Strand"] == "+", transcripts["End"])
        reversedTx["End"] = transcripts["End"].where(transcripts["Strand"] == "+", transcripts["Start"])
        self.geneRegulatory = regLogicGREAT(5000, 1000, 1000000)(reversedTx)
        # Merge regulatory regions with GO annotations
        self.fused = dict()
        for c in self.goClasses:
            self.fused[c] = self.geneRegulatory.merge(self.goClasses[c], on=self.gtfGeneCol)
            self.fused[c].rename({4:"GO_Term"}, axis=1, inplace=True)
        # Setup gene-GO matrix for clustering
        self.matrices = dict()
        for c in self.goClasses:
            geneFa, genes = pd.factorize(self.goClasses[c][gtfGeneCol])
            goFa, gos = pd.factorize(self.goClasses[c]["name"])
            data = np.ones_like(goFa, dtype="bool")
            mat = coo_matrix((data, (geneFa, goFa)), shape=(len(genes), len(gos))).toarray().T
            self.matrices[c] = pd.DataFrame(mat)
            self.matrices[c].columns = genes
            self.matrices[c].index = gos


    def __getClusters__(self, enrichedGOs, enrichCat):
        pass
    

    def findEnriched(self, query, background=None, clusterize=False, sources=[]):
        enrichs = {}
        regPR = pr.PyRanges(self.geneRegulatory.rename({self.gtfGeneCol:"Name"}, axis=1))
        intersectBg = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(background, useSummit=False))
        intersectQuery = overlap_utils.countOverlapPerCategory(regPR, overlap_utils.dfToPrWorkaround(query, useSummit=False))
        queryCounts = intersectBg.copy() * 0
        queryCounts.loc[intersectQuery.index] = intersectQuery
        obsGenes = np.isin(queryCounts.index, self.matrices["biological_process"].columns)
        queryGenes = np.isin(intersectQuery.index, self.matrices["biological_process"].columns)
        obsMatrix = self.matrices["biological_process"][queryCounts.index[obsGenes]]
        ratios = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]])
        r2 = pd.DataFrame(queryCounts.loc[queryCounts.index[obsGenes]]/intersectBg.loc[queryCounts.index[obsGenes]])
        expected = intersectBg.loc[queryCounts.index[obsGenes]]
        pvals = pd.Series()
        alpha = (np.var(intersectBg.values) - np.mean(intersectBg.values))/np.square(np.mean(intersectBg.values))
        i = 0
        for gos, hasAnnot in (obsMatrix.iterrows()):
            i += 1
            if hasAnnot.loc[intersectQuery.index[queryGenes]].sum() >= 3:
                try:
                    df = pd.DataFrame(np.array([hasAnnot.T.astype(float)*2.0-1.0, np.ones_like(expected)]).T, 
                                    columns=["GS", "Intercept"], index=queryCounts.index[obsGenes])
                    model = sm.GLM(ratios, df, family=NegativeBinomial(alpha=alpha)).fit(disp=0)
                    # Get one sided pvalues
                    pvals[gos] = t(model.df_resid).sf(model.tvalues["GS"])
                except: 
                    print("Failed regression for : ", gos)
                    continue
        qvals = pvals.copy()
        qvals.loc[:] = fdrcorrection(qvals)[1]
        return qvals.sort_values()[qvals < 0.05]

        




    
