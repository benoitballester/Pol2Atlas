# %%
import pandas as pd

# Extension of the genomic context file given (in both sides, bp)
genomicContextExtent = 1000

gencode = pr.read_gtf("/scratch/pdelangen/projet_these/data/annotation/gencode.v38.annotation.gtf")
gencode = gencode.as_df()
# %%
transcripts = gencode[gencode["Feature"] == "transcript"]
tp = dict([(k, x) for k, x in transcripts.groupby("Strand")])
# %%
TSSs = []
bedTSS = []
bedInterg = []
for s in tp.keys():
    if s == "+":
        dfTSSs = pd.DataFrame()
        dfTSSs[0] = tp[s][0]
        dfTSSs[1] = np.maximum(tp[s][3]-genomicContextExtent, 0)
        dfTSSs[2] = tp[s][3] + genomicContextExtent
        dfTSSs[3] = 
        bedTSS.append(dfTSSs)
    else:
        dfTSSs = pd.DataFrame()
        dfTSSs[0] = tp[s][0]
        dfTSSs[1] = np.maximum(tp[s][4]-genomicContextExtent, 0)
        dfTSSs[2] = tp[s][4] + genomicContextExtent
        bedTSS.append(dfTSSs)

# %%
