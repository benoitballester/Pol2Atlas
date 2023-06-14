# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from lib.peakMerge import peakMerger
from lib.utils import plot_utils, overlap_utils, utils
import numpy as np
from settings import params, paths
import pyranges as pr
import pickle
import os

topK = 5

utils.createDir(paths.outputDir + "per_cluster_enrichs/")

merger = pickle.load(open(paths.outputDir + "merger", "rb"))

enrich_dnase = pd.read_csv(paths.outputDir + "cluster_enrichments/dnaseIndex_Q.tsv", sep="\t")
enrich_dnase.set_index("Unnamed: 0", inplace=True)
enrich_dnase = -np.log10(np.clip(enrich_dnase, 1.0, 1e-300))

enrich_reps = pd.read_csv(paths.outputDir + "cluster_enrichments/repeats_Q.tsv", sep="\t")
enrich_reps.set_index("Unnamed: 0", inplace=True)
enrich_reps = -np.log10(np.clip(enrich_reps, 1.0, 1e-300))

enrich_remap = pd.read_csv(paths.outputDir + "cluster_enrichments/remap_Q.tsv", sep="\t")
enrich_remap.set_index("Unnamed: 0", inplace=True)
enrich_remap = -np.log10(np.clip(enrich_remap, 1.0, 1e-300))
enrich_remap_fc = pd.read_csv(paths.outputDir + "cluster_enrichments/remap_fc.tsv", sep="\t")
enrich_remap_fc.set_index("Unnamed: 0", inplace=True)

annotationDf = pd.read_csv(paths.annotationFile, sep="\t", index_col=0)
annotations, eq = pd.factorize(annotationDf.loc[merger.labels]["Annotation"], sort=True)
palette, colors = plot_utils.applyPalette(annotationDf.loc[merger.labels]["Annotation"], 
                                                eq, paths.polIIannotationPalette)
cluster_labels = pd.read_csv(paths.dataFolder + "dataset_annotation/cluster_annotation.csv",
                             sep="\t", index_col="Cluster")
# %%
lookupTable = pd.read_csv(f"{paths.ldscFilesPath}/LDSC Sumstat Manifest for Neale UKB GWAS - ukb31063_ldsc_sumstat_manifest.tsv", sep="\t")[["phenotype", "description"]]
lookupTable.set_index("phenotype", inplace=True)
lookupTable.index = lookupTable.index.astype(str)

allFiles = os.listdir(paths.outputDir + "ldsc/")
allFiles = [f for f in allFiles if f.endswith(".results")]
results = dict()
for f in allFiles:
    df = pd.read_csv(paths.outputDir + f"ldsc/{f}", sep="\t")
    df["Category"] = df["Category"].str.split("/cluster_",2,True)[1].str.split("L2_0",2,True)[0].astype("str")
    df.set_index("Category", inplace=True)
    cats = df.index
    trait = f.split(".")[0]
    results[np.unique(lookupTable.loc[trait]["description"])[0]] = df
enrichDF = pd.DataFrame()
pvalDF = pd.DataFrame()
for k in results:
    enrichDF[k] = results[k]["Enrichment"]
    pvalDF[k] = results[k]["Enrichment_p"]
qvalDf = pvalDF.copy()*np.prod(pvalDF.shape)
# %%
# Barplot of annotations
for c in np.unique(merger.clustered[0]):
    # Init figure
    fig, axs = plt.subplots(4, 2, figsize=(16, 9))
    inClust = merger.clustered[0] == c
    signalPerCategory = np.zeros(np.max(annotations)+1)
    signalPerAnnot = np.array([np.sum(merger.matrix[:, i == annotations]) for i in range(np.max(annotations)+1)])
    for i in range(np.max(annotations)+1):
        signalPerCategory[i] = np.sum(merger.matrix[inClust][:, annotations == i])/signalPerAnnot[i]
    maxSignal = np.argmax(signalPerCategory)
    normSignal = signalPerCategory/signalPerCategory.sum()
    runningSum = 0
    for j, p in enumerate(normSignal):
        axs[0,0].barh(0, p, left=runningSum, color=palette[j])
        runningSum += p
    axs[0,0].spines['right'].set_visible(False)
    axs[0,0].spines['left'].set_visible(False)
    axs[0,0].spines['top'].set_visible(False)
    axs[0,0].tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    axs[0,0].tick_params(axis="x", labelsize=6)
    axs[0,0].tick_params(length=3., width=1.2)
    axs[0,0].set_xlabel("Fraction of peaks in cluster", fontsize=10)
    axs[0,0].set_aspect(0.1)
    foldChange = signalPerCategory[maxSignal] / np.mean(signalPerCategory[list(range(len(signalPerCategory))).remove(maxSignal)])
    numPeaks = np.sum(inClust)
    axs[0,0].set_title(f'Top system : {eq[maxSignal]} (Fold change : {int(foldChange*100)/100})', fontsize=12, color=palette[maxSignal], fontweight='bold')
    # Normalized biotype origin
    biotypes = np.array([l.split(".")[2].split("_")[0] for l in merger.labels])
    specificity = pd.Series(index=np.unique(biotypes), dtype="float")
    for b in np.unique(biotypes):
        specificity[b] = np.mean(merger.matrix[inClust][:, b==biotypes]) / np.mean(merger.matrix[~inClust][:, b==biotypes])
    specificity = specificity.sort_values(ascending=False)[:topK]
    sns.barplot(y=specificity.index, x=specificity.values, color=sns.color_palette()[0], ax=axs[0,1])
    axs[0,1].set_ylabel("Biosamples", fontweight="bold")
    axs[0,1].set_xlabel("Specificity")
    # DNase index meuleman
    clust_log_Q = enrich_dnase[str(c)]
    clust_log_Q = clust_log_Q[clust_log_Q > -np.log10(0.05)]
    clust_log_Q = clust_log_Q.sort_values(ascending=False)[:topK]
    if len(clust_log_Q) > 0:
        sns.barplot(y=clust_log_Q.index, x=clust_log_Q.values, color=sns.color_palette()[0], ax=axs[1,0])
    axs[1,0].set_ylabel("ENCODE DNase", fontweight="bold")
    axs[1,0].set_xlabel("-log10(FDR)")
    # GO enrichments
    go_enrich = pd.read_csv(paths.outputDir + f"cluster_enrichments/go_enrich_{c}.csv", index_col="Name")
    go_enrich = go_enrich["-log10(qval)"][go_enrich["BH corrected p-value"] < 0.05]
    go_enrich = go_enrich.sort_values(ascending=False)[:topK]
    capped_term = [i if len(i) < 50 else i[:50]+"..." for i in go_enrich.index]
    if len(go_enrich) > 0:
        sns.barplot(y=capped_term, x=go_enrich.values, color=sns.color_palette()[0], ax=axs[1,1])
    axs[1,1].set_ylabel("Nearby genes GO term", fontweight="bold")
    axs[1,1].set_xlabel("-log10(FDR)")
    # HOMER 
    try:
        homer_enrich = pd.read_csv(paths.outputDir + f"homer_motifs/clusters_bed_cluster_{c}/knownResults.txt", sep="\t", index_col="Motif Name")
        homer_enrich.index = [i.split("/")[0].split("(")[0] for i in homer_enrich.index]
        homer_enrich = -homer_enrich["Log P-value"][homer_enrich["q-value (Benjamini)"] < 0.05][:topK]
        if len(homer_enrich) > 0:
            sns.barplot(y=homer_enrich.index, x=homer_enrich.values, color=sns.color_palette()[0], ax=axs[2,0])
    except FileNotFoundError:
        pass
    axs[2,0].set_ylabel("HOMER Motifs", fontweight="bold")
    axs[2,0].set_xlabel("-log10(FDR)")
    # ReMap enrichments
    clust_log_Q = enrich_remap[str(c)]
    clust_fc = enrich_remap_fc[str(c)][(clust_log_Q > -np.log10(0.05)).values]
    clust_log_Q = clust_log_Q[clust_log_Q > -np.log10(0.05)]
    order = np.lexsort([-clust_fc.values,-clust_log_Q.values])
    clust_log_Q = clust_log_Q.iloc[order][:topK]
    if len(clust_log_Q) > 0:
        sns.barplot(y=clust_log_Q.index, x=clust_log_Q.values, color=sns.color_palette()[0], ax=axs[2,1])
    axs[2,1].set_ylabel("ReMap TFs", fontweight="bold")
    axs[2,1].set_xlabel("-log10(FDR)")
    # GWAS enrichments
    if str(c) in qvalDf.index:
        clust_log_Q = -np.log10(qvalDf.loc[str(c)])
        clust_fc = enrichDF.loc[str(c)][(clust_log_Q > -np.log10(0.05)).values]
        clust_fc = clust_fc[clust_fc > 1.0]
        clust_log_Q = clust_fc.sort_values()[::-1][:topK]
        capped_term = [i if len(i) < 50 else i[:50]+"..." for i in clust_log_Q.index]
        if len(clust_log_Q) > 0:
            sns.barplot(y=capped_term, x=clust_log_Q.values, color=sns.color_palette()[0], ax=axs[3,0])
    axs[3,0].set_xlabel("Heritability enrichment")
    axs[3,0].set_ylabel("UK BioBank GWAS traits", fontweight="bold")
    # Repeats enrichments
    clust_log_Q = enrich_reps[str(c)]
    clust_log_Q = clust_log_Q[clust_log_Q > -np.log10(0.05)]
    clust_log_Q = clust_log_Q.sort_values(ascending=False)[:topK]
    if len(clust_log_Q) > 0:
        sns.barplot(y=clust_log_Q.index, x=clust_log_Q.values, color=sns.color_palette()[0], ax=axs[3,1])
    axs[3,1].set_ylabel("Repeats", fontweight="bold")
    axs[3,1].set_xlabel("-log10(FDR)")
    clust_ann = cluster_labels.loc[c]
    fig.suptitle(f"Cluster #{c} ({numPeaks} peaks)\nAnnotation:'{clust_ann[0]}' (confidence:{clust_ann[1]})", fontsize=30, fontweight="bold")
    fig.tight_layout(pad=2.0, w_pad=4.0, h_pad=4.0)
    fig.savefig(paths.outputDir + f"per_cluster_enrichs/cluster_{c}.pdf", bbox_inches="tight")
    plt.close()
# %%
clusters = pd.read_csv(paths.outputDir + "clusterConsensuses_Labels.txt", header=None).values.ravel()
cluster_quality = cluster_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Assuming cluster_quality is your dataframe and clusters is your numpy array

# Convert the clusters numpy array to a pandas Series for easier manipulation
clusters_series = pd.Series(clusters)

# Calculate the sizes of each cluster
cluster_sizes = clusters_series.value_counts().sort_index()

# One-hot encode the 'Annotation confidence' column
one_hot = pd.get_dummies(cluster_quality['Annotation confidence'])
cluster_quality = cluster_quality.join(one_hot)

# Create a new dataframe that combines the information
combined_df = pd.DataFrame({
    'Cluster_Size': cluster_sizes,
    'Low': cluster_quality['Low'],
    'Medium': cluster_quality['Medium'],
    'High': cluster_quality['High']
})

# Count the number of clusters of each quality and sort them
quality_counts = cluster_quality['Annotation confidence'].value_counts()[['High', 'Medium', 'Low']]

# Calculate the total number of observations for each quality level
obs_per_quality = clusters_series.map(cluster_quality['Annotation confidence']).value_counts()[['High', 'Medium', 'Low']]

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 12), sharey="row",
                                              gridspec_kw={'height_ratios': [1, 4, 1], 'width_ratios': [4, 1]})

fig.subplots_adjust(wspace=0.02, hspace=0.02)

# Create a bar plot on the top left subplot summarizing the number of clusters of each quality
ax1.bar(quality_counts.index, quality_counts.values)
ax1.set_xlabel('Cluster annotation Confidence')
ax1.set_ylabel('Number of Clusters')

# Create a heatmap on the middle left subplot
sns.heatmap(combined_df[['High', 'Medium', 'Low']], cmap='YlGnBu', ax=ax3, cbar=False, linewidths=0.5)

# Add a border around the heatmap
heatmap_border = patches.Rectangle((0, 0), 3, len(combined_df), fill=False, edgecolor='black', lw=3)
ax3.add_patch(heatmap_border)
ax3.set_xlabel('Cluster annotation Confidence')
ax3.set_ylabel('Cluster')

# Create a horizontal bar plot on the middle right subplot
ax4.barh(combined_df.index[::-1]+0.5, combined_df['Cluster_Size'][::-1])
ax4.set_xlabel('Cluster Size')
ax4.set_ylabel('')
# To remove spines
for spine in ax4.spines.values():
    spine.set_visible(False)

# To remove ticks
ax4.tick_params(left = False, right = False, labelleft = False)

# Create a bar plot on the bottom left subplot summarizing the number of observations of each quality
ax5.bar(obs_per_quality.index, obs_per_quality.values)
ax5.set_xlabel('Cluster annotation Confidence')
ax5.set_ylabel('Number of consensus peaks')

# Hide the unused top and bottom right subplots
ax2.axis('off')
ax6.axis('off')

fig.tight_layout()  
plt.savefig(paths.outputDir + "per_cluster_enrichs/cluster_qual.pdf")
plt.show()
# %%
