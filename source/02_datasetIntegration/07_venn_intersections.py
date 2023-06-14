# %%
import sys
sys.path.append("./")
from settings import params, paths
import os
import numpy as np
import pandas as pd
import pyranges as pr
from lib.utils import overlap_utils
from sklearn.preprocessing import LabelEncoder

table = pd.read_csv(paths.outputDir + "intersectIntergPol2.tsv", sep="\t")
# %%
intersect = table.iloc[:, [6,7,8,11,12,13,14,15,16]]
intersect["ReMap CRMs"] = intersect["ReMap CRMs"] >= 10
# Define custom function for replacement
def replace_values(x):
    if pd.isna(x) or x == 0:
        return False
    else:
        return True

# Apply custom function to each element in the DataFrame
df_final = intersect.applymap(replace_values)

# %%
# Tail of gene 5kb
geneTail = pr.read_bed(paths.outputDir + "dist_to_genes/pol2_5000_TES_ext.bed", as_df=True)
df_final["Gene tail"] = False
df_final.loc[geneTail["Name"], "Gene tail"] = True
# %%
# Fully detailled
category_sets = {}
for c in df_final.columns:
    category_sets[c] = set(np.nonzero(df_final[c].values)[0])
import upsetplot
import matplotlib.pyplot as plt
data = upsetplot.from_contents(category_sets)
fig = upsetplot.plot(data, min_subset_size=180, show_percentages=True, orientation="vertical")
plt.savefig(paths.outputDir + "intersections_databases/upset_intersections_full_detail.pdf", 
            bbox_inches="tight")
plt.show()
# %%
# Barplot v2
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(dpi=300)
grps = {"Regulatory" : ("DNase meuleman", "ReMap CRMs"),
        "Promoter-like": ("Encode CCREs PLS", "Encode CCREs H3K4me3", "Fantom 5 TSSs", "LNCipedia Promoter"),
        "Enhancer-like": ("ENCODE STARR-seq", "Encode CCREs ELS", "Fantom 5 Enhancers")}
encodeEnhancer = table["Encode CCREs"].str.find("ELS").values > -0.5
encodePromoter = table["Encode CCREs"].str.find("PLS").values > -0.5
encodeH3K4Me3 = table["Encode CCREs"].str.find("H3K4me3").values > -0.5
df_final["Encode CCREs PLS"] = encodePromoter
df_final["Encode CCREs H3K4me3"] = encodeH3K4Me3
df_final["Encode CCREs ELS"] = encodeEnhancer
i = 0
tick_label = []
yticks = []
palette = iter(sns.color_palette()[1:])
for grp in grps:
    col = next(palette)
    idx = list(grps[grp])
    any_pct = df_final[idx].any(axis=1).mean(axis=None)
    plt.barh(i, any_pct, color=col, edgecolor="k", linewidth=2.0)
    tick_label.append(f"Any {grp}")
    yticks.append(i)
    for cat in idx:
        i -= 1
        pct = df_final[cat].mean(axis=None)
        plt.barh(i, pct, color=col)
        tick_label.append(cat)
        yticks.append(i)
    i -= 2
# Add LNCipedia
col = next(palette)
pct = df_final["LNCipedia"].mean(axis=None)
plt.barh(i, pct, color=col, edgecolor="k", linewidth=2.0)
yticks.append(i)
tick_label.append("LNCipedia transcript body")
plt.yticks(yticks, tick_label)
plt.xlim(0,1)
plt.xlabel("Fraction of RNAP2 consensuses")
plt.savefig(paths.outputDir + "intersections_databases/intersect_databases.pdf", bbox_inches="tight")
# %%
# Barplot splitted
import matplotlib.pyplot as plt
import seaborn as sns

grps = {"Regulatory" : ("DNase meuleman", "ReMap CRMs"),
        "Promoter-like": ("Encode CCREs PLS", "Encode CCREs H3K4me3", "Fantom 5 TSSs", "LNCipedia Promoter"),
        "Enhancer-like": ("ENCODE STARR-seq", "Encode CCREs ELS", "Fantom 5 Enhancers")}
encodeEnhancer = table["Encode CCREs"].str.find("ELS").values > -0.5
encodePromoter = table["Encode CCREs"].str.find("PLS").values > -0.5
encodeH3K4Me3 = table["Encode CCREs"].str.find("H3K4me3").values > -0.5
df_final["Encode CCREs PLS"] = encodePromoter
df_final["Encode CCREs H3K4me3"] = encodeH3K4Me3
df_final["Encode CCREs ELS"] = encodeEnhancer
palette = iter(sns.color_palette()[1:])
fig, ax = plt.subplots(4, 1, dpi=300)
k = 0
for grp in grps:
    i = 0
    tick_label = []
    yticks = []
    col = next(palette)
    idx = list(grps[grp])
    any_pct = df_final[idx].any(axis=1).mean(axis=None)
    ax[k].barh(i, any_pct, color=col, edgecolor="k", linewidth=2.0)
    tick_label.append(f"Any {grp}")
    yticks.append(i)
    for cat in idx:
        i -= 1
        pct = df_final[cat].mean(axis=None)
        ax[k].barh(i, pct, color=col)
        tick_label.append(cat)
        yticks.append(i)
    ax[k].set_yticks(yticks, tick_label)
    k += 1
# Add LNCipedia
i = 0
tick_label = []
yticks = []
col = next(palette)
yticks.append(i)
tick_label.append("LNCipedia transcript body")
pct = df_final["LNCipedia"].mean(axis=None)
ax[k].barh(i, pct, color=col, edgecolor="k", linewidth=2.0)
ax[k].set_yticks(yticks, tick_label)
ax[k].set_xlabel("Fraction of RNAP2 consensuses")
fig.tight_layout()
plt.savefig(paths.outputDir + "intersections_databases/intersect_databases_subplots.pdf", bbox_inches="tight")
# %%
# Simplified categories
simplified = pd.DataFrame(np.zeros(len(table), dtype=bool), 
                          columns=["Unanotated"])
encodeEnhancer = table["Encode CCREs"].str.find("ELS").values > -0.5
encodePromoter = (table["Encode CCREs"].str.find("PLS").values > -0.5) | (table["Encode CCREs"].str.find("H3K4me3").values > -0.5)
simplified["Promoter-like"] = encodePromoter | df_final["Fantom 5 TSSs"] | df_final["LNCipedia Promoter"]
simplified["LNC-like"] = df_final["LNCipedia"] & (~simplified.any(axis=1))
simplified["Enhancer-like"] = (encodeEnhancer | df_final["Fantom 5 Enhancers"] | df_final["ENCODE STARR-seq"]) & (~simplified.any(axis=1))
simplified["Regulatory"] = (df_final["ReMap CRMs"] | df_final["Encode CCREs"]) & (~simplified.any(axis=1))
simplified["Unanotated"] = ~simplified.any(axis=1)
props = simplified.mean(axis=0)
# %%
import seaborn as sns
colors = sns.color_palette()[:4]
donutSize = 0.33
# Create the pie chart with customizations
patches, texts, autotexts = plt.pie(props, labels=props.index, autopct='%1.1f%%', startangle=90,
         textprops={'fontsize': 12, 'weight': 'bold'}, pctdistance= 1.0-donutSize*0.5,
         wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white'})
for i, text in enumerate(texts):
    text.set_color(patches[i].get_facecolor())
centre_circle = plt.Circle((0, 0), 1.0-donutSize, color='white')
plt.gca().add_artist(centre_circle)
plt.savefig(paths.outputDir + "intersections_databases/pie_chart_simplified_regulatory.pdf",
            bbox_inches="tight")
plt.show()
# %%
# Simplified categories
simplified = pd.DataFrame(np.zeros(len(table), dtype=bool), 
                          columns=["Unannotated"])
encodeEnhancer = table["Encode CCREs"].str.find("ELS").values > -0.5
encodePromoter = (table["Encode CCREs"].str.find("PLS").values > -0.5) | (table["Encode CCREs"].str.find("H3K4me3").values > -0.5)
simplified["Promoter-like"] = encodePromoter | df_final["Fantom 5 TSSs"] | df_final["LNCipedia Promoter"]
simplified["LNC-like"] = df_final["LNCipedia"]
simplified["Enhancer-like"] = (encodeEnhancer | df_final["Fantom 5 Enhancers"] | df_final["ENCODE STARR-seq"])
simplified["Regulatory"] = (df_final["ReMap CRMs"] | df_final["Encode CCREs"]) & (~simplified.any(axis=1))
simplified["Gene tail"] = df_final["Gene tail"]
simplified["Unannotated"] = ~simplified.any(axis=1)
props = simplified.mean(axis=0)
category_sets = {}
for c in simplified.columns:
    category_sets[c] = set(np.nonzero(simplified[c].values)[0])
import upsetplot
import matplotlib.pyplot as plt
data = upsetplot.from_contents(category_sets)
fig = upsetplot.plot(data, min_subset_size=180, sort_by="cardinality",
                     sort_categories_by="-cardinality", show_percentages=True)
plt.savefig(paths.outputDir + "intersections_databases/upset_simplified regulatory.pdf",
            bbox_inches="tight")
plt.show()
# %%
props = pd.Series(dict([(cat,len(items)) for cat, items in category_sets.items()]))
props.sort_values(inplace=True, ascending=False)
sns.barplot(x=props.values/len(simplified), y=props.index, color=sns.color_palette()[0])
for i in range(len(props)):
    plt.text(y=i, x=props.values[i]/len(simplified)+0.02, 
             s=f"{np.around(props.values[i]/len(simplified)*100, 1)} %", va="center")
plt.xlim(0,1)
plt.xlabel("Fraction of RNAP2 consensuses")
plt.ylabel("Category")
plt.savefig(paths.outputDir + "intersections_databases/barplot_simplified regulatory_non_excl.pdf",
            bbox_inches="tight")
# %%
consensuses = pr.read_bed(paths.outputDir + "consensuses.bed", as_df=True)
consensuses["Name"] = ["/".join(simplified.columns[r]) for r in simplified.values]
consensuses.to_csv(paths.tempDir + "consensuses_func_annot.tsv", sep="\t", header="None", index=None)
# %%
