# %%
import plotly.express as px
import numpy as np
import pandas as pd
from settings import params, paths

colors = pd.read_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/GTex/colors.txt", 
                        sep="\t", index_col="tissue_site_detail", dtype={"color_hex":"string"})
colors.drop_duplicates(inplace=True)
# %%
main = colors.drop_duplicates("tissue_site")
main = main[["tissue_site", "color_rgb", "color_hex"]]
main.index = main["tissue_site"]
# %%
fig = px.colors.qualitative.swatches()
fig.show()
# %%
px.colors.qualitative.Plotly
# %%
colormap = dict(zip(main["tissue_site"], main["color_hex"])) 
main["y"] = 1.0
fig = px.bar(main, y="y", x="tissue_site", color="tissue_site", color_discrete_map=colormap, width=1500)
fig.show()
# %%
colormap = dict(zip(colors.index, colors["color_hex"])) 
colors["y"] = 1.0
colors["detailed"] = colors.index
fig = px.bar(colors, y="y", x="detailed", color="detailed", color_discrete_map=colormap, width=1500)
fig.show()
# %%
allAnnots = np.unique(pd.read_csv(paths.annotationFile, 
                        sep="\t")["Annotation"])

# %%
embryo = pd.DataFrame([["Embryonic", "0,0,0", "#000000", 1.0]], index=["Embryonic"], columns=["tissue_site", "color_rgb", "color_hex", "y"])
eye = pd.DataFrame([["Eye", "119,119,255", "#7777FF", 1.0]], index=["Eye"], columns=["tissue_site", "color_rgb", "color_hex", "y"])
bone = pd.DataFrame([["Bone", "0,0,255", "#0000FF", 1.0]], index=["Bone"], columns=["tissue_site", "color_rgb", "color_hex", "y"])
trachea = pd.DataFrame([["Trachea", "153,255,0", "#99FF00", 1.0]], index=["Trachea"], columns=["tissue_site", "color_rgb", "color_hex", "y"])
main = pd.concat([main, embryo, eye, bone, trachea], axis=0)
# %%
colormap = dict(zip(main["tissue_site"], main["color_hex"])) 
fig = px.bar(main, y="y", x="tissue_site", color="tissue_site", color_discrete_map=colormap, width=1500)
fig.show()
# %%
main.to_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/palettes/main_annot.tsv", sep="\t")
oldFmt = main["color_rgb"].str.split(",", expand=True).astype(int)/255
oldFmt.columns = [["r", "g", "b"]]
oldFmt.index.name = "Annotation"
oldFmt.to_csv("/shared/projects/pol2_chipseq/pol2_interg_default/data_clean/palettes/main_annot_fmt2.csv", sep=",")
# %%
