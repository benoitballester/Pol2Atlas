import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from scipy.stats import norm
import pandas as pd 
import umap
from matplotlib.patches import Patch
from skimage.transform import rescale, resize, downscale_local_mean

def applyPalette(annot, avail, palettePath):
    annotPalette = pd.read_csv("annotation_palette.tsv", sep="\t", index_col=0)
    palette = annotPalette.loc[avail][["r","g","b"]].values
    colors = annotPalette.loc[annot][["r","g","b"]].values
    return palette, colors
    

def getPalette(labels, palette=None):
    """
    Selects a proper palette according to the annotation count.
    Uses "paired" from seaborn if the number of labels is under 12.
    Uses "tab20" without grey from seaborn if the number of labels is under 18.
    Uses a quasi-random sequence otherwise.
    Then applies palette according to integer labels

    Parameters
    ----------
    labels: ndarray of integers
        Label for each point.

    Returns
    -------
    palette: ndarray of shape (n_labels,3)
        0-1 rgb color values for each label

    colors: ndarray of shape (n_points,3)
        0-1 rgb color values for each point
    """
    numLabels = np.max(labels)
    if numLabels < 10:
        palette = np.array(sns.color_palette())
        colors = palette[labels]
    elif numLabels < 12:
        palette = np.array(sns.color_palette("Paired"))
        colors = palette[labels]
    elif numLabels < 18:
        # Exclude gray
        palette = sns.color_palette("tab20")
        palette = np.array(palette[:14] + palette[16:])
        colors = palette[labels]
    else:  
        # Too many labels, use random colors
        # Quasi-Random Sequence (has better rgb coverage than random)
        g = 1.22074408460575947536
        v = np.array([1/g, 1/g/g, 1/g/g/g])[:, None]
        palette = np.mod(v * (np.arange(numLabels + 1) + 1), 1.0).T
        colors = palette[labels]
    return palette, colors


def plotUmap(points, colors, forceVectorized=False):
    """
    Custom plotting function. The point size is automatically 
    selected according to the dataset size.
    Above 100 000 points, switches automatically from the vectorized 
    matplotlib backend to per pixel rendering on a 4000x4000 image. 
    Avoids overplotting (and also much faster than matplotlib).

    Parameters
    ----------
    points: ndarray of shape (n,2)
        Positions of the n points to plot.

    colors: array-like of shape (n,)
        The color assigned to each point.

    forceVectorized: Boolean (optional, default False)
        Forces the use of the matplotlib backend.
    """
    if (len(points) > 10000) and not forceVectorized:
        # Compute points per pixel for scalability to very large datasets
        alpha = 1.0
        size = 4000
        # Setup plotting window
        xLimMin = np.nanmin(points[:, 0])
        xLimMax = np.nanmax(points[:, 0])
        yLimMin = np.nanmin(points[:, 1])
        yLimMax = np.nanmax(points[:, 1])
        windowSize = (xLimMax-xLimMin, yLimMax-yLimMin)
        xLimMin = xLimMin - windowSize[0]*0.05
        xLimMax = xLimMax + windowSize[0]*0.05
        yLimMin = yLimMin - windowSize[1]*0.05
        yLimMax = yLimMax + windowSize[1]*0.05
        img = np.zeros((size,size,3))
        sums = np.zeros((size,size))
        # Map points onto pixel grid
        xCoord = ((points[:, 0] - xLimMin)/(xLimMax-xLimMin) * size)
        yCoord = ((points[:, 1] - yLimMin)/(yLimMax-yLimMin) * size)
        xCoordInt = ((points[:, 0] - xLimMin)/(xLimMax-xLimMin) * size).astype(int)
        yCoordInt = ((points[:, 1] - yLimMin)/(yLimMax-yLimMin) * size).astype(int)
        # Remove out of boundaries points
        kept = (xCoord > 0.5) & (xCoord < size-1.5) & (yCoord > 0.5) & (yCoord < size-1.5)
        # Each point = 2d gaussian
        # First bilinear interpolation
        # Bottom left
        f = 1.0 - np.abs(xCoordInt - xCoord) * np.abs(yCoordInt - yCoord)
        np.add.at(img, 
                  (xCoordInt[kept], yCoordInt[kept]), 
                  np.clip(colors[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept], yCoordInt[kept]), alpha*f[kept])
        # Upper left
        f = 1.0 - np.abs(xCoordInt - xCoord) * np.abs(1 + yCoordInt - yCoord)
        np.add.at(img, 
                  (xCoordInt[kept], 1+yCoordInt[kept]), 
                  np.clip(colors[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept], 1+yCoordInt[kept]), alpha*f[kept])
        # Bottom Right
        f = 1.0 - np.abs(xCoordInt + 1 - xCoord) * np.abs(yCoordInt - yCoord)
        np.add.at(img, 
                  (xCoordInt[kept]+1, yCoordInt[kept]), 
                  np.clip(colors[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept]+1, yCoordInt[kept]), alpha*f[kept])
        # Upper Right
        f = 1.0 - np.abs(1+xCoordInt - xCoord) * np.abs(1+yCoordInt - yCoord)
        np.add.at(img, 
                  (xCoordInt[kept]+1, yCoordInt[kept]+1), 
                  np.clip(colors[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept]+1, yCoordInt[kept]+1), alpha*f[kept])
        # Gaussian filter (automatically scale width with dataset size)
        sigma = 0.5/np.sqrt(len(points)/1e6)
        mult = 1.5/(norm.pdf(0,0,sigma)/0.4)    # Increase intensity
        img = cv2.GaussianBlur(img, (0,0), sigma, sigma) * mult
        sums = cv2.GaussianBlur(sums, (0,0), sigma, sigma) * mult
        alpha = np.power(sums/(1+sums),0.45)   # Rescale between 0 and 1
        plt.gca().set_aspect(1.0)
        mat = 1.0 - alpha[:, :, None] + alpha[:, :, None] * img / (1e-7+sums[:, :, None])
        plt.imshow(mat, interpolation="lanczos")
    else:
        plt.scatter(points[:, 0], points[:, 1], s=min(20.0,200/np.sqrt(len(points))),
                    linewidths=0.0, c=np.clip(colors,0.0,1.0))
        xScale = plt.xlim()[1] - plt.xlim()[0]
        yScale = plt.ylim()[1] - plt.ylim()[0]
        plt.gca().set_aspect(xScale/yScale)

def recursiveDownsample(mat, resX, resY):
    matDS = resize(mat.astype(float), (resX, resY), anti_aliasing=True)
    return matDS
        

def plotHC(matrix, labels, annotationFile=None, annotationPalette=None, rowOrder="umap", colOrder="umap"):
    """
    Plot ordered matrix with sample and consensus annotation

    Parameters
    ----------
    matrix: array-like

    rowOrder: "umap" or array-like
        If set to umap, performs UMAP to 1D then orders the rows based on the UMAP position.
        Otherwise uses the supplied order.
    
    colOrder: "umap" or array-like
        If set to umap, performs UMAP to 1D then orders the columns based on the UMAP position.
        Otherwise uses the supplied order.

    """
    if colOrder == "umap":
        consensuses1D = np.argsort(umap.UMAP(n_components=1, n_neighbors=50, min_dist=0.0, 
                                   low_memory=False, metric="dice").fit_transform(matrix).flatten())
    else:
        try:
            consensuses1D = np.array(colOrder).astype(int)
        except:
            raise TypeError("colOrder must be 'umap' or array-like")
    if rowOrder == "umap":
        samples1D = np.argsort(umap.UMAP(n_components=1, n_neighbors=50, min_dist=0.0, 
                               low_memory=False, metric="dice").fit_transform(matrix.T).flatten())
    else:
        try:
            samples1D = np.array(rowOrder).astype(int)
        except:
            raise TypeError("rowOrder must be 'umap' or array-like")
    # Draw pattern-ordered matrix 
    rasterRes = (min(4000, matrix.shape[0]), min(4000, matrix.shape[1]))
    rasterMat = resize(matrix[consensuses1D][:, samples1D].astype(float), rasterRes, anti_aliasing=True)
    rasterMat = (rasterMat - np.min(rasterMat)) / (np.max(rasterMat) - np.min(rasterMat))
    rasterMat = sns.color_palette("viridis", as_cmap=True)(rasterMat.T)[:,:,:3]
    #rasterMat = np.repeat(1-rasterMat.T[:,:,None], 3, 2)
    # Add sample annotation
    annotations = np.zeros(matrix.shape[1], "int64")
    eq = ["Non annotated"]
    if not annotationFile == None:
        annotationDf = pd.read_csv(annotationFile, sep="\t", index_col=0)
        annotations, eq = pd.factorize(annotationDf.loc[labels]["Annotation"],
                                        sort=True)
        if np.max(annotations) >= 18 and annotationPalette is None:
            print("Warning : Over 18 annotations, using random colors instead of a palette")
    annotationPalette = None
    if annotationPalette is None:
        palette, colors = getPalette(annotations)
    else:
        palette, colors = applyPalette(annotationDf.loc[self.labels]["Annotation"], 
                                                    eq, annotationPalette)
    # Add sample labels
    sampleLabelCol = np.zeros((matrix.shape[1], 1, 3))
    counts = np.zeros(rasterRes[1], dtype=int)
    for i in range(len(palette)):
        hasAnnot = (annotations[samples1D] == i).nonzero()[0]
        np.add.at(sampleLabelCol, hasAnnot, palette[i])
        np.add.at(counts, hasAnnot, 1)
    sampleLabelCol /= counts[:, None, None]
    sampleLabelCol = resize(sampleLabelCol, (rasterRes[1], int(rasterRes[0]/33.3)), anti_aliasing=True)
    # Big stacked barplot
    signalPerCategory = np.zeros((np.max(annotations)+1, len(matrix)))
    for i in range(np.max(annotations)+1):
        signalPerCategory[i, :] = np.mean(matrix[:, annotations == i], axis=1)
    runningSum = np.zeros(signalPerCategory.shape[1])
    barPlot = np.zeros((matrix.shape[1], signalPerCategory.shape[1],3))
    fractCount = signalPerCategory/np.sum(signalPerCategory, axis=0)*matrix.shape[1]
    for i, c in enumerate(fractCount):
        for j, f in enumerate(c):
            positions = np.round([runningSum[j],runningSum[j]+f]).astype("int")
            barPlot[positions[0]:positions[1], j] = palette[i]
            runningSum[j] += f
    barPlot = barPlot[:, consensuses1D]
    # Downsample the bar plot
    barPlotScale = 0.25
    # barPlotBlur = cv2.GaussianBlur(barPlot, ksize=(0,0), sigmaX=1.0*barPlot.shape[1]/rasterRes[0], sigmaY=1.0/barPlotScale)
    resized = recursiveDownsample(barPlot, int(rasterRes[1]*barPlotScale), rasterRes[0])
    # Combine everything
    blackPx = 5
    blackOutline = np.zeros((rasterRes[1], blackPx, 3))
    fullMat = np.concatenate((sampleLabelCol, blackOutline, rasterMat), axis=1)
    fill = np.ones((resized.shape[0],int(rasterRes[0]/33.3),3))
    blackOutline = np.zeros((resized.shape[0], blackPx, 3))
    bottom = np.concatenate((fill, blackOutline, resized), axis=1)
    blackOutline = np.zeros((blackPx, fullMat.shape[1], 3))
    figure = np.concatenate((fullMat, blackOutline, bottom), axis=0)
    # Plot
    plt.figure(figsize=(10,90/16), dpi=500)
    plt.imshow(figure, interpolation="lanczos")
    plt.gca().set_aspect(16/9)
    plt.yticks([rasterRes[1]*0.5], 
                [f"{matrix.shape[1]} Experiments"],
            fontsize=8, rotation=90, va="center")
    plt.xticks([int(rasterRes[0]/33.3)+rasterRes[0]*0.5], 
                [f"{matrix.shape[0]} Consensus Peaks"],
                fontsize=8, ha="center")
    patches = []
    for i in range(len(eq)):
        legend = Patch(color=palette[i], label=eq[i])
        patches.append(legend)
    plt.legend(handles=patches, prop={'size': 7}, bbox_to_anchor=(0,1.02,1,0.2),
                loc="lower left", mode="expand", ncol=6)
    


def labelsOnCond(labels, pcts, threshold):
    # Replace string label with empty string if pct is under threshold
    return [label if pct > threshold else "" for pct, label in zip(pcts, labels,)]


def donutPlot(donutSize, counts, nMult, labels, 
             nCenter, centerCaption, palette, fName=None, showPct=True, labelsCutThreshold=0.05, showTitle=True):
    plt.figure(1, (21,9))
    _, label, autopcts = plt.pie(counts, 
            labels=labelsOnCond(labels, counts/np.sum(counts), labelsCutThreshold), 
            colors=palette,
            autopct=lambda p:str(int(p*nMult/100+0.5))+"%"*showPct,
            pctdistance= 1.0-donutSize*0.5, shadow=False, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })

    for i in range(len(label)):
        plt.setp(label[i], **{'color':palette[i], 'weight':'bold', 'fontsize':40})
    plt.setp(autopcts, **{'color':'white', 'weight':'bold', 'fontsize':24})
    #Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),1.0-donutSize,color='black', fc='white',linewidth=0)
    totalTxt = plt.text(0, 0, str(nCenter),
                        {'weight':'bold', 'fontsize':60,
                        'ha':"center", 'va':"center"})
                        
    pol2Txt = plt.text(0, -0.15, centerCaption,
                        {'fontsize':18,
                        'ha':"center", 'va':"center"})
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    if fName is not None:
        plt.savefig("%s.pdf"%fName)
        plt.savefig("%s.png"%fName)
    plt.show()
    plt.close()




palette = sns.color_palette()
def stackedBarplot(counts, labels, plotTitle="",showPct=True, showNinPct=True, showTitle=True, savefig=False, palette=palette):
    plt.figure(figsize=(20,30))
    summed = 0
    tot = np.sum(counts)
    side = 0.1
    plt.ylim((0.0, tot))
    for x, l, i in zip(counts,labels, range(len(counts))):
        plt.bar(0, x, bottom=summed, width=0.1, color=palette[i])
        if showPct: 
            if showNinPct:
                plt.text(0, summed+x/2, str(int(x/tot*100)) + " %" + f" ({x})", 
                        va="center", ha="center",
                        color=(1.0,1.0,1.0), weight='bold', fontsize=40)
            else:
                
                plt.text(0, summed+x/2, str(int(x/tot*100)) + " %", 
                        va="center", ha="center",
                        color=(1.0,1.0,1.0), weight='bold', fontsize=40)
            
        else:
            plt.text(0, summed+x/2, x, 
                    va="center", ha="center",
                    color=(1.0,1.0,1.0), weight='bold', fontsize=40)
        plt.text(side, summed+x/2, l, 
                va="center", ha="center",
                color=palette[i], weight='bold', fontsize=60)
        side *= 1
        summed += x
    plt.axis("off", )
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.title(plotTitle, va="center", ha="center",
              color=(0.05,0.05,0.05), weight='bold', fontsize=60, y=1.05)
    plt.tight_layout()
    if savefig:
        plt.savefig(parameters.outputDir + "figures/descrPlots/%s.pdf"%plotTitle)
        plt.savefig(parameters.outputDir + "figures/descrPlots/%s.png"%plotTitle, dpi=300)
    plt.show()