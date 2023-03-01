import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from scipy.stats import norm
import pandas as pd 
from statsmodels.stats.multitest import fdrcorrection
import umap
from matplotlib.patches import Patch
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.patches as mpatch
from zepid.graphics import EffectMeasurePlot
import matplotlib

def applyPalette(annot, avail, palettePath, ret_labels=False):
    annotPalette = pd.read_csv(palettePath, sep=",", index_col=0)
    palette = annotPalette.loc[avail][["r","g","b"]]
    colors = annotPalette.loc[annot][["r","g","b"]].values
    if ret_labels:
        return palette.index, palette.values, colors
    return palette.values, colors
    

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

def plotUmapAlpha(points, colors, forceVectorized=False, lims=None):
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
        colorAlpha = np.hstack([colors, np.ones((len(colors),1))])
        # Compute points per pixel for scalability to very large datasets
        alpha = 1.0
        size = 4000
        # Setup plotting window
        if lims == None:
            xLimMin = np.nanmin(points[:, 0])
            xLimMax = np.nanmax(points[:, 0])
            yLimMin = np.nanmin(points[:, 1])
            yLimMax = np.nanmax(points[:, 1])
            windowSize = (xLimMax-xLimMin, yLimMax-yLimMin)
            xLimMin = xLimMin - windowSize[0]*0.05
            xLimMax = xLimMax + windowSize[0]*0.05
            yLimMin = yLimMin - windowSize[1]*0.05
            yLimMax = yLimMax + windowSize[1]*0.05
        else:
            xLimMin = lims[0]
            xLimMax = lims[1]
            yLimMin = lims[2]
            yLimMax = lims[3]
        img = np.zeros((size,size,4), dtype="float32")
        sums = np.zeros((size,size), dtype="float32")
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
                  np.clip(colorAlpha[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept], yCoordInt[kept]), alpha*f[kept])
        # Upper left
        f = 1.0 - np.abs(xCoordInt - xCoord) * np.abs(1 + yCoordInt - yCoord)
        np.add.at(img, 
                  (xCoordInt[kept], 1+yCoordInt[kept]), 
                  np.clip(colorAlpha[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept], 1+yCoordInt[kept]), alpha*f[kept])
        # Bottom Right
        f = 1.0 - np.abs(xCoordInt + 1 - xCoord) * np.abs(yCoordInt - yCoord)
        np.add.at(img, 
                  (xCoordInt[kept]+1, yCoordInt[kept]), 
                  np.clip(colorAlpha[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept]+1, yCoordInt[kept]), alpha*f[kept])
        # Upper Right
        f = 1.0 - np.abs(1+xCoordInt - xCoord) * np.abs(1+yCoordInt - yCoord)
        np.add.at(img, 
                  (xCoordInt[kept]+1, yCoordInt[kept]+1), 
                  np.clip(colorAlpha[kept]*alpha*f[kept][:, None], 0.0, 1.0))
        np.add.at(sums, (xCoordInt[kept]+1, yCoordInt[kept]+1), alpha*f[kept])
        # Gaussian filter (automatically scale width with dataset size)
        sigma = 0.5/np.sqrt(len(points)/1e6)
        mult = 1.5/(norm.pdf(0,0,sigma)/0.4)    # Increase intensity
        img = cv2.GaussianBlur(img, (0,0), sigma, sigma) * mult
        sums = cv2.GaussianBlur(sums, (0,0), sigma, sigma) * mult
        alpha = np.power(sums/(1+sums),0.45)   # Rescale between 0 and 1
        plt.gca().set_aspect(1.0)
        plt.gca().patch.set_alpha(0.0)
        plt.gcf().patch.set_alpha(0.0)
        mat = img / (1e-7+sums[:, :, None])
        mat[:,:,3] = alpha
        plt.imshow(mat, interpolation="lanczos")
    else:
        s0 = 3.5*np.linalg.norm(plt.gcf().get_size_inches())
        plt.scatter(points[:, 0], points[:, 1], s=s0*min(1.0,10/np.sqrt(len(points))),
                    linewidths=0.0, c=np.clip(colors,0.0,1.0))
        xScale = plt.xlim()[1] - plt.xlim()[0]
        yScale = plt.ylim()[1] - plt.ylim()[0]
        plt.gca().set_aspect(xScale/yScale)

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
        img = np.zeros((size,size,3), dtype="float32")
        sums = np.zeros((size,size), dtype="float32")
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
        s0 = 3.5*np.linalg.norm(plt.gcf().get_size_inches())
        plt.scatter(points[:, 0], points[:, 1], s=s0*min(1.0,10/np.sqrt(len(points))),
                    linewidths=0.0, c=np.clip(colors,0.0,1.0))
        xScale = plt.xlim()[1] - plt.xlim()[0]
        yScale = plt.ylim()[1] - plt.ylim()[0]
        plt.gca().set_aspect(xScale/yScale)


def plotHCProps(matrix, labels, matPct=None, annotationPalette=None, rowOrder="umap", colOrder="umap", cmap=None, hq=True, labelsPct=None, rescale=True):
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
    if labelsPct is None:
        labelsPct = labels
    if matPct is None:
        matPct = matrix
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
    # Add sample annotation
    annotations = np.zeros(matrix.shape[1], "int64")
    eq = ["Non annotated"]
    annotations, eq = pd.factorize(labels,
                                    sort=True)
    eq = pd.Index(eq)
    if np.max(annotations) >= 18 and annotationPalette is None:
        print("Warning : Over 18 annotations, using random colors instead of a palette")
    if annotationPalette is None:
        palette, colors = getPalette(annotations)
    else:
        palette, colors = applyPalette(labels, eq, annotationPalette)
    # Big stacked barplot
    annotations2 = eq.get_indexer(labelsPct)
    signalPerCategory = np.zeros((np.max(annotations2)+1, len(matPct)), dtype="float32")
    signalPerAnnot = np.array([np.sum(matPct[:, i == annotations2]) for i in range(np.max(annotations2)+1)])
    for i in np.unique(annotations2):
        signalPerCategory[i, :] = np.sum(matPct[:, annotations2 == i], axis=1) / signalPerAnnot[i]
    signalPerCategory /= np.sum(signalPerCategory, axis=0)
    runningSum = np.zeros(signalPerCategory.shape[1], dtype="float32")
    barPlot = np.zeros((matPct.shape[1], signalPerCategory.shape[1],3), dtype="float32")
    fractCount = signalPerCategory/np.sum(signalPerCategory, axis=0)*matPct.shape[1]
    for i, c in enumerate(fractCount):
        for j, f in enumerate(c):
            positions = np.round([runningSum[j],runningSum[j]+f]).astype("int")
            barPlot[positions[0]:positions[1], j] = palette[i]
            runningSum[j] += f
    barPlot = barPlot[:, consensuses1D]
    # Downsample the bar plot
    rasterRes = (4000, 2000)
    barPlotScale = 0.25
    # barPlotBlur = cv2.GaussianBlur(barPlot, ksize=(0,0), sigmaX=1.0*barPlot.shape[1]/rasterRes[0], sigmaY=1.0/barPlotScale)
    resized = resize(barPlot, (int(rasterRes[1]*barPlotScale), rasterRes[0]), anti_aliasing=hq)
    # Plot
    plt.figure(figsize=(10,90/16), dpi=500)
    plt.imshow(resized, interpolation="lanczos")
    plt.tick_params(
            axis='both', 
            which='both',    
            bottom=False,   
            top=False,         
            left=False,
            labelleft=False,
            labelbottom=False)


def plotHC(matrix, labels, matPct=None, annotationPalette=None, rowOrder="umap", colOrder="umap", cmap=None, hq=True, labelsPct=None, rescale=True):
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
    if labelsPct is None:
        labelsPct = labels
    if matPct is None:
        matPct = matrix
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
    rasterRes = (4000, 2000)
    rasterMat = resize(matrix[consensuses1D][:, samples1D].astype("float32"), (matrix.shape[0], 2000), anti_aliasing=False, order=int(matrix.shape[1]>2000))
    rasterMat = resize(rasterMat, rasterRes, anti_aliasing=hq, order=1)
    if rescale and not rescale == "3SD":
        rasterMat = (rasterMat - np.percentile(rasterMat, 0.5)) / (np.percentile(rasterMat, 99.5) - np.percentile(rasterMat, 0.5))
    if rescale == "3SD":
        rasterMat = np.clip(rasterMat + 3,0,6.0)/6
    # rasterMat = sns.color_palette("viridis", as_cmap=True)(rasterMat.T)[:,:,:3]
    if cmap is None:
        rasterMat = np.repeat(1-rasterMat.T[:,:,None], 3, 2)
    else:
        rasterMat = sns.color_palette(cmap, as_cmap=True)(rasterMat.T)[:,:,:3]
    # Add sample annotation
    annotations = np.zeros(matrix.shape[1], "int64")
    eq = ["Non annotated"]
    annotations, eq = pd.factorize(labels,
                                    sort=True)
    eq = pd.Index(eq)
    if np.max(annotations) >= 18 and annotationPalette is None:
        print("Warning : Over 18 annotations, using random colors instead of a palette")
    if annotationPalette is None:
        palette, colors = getPalette(annotations)
    else:
        palette, colors = applyPalette(labels, eq, annotationPalette)
    # Add sample labels
    sampleLabelCol = np.zeros((matrix.shape[1], 1, 3), dtype="float32")
    for i in range(len(palette)):
        hasAnnot = (annotations[samples1D] == i).nonzero()[0]
        np.add.at(sampleLabelCol, hasAnnot, palette[i])
    sampleLabelCol = resize(sampleLabelCol, (rasterRes[1], int(rasterRes[0]/33.3)), anti_aliasing=hq, order=0)
    # Big stacked barplot
    annotations2 = eq.get_indexer(labelsPct)
    signalPerCategory = np.zeros((np.max(annotations2)+1, len(matPct)), dtype="float32")
    signalPerAnnot = np.array([np.sum(matPct[:, i == annotations2]) for i in range(np.max(annotations2)+1)])
    for i in np.unique(annotations2):
        signalPerCategory[i, :] = np.sum(matPct[:, annotations2 == i], axis=1) / signalPerAnnot[i]
    signalPerCategory /= np.sum(signalPerCategory, axis=0)
    runningSum = np.zeros(signalPerCategory.shape[1], dtype="float32")
    barPlot = np.zeros((matPct.shape[1], signalPerCategory.shape[1],3), dtype="float32")
    fractCount = signalPerCategory/np.sum(signalPerCategory, axis=0)*matPct.shape[1]
    for i, c in enumerate(fractCount):
        for j, f in enumerate(c):
            positions = np.round([runningSum[j],runningSum[j]+f]).astype("int")
            barPlot[positions[0]:positions[1], j] = palette[i]
            runningSum[j] += f
    barPlot = barPlot[:, consensuses1D]
    # Downsample the bar plot
    barPlotScale = 0.25
    # barPlotBlur = cv2.GaussianBlur(barPlot, ksize=(0,0), sigmaX=1.0*barPlot.shape[1]/rasterRes[0], sigmaY=1.0/barPlotScale)
    resized = resize(barPlot, (int(rasterRes[1]*barPlotScale), rasterRes[0]), anti_aliasing=hq)
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


def capTxtLen(txt, maxlen):
    try:
        if len(txt) < maxlen:
            return txt
        else:
            return txt[:maxlen] + '...'
    except:
        return "N/A"

def enrichBarplot(ax, enrichFC, enrichQval, title="", order_by="fc", alpha=0.05, fcMin = 1.0, topK=10, cap=324.0):
    selected = (enrichQval < alpha) & (enrichFC > fcMin)
    if order_by == "fc":
        ordered = enrichFC[selected].sort_values(ascending=False)[:topK]
    elif order_by == "qval":
        ordered = -np.log10(enrichQval[selected]).sort_values(ascending=True)[:topK]
    else:
        print("Wrong order_by")
        return None
    terms = ordered.index
    t = [capTxtLen(term, 50) for term in terms]
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(length=3, width=1.2)
    ax.barh(range(len(terms)), np.minimum(ordered[::-1],cap))
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(t[::-1], fontsize=5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if order_by == "fc":
        ax.set_xlabel("Fold Change", fontsize=8)
    elif order_by == "qval":
        ax.set_xlabel("-log10(Corrected P-value)", fontsize=8)
    ax.set_title(title, fontsize=10)





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
            pctdistance= 1.0-donutSize*0.5, shadow=False, labeldistance=1.1, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })

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
        plt.savefig(params.outputDir + "figures/descrPlots/%s.pdf"%plotTitle)
        plt.savefig(params.outputDir + "figures/descrPlots/%s.png"%plotTitle, dpi=300)
    plt.show()


def plotDeHM(matrix, labels, isDE, resHM=(4096,4096), zClip=2):
    plt.figure(figsize=(10,90/16), dpi=300)
    reordered = matrix
    resized = resize(reordered, (4096, reordered.shape[1]), anti_aliasing=False)
    resized = resize(reordered, resHM, anti_aliasing=True)
    resized = resized + zClip
    coloredResized = sns.color_palette("coolwarm", as_cmap=True)(resized/zClip/2)[:,:,:3]
    deBar = resize(isDE[None, :], (200/(resHM[0]/4096), resHM[1]), anti_aliasing=True)
    deBar = np.repeat(deBar[:,:,None], 3, 2)
    blackBar = np.zeros((15,resHM[1],3))
    coloredResized = np.concatenate([coloredResized, blackBar, deBar], axis=0)
    plt.imshow(coloredResized, interpolation="lanczos")
    plt.yticks([resHM[1] * np.mean(1-labels) + resHM[1]*0.5*np.mean(labels), resHM[1] * np.mean(1-labels) * 0.5, resHM[0] + 15 + 100/(resHM[0]/4096)], 
                [f"{np.sum(labels)} Cancer samples", f"{np.sum(1-labels)} Normal samples", "DE"],
            fontsize=8, rotation=90, va="center")
    plt.xticks([resHM[0]*0.5], 
                [f"{matrix.shape[1]} Consensus Peaks, ranked by mean difference"],
                fontsize=8, ha="center")
    plt.title("Z-scores (capped +- 2SD)")
    plt.gca().set_aspect(9/16)
    palette = sns.color_palette()

def ridgePlot(df):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    cols = df.columns
    # Initialize the FacetGrid object
    pal = sns.color_palette("Paired")
    g = sns.FacetGrid(df, row=cols[1], hue=cols[1], aspect=15, height=.5, palette=pal)
    clipVals = np.percentile(df[cols[0]],(0,99))
    # Draw the densities in a few steps
    g.map(sns.kdeplot, cols[0],
        bw_adjust=.5, clip_on=False,
        fill=True, alpha=1, linewidth=1.5, clip=clipVals)
    g.map(sns.kdeplot, cols[0], clip_on=False, color="w", lw=2, bw_adjust=.5, clip=clipVals)
    
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.2, 0.2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, cols[0])

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

def forestPlot(regDF):
    order = np.argsort(np.abs(regDF["coef"]))[::-1]
    smallDF = regDF.iloc[order]
    p = EffectMeasurePlot(label=list(smallDF.index), effect_measure=smallDF["exp(coef)"].values, 
                        lcl=smallDF["exp(coef) lower 95%"].values, ucl=smallDF["exp(coef) upper 95%"].values)
    p.labels(effectmeasure='HR')
    p.colors(pointshape="D")
    ax=p.plot(figsize=(10,10), min_value=smallDF["exp(coef) lower 95%"].values.min(), 
              max_value=smallDF["exp(coef) upper 95%"].values.max())
    plt.title("Cox regression model",loc="right",x=1, y=1.045)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(np.round(100*np.array([ax.get_xlim()[0], 1.0, ax.get_xlim()[1]]))/100)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Hazard Ratio (log scale)", fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)


def manhattanPlot(coords, chrInfo, pvalues, es, maxLogP=30, threshold="fdr", ylabel="-log10(p-value)",
                  fdrThreshold=0.05):
    fig, ax = plt.subplots(dpi=500)
    fractPos = (chrInfo.values.ravel()/np.sum(chrInfo.values).ravel())
    offsets = np.insert(np.cumsum(fractPos),0,0)
    for i, c in enumerate(chrInfo.index):
        usedIdx = coords.iloc[:, 0] == c
        coordSubset = coords[usedIdx]
        x = offsets[i] + (coordSubset.iloc[:,1]*0.5 + coordSubset.iloc[:,2]*0.5)/chrInfo.loc[c].values * fractPos[i]
        y = np.clip(-np.log10(pvalues[usedIdx]),0, maxLogP)
        ax.scatter(x,y, s=1.0, linewidths=0)
    ax.set_xticks(offsets[:-1]+0.5*fractPos, chrInfo.index, rotation=90, fontsize=8)
    if threshold is not None:
        if threshold == "fdr":
            sortedP = np.sort(pvalues)[::-1]
            fdrSig = np.searchsorted(fdrcorrection(np.sort(pvalues)[::-1], fdrThreshold)[0], True)
            if fdrSig > 0:
                threshold = -np.log10(sortedP[fdrSig])
            else:
                threshold = -np.log10(0.05/len(coords))
        ax.set_xlim(-0.02,1.02)
        ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], color=(0,0,0), linestyles="dashed")
            # plt.text(plt.xlim()[0], threshold*1.1, f"{fdrThreshold} FDR", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax
