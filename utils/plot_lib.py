"""
This script contains some plotting functions used in the analysis of mesoscope data
Matthijs Oude Lohuis, 2023-2026, Champalimaud Center
"""

import numpy as np
from operator import itemgetter
from  matplotlib.colors import LinearSegmentedColormap
import matplotlib
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings
from scipy.stats import pearsonr,ttest_rel
import copy
import matplotlib.patches as patches

def set_plot_basic_config():
    plt.rcParams.update({'font.size': 7, 
                        'font.family': 'sans-serif', 
                        'font.sans-serif': 'DejaVu Sans',
                        
                        'xtick.labelsize': 5,
                        'ytick.labelsize': 5, 
                        'ytick.major.pad': 0.8, 
                        'xtick.major.pad': 0.8,
                        'ytick.direction': 'out',     # direction: {in, out, inout}
                        'xtick.direction': 'out',     # direction: {in, out, inout}
                        'xtick.major.width': 0.8,
                        'ytick.major.width': 0.8,
                        'xtick.major.size': 1.5,       # major tick size in points
                        'ytick.major.size': 1.5,

                        'axes.titlesize': 6,
                        'axes.titlepad': 3, 
                        'axes.labelsize': 6,
                        'axes.labelpad': 0.8, 
                        'axes.linewidth': 0.8,     # edge line width
                        
                        'lines.linewidth': 1,
                        'lines.linestyle': '-', 
                        
                        'legend.frameon': False,     # if True, draw the legend on a background patch
                        'legend.fontsize': 5,        # legend.fontsize
                        
                        'figure.dpi': 600,       # figure dots per inch
                        'figure.autolayout': True,
                        'image.aspect': 'auto',
                        #axes.facecolor:     white   # axes background color
                        #axes.edgecolor:     black   # axes edge color
                        #axes.grid:          False   # display grid or not
                        #axes.grid.axis:     both    # which axis the grid should apply to
                        #axes.grid.which:    major   # grid lines at {major, minor, both} ticks
                        #axes.titlelocation: center  # alignment of the title: {left, right, center}
                        #axes.titlesize:     large   # font size of the axes title
                        #axes.titleweight:   normal  # font weight of title
                        #axes.titlecolor:    auto    # color of the axes title, auto falls back to
                                                    # text.color as default value
                     })

def my_savefig(fig,savedir,filename,formats=['png','pdf']):
    for fmt in formats:
        fig.savefig(os.path.join(savedir,filename +  '.' + fmt),format = fmt,dpi=600,bbox_inches='tight',transparent=True)
    # fig.savefig(os.path.join(savedir,filename +  '.png'),format = 'png',dpi=300,bbox_inches='tight',transparent=True)
    # fig.savefig(os.path.join(savedir,filename +  '.pdf'),format = 'pdf',dpi=300,bbox_inches='tight',transparent=True)

def shaded_error(x,y,yerror=None,ax=None,center='mean',error='std',color='black',
                 alpha=0.25,linewidth=2,linestyle='-',label=None):
    x = np.array(x)
    y = np.array(y)

    # if np.ndim(y)==1:
    #     y = y[np.newaxis,:]
    if ax is None:
        ax = plt.gca()
        
    if yerror is None:
        if center=='mean':
            ycenter = np.nanmean(y,axis=0)
        elif center=='median':
            ycenter = np.nanmedian(y,axis=0)
        else:
            print('Unknown error type')

        if error=='std':
            yerror = np.nanstd(y,axis=0)
        elif error=='sem':
            # N = np.shape(y)[0]
            N = np.sum(~np.isnan(y),axis=0)
            yerror = np.nanstd(y,axis=0) / np.sqrt(N)
        # elif error=='ci95':
        else:
            print('Unknown error type')
    else:
        ycenter = y
        yerror = np.array(yerror)

    h, = ax.plot(x,ycenter,color=color,linestyle=linestyle,label=label,linewidth=linewidth)
    ax.fill_between(x, ycenter-yerror, ycenter+yerror,color=color,alpha=alpha)

    return h

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def get_sig_asterisks(pvalue, return_ns=False):
    """
    Return a string of asterisks corresponding to the significance level of the p-value.

    Parameters
    ----------
    pvalue : float
        The p-value of the statistical test.
    return_ns : bool, optional
        If `True`, return "ns" for non-significant results. If `False`, return an empty string.
        Default is `False`.

    Returns
    -------
    asterisks : str
        A string of asterisks corresponding to the significance level of the p-value.
    """
    pvalue_thresholds = np.array([[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [10000, "ns" if return_ns else ""]])
    # Iterate through the thresholds and return the appropriate significance string
    for threshold, asterisks in pvalue_thresholds:
        if pvalue <= float(threshold):
            return asterisks
    # Default return if p-value is greater than 1
    return ""

def round_pval(pvalue, return_ns=False):
    """
    """
    pvalue_thresholds = np.array([[1e-4, "0.0001"], [1e-3, "0.001"], [1e-2, "0.01"], [0.05, "0.05"], [10000, "ns" if return_ns else ""]])
    # Iterate through the thresholds and return the appropriate significance string
    for threshold, asterisks in pvalue_thresholds:
        if pvalue <= float(threshold):
            return asterisks

def ax_nticks(ax, n):
    ax.locator_params(axis='x', nbins=n)
    ax.locator_params(axis='y', nbins=n)

def add_stat_annotation(ax, x1, x2, y, p, h=None,**kwargs):
    """
    Add statistical annotation to plot.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to which the annotation will be added.
    x1 : float
        The x-position of the first group.
    x2 : float
        The x-position of the second group.
    y : float
        The y-position of the bar.
    p : float
        The p-value of the statistical test.
    h : float, optional
        The height of the bar. If None, the height will be set to y/10.

    Notes
    -----
    The function draws a line connecting the two groups and adds a label with
    the p-value of the statistical test. The label is centered above the line.
    """
    if h is None:
        h = y / 10
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
    ax.text((x1 + x2) * .5, y + h, get_sig_asterisks(p, return_ns=True),
            ha='center', va='bottom', **kwargs)

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    if len(c) != len(x) - 1:
        warnings.warn(
            "The c argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping
    lc.set_array(c)

    return ax.add_collection(lc)



def plot_mod_plane(celldata,iplane=0,cellfield='modulation',
                    id_sig=True,id_looped=False,radiuslooped=False,radius=50):    
    cmap = sns.color_palette('bwr',as_cmap=True)
    cmap_lims       = np.max(np.abs(np.nanpercentile(celldata[cellfield], [1, 99])))
    cmap_lims       = np.round(np.array([-cmap_lims,cmap_lims]),1)
    fig,axes        = plt.subplots(1,1,figsize=(3.3,3))
    ax = axes
    idx_plane         = celldata['plane_idx']==iplane
    area = celldata['roi_name'][idx_plane].values[0]
    depth = celldata['depth'][idx_plane].values[0]
    # circlesize = (radius * 600/512)**2
    # circlesize = radius**2
    rad = 5
    # sns.scatterplot(data = celldata[idx_plane],x='yloc',y='xloc',hue_norm=(cmap_lims[0],cmap_lims[1]),
                # hue=celldata[cellfield][idx_plane],ax=ax,palette=cmap,s=35,edgecolor="none")
    if id_sig:
        idx_pos = np.logical_and(celldata['sigmod']==1,idx_plane)
        sns.scatterplot(data = celldata[idx_pos],x='yloc',y='xloc',ax=ax,facecolor=None,s=(rad+2)**2,edgecolor="r",linewidth=1)
        idx_neg = np.logical_and(celldata['sigmod']==-1,idx_plane)
        sns.scatterplot(data = celldata[idx_neg],x='yloc',y='xloc',ax=ax,facecolor=None,s=(rad+2)**2,edgecolor="b",linewidth=1)
    
    sns.scatterplot(data = celldata[idx_plane],x='yloc',y='xloc',hue_norm=(cmap_lims[0],cmap_lims[1]),
                hue=celldata[cellfield][idx_plane],ax=ax,palette=cmap,s=rad**2,edgecolor=None,linewidth=0)
 
    idx_labeled = np.logical_and(celldata['redcell']==1,idx_plane)
    idx_unlabeled = np.logical_and(celldata['redcell']==0,idx_plane)

    if id_looped:
        sns.scatterplot(data = celldata[idx_labeled],x='yloc',y='xloc',hue_norm=(cmap_lims[0],cmap_lims[1]),
                    hue=celldata[cellfield][idx_labeled],ax=ax,palette=cmap,s=rad**2,edgecolor="k",linewidth=1)

        sns.scatterplot(data = celldata[idx_unlabeled],x='yloc',y='xloc',hue_norm=(cmap_lims[0],cmap_lims[1]),
                    hue=celldata[cellfield][idx_unlabeled],ax=ax,palette=cmap,s=rad**2,edgecolor="none")
    
    if radiuslooped:
        for idx in np.where(idx_labeled)[0]:
            # 3. Create Circle Patch
            # The radius parameter corresponds to data units (x units)
            center_x = celldata['xloc'].iloc[idx]
            center_y = celldata['yloc'].iloc[idx]
            # circle = patches.Circle((center_x, center_y), radius, fill=False, edgecolor='k', linewidth=1)
            circle = patches.Circle((center_y, center_x), radius, fill=False, edgecolor='k', linewidth=1)
            ax.add_artist(circle)
            ax.set_aspect(1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])  # Shrink current axis's height by 10% on the bottom
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0,600])
    ax.set_ylim([0,600])
    ax.set_title('%s, plane %d, depth %d (um)' % (area,iplane,depth),fontsize=10)
    ax.set_facecolor("grey")
    ax.invert_yaxis()

    norm = plt.Normalize(cmap_lims[0],cmap_lims[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if np.any(celldata[cellfield][idx_plane]):
        ax.get_legend().remove()
        # Remove the legend and add a colorbar (optional)
        handle = ax.figure.colorbar(sm,ax=ax,pad=0.02,shrink=0.5)
        handle.ax.set_yticks([-cmap_lims[1],0,cmap_lims[1]])
        handle.ax.tick_params(labelsize=7)
        handle.ax.set_ylabel('modulation',fontsize=8)
    plt.tight_layout()

    return fig

def add_corr_results(ax, x,y,pos=[0.2,0.1],fontsize=8):
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    r,p = pearsonr(x[~nas], y[~nas])

    print('Correlation (r=%1.2f): p=%.3f' % (r,p))
    if p<0.05:
        ax.text(pos[0],pos[1],'r=%1.2f\np<0.05' % r,transform=ax.transAxes,ha='center',va='center',fontsize=fontsize,color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')
    else: 
        ax.text(pos[0],pos[1],'p=n.s.',transform=ax.transAxes,ha='center',va='center',fontsize=fontsize,color='k')

def add_paired_ttest_results(ax, x,y,pos=[0.2,0.1],fontsize=8):
    nas = np.logical_or(np.isnan(x), np.isnan(y))
    t,p = ttest_rel(x[~nas], y[~nas])

    print('Paired t-test: p=%.3f' % (p))
    ax.text(pos[0],pos[1],'p<%s' % round_pval(p,return_ns=True),transform=ax.transAxes,ha='center',va='center',fontsize=fontsize,color='k') #ax.text(0.2,0.1,'p<0.05',transform=ax.transAxes,ha='center',va='center',fontsize=10,color='red')

def my_legend_strip(ax):
    leg = ax.get_legend()

    for i,t in enumerate(leg.texts):
        if isinstance(leg.legendHandles[i],matplotlib.lines.Line2D):
            c = leg.legendHandles[i].get_color()
        elif isinstance(leg.legendHandles[i],matplotlib.collections.PathCollection):
            c = leg.legendHandles[i].get_facecolor()
        # c = leg.legendHandles[i].get_facecolor()
        t.set_color(c)
    for handle in leg.legendHandles:
        handle.set_visible(False)
    leg.get_frame().set_visible(False)

def ax_3d_makeup(ax,Xp,lab='PC'):

    nticks = 5
    ax.grid(True)
    ax.set_facecolor('white')
    minperc = 1
    maxperc = 99
    ax.set_xticks(np.linspace(np.percentile(Xp[:,0],minperc),np.percentile(Xp[:,0],maxperc),nticks))
    ax.set_yticks(np.linspace(np.percentile(Xp[:,1],minperc),np.percentile(Xp[:,1],maxperc),nticks))
    ax.set_zticks(np.linspace(np.percentile(Xp[:,2],minperc),np.percentile(Xp[:,2],maxperc),nticks))
    
    ax.set_xlim(np.percentile(Xp[:,0],[minperc,maxperc]))
    ax.set_ylim(np.percentile(Xp[:,1],[minperc,maxperc]))
    ax.set_zlim(np.percentile(Xp[:,2],[minperc,maxperc]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Get rid of colored axes planes, remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlabel('%s1' % lab, fontsize=6)  # give labels to axes
    ax.set_ylabel('%s2' % lab, fontsize=6)
    ax.set_zlabel('%s3' % lab, fontsize=6)


#function to add colorbar for imshow data and axis
def add_colorbar_outside(im,ax):
    fig = ax.get_figure()
    bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    width = 0.01
    eps = 0.01 #margin between plot and colorbar
    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(im, cax=cax)
    return cbar

################################################################
## Series of function that spit out lists of colors for different combinations of 
## areas, protocols, mice, stimuli, etc. 

def get_clr_areas(areas):
    # palette       = {'V1'  : sns.xkcd_rgb['seaweed'],
    #                 'PM' : sns.xkcd_rgb['barney'],
    #                 'AL' : sns.xkcd_rgb['clear blue'],
    #                 'RSP' : sns.xkcd_rgb['orangered']}
    palette       = {
                    'V1'  : '#00CC99',
                    'PM' : '#9933FF',
                    'AL' : sns.xkcd_rgb['clear blue'],
                    'RSP' : sns.xkcd_rgb['orangered']}
    return itemgetter(*areas)(palette)

def sort_areas(areas):
    areas_order = ['V1', 'PM', 'AL', 'RSP']
    areas = [x for x in areas_order if x in areas]
    return areas

def get_clr_area_pairs(areapairs):
    palette       = {'V1-V1'  : sns.xkcd_rgb['seaweed'],
                    'PM-V1' : sns.xkcd_rgb['peacock blue'],
                    'V1-PM' : sns.xkcd_rgb['orangered'],
                    'PM-PM' : sns.xkcd_rgb['barney'],
                    'V1-AL' : sns.xkcd_rgb['scarlet'],
                    'V1-RSP' : sns.xkcd_rgb['grape'],
                    'PM-AL' : sns.xkcd_rgb['cerulean'],
                    'PM-RSP' : sns.xkcd_rgb['violet'],
                    'RSP-AL' : sns.xkcd_rgb['khaki'],
                    ' ' : sns.xkcd_rgb['black']}
    return itemgetter(*areapairs)(palette)

def get_clr_labeled(labeled=['unl','lab']):
    palette       = {'unl': '#818181',
                    'lab' : '#FF4C4D'}
    # clrs            = ['black','red']
    return itemgetter(*labeled)(palette)


def arealabeled_to_figlabels(arealabeled):
    if type(arealabeled) == str or type(arealabeled) == np.str_:
        arealabeled = [arealabeled]
    # arealabeled_fig = np.array(np.shape(arealabeled),dtype=object)
    table       = {'V1unl': "$V1_{ND}$",
        'V1lab' : "$V1_{PM}$",
        'PMunl' : "$PM_{ND}$",
        'PMlab' : "$PM_{V1}$",
        'V1_UNL': "$V1_{ND}$",
        'V1_LAB' : "$V1_{PM}$",
        'PM_UNL' : "$PM_{ND}$",
        'PM_LAB' : "$PM_{V1}$",
        'ALunl':    "$AL_{ND}$",
        'ALlab' :   "$AL_{PM}$",
        'RSPunl' :  "$RSP_{ND}$",
        'RSPlab' :  "$RSP_{PM}$"
        }
    return itemgetter(*arealabeled)(table)

def arealabelpair_to_figlabel(arealabelpairs):
    if type(arealabelpairs) == str or type(arealabelpairs) == np.str_:
        arealabelpairs = [arealabelpairs]
    
    arealabelpair_fig = ['-'.join(arealabeled_to_figlabels([arealabel.split('-')[0],arealabel.split('-')[1]])) for arealabel in arealabelpairs]
    return arealabelpair_fig


def get_clr_labelpairs(pairs):
    palette       = {'unl-unl': sns.xkcd_rgb['grey'],
        'unl-lab' : sns.xkcd_rgb['rose'],
        'lab-unl' : sns.xkcd_rgb['orange'],
        'lab-lab' : sns.xkcd_rgb['red'],
        ' ' : sns.xkcd_rgb['black']}
    # palette       = {'0-0': sns.xkcd_rgb['grey'],
    #     '0-1' : sns.xkcd_rgb['rose'],
    #     '1-0' : sns.xkcd_rgb['rose'],
    #     '1-1' : sns.xkcd_rgb['red']}
    return itemgetter(*pairs)(palette)

def get_clr_area_labeled(area_labeled):
    palette       = {
        'V1unl': '#00CC99',
        'V1lab' : '#006149',
        'PMunl' : '#9933FF',
        'PMlab' : '#4C0099',
        'V1_UNL': '#00CC99',
        'V1_LAB' :  '#006149',
        'PM_UNL' :  '#9933FF',
        'PM_LAB' :  '#4C0099',
        'ALunl': sns.xkcd_rgb['clear blue'],
        'ALlab' : sns.xkcd_rgb['burnt orange'],
        'RSPunl' : sns.xkcd_rgb['sienna'],
        'RSPlab' : sns.xkcd_rgb['crimson']}
    return itemgetter(*area_labeled)(palette)

def get_clr_arealabelpairs(arealabelpairs):
    palette       = {
        'V1lab-V1unl-PMunlL2/3': '#9933FF',
        'V1lab-V1unl-PMlabL2/3': '#4C0099',
        'V1lab-V1unl-PMunlL5':  '#9933FF',
        'V1lab-V1unl-PMlabL5':  '#4C0099',
        'PMlab-PMunl-V1unlL2/3': '#00CC99',
        'PMlab-PMunl-V1labL2/3': '#006149',
        'PMlab-PMunl-V1unlL5':  '#00CC99',
        'PMlab-PMunl-V1labL5':  '#006149',
                    }

    return itemgetter(*arealabelpairs)(palette)

def get_clr_arealayers(arealayers):
    palette       = {
        'PMunlL2/3': '#9933FF',
        'PMlabL2/3': '#4C0099',
        'PMunlL5':  '#9933FF',
        'PMlabL5':  '#4C0099',
        'V1unlL2/3': '#00CC99',
        'V1labL2/3': '#006149',
        'V1unlL5':  '#00CC99',
        'V1labL5':  '#006149',
                    }

    return itemgetter(*arealayers)(palette)

def get_clr_area_low_high():
    c = np.array([["#C489FF", "#4C0099"],  # V1lab-V1unl-PMunl
                   ["#20FFC7", "#006149"]])  # PMlab-PMunl-V1unl
    return c

def get_clr_area_labelpairs(area_labelpairs):
    palette       = {
        # 'V1unl-V1unl': sns.xkcd_rgb['mint'],
        # 'V1unl-V1lab': sns.xkcd_rgb['light green'],
        # 'V1lab-V1lab': sns.xkcd_rgb['chartreuse'],
        'V1unl-V1unl': sns.xkcd_rgb['green'],
        'V1unl-V1lab': sns.xkcd_rgb['dark cyan'],
        'V1lab-V1lab': sns.xkcd_rgb['forest green'],

        'PMunl-PMunl': sns.xkcd_rgb['lilac'],
        'PMunl-PMlab': sns.xkcd_rgb['orchid'],
        'PMlab-PMlab': sns.xkcd_rgb['plum'],
        
        'V1unl-PMunl': sns.xkcd_rgb['tangerine'],
        'V1unl-PMlab': sns.xkcd_rgb['orange brown'],
        'V1lab-PMunl': sns.xkcd_rgb['reddish orange'],
        'V1lab-PMlab': sns.xkcd_rgb['crimson'],
        
        'PMunl-V1unl': sns.xkcd_rgb['dark sea green'],
        'PMunl-V1lab': sns.xkcd_rgb['minty green'],
        'PMlab-V1unl': sns.xkcd_rgb['teal blue'],
        'PMlab-V1lab': sns.xkcd_rgb['true blue'],

        'V1unl-ALunl': sns.xkcd_rgb['teal'],
        'V1lab-ALunl': sns.xkcd_rgb['neon blue'],

        'V1unl-RSPunl': sns.xkcd_rgb['peach'],
        'V1lab-RSPunl': sns.xkcd_rgb['powder blue'],

        'PMunl-ALunl': sns.xkcd_rgb['teal'],
        'PMlab-ALunl': sns.xkcd_rgb['neon blue'],

        'PMunl-RSPunl': sns.xkcd_rgb['peach'],
        'PMlab-RSPunl': sns.xkcd_rgb['powder blue'],
                    }
    
    return itemgetter(*area_labelpairs)(palette)

def get_clr_layerpairs(pairs):
    palette       = {'L2/3-L2/3': sns.xkcd_rgb['teal'],
        'L2/3-L4': sns.xkcd_rgb['neon blue'],
        'L2/3-L5': sns.xkcd_rgb['lilac'],
        'L4-L2/3': sns.xkcd_rgb['peach'],
        'L4-L4': sns.xkcd_rgb['powder blue'],
        'L4-L5': sns.xkcd_rgb['navy'],
        'L5-L2/3': sns.xkcd_rgb['deep purple'],
        'L5-L4': sns.xkcd_rgb['light grey'],
        'L5-L5': sns.xkcd_rgb['royal blue'],
        ' ' : sns.xkcd_rgb['black']}
    return itemgetter(*pairs)(palette)

def get_clr_arealayerpairs(pairs):
    palette       = {
        'V1L2/3-PML2/3': sns.xkcd_rgb['teal'],
        'V1L2/3-PML5': sns.xkcd_rgb['lilac'],
        'V1L5-V1L2/3': sns.xkcd_rgb['forest green'],
        'V1L2/3-V1L5': sns.xkcd_rgb['apple green'],
        'PML5-PML2/3': sns.xkcd_rgb['deep purple'],
        'PML5-V1L2/3': sns.xkcd_rgb['royal blue'],
        'PML2/3-V1L2/3': sns.xkcd_rgb['powder blue'],
        'PML2/3-PML5': sns.xkcd_rgb['navy'],
        ' ' : sns.xkcd_rgb['black']}
    return itemgetter(*pairs)(palette)

def get_clr_recombinase(enzymes):
    palette       = {'non': 'gray',
        'cre' : 'orangered',
        'flp' : 'indianred'}
    return itemgetter(*enzymes)(palette)


def get_clr_animal_id(animal_ids):
    # clrs            = sns.color_palette('inferno', len(signals))

    clrs = sns.color_palette(palette='tab10', n_colors=len(animal_ids))

    # animal_ids  = np.array(['LPE10884', 'LPE11081', 'LPE11086', 'LPE11495', 'LPE11622',
    #    'LPE11623', 'LPE11997', 'LPE11998', 'LPE12013', 'LPE12223',
    #    'LPE12385'], dtype=object)
    
    return clrs
