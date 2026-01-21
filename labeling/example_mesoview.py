"""
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025

This script makes an average image of the mesoview data for each session
2Pram Mesoscope data

"""
#%% Import packages
import os

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from loaddata.get_data_folder import *
import numpy as np
from tifffile import imread
# from utils.twoplib import split_mROIs
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from labeling.label_lib import bleedthrough_correction, estimate_correc_coeff
import time

from utils.plot_lib import *
from utils.imagelib import *
from labeling.tdTom_labeling_cellpose import *

savedir         = 'E:\\OneDrive\\PostDoc\\Figures\\\Affine_FF_vs_FB\\Labeling\\'

#%% Set parameters
#Plotting and parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

# clr_rchan = np.array(ImageColor.getcolor('#ff0040', "RGB")) / 255
# clr_gchan = np.array(ImageColor.getcolor('#00ffbf', "RGB")) / 255
clr_rchan = [1,0,0]
clr_gchan = [0,1,0]

cmred = LinearSegmentedColormap.from_list(
        "Custom", [(0, 0, 0), (1, 0, 0)], N=100)
cmgreen = LinearSegmentedColormap.from_list(
        "Custom", [(0, 0, 0), (0, 1, 0)], N=100)

#%% Loop over all selected animals and folders

animal_id = 'LPE10919'
sessiondate = '2023_12_16'

#%% 
animal_id = 'LPE09830'
sessiondate = '2023_04_10'

#%% 
animal_id = 'LPE09661'
sessiondate = '2023_03_20'

#%% 
animal_id = 'LPE11998'
sessiondate = '2024_04_23'

#%% 
animal_id = 'LPE11622'
sessiondate = '2024_02_22'

#%% 
rawdatadir      = "E:\\Procdata\\OV"
greentif = os.path.join(rawdatadir,animal_id,animal_id + '_' + sessiondate + '_green.tif')
redtif = os.path.join(rawdatadir,animal_id,animal_id + '_' + sessiondate + '_red.tif')


#%%
def estimate_correc_coeff(greendata,reddata):
    # Fit linear regression via least squares with numpy.polyfit
    coeff, offset = np.polyfit(reddata.flatten(),greendata.flatten(), deg=1)
    return coeff

def bleedthrough_correction(data_green,data_red,coeff=None,gain1=0.6,gain2=0.4):
    offset             = np.percentile(data_red.flatten(),5)
    data_green_corr    = data_green - coeff * (data_red-offset)
    return data_green_corr

#%% 
coeff = estimate_correc_coeff(mimg,mimg2)
mimg  = bleedthrough_correction(mimg,mimg2,coeff=coeff)
# mimg_corr  = bleedthrough_correction(mimg,mimg2,coeff=0.4)

mimg    = im_norm8(mimg,min=0,max=100) 
gchan = (mimg - np.min(mimg)) / (np.max(mimg) - np.min(mimg))
plt.imshow(gchan,cmap=cmgreen,vmin=0,vmax=1)

#%% 

mimg = imread(greentif)
mimg2 = imread(redtif)

lowgreenprc = 5 #scaling minimum percentile
uppgreenprc = 99.8 #scaling maximum percentile

lowredprc  = 0 #scaling minimum percentile
uppredprc  = 98 #scaling maximum percentile

mimg    = im_norm(mimg,min=lowgreenprc,max=uppgreenprc) 
mimg2   = im_norm(mimg2,min=lowredprc,max=uppredprc) 

rchan = (mimg2 - np.min(mimg2)) / (np.max(mimg2) - np.min(mimg2))
gchan = (mimg - np.min(mimg)) / (np.max(mimg) - np.min(mimg))

#Flip axes:
gchan = np.rot90(gchan)
rchan = np.rot90(rchan)

gchan = np.flipud(gchan)
rchan = np.flipud(rchan)

fig, axes = plt.subplots(1,3,figsize=(30*cm,10*cm))

axes[0].imshow(gchan,cmap=cmgreen,vmin=0,vmax=1,aspect=1)

axes[1].imshow(rchan,cmap=cmred,vmin=0,vmax=1)

im3 = rchan[:,:,np.newaxis] * clr_rchan + gchan[:,:,np.newaxis] * clr_gchan

axes[2].imshow(im3,vmin=0,vmax=1)

for ax in axes.flatten():
    ax.set_axis_off()
    ax.set_aspect('equal')

fig.suptitle(f'{animal_id}')
plt.tight_layout(rect=[0, 0, 1, 1])

# Save the full figure...
my_savefig(fig,savedir,f'Window_{animal_id}_{sessiondate}_merge', formats = ['png','pdf'])

