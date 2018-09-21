"""
Script to print the cutouts of the sources in a certain file, 
for example flux_bins2_4.fits.

"""

import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 
import math

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# import seaborn as sns
# sns.set()
# plt.rc('text', usetex=True)

import pyfits as pf

from utils import (angular_dispersion_vectorized_n, distanceOnSphere, load_in
					, rotate_point, PositionAngle)
from utils_orientation import find_orientationNN, find_orientationMG

import difflib

FieldNames = [
	'P11Hetdex12', 'P173+55', 'P21', 'P8Hetdex', 'P30Hetdex06', 'P178+55', 
	'P10Hetdex', 'P218+55', 'P34Hetdex06', 'P7Hetdex11', 'P12Hetdex11', 'P16Hetdex13', 
	'P25Hetdex09', 'P6', 'P169+55', 'P187+55', 'P164+55', 'P4Hetdex16', 'P29Hetdex19', 'P35Hetdex10', 
	'P3Hetdex16', 'P41Hetdex', 'P191+55', 'P26Hetdex03', 'P27Hetdex09', 'P14Hetdex04', 'P38Hetdex07', 
	'P182+55', 'P33Hetdex08', 'P196+55', 'P37Hetdex15', 'P223+55', 'P200+55', 'P206+50', 'P210+47', 
	'P205+55', 'P209+55', 'P42Hetdex07', 'P214+55', 'P211+50', 'P1Hetdex15', 'P206+52',
	'P15Hetdex13', 'P22Hetdex04', 'P19Hetdex17', 'P23Hetdex20', 'P18Hetdex03', 'P39Hetdex19', 'P223+52',
	'P221+47', 'P223+50', 'P219+52', 'P213+47', 'P225+47', 'P217+47', 'P227+50', 'P227+53', 'P219+50'
 ]

def plot_source_finding(NN,file,filename,i):
	"""
	Plots the orientation finding for a single source.

	Arguments:
	NN -- Bool indicating if we want an NN source or MG source
	file -- Fieldname the source is in
	i -- new_Index in the file 

	"""

	if NN:
		# file = 'P3Hetdex16'
		# i = 334
		
		Source_Name, Mosaic_ID = load_in(filename,'Source_Name', 'Mosaic_ID')
		RA, DEC, NN_RA, NN_DEC, NN_dist, Total_flux, E_Total_flux, new_NN_index, Maj = load_in(filename,'RA','DEC','new_NN_RA','new_NN_DEC','new_NN_distance(arcmin)','Total_flux', 'E_Total_flux','new_NN_index','Maj')
		source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'
		head = pf.getheader(source)
		hdulist = pf.open(source)

		sname = Source_Name[i]
		print sname

		find_orientationNN(i,'',RA[i],DEC[i],NN_RA[i],NN_DEC[i],NN_dist[i],Maj[i],(3/60.),plot=True,head=head,hdulist=hdulist)

	else:
		Source_Name, Mosaic_ID = load_in(filename,'Source_Name', 'Mosaic_ID')
		RA, DEC, Maj, Min, Total_flux , E_Total_flux = load_in(filename,'RA','DEC', 'Maj', 'Min', 'Total_flux', 'E_Total_flux')

		multiple_gaussian_indices = (np.where(S_Code == 'M')[0])

		source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'
		head = pf.getheader(source)
		hdulist = pf.open(source)

		sname = Source_Name[i]
		print sname

		find_orientationMG(i,'',RA[i],DEC[i],Maj[i],Min[i],(3/60.),plot=True,head=head,hdulist=hdulist)

def plot_flux_bins2_4():
	filename = '/data1/osinga/figures/statistics/deal_with_overlap/flux_bins2_4.fits'
	tdata = Table(fits.open(filename)[1].data)
	
	MG_index = np.isnan(tdata['new_NN_distance(arcmin)']) 

	for i in range(len(tdata)): # Think about this i and the i in plot_source_finding
		if MG_index[i] == 0:
			NN = True
		else:
			NN = False

		# workaround to get MosaicID since the string is cutoff after 8 characters...
		MosaicID = difflib.get_close_matches(tdata['Mosaic_ID'][i],FieldNames,n=1)[0]
		trying = 1 
		while MosaicID[:8] != tdata['Mosaic_ID'][i]: # check to see if difflib got the right string		
			trying +=1
			MosaicID = difflib.get_close_matches(tdata['Mosaic_ID'][i],FieldNames,n=trying)[trying-1]

		plot_source_finding(NN,MosaicID,filename,i)

plot_flux_bins2_4()