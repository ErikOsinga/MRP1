import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, join, vstack

from scipy import constants as S
from scipy.spatial import cKDTree
from scipy.stats import norm

from shutil import copy2

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# import seaborn as sns
# sns.set()
# plt.rc('text', usetex=True)

import pyfits as pf

from utils import (angular_dispersion_vectorized_n, distanceOnSphere, load_in
					, rotate_point, PositionAngle, deal_with_overlap)
from utils_orientation import find_orientationNN, find_orientationMG

import difflib

# import treecorr

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


def hist_n(Sn_data,mcdata,n=35):

	# plt.title('Number of nearest neighbours n = '+str(n)+ '\nSignificance Level: '+str(Sn_data['SL'][10]))  
	plt.hist(mcdata['S_'+str(n)],label='Monte Carlo simulations')  
	plt.axvline(Sn_data[n-1], label='Data',color='red')
	plt.xlabel('S_'+str(n))          
	plt.ylabel('count')   
	plt.legend()  
	plt.show()   

def Sn_vs_n(tdata,Sn_mc,Sn_data,filename,ending_n=180):
	"""
	Make a plot of the SL (Significance level) statistic vs n.
	"""

	starting_n = 1 # In the data file the S_1 is the first column
	n_range = range(0,ending_n) # index 0 corresponds to Sn_1 

	all_sn = []
	all_sn_mc = []
	all_sl = [] # (length n list containing SL_1 to SL_80)
	all_std = [] # contains the standard deviations of the MC simulations
	N = len(tdata)
	sigma = (0.33/N)**0.5
	for n in n_range:
		# print 'Now doing n = ', n+starting_n , '...'  
		Sn_mc_n = np.average(Sn_mc['S_'+str(n+starting_n)])  
		Sn_data_n = Sn_data[n]
		print Sn_data_n, Sn_mc_n
		SL = 1 - norm.cdf(   (Sn_data_n - Sn_mc_n) / (sigma)   )
		all_sn.append(Sn_data_n)
		all_sl.append(SL)
		all_sn_mc.append(Sn_mc_n)
		all_std.append(np.std(Sn_mc['S_'+str(n+starting_n)]))
		print SL
	
	Results = Table()
	Results['n'] = range(starting_n,ending_n+1)
	Results['Sn_data'] = all_sn
	Results['SL'] = all_sl
	Results['Sn_mc'] = all_sn_mc
	Results.write('./Statistics_'+filename+'_results.fits',overwrite=True)
	
	fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
	axarr[0].plot(range(starting_n,ending_n+1),np.log(all_sl))
	print np.argmin(all_sl)
	axarr[0].set_title('SL vs n for '+filename)
	axarr[0].set_ylabel('Log_e SL')
	axarr[0].set_xlim(15,80)
	axarr[0].set_ylim(-4,0)
	
	axarr[1].plot(range(starting_n,ending_n+1),all_std)
	axarr[1].set_xlabel('n')
	axarr[1].set_ylabel('sigma')

	plt.savefig('./SL_vs_n_'+filename+'.png')
	plt.show()

def angular_radius_vs_n(tdata,filename,n=180,starting_n=2):
	"""
	Make a plot of the angular separation vs the amount of neighbours n
	"""

	n_range = range(starting_n,n+1) 

	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['position_angle'])

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
	y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
	z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T
	
	#make a KDTree for quick NN searching	
	coordinates_tree = cKDTree(coordinates,leafsize=16)

	# for every source: find n closest neighbours, calculate median dist to furthest
	furthest_nearestN = np.zeros((N,n),dtype='int') # array of shape (N,n) that contains index of furthest_nearestN
	for i in range(N):
		indices = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1] # include source itself
		furthest_nearestN[i] = indices
		# source = [RAs[i],DECs[i]]

	temp = np.arange(0,N).reshape(N,1) - np.zeros(n,dtype='int') # Array e.g.
		# to make sure we calc the distance between				#[[0,0,0,0],
		# the current source and the neighbours					#[[1,1,1,1]]
	distance = distanceOnSphere(RAs[temp],DECs[temp]
						,RAs[furthest_nearestN],DECs[furthest_nearestN])
	
	median = np.median(distance,axis=0)
	# std = np.std(distance,axis=0)

	assert median.shape == (n,) # array of the median of (the distance to the furthest 
								# neighbour n for all sources), for every n. 

	plt.plot(n_range,median[starting_n-1:]) # index 1 corresponds to n=2
	plt.title('Angular radius vs n for ' + filename)
	plt.ylabel('Mean angular radius (deg)')
	plt.xlabel('n')
	# plt.savefig('/data1/osinga/figures/statistics/show/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	plt.savefig('./angular_radius_vs_n_'+filename+'.png',overwrite=True)
	plt.show()
		
def select_on_size(tdata,cutoff=6):
	"""
	Outputs a table that only contains items below a given angular size cutoff
	Cutoff is given in arcseconds. Default = 6 arcseconds
	Selection is done differently based on the source:
		- MG: Semimajor axis has to be > cutoff/2
		- NN: Distance between NN has to be > cutoff
	"""

	cutoff_arcmin = cutoff/60.

	MG_index = np.isnan(tdata['new_NN_distance(arcmin)']) 
	NN_index = np.invert(MG_index)

	interestingMG = tdata[MG_index]['Maj'] > cutoff/2. #arcsec (/2 because Maj is semi-major axis)
	interestingNN = tdata[NN_index]['new_NN_distance(arcmin)'] > cutoff_arcmin

	tdataMG = tdata[MG_index][interestingMG]
	tdataNN = tdata[NN_index][interestingNN]

	tdata_interesting = vstack([tdataNN,tdataMG])

	print 'Selected data has ' + str(len(tdata_interesting)) + ' sources left'

	return tdata_interesting

def biggest_selection():
	"""
	Calculate the Sn statistic for the biggest_selection data
	Produce the Sn vs n plot.
	Produce the median angular radius vs n plot.
	"""

	# open the actual data
	tdata = fits.open('/data1/osinga/data/monte_carlo/biggest_selection.fits')
	tdata = Table(tdata[1].data)
	# open the monte carlo data
	Sn_mc = fits.open('/data1/osinga/data/monte_carlo/results/Sn_monte_carlo_biggest_selection_180.fits')
	Sn_mc = Table(Sn_mc[1].data)
	# an array of shape (1xn) containing S1 to S_n
	Sn_data = angular_dispersion_vectorized_n(tdata,180)

	# calculate Sn_vs_n plot
	Sn_vs_n(tdata,Sn_mc,Sn_data,'biggest_selection2')

	angular_radius_vs_n(tdata,'biggest_selection2',220)

def Size():
	"""
	Calculate the Sn statistic for the Size data (selection on > 30 arcseconds)
	Produce the Sn vs n plot
	Produce the median angular radius vs n plot.
	"""

	# open the monte carlo data
	Sn_mc = fits.open('/data1/osinga/data/monte_carlo/results/Sn_monte_carlo_size.fits')
	Sn_mc = Table(Sn_mc[1].data)

	sdata = select_on_size(tdata,cutoff=30)

	# array containing S_1 to S_80
	Sn_data = angular_dispersion_vectorized_n(sdata)

	Sn_vs_n(sdata,Sn_mc,Sn_data,filename='Size')

	angular_radius_vs_n(sdata,filename='Size')

def FIRSTdata():
	"""
	Calculate the Sn statistic for the FIRST data (Omar Contigiani.)
	Produce the Sn vs n plot
	Produce the median angular radius vs n plot.
	"""

	# open the actual data
	fdata = fits.open('/data1/osinga/data/omar_data/catalog_fixed.fits')
	fdata = Table(fdata[1].data)
	# open the monte carlo data
	Sn_mc = fits.open('/data1/osinga/data/monte_carlo/results/FIRSTdata/Sn_monte_carlo_Firstdata.fits')
	Sn_mc = Table(Sn_mc[1].data)
	# calculate the SN for the actual data
	Sn_data = angular_dispersion_vectorized_n(fdata,80)

	# hist_n(Sn_data,Sn_mc,n=35)

	Sn_vs_n(fdata,Sn_mc,Sn_data,'FIRSTdata',80)

	# angular_radius_vs_n(fdata,'FIRSTdata',80)

def MATCH():
	"""
	Plot the (faulty because different sources) match between First and LOFAR
	"""
	tdata = '/data1/osinga/figures/statistics/flux_bins2/flux_bins2_4.fits'
	tdata = Table(fits.open(tdata)[1].data)

	Sn_mc = '/data1/osinga/figures/statistics/flux_bins2/Sn_monte_carlo_flux_bins2_4.fits'
	Sn_mc = Table(fits.open(Sn_mc)[1].data)

	Sn_data = angular_dispersion_vectorized_n(tdata,180)

	hist_n(Sn_data,Sn_mc,n=80)


# biggest_selection()
# Size()
# FIRSTdata()
# MATCH()

def plot_source_finding(NN,file,source_name,plot=False):
	"""
	Plots the orientation finding for a single source.

	Arguments:
	NN -- Bool indicating if we want an NN source or MG source
	file -- Fieldname the source is in
	source_name -- name of the source in the Table 
	plot -- whether to plot. 

	"""

	if NN:
		# file = 'P3Hetdex16'
		# i = 334
		
		prefix = '/data1/osinga/data/NN/'
		Source_Data = prefix+file+'NearestNeighbours_efficient_spherical2.fits'
		Source_Name, Mosaic_ID = load_in(Source_Data,'Source_Name', 'Mosaic_ID')
		RA, DEC, NN_RA, NN_DEC, NN_dist, Total_flux, E_Total_flux, new_NN_index, Maj = load_in(Source_Data,'RA','DEC','new_NN_RA','new_NN_DEC','new_NN_distance(arcmin)','Total_flux', 'E_Total_flux','new_NN_index','Maj')
		source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'
		head = pf.getheader(source)
		hdulist = pf.open(source)

		i = np.where(Source_Name == source_name)[0][0]

		print Source_Name[i]

		try: 
			error = find_orientationNN(i,'',RA[i],DEC[i],NN_RA[i],NN_DEC[i],NN_dist[i],Maj[i],(3/60.),plot=plot,head=head,hdulist=hdulist)[-1]
		except UnboundLocalError:
			error = 10e6

	else:
		prefix = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/CATALOG-DISTRIBUTED-24072017/'
		Source_Data = prefix+file+'cat.srl.fits'
		Source_Name, S_Code, Mosaic_ID = load_in(Source_Data,'Source_Name', 'S_Code', 'Mosaic_ID')
		RA, DEC, Maj, Min, Total_flux , E_Total_flux = load_in(Source_Data,'RA','DEC', 'Maj', 'Min', 'Total_flux', 'E_Total_flux')

		multiple_gaussian_indices = (np.where(S_Code == 'M')[0])

		source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'
		head = pf.getheader(source)
		hdulist = pf.open(source)

		i = np.where(Source_Name == source_name)[0][0]

		print Source_Name[i]
		try:
			error = find_orientationMG(i,'',RA[i],DEC[i],Maj[i],Min[i],(3/60.),plot=plot,head=head,hdulist=hdulist)[-1]
		except UnboundLocalError: # means only 1 maximum in the min and max orientation angle
			error = 10e6

	return error

def setup_source_finding(source_name):
	"""
	This function is used to plot source orientation of a given source_name

	Arguments:
	source_name -- The name of the source that will be plot

	"""

	tdata = Table(fits.open('/data1/osinga/data/monte_carlo/biggest_selection.fits')[1].data)
	tdata = deal_with_overlap(tdata)
	source = np.where(tdata['Source_Name'] == source_name)[0]
	if len(source) != 1:
		print len(source)
		raise ValueError ("This source was not defined anymore in tdata")
	i = source[0]
	source = tdata[i]
	# workaround since the string is cutoff after 8 characters...
	MosaicID = difflib.get_close_matches(tdata['Mosaic_ID'][i],FieldNames,n=1)[0]
	# check to see if difflib got the right string		
	trying = 1
	while MosaicID[:8] != tdata['Mosaic_ID'][i]:
		trying +=1
		MosaicID = difflib.get_close_matches(tdata['Mosaic_ID'][i],FieldNames,n=trying)[trying-1]

	NN = np.isnan(source['new_NN_distance(arcmin)'])^1

	error = plot_source_finding(NN,MosaicID,source_name,False)

def bins():
	"""
	Plot the angular size distribution for the flux bins and the flux distribution
	for the angular size bins.
	"""

	flux_bins = dict()
	size_bins = dict()
	for i in range(5):
		filename = './flux_bins2_'+str(i)+'.fits'
		flux_bins[i] = Table(fits.open(filename)[1].data)
		filename = './size_bins2_'+str(i)+'.fits'
		size_bins[i] = Table(fits.open(filename)[1].data)

	for i in range(5):
		# the Flux bins
		print ('Flux bin %i'%i)
		tdata = flux_bins[i]

		MG_index = np.isnan(tdata['new_NN_distance(arcmin)']) 
		NN_index = np.invert(MG_index)
		NNsize = tdata[MG_index]['Maj'] * 2  #arcsec (*2 because Maj is semi-major axis)
		MGsize = tdata[NN_index]['new_NN_distance(arcmin)'] * 60. # arcsec
		all_sizes = np.append(NNsize,MGsize)


		plt.title('Size distribution of Flux bin %i' % i
			+ '\nBridgeless (NN): '+str(np.sum(NN_index)) + ' Bridge (MG): ' + str(np.sum(MG_index)))
		plt.hist(all_sizes)
		plt.xlabel('Size (arcsec)')
		plt.ylabel('Count')
		plt.yscale('log')
		plt.savefig('./sizedist_flux2_'+str(i))
		plt.close()

		print ('Overlap: ', len(tdata['Source_Name']) - len(np.unique(tdata['Source_Name'])))

		# the Size bins
		print ('Size bin %i'%i)
		tdata = size_bins[i]
		
		MG_index = np.isnan(tdata['new_NN_distance(arcmin)']) 
		NN_index = np.invert(MG_index)

		plt.title('Flux distribution of Size bin %i' % i 
			+ '\nBridgeless (NN): '+str(np.sum(NN_index)) + ' Bridge (MG): ' + str(np.sum(MG_index)))

		all_fluxes = tdata['Peak_flux']
	
		print ('Overlap: ', len(tdata['Source_Name']) - len(np.unique(tdata['Source_Name'])))

		plt.hist(all_fluxes)
		plt.xlabel('Flux (mJy/beam)')
		plt.ylabel('Count')
		plt.yscale('log')
		plt.savefig('./fluxdist_size2_'+str(i))
		plt.close()


if __name__ == '__main__':
	# bins()
	# plot_flux_bins2_4()

	tdata = Table(fits.open('/data1/osinga/data/monte_carlo/biggest_selection.fits')[1].data)
	tdata = deal_with_overlap(tdata)

	# setup_source_finding('ILTJ113337.273+465836.71')

	# plot_source_finding(False,'P1Hetdex15','ILTJ135210.962+480947.16') # the source with interesting flux vs orientation. 

	all_error = []
	for i in range(len(tdata)):
		# source_name = 'ILTJ113636.138+480928.25'
		source_name = tdata['Source_Name'][i]
		error = setup_source_finding(source_name)
		if error < 10e6:
			all_error.append(error)

	print np.median(all_error)
	print np.mean(all_error)