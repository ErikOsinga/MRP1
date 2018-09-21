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
import pywcs as pw

from utils import (angular_dispersion_vectorized_n, distanceOnSphere, load_in
					, rotate_point, PositionAngle, deal_with_overlap, deal_with_overlap_2, FieldNames, tableMosaic_to_full)
from utils_orientation import find_orientationNN, find_orientationMG

import difflib

# import treecorr

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
			error = [10e6,10e6]

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
			error = [10e6,10e6]

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
	errorRA = error[0]
	errorDEC = error[1]

	return errorRA, errorDEC

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

def something_with_errorprop():
	all_errorRA = []
	all_errorDEC = []
	for i in range(len(tdata)):
		# source_name = 'ILTJ113636.138+480928.25'
		source_name = tdata['Source_Name'][i]
		errorRA, errorDEC = setup_source_finding(source_name)
		if errorRA < 10e6:
			all_errorRA.append(errorRA)
			all_errorDEC.append(errorDEC)

	print np.median(all_errorRA)
	print np.mean(all_errorRA)

	print '\n'

	print np.median(all_errorDEC)
	print np.mean(all_errorDEC)

def plot_source_finding_latest_catalog(NN,file,source_name,plot=False):
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
		
		prefix = '/data1/osinga/value_added_catalog_1_1b_thesis/source_filtering/NN/'
		Source_Data = prefix+file+'NearestNeighbours_efficient_spherical2.fits'
		Source_Name, Mosaic_ID = load_in(Source_Data,'Source_Name', 'Mosaic_ID')
		RA, DEC, NN_RA, NN_DEC, NN_dist, Total_flux, E_Total_flux, new_NN_index, Min, NN_Min = load_in(Source_Data,'RA','DEC','new_NN_RA','new_NN_DEC','new_NN_distance(arcmin)','Total_flux', 'E_Total_flux','new_NN_index','Min','new_NN_Min')
		source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'
		head = pf.getheader(source)
		hdulist = pf.open(source)

		i = np.where(Source_Name == source_name)[0][0]

		print Source_Name[i], 'Nearest Neighbour source'
		print RA[i],DEC[i]

		wcs = pw.WCS(hdulist[0].header)
		skycrd = np.array([[RA[i],DEC[i],0,0]], np.float_)
		pixel = wcs.wcs_sky2pix(skycrd, 1)

		# Some pixel coordinates of interest.
		x = int(pixel[0][0])
		y = int(pixel[0][1])
		rad = 60
		# plt.imshow((fits.open(source)[0].data)[x-rad:x+rad,y-rad:y+rad],origin='lower')
		# plt.show()
		try: 
			error = find_orientationNN(i,source,RA[i],DEC[i],NN_RA[i],NN_DEC[i],NN_dist[i],Min[i],NN_Min[i],(3/60.),plot=plot,head=head,hdulist=hdulist)[-1]
		except UnboundLocalError:
			error = [10e6,10e6]

	else:
		Source_Data = '/data1/osinga/value_added_catalog1_1b/LOFAR_HBA_T1_DR1_catalog_v0.9.srl.fixed.fits'
		Source_Name, S_Code, Mosaic_ID = load_in(Source_Data,'Source_Name', 'S_Code', 'Mosaic_ID')
		RA, DEC, Maj, Min, Total_flux , E_Total_flux = load_in(Source_Data,'RA','DEC', 'Maj', 'Min', 'Total_flux', 'E_Total_flux')

		multiple_gaussian_indices = (np.where(S_Code == 'M')[0])

		source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'
		head = pf.getheader(source)
		hdulist = pf.open(source)

		i = np.where(Source_Name == source_name)[0][0]

		wcs = pw.WCS(hdulist[0].header)
		skycrd = np.array([[RA[i],DEC[i],0,0]], np.float_)
		pixel = wcs.wcs_sky2pix(skycrd, 1)
		# Some pixel coordinates of interest.
		x = int(pixel[0][0])
		y = int(pixel[0][1])
		
		print Source_Name[i], 'MG source'
		print RA[i],DEC[i]

		try:
			error = find_orientationMG(i,source,RA[i],DEC[i],Maj[i],Min[i],(3/60.),plot=plot,head=head,hdulist=hdulist)[-1]
		except UnboundLocalError: # means only 1 maximum in the min and max orientation angle
			error = [10e6,10e6]

	return error

def check_NN_vs_MG():
	"""
	Shows the NN and MG source finding algorithms for all the overlapping (with the NN) sources in the latest biggest_selection
	aka, finds the sources whose nearest neighbour are also classified as a legit MG source.
	"""

	tdata = Table(fits.open('/data1/osinga/value_added_catalog/source_filtering/biggest_selection_latest.fits')[1].data)
	# plot_source_finding_latest_catalog(True,'P11Hetdex12','ILTJ113503.51+482613.9',plot=True) # the source that is NN and MG (the other way around)
	
	all_names, all_indices, mosaic_ids, _ = deal_with_overlap_2(tdata)

	for i in range(len(all_names)):
		print '\n %i' % i
		plot_source_finding_latest_catalog(True,tableMosaic_to_full(mosaic_ids[i]),all_names[i],plot=True) # NN
		plot_source_finding_latest_catalog(False,tableMosaic_to_full(mosaic_ids[i]),all_names[i],plot=True) # MG

def selection_no_valueadded():
	"""	 Plot all the sources that do not exist in the value-added catalog"""
	# All the sources that are in the biggest_selection but not in the value added catalog
	BS_no_VA = Table(fits.open('/data1/osinga/value_added_catalog/not_in_VA_biggest_selection.fits')[1].data)

	# Boolean indicating NN or MG source
	NN = np.invert(np.isnan(BS_no_VA['new_NN_RA']))
	print np.where(NN == False)
	print 'Number of NN sources: %i' % np.sum(NN)
	for i in range(749,len(BS_no_VA)):
		# i = np.random.randint(0,len(BS_no_VA))
		print i
		plot_source_finding_latest_catalog(NN[i],tableMosaic_to_full(BS_no_VA['Mosaic_ID'][i])
			,BS_no_VA['Source_Name'][i],plot=True)



if __name__ == '__main__':
	# selection_no_valueadded()

	# file = '/data1/osinga/value_added_catalog1_1b/value_added_selection.fits'
	# file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/value_added_selection.fits'
	# file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/value_added_selection_MG.fits'
	# file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/biggest_selection_with_overlap.fits'
	file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/source_filtering/all_NN_sources.fits' #
	tdata = Table(fits.open(file)[1].data)
	tdata['Source_Name_1'] = tdata['Source_Name']
	tdata['Source_Name'] = tdata['Source_Name_1']
	tdata['PA_1'] = tdata['PA']
	tdata['final_PA'] = tdata['position_angle']
	NN = np.invert(np.isnan(tdata['new_NN_RA']))
	# iis = [5744, 5743, 6317, 6316, 5849, 5850, 6583, 6584, 4101, 4100]
	
	# all 39 source name1s that are NN and have the wrong components
	# iis = ['ILTJ104730.34+530706.5', 'ILTJ105800.45+484339.6', 'ILTJ105956.46+483410.5', 'ILTJ110015.47+471048.8', 'ILTJ110718.36+504812.0', 'ILTJ110936.24+510320.0', 'ILTJ112205.56+484429.8', 'ILTJ112749.69+531707.6', 'ILTJ113908.87+553941.0', 'ILTJ113957.84+540808.7', 'ILTJ114915.54+482408.6', 'ILTJ120900.69+464147.4', 'ILTJ120954.18+491224.1', 'ILTJ121726.95+484557.2', 'ILTJ122938.96+501137.2', 'ILTJ124139.51+543859.9', 'ILTJ124229.23+502822.4', 'ILTJ124754.90+514359.8', 'ILTJ125822.71+534315.1', 'ILTJ130109.26+510620.8', 'ILTJ130647.79+463345.5', 'ILTJ130850.67+543820.5', 'ILTJ131251.93+471440.2', 'ILTJ132219.83+495805.6', 'ILTJ132418.13+492636.1', 'ILTJ132642.48+495214.7', 'ILTJ133121.62+554055.9', 'ILTJ133359.24+480410.3', 'ILTJ133637.18+533143.9', 'ILTJ134110.72+512125.3', 'ILTJ134153.00+474005.9', 'ILTJ134706.34+534003.2', 'ILTJ141842.68+542241.1', 'ILTJ142437.53+480513.3', 'ILTJ143647.77+474658.0', 'ILTJ144258.31+502421.7', 'ILTJ144818.16+563622.6', 'ILTJ145018.89+473630.5', 'ILTJ150118.14+471717.8']
	# investigate componentmatch PA vs Position angle
	# iis = ['ILTJ110011.82+484820.2','ILTJ111430.49+520441.5','ILTJ110306.76+562754.4','ILTJ110037.08+502724.0']
	# iis = ['ILTJ104936.51+471412.8']
	# iis = ['ILTJ105020.88+474223.3']


	# iis = ['ILTJ105036.49+532219.0'] # NNalgorithm.pdf (v1)
	iis = ['ILTJ135100.92+474938.1'] #NNalgorithm_v2.pdf
	# iis = ['ILTJ134706.37+482807.6'] # MGalgorithm.pdf


	# err_orientation cutoff showing rejected sources
	# tdata = Table(fits.open('../NN_excluded_by_orientation_cutoff.fits')[1].data)
	# iis = ['ILTJ113321.20+470144.7','ILTJ113055.16+483650.4','ILTJ112538.30+465258.2'] # err_orientation cutoff
	# iis = tdata['Source_Name']
	# tdata['PA_1'] = tdata['PA']
	# NN = np.invert(np.isnan(tdata['new_NN_RA']))

	# NN sources that are excluded because they are also good MG sources.
	# tdata = Table(fits.open('../NN_excluded_by_also_being_MG.fits')[1].data)
	# iis = tdata['Source_Name']
	# tdata['PA_1'] = tdata['PA']

	# iis = ['ILTJ110226.26+534327.3']


	# iis = ['ILTJ110005.66+530118.3','ILTJ110005.49+530116.9']
	# iis = ['ILTJ131441.36+473224.6','ILTJ131441.47+473213.3'] # source to remove after distance check
	# source to remove after VA catalog merge and then distance check
	# iis = ['ILTJ142701.92+554824.3','ILTJ142701.70+554817.3'] 
											

	# Bright sources
	# tdata = tdata[tdata['Peak_flux_2'] > 9.41]

	# Faint sources
	# tdata = tdata[tdata['Peak_flux_2'] < 3]
	# iis = tdata['Source_Name']

	for i in iis:
		i = np.where(tdata['Source_Name'] == i)[0][0]
		mosaicid = tdata['Mosaic_ID'][i]

		print 'PA_1: %f, position_angle: %f' %(tdata['PA_1'][i],tdata['position_angle'][i])
		# print tdata['Mosaic_ID'][i]
		mosaicid = tableMosaic_to_full(tdata['Mosaic_ID'][i])
		plot_source_finding_latest_catalog(NN[i],mosaicid,tdata['Source_Name'][i],plot=True)
		
		# for NN excluded by also being MG
		# plot_source_finding_latest_catalog(True,mosaicid,tdata['Source_Name'][i],plot=True)

	# setup_source_finding('ILTJ113337.273+465836.71')

	# plot_source_finding(False,'P1Hetdex15','ILTJ135210.962+480947.16') # the source with interesting flux vs orientation. 

