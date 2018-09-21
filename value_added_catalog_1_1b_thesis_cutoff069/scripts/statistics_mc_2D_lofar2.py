import sys
sys.path.insert(0,'/net/reusel/data1/osinga/anaconda2')

import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.table import Table, join, vstack

from scipy.spatial import cKDTree
from scipy.stats import norm

import matplotlib
from matplotlib import pyplot as plt

from utils import angular_dispersion_vectorized_n, distanceOnSphere, angular_dispersion_vectorized_n_parallel_transport

from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins)

from astropy.io import ascii

parallel_transport = False
print ('Using parallel_transport = %s' %parallel_transport)

position_angle = sys.argv[2]
if position_angle == 'True':
	position_angle = True
else:
	position_angle = False
if position_angle:
	print ('Using position_angle')
else: 
	print ('Using final_PA')

def histedges_equalN(x, nbin):
	"""
 	Make nbin equal height bins
 	Call plt.hist(x, histedges_equalN(x,nbin))
	"""
	npt = len(x)
	return np.interp(np.linspace(0,npt,nbin+1),
					np.arange(npt),
					np.sort(x))

def select_size_bins_cuts_biggest_selection(tdata,Write=False):
	"""
	Makes 4 size bins from the tdata, same bins as biggest_selection

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'3':bin5) -- 4 tables 
											containing the selected sources in each bin
	"""

	tdata_bs = Table(fits.open('../biggest_selection.fits')[1].data)
	try:
		size = np.asarray(tdata_bs['size']) 
	except KeyError:
		print ('using size: Maj*2 or NN_dist')
		MGwhere = np.isnan(tdata_bs['new_NN_RA'])
		NNwhere = np.invert(MGwhere)
		MGsources = tdata_bs[MGwhere]
		NNsources = tdata_bs[NNwhere]

		sizeNN = np.asarray(NNsources['new_NN_distance(arcmin)']) * 60 # to arcsec
		sizeMG = np.asarray(MGsources['Maj'])*2 # in arcsec

		size = np.concatenate((sizeNN,sizeMG))

	fig3 = plt.figure(3)
	ax = fig3.add_subplot(111)

	# get the size bins from biggest_selection
	n, bins, patches = ax.hist(size,histedges_equalN(size,4))
	plt.close(fig3)
	# print bins/60.
	print (n)
	# print ('Size bins (arcmin):',bins/60)

	try:
		size = np.asarray(tdata['size']) 
	except KeyError:
		print ('using size: Maj*2 or NN_dist')
		MGwhere = np.isnan(tdata['new_NN_RA'])
		NNwhere = np.invert(MGwhere)
		MGsources = tdata_bs[MGwhere]
		NNsources = tdata_bs[NNwhere]

		sizeNN = np.asarray(NNsources['new_NN_distance(arcmin)']) * 60 # to arcsec
		sizeMG = np.asarray(MGsources['Maj'])*2 # in arcsec

		size = np.concatenate((sizeNN,sizeMG))
	
	a = dict()
	for i in range(len(bins)-1): 
		# select from current tdata with the bins from tdata_bs 
		a[str(i)] = tdata[(bins[i]<size)&(size<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./biggest_selection_SIZEssdfsdf%i.fits'%i,overwrite=True)
	return a

def select_flux_bins_cuts_biggest_selection(tdata,Write=False):
	"""
	Makes 4 flux bins from the tdata, equal freq.

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'3':bin5) -- 4 tables 
											containing the selected sources in each bin
	"""
	
	tdata_bs = Table(fits.open('../biggest_selection.fits')[1].data)
	try:
		flux = np.asarray(tdata_bs['Peak_flux_2']) 
	except KeyError:
		print ('using Peak_flux')
		flux = np.asarray(tdata_bs['Peak_flux'])

	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	# get flux bins from tdata_bs
	n, bins, patches = ax.hist(flux,histedges_equalN(flux,4))
	plt.close(fig2)
	# print ('Flux bins:',bins)

	try:
		flux = np.asarray(tdata['Peak_flux_2']) 
	except KeyError:
		print ('using Peak_flux')
		flux = np.asarray(tdata['Peak_flux'])
	
	a = dict()
	for i in range(len(bins)-1): 
		# select from tdata with the bins from tdata_bs 
		a[str(i)] = tdata[(bins[i]<flux)&(flux<bins[i+1])]		
		# print ('Number in bin %i:'%i,len(a[str(i)]))

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selectionDFSDG_%i.fits'%i,overwrite=True)
	return a

def tick_function(X):
	return ["%.2f" % z for z in X]

def Sn_vs_n_test(tdata,Sn_mc,Sn_data,filename,angular_radius,ending_n=180):
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
	jain_sigma = (0.33/N)**0.5
	for n in n_range:
		# print 'Now doing n = ', n+starting_n , '...'  
		Sn_mc_n = np.asarray(Sn_mc['S_'+str(n+starting_n)])
		av_Sn_mc_n = np.mean(Sn_mc_n)
		sigma = np.std(Sn_mc_n)

		Sn_data_n = Sn_data[n]
		# print Sn_data_n, av_Sn_mc_n
		SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
		all_sn.append(Sn_data_n)
		all_sl.append(SL)
		all_sn_mc.append(av_Sn_mc_n)
		all_std.append(sigma)
	
	Results = Table()
	Results['n'] = range(starting_n,ending_n+1)
	Results['Sn_data'] = all_sn
	Results['SL'] = all_sl
	Results['Sn_mc'] = all_sn_mc
	Results['angular_radius'] = angular_radius
	Results['sigma_S_n'] = all_std
	Results['jain_sigma'] = [jain_sigma]
	if parallel_transport:
		print ('no writing')
		# Results.write('./data/2D/Statistics_PT'+filename+'_results.fits',overwrite=True)
	else:
		print ('no writing')
		# Results.write('./data/2D/Statistics_'+filename+'_results.fits',overwrite=True)
	
	fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
	axarr[0].plot(range(starting_n,ending_n+1),np.log10(all_sl))
	# print np.argmin(all_sl)
	axarr[0].set_title('SL vs n for '+filename+'\n\n')
	axarr[0].set_ylabel(r'$\log_{10}$ SL')
	axarr[0].set_xlim(2,n)
	
	if angular_radius is not None:
		ax2 = axarr[0].twiny()
		ax2.set_xlabel('angular_radius (degrees)')
		xticks = axarr[0].get_xticks()
		ax2.set_xticks(xticks)
		print (np.asarray(xticks,dtype='int'))
		xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
		ax2.set_xticklabels(tick_function(xticks2))
	plt.subplots_adjust(top=0.850)	

	axarr[1].plot(range(starting_n,ending_n+1),all_std)
	axarr[1].set_xlabel('n')
	axarr[1].set_ylabel('sigma')
	axarr[0].set_ylim(-2.5,0)

	if parallel_transport:
		print ('PT')
		# plt.savefig('./figures/2D/SL_vs_n_PT'+filename+'.png',overwrite=True)
	else:
		print ('no PT')
		# plt.savefig('./figures/2D/SL_vs_n_'+filename+'.png',overwrite=True)
	plt.show()
	plt.close()

def Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,ending_n=180):
	"""
	Make a plot of the SL (Significance level) statistic vs n.
	"""

	# hard fix for when n > amount of sources in a bin 
	if ending_n > len(tdata):
		print ('ending_n = %i, but this tdata only contains N=%i sources'%(ending_n,len(tdata)))
		ending_n = len(tdata)-1
		print ('Setting ending_n=%i'%ending_n)


	starting_n = 1 # In the data file the S_1 is the first column
	n_range = range(0,ending_n) # index 0 corresponds to Sn_1 

	all_sn = []
	all_sn_mc = []
	all_sl = [] # (length n list containing SL_1 to SL_80)
	all_std = [] # contains the standard deviations of the MC simulations
	N = len(tdata)
	jain_sigma = (0.33/N)**0.5
	for n in n_range:
		# print 'Now doing n = ', n+starting_n , '...'  
		Sn_mc_n = np.asarray(Sn_mc['S_'+str(n+starting_n)])
		av_Sn_mc_n = np.mean(Sn_mc_n)
		sigma = np.std(Sn_mc_n)
		if sigma == 0:
			print ('Using jain sigma')
			sigma = jain_sigma
		Sn_data_n = Sn_data[n]
		# print Sn_data_n, av_Sn_mc_n
		SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
		all_sn.append(Sn_data_n)
		all_sl.append(SL)
		all_sn_mc.append(av_Sn_mc_n)
		all_std.append(sigma)
	
	Results = Table()
	Results['n'] = range(starting_n,ending_n+1)
	Results['Sn_data'] = all_sn
	Results['SL'] = all_sl
	Results['Sn_mc'] = all_sn_mc
	Results['angular_radius'] = angular_radius
	Results['sigma_S_n'] = all_std
	Results['jain_sigma'] = [jain_sigma]
	if parallel_transport:
		Results.write('./data/2D/Statistics_PT'+filename+'_results.fits',overwrite=True)
	else:
		Results.write('./data/2D/Statistics_'+filename+'_results.fits',overwrite=True)
	
	try:
		fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
		axarr[0].plot(range(starting_n,ending_n+1),np.log10(all_sl))
		# print np.argmin(all_sl)
		axarr[0].set_title('SL vs n for '+filename+'\n\n')
		axarr[0].set_ylabel(r'$\log_{10}$ SL')
		axarr[0].set_xlim(2,n)
		axarr[0].set_ylim(-2.5,0)
		
		if angular_radius is not None:
			ax2 = axarr[0].twiny()
			ax2.set_xlabel('angular_radius (degrees)')
			xticks = axarr[0].get_xticks()
			ax2.set_xticks(xticks)
			print (np.asarray(xticks,dtype='int'))
			xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
			ax2.set_xticklabels(tick_function(xticks2))
		plt.subplots_adjust(top=0.850)	

		axarr[1].plot(range(starting_n,ending_n+1),all_std)
		axarr[1].set_xlabel('n')
		axarr[1].set_ylabel('sigma')

		if parallel_transport:
			plt.savefig('./figures/2D/SL_vs_n_PT'+filename+'.png',overwrite=True)
		else:
			plt.savefig('./figures/2D/SL_vs_n_'+filename+'.png',overwrite=True)

	except IndexError: # try again, but with xlim on n -= 60
		
		temp_n = np.asarray(xticks,dtype='int')[-2]
		# temp_n = 3800
		print ('Index error, setting n=',temp_n)

		plt.close()

		fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
		axarr[0].plot(range(starting_n,ending_n+1),np.log10(all_sl))
		# print np.argmin(all_sl)
		axarr[0].set_title('SL vs n for '+filename+'\n\n')
		axarr[0].set_ylabel(r'$\log_{10}$ SL')
		axarr[0].set_xlim(2,temp_n)
		axarr[0].set_ylim(-2.5,0)
		
		# if filename == 'value_added_selection_MG_PA':
			# axarr[0].set_xticks(np.arange(1000,4000,200))

		if angular_radius is not None:
			ax2 = axarr[0].twiny()
			ax2.set_xlabel('angular_radius (degrees)')
			xticks = axarr[0].get_xticks()
			ax2.set_xticks(xticks)
			print (np.asarray(xticks,dtype='int'))
			xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
			ax2.set_xticklabels(tick_function(xticks2))
		plt.subplots_adjust(top=0.850)	

		axarr[1].plot(range(starting_n,ending_n+1),all_std)
		axarr[1].set_xlabel('n')
		axarr[1].set_ylabel('sigma')

		if parallel_transport:
			plt.savefig('./figures/2D/SL_vs_n_PT'+filename+'.png',overwrite=True)
		else:
			plt.savefig('./figures/2D/SL_vs_n_'+filename+'.png',overwrite=True)



	# axarr[0].set_xlim(2,len(tdata))

	plt.show()
	plt.close()

def angular_radius_vs_n(tdata,filename,n=180,starting_n=2):
	"""
	Make a plot of the angular separation vs the amount of neighbours n
	"""
	# hard fix for when n > amount of sources in a bin 
	if n > len(tdata):
		print ('Announcement about %s'%filename)
		print ('n = %i, but this bin only contains N=%i sources'%(n,len(tdata)))
		n = len(tdata)-1
		print ('Setting n=%i'%n)

	n_range = range(starting_n,n+1) 

	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['final_PA'])

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
		indices = coordinates_tree.query(coordinates[i],k=n,p=2)[1] # include source itself
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
	plt.ylabel('Median angular radius (deg)')
	plt.xlabel('n')
	# plt.savefig('/data1/osinga/figures/statistics/show/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	# plt.savefig('./figures/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	plt.close()
		
	return median

def statistics(filename,tdata,redshift=False):
	if parallel_transport:
		if n < 998:
			Sn_mc = Table(fits.open('./data/2D/Sn_monte_carlo_PT'+filename+'.fits')[1].data)
		else: 
			Sn_mc = ascii.read('./data/2D/Sn_monte_carlo_PT'+filename+'.csv')

	else:
		if n < 998:
			Sn_mc = Table(fits.open('./data/2D/Sn_monte_carlo_'+filename+'.fits')[1].data)
		else:
			Sn_mc = ascii.read('./data/2D/Sn_monte_carlo_'+filename+'.csv')
	# calculate angular radius for number of nn, 
	angular_radius = angular_radius_vs_n(tdata,filename,n)
	if parallel_transport:
		Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n,redshift)
	else:
		Sn_data = angular_dispersion_vectorized_n(tdata,n,redshift)
	Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,n)

n = 3890

if __name__ == '__main__':

	'''
	#Running just one file

	filename = sys.argv[1]
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	tdata['RA'] = tdata['RA_2']
	tdata['DEC'] = tdata['DEC_2']

	print ('Using redshift.., but only for selecting sources') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	filename += '_redshift_' ## edited for not actually using z

	tdata_original = tdata
	filename_original = filename

	filename += '_astropy'

	# fluxbins1 = select_flux_bins1(tdata_original)
	# tdata = fluxbins1['2']
	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux2'
	tdata = tdata[z_available]
	statistics(filename,tdata)
	'''

	
	#Running all the statistics without redshift
	redshift = False
	equal_width = False # If equal_width is false use the same flux and size cuts as the initial subsample

	filename = sys.argv[1]
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only

	if position_angle: filename += '_PA' # only when using 'position_angle'

	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass # then we're doing biggest_selection


	tdata_original = tdata
	filename_original = filename

	statistics(filename,tdata,redshift)


	# if equal_width: 
	# 	fluxbins = select_flux_bins1(tdata_original)
	# else:
	# 	fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
	# for key in fluxbins:
	# 	tdata = fluxbins[key]
	# 	filename = filename_original + 'flux%s'%key
		
	# 	statistics(filename,tdata,redshift)

	# if equal_width:
	# 	sizebins = select_size_bins1(tdata_original)
	# else:
	# 	sizebins = select_size_bins_cuts_biggest_selection(tdata_original)
	# for key in sizebins:
	# 	tdata = sizebins[key]
	# 	filename = filename_original + 'size%s'%key
		
	# 	statistics(filename,tdata,redshift)






	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux11'

	# statistics(filename,tdata,redshift)
	

	'''
	#Running all the statistics with redshift
	redshift = True 

	filename = sys.argv[1]
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	tdata['RA'] = tdata['RA_2']
	tdata['DEC'] = tdata['DEC_2']

	print ('Using redshift..')
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	# filename += '_only_redshift_sources_' ## edited for not actually using z
	if position_angle: filename += '_PA' # only when using 'position_angle'
	filename += '_redshift_'

	tdata_original = tdata
	filename_original = filename

	tdata = tdata[z_available]

	tdata_original = tdata # use redshift data for power bins and size bins !!
	statistics(filename,tdata,redshift)


	fluxbins = select_power_bins(tdata_original)
	for key in fluxbins:
		tdata = fluxbins[key]
		filename = filename_original + 'power%s'%key
		
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available]

		statistics(filename,tdata,redshift)

	sizebins = select_physical_size_bins(tdata_original)
	for key in sizebins:
		tdata = sizebins[key]
		filename = filename_original + 'physicalsize%s'%key
		
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available]

		statistics(filename,tdata,redshift)


	#Running all the statistics with redshift, but not using redshift
	redshift = False

	filename = sys.argv[1]
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
	tdata['RA'] = tdata['RA_2']
	tdata['DEC'] = tdata['DEC_2']

	print ('Using redshift.. but only for selection')
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	if position_angle: filename += '_PA' # only when using 'position_angle'
	filename += '_only_redshift_sources_' ## edited for not actually using z

	tdata_original = tdata
	filename_original = filename

	tdata = tdata[z_available]

	tdata_original = tdata # use redshift data for power bins and size bins !!
	statistics(filename,tdata,redshift)


	fluxbins = select_power_bins(tdata_original)
	for key in fluxbins:
		tdata = fluxbins[key]
		filename = filename_original + 'power%s'%key
		
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available]

		statistics(filename,tdata,redshift)

	sizebins = select_physical_size_bins(tdata_original)
	for key in sizebins:
		tdata = sizebins[key]
		filename = filename_original + 'physicalsize%s'%key
		
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available]

		statistics(filename,tdata,redshift)

	# fluxbins11 = select_flux_bins11(tdata_original)
	# tdata = fluxbins11
	# filename = filename_original + 'flux11'

	# z_available = np.invert(np.isnan(tdata['z_best']))
	# print ('Number of sources with available redshift:', np.sum(z_available))
	# tdata = tdata[z_available]

	# statistics(filename,tdata,redshift)
	'''