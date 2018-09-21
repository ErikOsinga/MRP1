import sys
sys.path.insert(0, '/net/reusel/data1/osinga/anaconda2')
import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt

from utils import (angular_dispersion_vectorized_n, load_in, distanceOnSphere
	, deal_with_overlap, FieldNames, parallel_transport,
	angular_dispersion_vectorized_n_parallel_transport)

from astropy import visualization

import uncertainties as unc 
from uncertainties import unumpy

from scipy.spatial import cKDTree
from scipy.stats import norm

directory = '.'


def histedges_equalN(x, nbin):
	"""
 	Make nbin equal height bins
 	Call plt.hist(x, histedges_equalN(x,nbin))
	"""
	npt = len(x)
	return np.interp(np.linspace(0,npt,nbin+1),
					np.arange(npt),
					np.sort(x))

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
	
	# find the Peak flux of tdata_bs, based on isolated or connected lobe
	MGwhere = np.isnan(tdata_bs['new_NN_RA'])
	NNwhere = np.invert(MGwhere)
	MGsources = tdata_bs[MGwhere]
	NNsources = tdata_bs[NNwhere]
	fluxMG = np.asarray(MGsources['Peak_flux'])
	# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
	use_current = NNsources['Peak_flux'] > NNsources['new_NN_Peak_flux']
	fluxMG = np.asarray(MGsources['Peak_flux'])
	fluxNN = []
	for i in range(len(NNsources)):
		if use_current[i]:
			fluxNN.append(NNsources['Peak_flux'][i])
		else: # use NN peak flux
			fluxNN.append(NNsources['new_NN_Peak_flux'][i])
	fluxNN = np.asarray(fluxNN)
	assert len(fluxNN) == len(NNsources)

	# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
	flux = np.concatenate((fluxNN,fluxMG))

	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	# get flux bins from tdata_bs
	n, bins, patches = ax.hist(flux,histedges_equalN(flux,4))
	plt.close(fig2)
	print ('Flux bins:',bins)

	# find the Peak flux of tdata, based on isolated or connected lobe
	MGwhere = np.isnan(tdata['new_NN_RA'])
	NNwhere = np.invert(MGwhere)
	MGsources = tdata[MGwhere]
	NNsources = tdata[NNwhere]
	try:
		fluxMG = np.asarray(MGsources['Peak_flux_2'])
		# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
		use_current = NNsources['Peak_flux_2'] > NNsources['new_NN_Peak_flux']
		fluxNN = []
		for i in range(len(NNsources)):
			if use_current[i]:
				fluxNN.append(NNsources['Peak_flux_2'][i])
			else: # use NN peak flux
				fluxNN.append(NNsources['new_NN_Peak_flux'][i])
		fluxNN = np.asarray(fluxNN)
		assert len(fluxNN) == len(NNsources)
		print ('using Peak_flux_2')

	except KeyError:
		print ('using Peak_flux, so this is biggest_selection')
		fluxMG = np.asarray(MGsources['Peak_flux'])
		# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
		# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
		use_current = NNsources['Peak_flux'] > NNsources['new_NN_Peak_flux']
		fluxMG = np.asarray(MGsources['Peak_flux'])
		fluxNN = []
		for i in range(len(NNsources)):
			if use_current[i]:
				fluxNN.append(NNsources['Peak_flux'][i])
			else: # use NN peak flux
				fluxNN.append(NNsources['new_NN_Peak_flux'][i])
		fluxNN = np.asarray(fluxNN)
		assert len(fluxNN) == len(NNsources)

	flux = np.concatenate((fluxNN,fluxMG))

	a = dict()
	for i in range(len(bins)-1): 
		# select from tdata with the bins from tdata_bs 
		a[str(i)] = tdata[(bins[i]<flux)&(flux<bins[i+1])]		
		print ('Number in bin %i:'%i,len(a[str(i)]))

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selection_redshift_FLUX_%i.fits'%i,overwrite=True)
	return a

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
		size = np.asarray(tdata_bs['size_thesis']) 
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
		size = np.asarray(tdata['size_thesis']) 
	except KeyError:
		print ('using size: Maj*2 or NN_dist')
		MGwhere = np.isnan(tdata['new_NN_RA'])
		NNwhere = np.invert(MGwhere)
		MGsources = tdata[MGwhere]
		NNsources = tdata[NNwhere]

		sizeNN = np.asarray(NNsources['new_NN_distance(arcmin)']) * 60 # to arcsec
		if len(MGsources) != 0:
			sizeMG = np.asarray(MGsources['Maj'])*2 # in arcsec
		else:
			print ('No MG sources found')
			sizeMG = np.asarray([])

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

def angular_radius_vs_n(tdata,filename,n=180,starting_n=2):
	"""
	Make a plot of the angular separation vs the amount of neighbours n
	"""

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

	#plt.plot(n_range,median[starting_n-1:]) # index 1 corresponds to n=2
	#plt.title('Angular radius vs n for ' + filename)
	#plt.ylabel('Mean angular radius (deg)')
	#plt.xlabel('n')
	# plt.savefig('/data1/osinga/figures/statistics/show/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	#plt.savefig('./angular_radius_vs_n_'+filename+'.pdf',overwrite=True)
	#plt.close()
		
	return median

def parallel_transport_uncertainties(RA_t,DEC_t,RA_s,DEC_s,PA_s):
	"""
	Parallel transports source(s) s at RA_s, DEC_s with position angle PA_s to target
	position t, with RA_t and DEC_t. See Jain et al. 2004

	Arguments:
	RA_t, DEC_t -- coordinates of the target in degrees
	RA_s, DEC_s -- numpy array with coordinates of the source(s) in degrees
	PA_s -- numpy array with position angle of the source(s) in degrees

	Returns:
	PA_transport -- numpy array with the parallel transported angle(s) PA_s to target position t
	 				in degrees

	"""

	PA_s = unumpy.radians(PA_s)
	RA_s = np.radians(RA_s)
	DEC_s = np.radians(DEC_s)
	RA_t = np.radians(RA_t)
	DEC_t = np.radians(DEC_t)
	
	# define the radial unit vectors u_rs and u_rt
	x = np.cos(RA_s) * np.cos(DEC_s)
	y = np.sin(RA_s) * np.cos(DEC_s)
	z = np.sin(DEC_s)
	u_rs = -1 * np.array([x,y,z]) #pointing towards center of sphere 

	x = np.cos(RA_t) * np.cos(DEC_t)
	y = np.sin(RA_t) * np.cos(DEC_t)
	z = np.sin(DEC_t)
	u_rt = -1 * np.array([x,y,z]).reshape(1,3) #pointing towards center of sphere 

	xi_1 = np.arccos( (np.sin(DEC_t)*np.sin(RA_t - RA_s)) / (np.sqrt(1-(np.dot(u_rt,u_rs))**2 )) )[0]
	temp1 = ( (-1 * np.sin(DEC_s)*np.cos(DEC_t)+np.cos(DEC_s)*np.sin(DEC_t)*np.cos(RA_t-RA_s) ) 
				/ np.sqrt( 1 - (np.dot(u_rt,u_rs)**2)) )[0]
	xi_2 = np.arccos( (-1* np.sin(DEC_s)*np.sin(RA_s - RA_t)) / (np.sqrt(1-(np.dot(u_rt,u_rs))**2 )) )[0]
	temp2 = ( (-1 * np.sin(DEC_t)*np.cos(DEC_s)+np.cos(DEC_t)*np.sin(DEC_s)*np.cos(RA_t-RA_s) )
				 / np.sqrt( 1 - (np.dot(u_rt,u_rs)**2)) )[0]

	# Make sure that temp1 and temp2 are also never NaN
	assert np.sum(np.isnan(temp1)) == 0
	assert np.sum(np.isnan(temp2)) == 0

	# if vectors are too close NaN appears, use 'neglect parallel transport', it's super effective.
	wherenan1 = np.isnan(xi_1) 
	wherenan2 = np.isnan(xi_2)
	wherenan = np.logical_or(wherenan1,wherenan2)

	for j in range(len(wherenan)):
		if wherenan[j] == True:
			# if vectors are too close, parallel transport is negligible
			xi_1[j] = 0
			xi_2[j] = 0 

		if temp1[j] < 0:
			xi_1[j] *= -1
		if temp2[j] > 0:  		# REKEN DIT NOG FF NA
			xi_2[j] *= -1

	PA_transport = PA_s + xi_2 - xi_1 
	# fix it back into [0,pi] range if it went beyond it
	for j in range(len(RA_s)):
		if PA_transport[j] > np.pi:
			# print ('PT : Reducing angle by pi')
			PA_transport[j] -= np.pi
		if PA_transport[j] < 0:
			# print ('PT : Increasing angle by pi')
			PA_transport[j] += np.pi

	return unumpy.degrees(PA_transport)
	
def angular_dispersion_uncertainties(tdata,n,position_angles,redshift=False,PT=False):
	
	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])

	# hard fix for when n > N
	if n > len(tdata):
		print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
		n = len(tdata)-1
		print ('Setting n=%i'%n)

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	if redshift:
		Z = tdata['z_best']
		'''
		H = 73450 # m/s/Mpc = 73.45 km/s/Mpc
		# but is actually unimportant since only relative distances are important
		from scipy.constants import c # m/s
		# assume flat Universe with q0 = 0.5 (see Hutsemekers 1998)
		# I think assuming q0 = 0.5 means flat universe
		r = 2.0*c/H * ( 1-(1+Z)**(-0.5) ) # comoving distance
		'''
		from astropy.cosmology import Planck15
		r = Planck15.comoving_distance(Z) #better to just use this calculator
		x = r * np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
		y = r * np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
		z = r * np.sin(np.radians(DECs))	
	else:
		x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
		y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
		z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T
	
	#make a KDTree for quick NN searching	
	coordinates_tree = cKDTree(coordinates,leafsize=16)

	# for every source: find n closest neighbours, calculate max dispersion using PT
	position_angles_array = np.zeros((N,n)) # array of shape (N,n) that contains position angles
	position_angles_array = unumpy.uarray(position_angles_array,0)
	for i in range(N):
		if PT:
			index_NN = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1] # include source itself
			# print index_NN # if this gives an error we should check whether we have the right sources (redshift selection)
			# Transport every nearest neighbour to the current source i
			angles_transported = parallel_transport_uncertainties(RAs[i],DECs[i],RAs[index_NN[1:]],DECs[index_NN[1:]],position_angles[index_NN[1:]])
			# Then concatenate the transported angles to the current source position angle and store it
			position_angles_array[i] = np.concatenate(([position_angles[i]],angles_transported))
		else:
			index_NN = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1] # include source itself
			position_angles_array[i] = position_angles[index_NN] 

	assert position_angles_array.shape == (N,n)

	n_array = np.asarray(range(1,n+1)) # have to divide different elements by different n

	x = unumpy.radians(2*position_angles_array) # used to use numexpr to speed it up quite significantly

	di_max = 1./n_array * ( (np.cumsum(unumpy.cos(x),axis=1))**2 
				+ (np.cumsum(unumpy.sin(x),axis=1))**2 )**0.5
	
	assert di_max.shape == (N,n) # array of max_di for every source, for every n

	Sn = 1./N * np.sum(di_max,axis=0) # array of shape (1xn) containing S_1 (nonsense)
										# to S_n
	return Sn

def error_prop(file):
	''' histogram of the deviation in SL_Number '''

	file = directory+'/%s.fits'%file
	all_sl = Table(fits.open(file)[1].data)

	number = 35
	for number in range(0,180):
		sl_number = [] # e.g., SL_35 

		original = all_sl['original'][number]
		
		# Loop over the 100 gaussian errors
		for i in range(0,100):
			sl_number.append( all_sl[str(i)][number] )


		sl_number = np.log10(sl_number)
		original = np.log10(original)

		plt.title('Error by simulating (100x) Gaussian error with a std of 5 degrees.')
		plt.hist(sl_number,label='Added Gaussian noise',bins=20)
		# visualization.hist(sl_number,bins='scott',histtype=u'step',color='black',label='Added Gaussian noise')


		plt.axvline(original, label='Original',c='k',ls='dashed')
		plt.legend(fontsize=14)
		plt.xlabel(r'$Log_{10}$ SL for'+' n = %i'%(number+1),fontsize=14)
		plt.ylabel('Counts',fontsize=14)
		ax = plt.gca()
		ax.tick_params(labelsize=12)
		plt.tight_layout()
		if 'MG' in file:
			if 'Z' in file:
				plt.savefig('./MG_Z_histograms/error_%i'%number)
			else:
				plt.savefig('./MG_histograms/error_%i'%number)
		else: 
			plt.savefig('./histograms/error_%i'%number)
		plt.close()

def error_in_SL_plot(file):
	''' Sl_vs_n plot with deviation in it '''
	all_sl = Table(fits.open(directory+'/%s.fits'%file)[1].data)

	# to open with numpy array
	all_sl['100'] = all_sl['original']
	del all_sl['original'] # Needs to be float
	all_sl.write('./%s.csv'%file,overwrite=True)

	all_sl_np = np.loadtxt('./%s.csv'%file,delimiter=',')
	# Take all rows except the first from the 100-th column
	all_sl_original = all_sl_np[1:,100] # This is original sl
	# All rows except the first from the other 100 colums (0-99) 
	all_sl_gauss = all_sl_np[1:,:100] # this is the Gaussian errors

	starting_n = 1 
	ending_n = 180
	n_range = range(starting_n,ending_n+1)

	fig, ax = plt.subplots()

	ax.plot(n_range,np.log10(all_sl_original),color='k',label='original data') # plot original data

	def plot_min_max():
		'''
		# to find minimum and maximum is wrong, but just as an indication (proof of concept)
		# better to do 25 percentile or 1 sigma 
		'''
		all_sl_min = []
		all_sl_max = []

		for i in range(ending_n):
			# find the min/max of the SL from 100 gaussian errors 
			# of row i (corresponds to n_n = i)
			all_sl_min.append(np.min(all_sl_gauss[i]))
			all_sl_max.append(np.max(all_sl_gauss[i]))

		# plot min and max
		ax.plot(n_range,np.log10(all_sl_min),color='k',ls='dashed',label='Gaussian error min/max')
		ax.plot(n_range,np.log10(all_sl_max),color='k',ls='dashed')

	def plot_percentiles():
		''' 25 and 75 percentile '''

		all_sl_25 = []
		all_sl_75 = []

		for i in range(ending_n):
			# find the min/max of the SL from 100 gaussian errors 
			# of row i (corresponds to n_n = i)
			percentiles = np.percentile(all_sl_gauss[i],[25,75])
			all_sl_25.append(percentiles[0])
			all_sl_75.append(percentiles[1])

		# plot min and max
		ax.plot(n_range,np.log10(all_sl_25),ls='dashed',label='Gaussian error 25 percentile')
		ax.plot(n_range,np.log10(all_sl_75),ls='dashed',label='Gaussian error 75 percentile')

	plot_percentiles()

	plt.legend(fontsize=14)
	ax.tick_params(labelsize=12)
	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	plt.xlim(1,ending_n)
	plt.ylim(-3.82,0)
	plt.tight_layout()
	plt.savefig('./gaussian_error_%s'%file)
	plt.show()

def distribution_of_Sn(file):
	''' S_number plot with deviation in it '''
	all_sn = Table(fits.open(directory+'/%s.fits'%file)[1].data)

	# to open with numpy array
	all_sn['100'] = all_sn['original']
	del all_sn['original'] # Needs to be float
	all_sn.write('./%s.csv'%file,overwrite=True)

	all_sn_np = np.loadtxt('./%s.csv'%file,delimiter=',')
	# The first row is the headers
	# Take all rows except the first from the 100-th column
	all_sn_original = all_sn_np[1:,100] # This is original sl
	# All rows except the first from the other 100 colums (0-99) 
	all_sn_gauss = all_sn_np[1:,:100] # this is the Gaussian errors

	starting_n = 1 
	ending_n = 180
	n_range = range(starting_n,ending_n+1)

	number = 35

	fig, ax = plt.subplots()

	plt.axvline(x=all_sn_original[number],ymin=0,ymax=1, color = 'r', label='Original data')

	plt.hist(all_sn_gauss[number],label='Added Gaussian noise')

	plt.legend(fontsize=14)
	ax.tick_params(labelsize=12)
	ax.set_xlabel(r'$S_{%i}$'%number,fontsize=14)
	ax.set_ylabel('Counts',fontsize=14)
	# plt.xlim(1,ending_n)
	# plt.ylim(-3.82,0)
	plt.tight_layout()
	# plt.savefig('./gaussian_error_SN%s'%file)
	plt.show()


def uncertainty_in_Sn(file):
	tdata = Table(fits.open('../%s.fits'%file)[1].data)
	error = np.random.normal(0,5,len(tdata)) # add gaussian noise centered on the PA with std 5.
	position_angles = tdata['position_angle']

	# unumpy deals with standard deviation so use np.abs
	position_angles = unumpy.uarray(position_angles,np.abs(error))

	# Number of S_{n_n} to investigate n_n = number
	number = 60 

	# array of Sn with uncertainties shape: (n,)
	Sn = angular_dispersion_uncertainties(tdata,number,position_angles)

	print ('Sn %i: %.3f, standard deviation: %.3f'%(number,Sn[number].nominal_value,Sn[number].std_dev) )

	# vergelijk het met random data, gegenereerd door error-loze hoeken the shufflen
	if 'MG' in file: 
		filename = 'data_original'
		Sn_mc = Table(fits.open('data/Sn_monte_carlo_'+filename+'.fits')[1].data) 
	else:
		filename = 'MGdata_original'
		Sn_mc = Table(fits.open('data/Sn_monte_carlo_MG'+filename+'.fits')[1].data) 

	n = number
	starting_n = 1

	Sn_mc_n = np.asarray(Sn_mc['S_'+str(n+starting_n)])
	Sn_data_n = Sn[n].nominal_value + Sn[n].std_dev
	av_Sn_mc_n = np.mean(Sn_mc_n)
	sigma = np.std(Sn_mc_n)
	SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
	print ('SL upper bound: %.3f'%SL)

	Sn_data_n = Sn[n].nominal_value - Sn[n].std_dev
	SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
	print ('SL lower bound: %.3f'%SL)


def uncertainty_in_Sn_VA_MG_redshift_2D():
	file = 'value_added_selection_MG'
	tdata = Table(fits.open('../%s.fits'%file)[1].data)
	print ('Using redshift sources..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]

	# get flux bin 3 of VA_MG
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']
	tdata['final_PA'] = tdata['position_angle']
	tdata['RA'], tdata['DEC'] = tdata['RA_2'], tdata['DEC_2']
	
	# the simulated original (no error) SN_MC
	# filename = 'MG_Zdata_original'
	# Sn_mc = './data/Sn_monte_carlo_MG'+filename+'.fits'  # needs rerunning with PT probably
	Sn_mc = './tmp/Sn_monte_carlo_PTvalue_added_selection_MG_PA_only_redshift_sources_flux3.fits'
	Sn_mc = Table(fits.open(Sn_mc)[1].data)
	n_begin = 60
	n = 140

	# for two figures in the discussion showing dist of Sn_mc and Sn_data
	def produce_Sn_plots():
		Sn_data = angular_dispersion_vectorized_n(tdata,n)

		plt.hist(Sn_mc['S_%i'%n],bins=25,histtype=u'step',color='black',label=r'$S_n\vert_{MC}$')
		
		# print (Sn_data[n-1])
		# plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed', label='Data no PT')

		Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n)
		print (Sn_data[n-1])
		plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed',color='k',label='Data')
		
		ax = plt.gca() 
		ax.tick_params(labelsize=12)
		ax.set_xlabel(r'$S_{%i}$'%n,fontsize=14)	
		ax.set_ylabel(r'Counts',fontsize=14)
		ax.legend(fontsize=14)
		title = 'Distribution of S_%i'%n
		title = ' '
		plt.title(title,fontsize=16)
		if n == 140: plt.xlim(0.040,0.12151429413439714)
		plt.tight_layout()

		plt.show()

		Sn_mc_n = np.asarray(Sn_mc['S_'+str(n)])
		Sn_data_n = Sn_data[n-1]
		av_Sn_mc_n = np.mean(Sn_mc_n)
		sigma = np.std(Sn_mc_n)

		print sigma
		print ('S_%i = %.3f'%(n,Sn_data_n))
		
		SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL = %.3f for n= %i, calculated with PT'%(np.log10(SL),n) )

	produce_Sn_plots()
	''' for calculating the change when position angles are given Gaussian noise '''
	def produce_error_on_SN():
		error = np.random.normal(0,5,len(tdata)) # add gaussian noise centered on the PA with std 5.
		
		position_angles = tdata['position_angle']

		std = 10
		print ('Using std = ',std)
		# unumpy deals with standard deviation so use np.abs
		position_angles = unumpy.uarray(position_angles,std) #np.abs(error)

		# Number of S_{n_n} to investigate number is one less than n because index
		number = n -1

		# array of Sn with uncertainties shape: (n,)
		Sn_data = angular_dispersion_uncertainties(tdata,number+1,position_angles,PT=True)

		print ('Sn %i: %.3f, standard deviation: %.8f'%(number+1,Sn_data[number].nominal_value,Sn_data[number].std_dev) )

		print ('Does parallel_transport')

		av_Sn_mc_n = np.mean(Sn_mc['S_%i'%n])
		sigma = np.std(Sn_mc['S_%i'%n])

		SL = 1 - norm.cdf(   (Sn_data[number].nominal_value - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL data : %.3f'%np.log10(SL))

		upperboundSn = Sn_data[number].nominal_value + Sn_data[number].std_dev
		SL = 1 - norm.cdf(   (upperboundSn - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL upper bound: %.3f'%np.log10(SL))
		
		lowerboundSn = Sn_data[number].nominal_value - Sn_data[number].std_dev

		SL = 1 - norm.cdf(   (lowerboundSn - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL lower bound: %.3f'%np.log10(SL))



	# produce_error_on_SN()

def uncertainty_in_Sn_VA_MG_redshift_3D():
	file = 'value_added_selection_MG'
	tdata = Table(fits.open('../%s.fits'%file)[1].data)
	print ('Using redshift sources..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]

	# get flux bin 3 of VA_MG
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']
	tdata['final_PA'] = tdata['position_angle']
	tdata['RA'], tdata['DEC'] = tdata['RA_2'], tdata['DEC_2']
	
	# the simulated original (no error) SN_MC
	# filename = 'MG_Zdata_original'
	# Sn_mc = './data/Sn_monte_carlo_MG'+filename+'.fits'  # needs rerunning with PT probably
	Sn_mc = './tmp/Sn_monte_carlo_PTvalue_added_selection_MG_PA_redshift_flux3.fits'
	Sn_mc = Table(fits.open(Sn_mc)[1].data)
	n_begin = 60
	n = 60

	# for two figures in the discussion showing dist of Sn_mc and Sn_data
	def produce_Sn_plots():
		Sn_data = angular_dispersion_vectorized_n(tdata,n)

		plt.hist(Sn_mc['S_%i'%n],bins=100,histtype=u'step',color='black',label=r'$S_n\vert_{MC}$')
		
		# print (Sn_data[n-1])
		# plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed', label='Data no PT')

		Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n,redshift=True)
		print (Sn_data[n-1])
		plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed',color='k',label='Data')
		
		ax = plt.gca() 
		ax.tick_params(labelsize=12)
		ax.set_xlabel(r'$S_{%i}$'%n,fontsize=14)	
		ax.set_ylabel(r'Counts',fontsize=14)
		ax.legend(fontsize=14)
		title = 'Distribution of S_%i'%n
		title = ' '
		plt.title(title,fontsize=16)
		if n == 140: plt.xlim(0.040,0.12151429413439714)
		plt.tight_layout()

		plt.show()

		Sn_mc_n = np.asarray(Sn_mc['S_'+str(n)])
		Sn_data_n = Sn_data[n-1]
		av_Sn_mc_n = np.mean(Sn_mc_n)
		sigma = np.std(Sn_mc_n)

		print ('S_%i = %.3f'%(n,Sn_data_n))
		
		SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL = %.3f for n= %i, calculated with PT'%(np.log10(SL),n) )

	# produce_Sn_plots()
	''' for calculating the change when position angles are given Gaussian noise '''
	def produce_error_on_SN():
		error = np.random.normal(0,5,len(tdata)) # add gaussian noise centered on the PA with std 5.
		
		position_angles = tdata['position_angle']
		
		std = 10
		print ('Using std = ',std)
		
		# unumpy deals with standard deviation so use np.abs
		position_angles = unumpy.uarray(position_angles,std) #np.abs(error)

		# Number of S_{n_n} to investigate number is one less than n because index
		number = n -1

		# array of Sn with uncertainties shape: (n,)
		Sn_data = angular_dispersion_uncertainties(tdata,number+1,position_angles,PT=True,redshift=True)

		print ('Sn %i: %.3f, standard deviation: %.8f'%(number+1,Sn_data[number].nominal_value,Sn_data[number].std_dev) )

		print ('Does parallel_transport')

		av_Sn_mc_n = np.mean(Sn_mc['S_%i'%n])
		sigma = np.std(Sn_mc['S_%i'%n])

		SL = 1 - norm.cdf(   (Sn_data[number].nominal_value - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL data : %.3f'%np.log10(SL))

		upperboundSn = Sn_data[number].nominal_value + Sn_data[number].std_dev
		SL = 1 - norm.cdf(   (upperboundSn - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL upper bound: %.3f'%np.log10(SL))
		
		lowerboundSn = Sn_data[number].nominal_value - Sn_data[number].std_dev

		SL = 1 - norm.cdf(   (lowerboundSn - av_Sn_mc_n) / (sigma)   )
		print ('lgo10 SL lower bound: %.3f'%np.log10(SL))



	produce_error_on_SN()

def uncertainty_in_Sn_VA_redshift_2D():
	file = 'value_added_selection'
	tdata = Table(fits.open('../%s.fits'%file)[1].data)
	print ('Using redshift sources..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]

	# get flux bin 3 of VA_MG
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']
	tdata['final_PA'] = tdata['position_angle']
	tdata['RA'], tdata['DEC'] = tdata['RA_2'], tdata['DEC_2']
	
	# the simulated original (no error) SN_MC
	# filename = 'MG_Zdata_original'
	# Sn_mc = './data/Sn_monte_carlo_MG'+filename+'.fits'  # needs rerunning with PT probably
	Sn_mc = './tmp/Sn_monte_carlo_PTvalue_added_selection_PA_only_redshift_sources_flux3.fits'
	Sn_mc = Table(fits.open(Sn_mc)[1].data)
	n_begin = 60
	n = 500

	# for two figures in the discussion showing dist of Sn_mc and Sn_data
	def produce_Sn_plots():
		Sn_data = angular_dispersion_vectorized_n(tdata,n)

		plt.hist(Sn_mc['S_%i'%n],bins=25,histtype=u'step',color='black',label=r'$S_n\vert_{MC}$')
		
		# print (Sn_data[n-1])
		# plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed', label='Data no PT')

		Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n)
		print (Sn_data[n-1])
		plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed',color='k',label='Data')
		
		ax = plt.gca() 
		ax.tick_params(labelsize=12)
		ax.set_xlabel(r'$S_{%i}$'%n,fontsize=14)	
		ax.set_ylabel(r'Counts',fontsize=14)
		ax.legend(fontsize=14)
		title = 'Distribution of S_%i'%n
		title = ' '
		plt.title(title,fontsize=16)
		if n == 140: plt.xlim(0.040,0.12151429413439714)
		plt.tight_layout()

		plt.show()

		Sn_mc_n = np.asarray(Sn_mc['S_'+str(n)])
		Sn_data_n = Sn_data[n-1]
		av_Sn_mc_n = np.mean(Sn_mc_n)
		sigma = np.std(Sn_mc_n)

		print sigma
		print ('S_%i = %.3f'%(n,Sn_data_n))
		
		SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL = %.3f for n= %i, calculated with PT'%(np.log10(SL),n) )

	produce_Sn_plots()
	''' for calculating the change when position angles are given Gaussian noise '''
	def produce_error_on_SN():
		error = np.random.normal(0,5,len(tdata)) # add gaussian noise centered on the PA with std 5.
		
		position_angles = tdata['position_angle']

		std = 5
		print ('Using std = ',std)
		# unumpy deals with standard deviation so use np.abs
		position_angles = unumpy.uarray(position_angles,std) #np.abs(error)

		# Number of S_{n_n} to investigate number is one less than n because index
		number = n -1

		# array of Sn with uncertainties shape: (n,)
		Sn_data = angular_dispersion_uncertainties(tdata,number+1,position_angles,PT=True)

		print ('Sn %i: %.3f, standard deviation: %.8f'%(number+1,Sn_data[number].nominal_value,Sn_data[number].std_dev) )

		print ('Does parallel_transport')

		av_Sn_mc_n = np.mean(Sn_mc['S_%i'%n])
		sigma = np.std(Sn_mc['S_%i'%n])

		SL = 1 - norm.cdf(   (Sn_data[number].nominal_value - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL data : %.3f'%np.log10(SL))

		upperboundSn = Sn_data[number].nominal_value + Sn_data[number].std_dev
		SL = 1 - norm.cdf(   (upperboundSn - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL upper bound: %.3f'%np.log10(SL))
		
		lowerboundSn = Sn_data[number].nominal_value - Sn_data[number].std_dev

		SL = 1 - norm.cdf(   (lowerboundSn - av_Sn_mc_n) / (sigma)   )
		print ('lgo10 SL lower bound: %.3f'%np.log10(SL))



	produce_error_on_SN()


def uncertainty_in_Sn_VA_MG_redshift_2D_60_140():
	'''For in the paper, plotting n=60 and n=140 in same plot'''
	file = 'value_added_selection_MG'
	tdata = Table(fits.open('../%s.fits'%file)[1].data)
	print ('Using redshift sources..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]

	# get flux bin 3 of VA_MG
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']
	tdata['final_PA'] = tdata['position_angle']
	tdata['RA'], tdata['DEC'] = tdata['RA_2'], tdata['DEC_2']
	
	# the simulated original (no error) SN_MC
	# filename = 'MG_Zdata_original'
	# Sn_mc = './data/Sn_monte_carlo_MG'+filename+'.fits'  # needs rerunning with PT probably
	Sn_mc = './tmp/Sn_monte_carlo_PTvalue_added_selection_MG_PA_only_redshift_sources_flux3.fits'
	Sn_mc = Table(fits.open(Sn_mc)[1].data)
	n_begin = 60
	n = 140

	# for two figures in the discussion showing dist of Sn_mc and Sn_data
	def produce_Sn_plots():
		Sn_data = angular_dispersion_vectorized_n(tdata,n)

		prop_cycle = plt.rcParamsDefault['axes.prop_cycle']
		colors = prop_cycle.by_key()['color']

		if n == 60:
			plt.hist(Sn_mc['S_%i'%n],bins=25,histtype=u'step',color=colors[0],label=r'$S_{%i}\vert_{MC}$'%n)
		elif n == 140:
			plt.hist(Sn_mc['S_%i'%n],bins=25,histtype=u'step',color=colors[1],label=r'$S_{%i}\vert_{MC}$'%n)

		# print (Sn_data[n-1])
		# plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed', label='Data no PT')

		Sn_data = angular_dispersion_vectorized_n_parallel_transport(tdata,n)
		print (Sn_data[n-1])
		if n == 60:
			plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed',label=r'$S_{60}$',color=colors[0])
		elif n == 140:
			plt.axvline(x=Sn_data[n-1],ymin=0,ymax=1,ls='dashed',label=r'$S_{140}$',color=colors[1])
		
		ax = plt.gca() 
		ax.tick_params(labelsize=12)
		ax.set_xlabel(r'$S_n$',fontsize=14)	
		ax.set_ylabel(r'Counts',fontsize=14)
		ax.legend(fontsize=14)
		title = 'Distribution of S_%i'%n
		title = ' '
		plt.title(title,fontsize=16)
		# if n == 140: plt.xlim(0.040,0.12151429413439714)


		Sn_mc_n = np.asarray(Sn_mc['S_'+str(n)])
		Sn_data_n = Sn_data[n-1]
		av_Sn_mc_n = np.mean(Sn_mc_n)
		sigma = np.std(Sn_mc_n)

		print sigma
		print ('S_%i = %.3f'%(n,Sn_data_n))
		
		SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
		print ('log10 SL = %.3f for n= %i, calculated with PT'%(np.log10(SL),n) )

	n = 60
	produce_Sn_plots()
	n = 140
	produce_Sn_plots()

	plt.tight_layout()
	plt.show()

def SL_test_bij_mean_en_ver_weg():

	# only redshift sources n = 140
	Sn_data_n = 0.10853961910118307
	av_Sn_mc_n = 0.0744707874591625
	sigma = 0.01011699520823149

	SL = (1 - norm.cdf( (Sn_data_n - av_Sn_mc_n) / sigma ))
	print ('SL = ', SL)

	std = 0.003 # test

	# upper limit
	SL = (1 - norm.cdf( (Sn_data_n+std - av_Sn_mc_n) / sigma))
	print ('SL upper = ', SL)
	# lower limit 
	SL = (1 - norm.cdf( (Sn_data_n-std - av_Sn_mc_n) / sigma))
	print ('SL lower = ', SL)

	# redshift sources n = 140
	Sn_data_n = 0.0714254169327698
	av_Sn_mc_n = 0.07593616455063615
	sigma = 0.010336677073948232

	SL = (1 - norm.cdf( (Sn_data_n - av_Sn_mc_n) / sigma ))
	print ('SL = ', SL)

	# upper limit
	SL = (1 - norm.cdf( (Sn_data_n+std - av_Sn_mc_n) / sigma))
	print ('SL upper = ', SL)
	# lower limit 
	SL = (1 - norm.cdf( (Sn_data_n-std - av_Sn_mc_n) / sigma))
	print ('SL lower = ', SL)


# Nutteloos, <Sn_mc> volgt wel n binnen 1 dataset, sigma_n volgt wel een beetje 0.33/sqrt(N)
# maar, dan moet je alsnog elke dataset apart doen
def fit_random_data(file):
	''' See if there is a fit between Sn_MC ~ n (and N) and sigma_n ~ N'''

	# Look at the five subsets

	# first investigate how Sn_MC goes as function of nearest neighbours 
	def sn_mc_vs_n():
		filename = 'biggest_selection'
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)

		all_N = []
		all_Sn_MC = []
		N = len(tdata)

		Sn_mc = Table(fits.open('../scripts/data/Sn_monte_carlo_PT'+filename+'.fits')[1].data) 
		n = len(Sn_mc.colnames) # the amount of columns is the amount of probed neighbours
		print ('%s, N = %i, n=%i'%(filename,N,n))
		n_range = range(1,n+1)

		# calculate <Sn_MC> for all n
		all_avg_Sn = []
		for i in n_range:
			all_avg_Sn.append(np.mean(Sn_mc['S_'+str(i)]))

		plt.plot(n_range,all_avg_Sn,label='<Sn_mc>')

		# S_n = S_20 * 1/sqrt(n) * sqrt(20)
		dependency = all_avg_Sn[19]/np.sqrt(n_range) * np.sqrt(20)

		plt.plot(n_range,dependency,label='1/sqrt(n)',ls='dashed')
		plt.xlabel('n',fontsize=14)
		plt.ylabel('Sn',fontsize=14)
		plt.legend(fontsize=14)
		# plt.xlim(20,n)
		plt.show()

	# sn_mc_vs_n()

	# then investigate how sigma goes as function of N
	all_N = []
	all_median_std_Sn = []
	all_Sn_MC_100 = []
	
	def find_median_std_N(filename):
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		tdata_original = tdata
		def helper_func(filename):
			N = len(tdata)
			Sn_mc = Table(fits.open('../scripts/data/2D/Sn_monte_carlo_PT'+filename+'.fits')[1].data) 
			n = len(Sn_mc.colnames) # the amount of columns is the amount of probed neighbours
			print ('%s, N = %i, n=%i'%(filename,N,n))
			n_range = range(1,n+1)

			all_std_Sn = []
			for i in n_range:
				all_std_Sn.append(np.std(Sn_mc['S_'+str(i)]))

			all_median_std_Sn.append(np.median(all_std_Sn))
			all_N.append(N)
			all_Sn_MC_100.append(np.mean(Sn_mc['S_100']))

		filename += '_PA'
		filename_original = filename
		helper_func(filename)
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
		for key in fluxbins:
			tdata = fluxbins[key]
			filename = filename_original + 'flux%s'%key
			helper_func(filename)

		sizebins = select_size_bins_cuts_biggest_selection(tdata_original)
		for key in sizebins:
			tdata = sizebins[key]
			filename = filename_original + 'size%s'%key
			if len(tdata) != 0:
				helper_func(filename)

	filename = 'biggest_selection'
	find_median_std_N(filename)

	filename = 'value_added_selection'
	find_median_std_N(filename)

	filename = 'value_added_selection_MG'
	find_median_std_N(filename)

	filename = 'value_added_selection_NN'
	find_median_std_N(filename)

	filename = 'value_added_compmatch'
	find_median_std_N(filename)

	argsort = np.argsort(all_N)
	all_N = np.asarray(all_N)[argsort]
	all_median_std_Sn = np.asarray(all_median_std_Sn)[argsort]
	all_Sn_MC_100 = np.asarray(all_Sn_MC_100)[argsort]

	plt.plot(all_N,all_median_std_Sn,label='data')
	dependency = all_median_std_Sn[0] / np.sqrt(all_N) * np.sqrt(all_N[0])
	plt.plot(all_N,dependency, label='0.33/sqrt(N)')
	plt.xlabel('N',fontsize=14)
	plt.ylabel(r'$\sigma_n$',fontsize=14)
	plt.legend(fontsize=14)
	plt.show()

	plt.plot(all_N,all_Sn_MC_100,label='<S_100_mc>')
	plt.xlabel('N')
	plt.ylabel('S_100_mc')
	plt.show()


# error_prop('all_sl_MG')
# error_prop('all_sl')
# error_prop('all_sl_MG_Z')

# error_in_SL_plot('all_sl')

# distribution_of_Sn('all_sn_MG_Z')

# uncertainty_in_Sn('value_added_selection_MGflux3')
# uncertainty_in_Sn_VA_MG_redshift_2D()
# uncertainty_in_Sn_VA_MG_redshift_3D()
# uncertainty_in_Sn_VA_redshift_2D()

# SL_test_bij_mean_en_ver_weg()

# fit_random_data('boeie')

uncertainty_in_Sn_VA_MG_redshift_2D_60_140()
