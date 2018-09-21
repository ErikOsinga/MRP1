import sys
sys.path.insert(0,'/net/reusel/data1/osinga/anaconda2')

import numpy as np 
import numexpr as ne

from astropy.io import fits
from astropy.table import Table, join, vstack

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process

import matplotlib.pyplot as plt

import math

from scipy.spatial import cKDTree

from utils import (select_on_size, load_in, distanceOnSphere
	, angular_dispersion_vectorized_n, deal_with_overlap
	, angular_dispersion_vectorized_n_parallel_transport)


######### SETUP ################ 
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata

# tdata = fits.open('./'+filename'.fits')
# tdata = Table(tdata[1].data)

# tdata = deal_with_overlap(tdata)

# Parameters
n_sim = 1000
n = 180
n_cores = 4 #multiprocessing.cpu_count()

PT = True

################################
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

def random_data(tdata):
	'''
	Makes random data with the same no. of sources in the same area
	Better: just shuffle the array of position angles
	'''
	maxra = np.max(tdata['RA'])
	minra = np.min(tdata['RA'])
	maxdec = np.max(tdata['DEC'])
	mindec = np.min(tdata['DEC'])
	minpa = np.min(tdata['position_angle'])
	maxpa = np.max(tdata['position_angle'])
	length = len(tdata)

	rdata=Table()
	rdata['RA'] = np.random.randint(minra,maxra,length)
	rdata['DEC'] = np.random.randint(mindec,maxdec,length)
	rdata['position_angle'] = np.random.randint(minpa,maxpa,length)
	return rdata

def parallel_datasets(number_of_simulations=n_sim/n_cores):
	print 'Number of simulations per core:' + str(number_of_simulations)
	np.random.seed() # changes the seed
	Sn_datasets = []
	
	ra = np.asarray(tdata['RA'])
	dec = np.asarray(tdata['DEC'])
	pa = np.asarray(tdata['final_PA'])
	length = len(tdata)

	max_ra = np.max(ra)
	min_ra = np.min(ra)
	max_dec = np.max(dec)
	min_dec = np.min(dec)
	
	rdata = Table()
	for i in range(number_of_simulations):
		np.random.shuffle(pa)
		rdata['RA'] = ra
		rdata['DEC'] = dec
		rdata['final_PA'] = pa
		if PT:
			Sn = angular_dispersion_vectorized_n_parallel_transport(rdata,n=n,redshift=False) # up to n nearest neighbours
		else:
			Sn = angular_dispersion_vectorized_n(rdata,n=n) # up to n nearest neighbours
		Sn_datasets.append(Sn)

	return Sn_datasets

def monte_carlo(totally_random=False,filename=''):
	'''
	Make (default) n_sim random data sets and calculate the Sn statistic

	If totally_random, then generate new positions and position angles instead
	of shuffeling the position angles among the sources
	'''
					# a 4 x 250 x n array for 1000 simulations on a 4 core system.
	Sn_datasets = [] # a n_core x n_sim/n_core x n array containing n_sim simulations of n different S_n
	print 'Starting '+ str(n_sim) + ' Monte Carlo simulations for n = 0 to n = ' + str(n)
	
	p = Pool(n_cores)
	# set up 4 processes that each do 1/num_cores simulations on a num_cores system
	print 'Using ' +str(n_cores) + ' cores'
	processes = [p.apply_async(parallel_datasets) for _ in range(n_cores)]
	# get the results into a list
	[Sn_datasets.append(process.get()) for process in processes]
	p.close()
		
	Sn_datasets = np.asarray(Sn_datasets)
	print 'Shape of Sn_datasets: ', Sn_datasets.shape
	Sn_datasets = Sn_datasets.reshape((n_sim,n))

	Result = Table()
	for i in range(1,n+1):
		Result['S_'+str(i)] = Sn_datasets[:,i-1] # the 0th element is S_1 and so on..

	if totally_random:
		np.save('./TR_Sn_monte_carlo_'+filename,Sn_datasets)
		Result.write('./TR_Sn_monte_carlo_'+filename+'.fits',overwrite=True)
	else:
		np.save('./data/Sn_monte_carlo_MG'+filename,Sn_datasets)
		Result.write('./data/Sn_monte_carlo_MG'+filename+'.fits',overwrite=True)


filename = '../value_added_selection_MG'
big_data = Table(fits.open('./'+filename+'.fits')[1].data)
big_data['final_PA'] = big_data['position_angle']
big_data['RA'], big_data['DEC'] = big_data['RA_2'], big_data['DEC_2']

# select only redshift data
print ('Using redshift..') ## edited for not actually using z
z_available = np.invert(np.isnan(big_data['z_best']))
z_zero = big_data['z_best'] == 0#
# also remove sources with redshift 0, these dont have 3D positions
z_available = np.logical_xor(z_available,z_zero)
print ('Number of sources with available redshift:', np.sum(z_available))
big_data = big_data[z_available]

#select highest flux bin
big_data = select_flux_bins_cuts_biggest_selection(big_data)['3']

X = 25
# X = 10
# for 0 to 10, 10 to 20, 20 to 30, etc 
# call this script with 0, 10, 20, 30 etc. (0,25,50,75 if X = 25)
number = int(sys.argv[1]  ) # to run on multiple computers (or nodes)
print number

for run in range(0+number,X+number):
	print 'Run number: %i' %run
	i = 0
	filename = 'MG_Zdata_%i'%run
	if PT: filename += 'PT'
	tdata = big_data
	error = np.random.normal(0,5,len(tdata)) # add gaussian noise centered on the PA with std 5.
	tdata['final_PA'] += error
	tdata.write('./data/tdata_with_error/MG_Z'+filename+'.fits',overwrite=True)
	# monte carlo is always executed for Table tdata and 'position_angle'
	monte_carlo(totally_random=False,filename=filename)

# also do without errors
print ('Doing data without errors:')
filename = 'MG_Zdata_original'
if PT: filename += 'PT'
tdata = big_data 
tdata.write('./data/tdata_with_error/MG'+filename+'.fits',overwrite=True)
monte_carlo(totally_random=False,filename=filename)