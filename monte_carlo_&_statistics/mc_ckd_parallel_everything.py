import sys

import numpy as np 
import numexpr as ne

from astropy.io import fits
from astropy.table import Table, join, vstack

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process

import math

from scipy.spatial import cKDTree

from utils import (select_on_size, load_in, distanceOnSphere
	, angular_dispersion_vectorized_n, deal_with_overlap)

######### SETUP ################ 
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata
# filename = sys.argv[1] # dont forget to check 

# tdata = fits.open('./'+filename'.fits')
# tdata = Table(tdata[1].data)

# tdata = deal_with_overlap(tdata)

# Parameters
n_sim = 1000
n = 180
n_cores = 20 #multiprocessing.cpu_count()

################################

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
	pa = np.asarray(tdata['position_angle'])
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
		rdata['position_angle'] = pa
		Sn = angular_dispersion_vectorized_n(rdata,n=n) # up to n nearest neighbours
		Sn_datasets.append(Sn)

	return Sn_datasets

def monte_carlo(tdata,totally_random=False,filename=''):
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
		np.save('./Sn_monte_carlo_'+filename,Sn_datasets)
		Result.write('./Sn_monte_carlo_'+filename+'.fits',overwrite=True)


# Simulate tdata (biggest_selection)
# monte_carlo(tdata,totally_random=False,filename=filename)
# Simulate all bins
for i in range(5):
	filename = 'size_bins2_'+str(i)
	tdata = Table(fits.open('./'+filename+'.fits')[1].data)
	tdata = deal_with_overlap(tdata)
	monte_carlo(tdata,totally_random=False,filename=filename)

	filename = 'flux_bins2_'+str(i)
	tdata = Table(fits.open('./'+filename+'.fits')[1].data)
	tdata = deal_with_overlap(tdata)
	monte_carlo(tdata,totally_random=False,filename=filename)




