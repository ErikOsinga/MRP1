import sys
sys.path.insert(0,'/net/reusel/data1/osinga/anaconda2')

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
	, angular_dispersion_vectorized_n, angular_dispersion_vectorized_n_parallel_transport)

from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins
		,select_power_bins_cuts_VA_selection, select_physical_size_bins_cuts_VA_selection
		,select_flux_bins_cuts_biggest_selection, select_size_bins_cuts_biggest_selection)

######### SETUP ################ 
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata

# Parameters
n_sim = 960
n = 800
n_cores = 4 #multiprocessing.cpu_count()

parallel_transport = True
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
	minpa = np.min(tdata['final_PA'])
	maxpa = np.max(tdata['final_PA'])
	length = len(tdata)

	rdata=Table()
	rdata['RA'] = np.random.randint(minra,maxra,length)
	rdata['DEC'] = np.random.randint(mindec,maxdec,length)
	rdata['final_PA'] = np.random.randint(minpa,maxpa,length)
	return rdata

def parallel_datasets(number_of_simulations=n_sim/n_cores):
	print 'Number of simulations per core:' + str(number_of_simulations)
	np.random.seed() # changes the seed
	Sn_datasets = []
	
	ra = np.asarray(tdata['RA'])
	dec = np.asarray(tdata['DEC'])
	pa = np.asarray(tdata['final_PA'])
	length = len(tdata)
	if redshift: z_best = np.asarray(tdata['z_best'])

	max_ra = np.max(ra)
	min_ra = np.min(ra)
	max_dec = np.max(dec)
	min_dec = np.min(dec)
	
	rdata = Table()
	for i in range(number_of_simulations):
		np.random.shuffle(pa)
		rdata['RA'] = ra
		rdata['DEC'] = dec
		if redshift: rdata['z_best'] = z_best
		rdata['final_PA'] = pa
		if parallel_transport:
			Sn = angular_dispersion_vectorized_n_parallel_transport(rdata,n=n,redshift=redshift) # up to n nearest neighbours
		else:
			Sn = angular_dispersion_vectorized_n(rdata,n=n,redshift=redshift) # up to n nearest neighbours
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
	if Sn_datasets.shape[2] < n:
		# hard fix when n > len(tdata)
		Sn_datasets = Sn_datasets.reshape((n_sim,len(tdata)-1)) 
	else:
		Sn_datasets = Sn_datasets.reshape((n_sim,n))

	Result = Table()

	if n > len(tdata):
		for i in range(1,len(tdata)):
			Result['S_'+str(i)] = Sn_datasets[:,i-1] # the 0th element is S_1 and so on..
	else:
		for i in range(1,n+1):
			Result['S_'+str(i)] = Sn_datasets[:,i-1] # the 0th element is S_1 and so on..

	if parallel_transport:
		# np.save('./data/Sn_monte_carlo_PT'+filename,Sn_datasets)
		if n < 998:
			Result.write('./data/Sn_monte_carlo_PT'+filename+'.fits',overwrite=True)
		else:
			print ('Creating csv file')
			Result.write('./data/Sn_monte_carlo_PT'+filename+'.csv',overwrite=True)

	else:
		# np.save('./data/Sn_monte_carlo_'+filename,Sn_datasets)
		if n < 998:
			Result.write('./data/Sn_monte_carlo_'+filename+'.fits',overwrite=True)
		else:
			print ('Creating csv file')
			Result.write('./data/Sn_monte_carlo_'+filename+'.csv',overwrite=True)




# Running all statistics with redshift

equal_width = False # if false use cuts defined in VA biggest selection
redshift = True
 
filename = 'value_added_selection'
print (filename)
tdata = Table(fits.open('../%s.fits'%filename)[1].data)
if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
tdata['RA'] = tdata['RA_2']
tdata['DEC'] = tdata['DEC_2'] # cant use biggest_selection so no try except

# With redshift
print ('Using redshift..') ## edited for not actually using z
z_available = np.invert(np.isnan(tdata['z_best']))
z_zero = tdata['z_best'] == 0#
# also remove sources with redshift 0, these dont have 3D positions
z_available = np.logical_xor(z_available,z_zero)
print ('Number of sources with available redshift:', np.sum(z_available))
# filename += '_only_redshift_sources_' ## edited for not actually using z
if position_angle: filename += '_PA' # only when using 'position_angle'
filename += '_redshift_z_above1'

tdata_original = tdata
filename_original = filename

tdata = tdata[z_available]

spectroscopic = np.where(tdata['z_best'] > 1)
tdata = tdata[spectroscopic]
tdata_original = tdata
print ('Using 1 < redshift only')

# # THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
monte_carlo(totally_random=False,filename=filename)


fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
for key in fluxbins:
	tdata = fluxbins[key]
	filename = filename_original + 'flux%s'%key
	
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	assert np.sum(np.invert(z_available)) == 0, "These sources should all have redshift.."
	tdata = tdata[z_available]
	
	monte_carlo(totally_random=False,filename=filename)

sizebins = select_size_bins_cuts_biggest_selection(tdata_original)
for key in sizebins:
	tdata = sizebins[key]
	filename = filename_original + 'size%s'%key
	
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	assert np.sum(np.invert(z_available)) == 0, "These sources should all have redshift.."
	tdata = tdata[z_available]
	
	monte_carlo(totally_random=False,filename=filename)

########################################################################################

#Running all the statistics with redshift , but not using redshift
equal_width = False # if false use cuts defined in VA biggest selection
redshift = False
 
filename = 'value_added_selection'
print (filename)
tdata = Table(fits.open('../%s.fits'%filename)[1].data)
if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
tdata['RA'] = tdata['RA_2']
tdata['DEC'] = tdata['DEC_2']

# With redshift
print ('Using redshift.. but only for selecting sources') ## edited for not actually using z
z_available = np.invert(np.isnan(tdata['z_best']))
z_zero = tdata['z_best'] == 0#
# also remove sources with redshift 0, these dont have 3D positions
z_available = np.logical_xor(z_available,z_zero)
print ('Number of sources with available redshift:', np.sum(z_available))
if position_angle: filename += '_PA' # only when using 'position_angle'
filename += '_only_redshift_sources_z_above1'

tdata_original = tdata
filename_original = filename

tdata = tdata[z_available]

spectroscopic = np.where(tdata['z_best'] > 1)
tdata = tdata[spectroscopic]
tdata_original = tdata # use redshift data for power bins and size bins !!

print ('Using 1 < redshift only')

# # THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
monte_carlo(totally_random=False,filename=filename)

fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)
for key in fluxbins:
	tdata = fluxbins[key]
	filename = filename_original + 'flux%s'%key
	
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	assert np.sum(np.invert(z_available)) == 0, "These sources should all have redshift.."
	tdata = tdata[z_available]
	
	monte_carlo(totally_random=False,filename=filename)


sizebins = select_size_bins_cuts_biggest_selection(tdata_original)
for key in sizebins:
	tdata = sizebins[key]
	filename = filename_original + 'size%s'%key
	
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	assert np.sum(np.invert(z_available)) == 0, "These sources should all have redshift.."
	tdata = tdata[z_available]
	
	monte_carlo(totally_random=False,filename=filename)
