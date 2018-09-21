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

from utils import (select_on_size, load_in, distanceOnSphere, parallel_transport)

from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins)

'''
Correct because this uses the tricks that can only be used in a full sample
such as not having to calculate nearest neighbours and 
only having to calculate di for 1 source IF not using parallel transport
'''



######### SETUP ################ 
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata

# Parameters
n_sim = 10000 #1e3

filename = sys.argv[1]
print (filename)
tdata = Table(fits.open('../%s.fits'%filename)[1].data)

n = len(tdata)
n_cores = 4 #multiprocessing.cpu_count()

parallel_transportbool = True
print ('Using parallel_transport = %s' %parallel_transportbool)

position_angle = sys.argv[2]
if position_angle == 'True':
	position_angle = True
else:
	position_angle = False

if position_angle:
	print ('Using position_angle')
else: 
	print ('Using final_PA')

if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
if position_angle: filename += '_PA' # only when using 'position_angle'

filename += 'correct'

def select_uniform_sources(tdata):
	''' select uniform distributed number of position angles and assign them to tdata '''

	print ('Replacing all position angles with uniform selected position angles U[0,180)')
	number_of_sources = len(tdata)
	uniform_position_angles = np.random.rand(number_of_sources) * 180
	tdata['final_PA'] = uniform_position_angles

	return tdata

def select_random_sources(number_of_sources,tdata):
	random_selection = np.random.choice(len(tdata),size=number_of_sources,replace=False)
	
	tdata = tdata[random_selection]

	return tdata

def angular_dispersion(tdata):
	'''
	Calculates and returns the Sn statistic for tdata
	with number of sources n closest to source i
	
	# n = number of sources closest to source i, including itself
	# e.g. n=5 implies 4 nearest neighbours
	# N = number of sources

	Returns Sn, as a float.

	'''
	N = len(tdata)
	n = N
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['final_PA'])

	# No parallel trasnport
	position_angles_array = position_angles

	x = np.radians(2*position_angles_array) # use numexpr to speed it up quite significantly

	assert len(x) == N

	# now only calculate the dispersion for one source (same for all sources)
	di_max = 1./n * ( (np.sum(ne.evaluate('cos(x)')))**2 
				+ (np.sum(ne.evaluate('sin(x)')))**2 )**0.5
	
	Sn = di_max # Value containing Sn is same as di_max if n=N

	return Sn

def angular_dispersion_parallel_transport(tdata):
	'''
	Calculates and returns the Sn statistic for tdata
	with number of sources N (FULL SAMPLE) closest to source i
	
	# N = number of sources

	Returns S_N =Sum d_i,N|max for all sources, as a float.

	'''
	N = len(tdata)
	n = N
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['final_PA'])

	# for every source: find n closest neighbours, calculate max dispersion using PT
	position_angles_array = np.zeros((N,n)) # array of shape (N,n) that contains position angles
	# create indices for nearest neighbours by removing the current source in the loop (no reason to search for them)
	index_NN_initial = np.arange(len(tdata))
	for i in range(len(tdata)):
		sys.stdout.write("\r%i"%i)
		sys.stdout.flush()
		#transport all angles to source i, aka the nearest neighbours are all sources except source i
		index_NN = np.delete(index_NN_initial,i)
		angles_transported = parallel_transport(RAs[i],DECs[i],RAs[index_NN],DECs[index_NN],position_angles[index_NN])
		# Then concatenate the transported angles to the current source position angle and store it
		position_angles_array[i] = np.concatenate(([position_angles[i]],angles_transported))
	
	position_angles_array = np.array(position_angles_array)
	assert position_angles_array.shape == (N,n)

	x = np.radians(2*position_angles_array) # use numexpr to speed it up quite significantly

	di_max = 1./N * ( (np.sum(ne.evaluate('cos(x)'),axis=1))**2 
				+ (np.sum(ne.evaluate('sin(x)'),axis=1))**2 )**0.5

	assert di_max.shape == (N,) # array of max_di for every source

	Sn = 1./N * np.sum(di_max) # S_N
	
	return Sn

################################
def random_data(tdata):
	'''
	Makes random data with the same no. of sources in the same area
	Better: just shuffle the array of position angles, UNLESS TESTING GLOBALLY
	'''
	maxra = np.max(tdata['RA'])
	minra = np.min(tdata['RA'])
	maxdec = np.max(tdata['DEC'])
	mindec = np.min(tdata['DEC'])
	minpa = np.min(tdata['final_PA'])
	maxpa = np.max(tdata['final_PA'])

	length = len(tdata)
	rdata=Table()

	# keep the same positions 
	rdata['RA'] = np.asarray(tdata['RA'])
	rdata['DEC'] = np.asarray(tdata['DEC'])
	# but unform distribute angles along sources
	rdata['final_PA'] = np.random.rand(length)*180
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
		print ('Number simulation: %i/%i'%(i,number_of_simulations))
		# TOTALLY RANDOM SIMULATION
		rdata['RA'] = ra
		rdata['DEC'] = dec
		if redshift: rdata['z_best'] = z_best
		
		# keep the same positions 
		rdata['RA'] = np.asarray(tdata['RA'])
		rdata['DEC'] = np.asarray(tdata['DEC'])
		# but uniform distribute angles along sources
		rdata['final_PA'] = np.random.rand(length)*180
		
		if parallel_transportbool:
			Sn = angular_dispersion_parallel_transport(rdata)#,redshift # For n nearest neighbours
		else:
			Sn = angular_dispersion(rdata)#,redshift # For n nearest neighbours
		Sn_datasets.append(Sn)

	return Sn_datasets

def monte_carlo(totally_random=False,filename=''):
	'''
	Make (default) n_sim random data sets and calculate the Sn statistic

	If totally_random, then generate new positions and position angles instead
	of shuffeling the position angles among the sources
	'''
	if not totally_random:
		raise ValueError('Totally random is hardcoded, so try again with totally_random=True')

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
	Sn_datasets = Sn_datasets.reshape((n_sim,))

	Result = Table()
	Result['S_N'] = Sn_datasets # one row with 1000 simulated S_N values

	if parallel_transportbool:
		# np.save('./data/Sn_monte_carlo_PT'+filename,Sn_datasets)
		Result.write('./data/Sn_monte_carlo_PT_full_sample_'+filename+'.fits',overwrite=True)
	else:
		# np.save('./data/Sn_monte_carlo_'+filename,Sn_datasets)
		Result.write('./data/Sn_monte_carlo_full_sample_'+filename+'.fits',overwrite=True)



'''
#Running just one file
redshift = True

filename = sys.argv[1]
print (filename)
# tdata = Table(fits.open('../redshift_bins/%s.fits'%filename)[1].data)
tdata = Table(fits.open('../%s.fits'%filename)[1].data)

tdata['RA'] = tdata['RA_2']
tdata['DEC'] = tdata['DEC_2']

z_available = np.invert(np.isnan(tdata['z_best']))  
z_zero = tdata['z_best'] == 0asdasdaw#
# also remove sources with redshift 0, these dont have 3D positions
z_available = np.logical_xor(z_available,z_zero)
print ('Number of sources with available redshift:', np.sum(z_available))
filename += '_redshift_' 

tdata_original = tdata
filename_original = filename

filename += '_astropy'

# fluxbins11 = select_flux_bins11(tdata_original)
# tdata = fluxbins11
# filename = filename_original + 'flux11_360'
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
tdata = tdata[z_available]
monte_carlo(totally_random=False,filename=filename)
'''


#Running all the statistics without redshift
redshift = False

# filename = sys.argv[1]
# print (filename)
# tdata = Table(fits.open('../%s.fits'%filename)[1].data)
# tdata['final_PA'] = tdata['position_angle']

try:
	tdata['RA'] = tdata['RA_2']
	tdata['DEC'] = tdata['DEC_2']
except KeyError:
	pass

# if 'test' in filename:
# number_of_sources = 3000
# print ('Selecting %i random sources from '%number_of_sources,filename)
# tdata = select_random_sources(number_of_sources,tdata)

# elif 'uniform' in filename:
# tdata = select_uniform_sources(tdata) # dit is eigenlijk useless want dat doen we daarna toch ook al 

n = len(tdata)
tdata_original = tdata
filename_original = filename

# # THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
monte_carlo(totally_random=True,filename=filename)

'''
fluxbins = select_flux_bins1(tdata_original)
for key in fluxbins:
	tdata = fluxbins[key]
	filename = filename_original + 'flux%s'%key
	
	monte_carlo(totally_random=False,filename=filename)

sizebins = select_size_bins1(tdata_original)
for key in sizebins:
	tdata = sizebins[key]
	filename = filename_original + 'size%s'%key

	monte_carlo(totally_random=False,filename=filename)

fluxbins11 = select_flux_bins11(tdata_original)
tdata = fluxbins11
filename = filename_original + 'flux11'

monte_carlo(totally_random=False,filename=filename)
'''


'''
#Running all the statistics with redshift
redshift = True
 
filename = sys.argv[1]
print (filename)
tdata = Table(fits.open('../%s.fits'%filename)[1].data)
if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
tdata['RA'] = tdata['RA_2']
tdata['DEC'] = tdata['DEC_2']

# With redshift
print ('Using redshift..') ## edited for not actually using z
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
# # THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
monte_carlo(totally_random=False,filename=filename)


tdata_original = tdata # use redshift data for power bins and size bins !!!
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
	
	monte_carlo(totally_random=False,filename=filename)

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
	
	monte_carlo(totally_random=False,filename=filename)

# fluxbins11 = select_flux_bins11(tdata_original)
# tdata = fluxbins11
# filename = filename_original + 'flux11'

# z_available = np.invert(np.isnan(tdata['z_best']))
# print ('Number of sources with available redshift:', np.sum(z_available))
# tdata = tdata[z_available]

# monte_carlo(totally_random=False,filename=filename)


#Running all the statistics with redshift , but not using redshift
redshift = False
 
filename = sys.argv[1]
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
filename += '_only_redshift_sources_' ## edited for not actually using z

tdata_original = tdata
filename_original = filename

tdata = tdata[z_available]
# # THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
monte_carlo(totally_random=False,filename=filename)


# fluxbins = select_flux_bins(tdata_original) # use original tdata for fluxbins

tdata_original = tdata # use redshift data for power bins
fluxbins = select_power_bins(tdata_original) # use redshift data for power bins
for key in fluxbins:
	tdata = fluxbins[key]
	filename = filename_original + 'power%s'%key
	
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]
	
	monte_carlo(totally_random=False,filename=filename)

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
	
	monte_carlo(totally_random=False,filename=filename)

# fluxbins11 = select_flux_bins11(tdata_original)
# tdata = fluxbins11
# filename = filename_original + 'flux11'

# z_available = np.invert(np.isnan(tdata['z_best']))
# z_zero = tdata['z_best'] == 0#
# # also remove sources with redshift 0, these dont have 3D positions
# z_available = np.logical_xor(z_available,z_zero)
# print ('Number of sources with available redshift:', np.sum(z_available))
# tdata = tdata[z_available]

# monte_carlo(totally_random=False,filename=filename)



'''





