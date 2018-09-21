import sys
sys.path.insert(0,'/net/reusel/data1/osinga/anaconda2')

import numpy as np 
import numexpr as ne

from astropy.io import fits
from astropy.table import Table, join, vstack

import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process

from matplotlib import pyplot as plt

import math

from scipy.spatial import cKDTree

from utils import (select_on_size, load_in, distanceOnSphere
	, angular_dispersion_vectorized_n, angular_dispersion_vectorized_n_parallel_transport)

from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins)

######### SETUP ################ 
# THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata

# Parameters
n_sim = 960
n = 3890
n_cores = 60 #multiprocessing.cpu_count()

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
		# print ('Simulation number: %i/%i'%(i,number_of_simulations))
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
		# hard fix when n > len(tdata)
		for i in range(1,len(tdata)):
			Result['S_'+str(i)] = Sn_datasets[:,i-1] # the 0th element is S_1 and so on..
	else:
		for i in range(1,n+1):
			Result['S_'+str(i)] = Sn_datasets[:,i-1] # the 0th element is S_1 and so on..

	if parallel_transport:
		# np.save('./data/Sn_monte_carlo_PT'+filename,Sn_datasets)
		if n < 998:
			Result.write('./data/2D/Sn_monte_carlo_PT'+filename+'.fits',overwrite=True)
		else:
			print ('Creating csv file')
			Result.write('./data/2D/Sn_monte_carlo_PT'+filename+'.csv',overwrite=True)
	else:
		if n < 998:
			# np.save('./data/Sn_monte_carlo_'+filename,Sn_datasets)
			Result.write('./data/2D/Sn_monte_carlo_'+filename+'.fits',overwrite=True)
		else:
			print ('Creating csv file')
			Result.write('./data/2D/Sn_monte_carlo_'+filename+'.csv',overwrite=True)


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

equal_width = False

#Running all the statistics without redshift
redshift = False

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

# # THE FUNCTION MONTE_CARLO IS EXECUTED FOR TABLE tdata WITH RA DEC and final_PA
monte_carlo(totally_random=False,filename=filename)


if equal_width: 
	fluxbins = select_flux_bins1(tdata_original)
else:
	fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

for key in fluxbins:
	tdata = fluxbins[key]
	filename = filename_original + 'flux%s'%key
	
	monte_carlo(totally_random=False,filename=filename)

if equal_width:
	sizebins = select_size_bins1(tdata_original)
else:
	sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

for key in sizebins:
	tdata = sizebins[key]
	filename = filename_original + 'size%s'%key

	monte_carlo(totally_random=False,filename=filename)

# fluxbins11 = select_flux_bins11(tdata_original)
# tdata = fluxbins11
# filename = filename_original + 'flux11'

# monte_carlo(totally_random=False,filename=filename)



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





