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

from utils import distanceOnSphere, parallel_transport

from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins)


filename = sys.argv[1]
print (filename)

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

tdata = Table(fits.open('../%s.fits'%filename)[1].data)

if position_angle: tdata['final_PA'] = tdata['position_angle'] # overwrite to use 'position_angle' only
if position_angle: filename += '_PA' # only when using 'position_angle'

filename += 'correct'
# filename += 'correct_test'

def select_uniform_sources(tdata):
	''' select uniform distributed number of position angles and assign them to tdata '''

	# print ('Replacing all position angles with uniform selected position angles U[0,180)')
	number_of_sources = len(tdata)
	uniform_position_angles = np.random.rand(number_of_sources) * 180
	tdata['final_PA'] = uniform_position_angles

	return tdata

def select_random_sources(number_of_sources,tdata):
	'''select 'number of sources' sources from tdata, without replacement '''
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

	# #convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	# x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
	# y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
	# z = np.sin(np.radians(DECs))
	# coordinates = np.vstack((x,y,z)).T

	# #make a KDTree for quick NN searching	
	# from scipy.spatial import cKDTree
	# coordinates_tree = cKDTree(coordinates,leafsize=16)
	
	# # for every source: find n closest neighbours, calculate max dispersion
	# position_angles_array = np.zeros((N,n)) # array of shape (N,n) that contains position angles
	# for i in range(N):
	# 	index_NN = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1] # include source itself
	# 	# Transport every nearest neighbour to the current source i
	# 	angles_transported = parallel_transport(RAs[i],DECs[i],RAs[index_NN[1:]],DECs[index_NN[1:]],position_angles[index_NN[1:]])
	# 	# Then concatenate the transported angles to the current source position angle and store it
	# 	position_angles_array[i] = np.concatenate(([position_angles[i]],angles_transported))

	# position_angles_array = np.array(position_angles_array)

	# assert position_angles_array.shape == (N,n)

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

def tick_function(X):
	return ["%.2f" % z for z in X]

def Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,ending_n=180):
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
	if parallel_transportbool:
		Results.write('./data/Statistics_PT_full_sample_'+filename+'_results.fits',overwrite=True)
	else:
		Results.write('./data/Statistics_full_sample_'+filename+'_results.fits',overwrite=True)
	
	fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
	axarr[0].plot(range(starting_n,ending_n+1),np.log(all_sl))
	# print np.argmin(all_sl)
	axarr[0].set_title('SL vs n for '+filename+'\n\n')
	axarr[0].set_ylabel('Log_e SL')
	axarr[0].set_xlim(2,n)
	axarr[0].set_ylim(-4,0)
	
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

	if parallel_transportbool:
		plt.savefig('./figures/SL_vs_n_PT_full_sample_'+filename+'.png',overwrite=True)
	else:
		plt.savefig('./figures/SL_vs_n_full_sample_'+filename+'.png',overwrite=True)
	# plt.show()
	plt.close()

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

	plt.plot(n_range,median[starting_n-1:]) # index 1 corresponds to n=2
	plt.title('Angular radius vs n for ' + filename)
	plt.ylabel('Mean angular radius (deg)')
	plt.xlabel('n')
	# plt.savefig('/data1/osinga/figures/statistics/show/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	# plt.savefig('./figures/angular_radius_vs_n_'+filename+'.png',overwrite=True)
	plt.close()
		
	return median

def statistics(filename,tdata,redshift=False):
	if parallel_transportbool:
		Sn_mc = Table(fits.open('./data/Sn_monte_carlo_PT_full_sample_'+filename+'.fits')[1].data)
	else:
		Sn_mc = Table(fits.open('./data/Sn_monte_carlo_full_sample_'+filename+'.fits')[1].data)

	if parallel_transportbool:
		Sn_data = angular_dispersion_parallel_transport(tdata)
	else:
		Sn_data = angular_dispersion(tdata)

	Sn_mc = np.asarray(Sn_mc['S_N'])
	avg_Sn_mc = np.average(np.asarray(Sn_mc))
	
	N = len(tdata)
	old_sigma = (0.33/N)**0.5 # OLD and WRONG
	sigma = np.std(Sn_mc)
	print ('Old sigma: %f, New sigma: %f'%(old_sigma,sigma) )
	SL = 1 - norm.cdf(   (Sn_data - avg_Sn_mc) / (sigma)   )
	# print SL

	SL_old = 1 - norm.cdf(   (Sn_data - avg_Sn_mc) / (old_sigma)   )
	print ('SL for old sigma ....... %f'%SL_old)
	
	print ('Sn_data: %f, Sn_mc: %f, Significance Level: %f' %(Sn_data,avg_Sn_mc,SL))

	Result = Table()
	Result['Sn_mc'] = [avg_Sn_mc]
	Result['Sigma'] = sigma 
	Result['Sn_data'] = [Sn_data]
	Result['SL(per cent)'] = [SL*100]

	print ('Significance, per cent: %f'%(SL*100))
	
	Result.write('./%s_statistics_full_sample.fits'%filename,overwrite=True)
	Result.write('./%s_statistics_full_sample.csv'%filename,overwrite=True)

	return SL, Sn_data


if __name__ == '__main__':

	
	#Running all the statistics without redshift
	redshift = False
	tdata = tdata
	n = len(tdata)

	tdata['final_PA'] = tdata['position_angle']
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass

	tdata_original = tdata
	filename_original = filename

	statistics(filename,tdata,redshift)

	


	'''
	#Running all the statistics without redshift
	redshift = False
	n = len(tdata)

	tdata['final_PA'] = tdata['position_angle']
	try:
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
	except KeyError:
		pass

	tdata_initial = tdata
	all_SL = []
	all_SN_data = []
	i = 0
	while i < 10000:

		# if 'test' in filename:
		# number_of_sources = 3000
		# print ('Selecting %i random sources from '%number_of_sources,filename)
		# tdata = select_random_sources(number_of_sources,tdata_initial)	

		# elif 'uniform' in filename:
		tdata = select_uniform_sources(tdata_initial)

		tdata_original = tdata
		filename_original = filename
		
		sl_now, sn_data_now = statistics(filename,tdata,redshift) 
		all_SL.append(sl_now)
		all_SN_data.append(sn_data_now)

		i += 1

	np.save('./all_SL_%s'%filename,all_SL)
	np.save('./all_SN_data_%s'%filename,all_SN_data)
	'''
