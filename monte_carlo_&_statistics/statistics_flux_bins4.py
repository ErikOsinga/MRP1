import sys

import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, join, vstack

from scipy.spatial import cKDTree
from scipy.stats import norm

import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt


# import seaborn as sns
# sns.set()
# plt.rc('text', usetex=True)

from utils import angular_dispersion_vectorized_n, distanceOnSphere, deal_with_overlap


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
		Sn_mc_n = np.average(np.asarray(Sn_mc['S_'+str(n+starting_n)]))
		Sn_data_n = Sn_data[n]
		# print Sn_data_n, Sn_mc_n
		SL = 1 - norm.cdf(   (Sn_data_n - Sn_mc_n) / (sigma)   )
		all_sn.append(Sn_data_n)
		all_sl.append(SL)
		all_sn_mc.append(Sn_mc_n)
		all_std.append(np.std(np.asarray(Sn_mc['S_'+str(n+starting_n)])))
		# print SL
	
	Results = Table()
	Results['n'] = range(starting_n,ending_n+1)
	Results['Sn_data'] = all_sn
	Results['SL'] = all_sl
	Results['Sn_mc'] = all_sn_mc
	Results.write('./Statistics_'+filename+'_results.fits',overwrite=True)
	
	fig, axarr = plt.subplots(2, sharex=True, gridspec_kw= {'height_ratios':[3, 1]})
	axarr[0].plot(range(starting_n,ending_n+1),np.log(all_sl))
	# print np.argmin(all_sl)
	axarr[0].set_title('SL vs n for '+filename)
	axarr[0].set_ylabel('Log_e SL')
	axarr[0].set_xlim(2,180)
	axarr[0].set_ylim(-4,0)
	
	axarr[1].plot(range(starting_n,ending_n+1),all_std)
	axarr[1].set_xlabel('n')
	axarr[1].set_ylabel('sigma')

	plt.savefig('./SL_vs_n_'+filename+'.pdf',overwrite=True)
	plt.close()

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
	plt.savefig('./angular_radius_vs_n_'+filename+'.pdf',overwrite=True)
	plt.close()
		
	return median

n = 180

filename = 'biggest_selection'
tdata = Table(fits.open('./'+filename+'.fits')[1].data)
tdata = deal_with_overlap(tdata)
for i in range(20):
	# Open the size data
	threshold = (11.3 - i/5.)
	filename = 'flux_bins4_%.3f'%threshold
	ndata = tdata[tdata['Peak_flux'] > threshold]
	Sn_mc = Table(fits.open('./Sn_monte_carlo_'+filename+'.fits')[1].data)
	# Calculate the Sn and the plots
	angular_radius = angular_radius_vs_n(ndata,filename,n)
	Sn_data = angular_dispersion_vectorized_n(ndata,n)
	Sn_vs_n(ndata,Sn_mc,Sn_data,filename,angular_radius,n)
	