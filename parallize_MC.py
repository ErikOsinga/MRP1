import math
import numpy as np 
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from astropy.io import fits
from astropy.table import Table, join, vstack
import sys
sys.path.insert(0, './')
import ScipySpatialckdTree

# def monte_carlo(tdata):
'''
Make 1000 random data sets and calculate the Sn statistic
'''

def inner_product(alpha1,alpha2):
	'''
	Returns the inner product of position angles alpha1 and alpha2
	The inner product is defined in Jain et al. 2014
	
	Assumes input is given in degrees
	+1 indicates parallel -1 indicates perpendicular
	'''
	alpha1, alpha2 = math.radians(alpha1), math.radians(alpha2)
	return math.cos(2*(alpha1-alpha2))

def angular_dispersion(tdata,n=20):
	'''
	Calculates and returns the Sn statistic for tdata
	with number of sources n closest to source i
	
	# n = number of sources closest to source i
	# N = number of sources
	
	'''
	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['position_angle'])

	#convert RAs and DECs to an array that has following layout: [[ra1,dec1],[ra2,dec2],etc]
	coordinates = np.vstack((RAs,DECs)).T
	#make a KDTree for quick NN searching	
	import ScipySpatialckdTree
	coordinates_tree = ScipySpatialckdTree.KDTree_plus(coordinates,leafsize=16)

	d_max = 0
	i_max = 0
	# for i-th source (item) find n closest neighbours
	di_max_set = []
	thetha_max_set = []
	for i, item in enumerate(coordinates):
		# stores the indices of the n nearest neighbours as an array (including source i)
		# e.g. the first source has closest n=2 neighbours:  ' array([0, 3918, 3921]) '
		indices=coordinates_tree.query(item,k=n+1,p=3)[1]
		#coordinates of the n closest neighbours
		nearestN = coordinates[indices]
		nearestRAs = nearestN[:,0]
		nearestDECs = nearestN[:,1]
		# distance in arcmin
		# distance = distanceOnSphere(nearestRAs,nearestDECs,#coordinates of the nearest
		# 						item[0],item[1])*60 #coordinates of the current item

		max_di = 0
		max_theta =  0
		# find the angle for which the dispersion is maximized
		for theta in range(0,180):
			sum_inner_product = 0
			for j in indices:
				sum_inner_product += inner_product(theta,position_angles[j])
			dispersion = 1./n * sum_inner_product
			# should maximalize d_i for theta
			if dispersion > max_di:
				max_di = dispersion
				max_theta = theta

		thetha_max_set.append(max_theta)
		di_max_set.append(max_di)

		# sum_inner_product_cos = 0
		# sum_inner_product_sin = 0
		# for j in indices:
		# 	sum_inner_product_cos += math.cos(2*math.radians(position_angles[j]))
		# 	sum_inner_product_sin += math.sin(2*math.radians(position_angles[j]))
		# di_max = np.abs(max_di - 1./n * (sum_inner_product_cos**2 + sum_inner_product_sin**2)**(1./2))

	# Sn measures the average position angle dispersion of the sets containing every source
	# and its n neighbours. If the condition N>>n>>1 is statisfied then Sn is expected to be
	# normally distributed (Contigiani et al.) 
	Sn = 1./N * np.sum(np.asarray(di_max_set))
	sigma = (0.33/N)**0.5
	# SL = 1 - cdfnorm( (Sn - Snmc) / sigma )

	return Sn

def Monte_carlo(n):
	Sn_datasets = []
	amount_of_simulations = 1000
	print 'Starting '+str(amount_of_simulations)+' monte carlo simulations with n = '+ str(n) + '..'
	for dataset in range(0,amount_of_simulations):
		rdata = Table()
		np.random.shuffle(pa)
		rdata['RA'] = ra
		rdata['DEC'] = dec
		rdata['position_angle'] = pa
		Sn = angular_dispersion(rdata,n=n)
		Sn_datasets.append(Sn)

	Sn_datasets = np.asarray(Sn_datasets)
	temp = Table()
	temp['Sn'] = Sn_datasets
	temp.write('./Sn_monte_carlo_n='+str(n)+'.fits',overwrite=True)
	# print np.average(Sn_datasets)


tdata = fits.open('biggest_selection.fits')
tdata = Table(tdata[1].data)
ra = np.asarray(tdata['RA'])
dec = np.asarray(tdata['DEC'])
pa = np.asarray(tdata['position_angle'])
length = len(tdata)

n_range = range(15,81)
n_range_temp = range(77,81)
# second time running from top down so -4
n_range_temp = list(np.asarray(n_range_temp) - 4)

if __name__ == '__main__':
	print 'cpus: ', multiprocessing.cpu_count()
	p = Pool(multiprocessing.cpu_count())
	p.map(Monte_carlo, n_range_temp)
