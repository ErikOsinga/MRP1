import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.io import fits
from astropy.table import Table

def check_em(nn1_distance,nn2_distance):
	# bon = 0
	# not_so_bon = 0
	# for i in range(0,len(nn2_index)):
		
	# 	if (nn1_index[i] != nn2_index[i]):
	# 		print 'index fout van source met index: ' +str(i)
	# 		not_so_bon +=1
	# 	else:
	# 		bon += 1
	# print 'index aantal goed: ' + str(bon)
	# print 'index aantal fout: ' + str(not_so_bon)

	bon = 0
	not_so_bon = 0
	for i in range(0,len(nn1_distance)):
		
		if (nn1_distance[i] != nn2_distance[i]):
			print 'index fout van source met index: ' +str(i)
			not_so_bon +=1
		else:
			bon += 1
	print 'distance aantal goed: ' + str(bon)
	print 'distance aantal fout: ' + str(not_so_bon)



def load_in(nnpath,*arg):
	'''
	Is used to load in columns from nnpath Table,
	*arg is a tuple with all the arguments (columns) given after nnpath
	returns the columns as a tuple of numpy arrays
	'''

	nn1 = fits.open(nnpath)
	nn1 = Table(nn1[1].data)

	x = (np.asarray(nn1[arg[0]]),)
	for i in range (1,len(arg)):
		x += (np.asarray(nn1[arg[i]]),)

	return x

nn_inefficient_distance = load_in('/data1/osinga/data/P173+55NearestNeighbours_efficient_old.fits','NN_distance(arcmin)') # the euclidian --> spherical distance
nn_efficient_distance = load_in('/data1/osinga/data/P173+55NearestNeighbours_efficient_spherical1.fits','NN_distance(arcmin)')# the spherical --> spherical distance

# check_em(nn_inefficient_index,nn_inefficient_distance,nn_efficient_index,nn_efficient_distance)
# nn_efficienter_index,nn_efficienter_distance = load_in('/data1/osinga/data/P173+55NearestNeighbours_efficient_Spherical_test2.fits')
check_em(nn_inefficient_distance[0],nn_efficient_distance[0])



