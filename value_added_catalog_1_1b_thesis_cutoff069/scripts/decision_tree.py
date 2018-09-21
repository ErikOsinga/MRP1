import sys
sys.path.insert(0, '/net/reusel/data1/osinga/anaconda2')
import numpy as np 

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt


dataNN = Table(fits.open('../source_filtering/all_NN_sources.fits')[1].data)
dataMG = Table(fits.open('../source_filtering/all_multiple_gaussians.fits')[1].data)

def print_tree_counts():
	tdata_MG = Table(fits.open('../source_filtering/all_multiple_gaussians.fits')[1].data)
	tdata_NN = Table(fits.open('../source_filtering/all_NN_sources.fits')[1].data)

	print ('Top of the tree:')
	mutuals = tdata_NN
	bridged = tdata_MG
	print ('Mutuals: %i' %len(mutuals))
	print ('Bridged: %i' %len(bridged))
	print ('\n')

	print ('Reject due to 1 flux maximum')
	selectionNN = np.where(tdata_NN['amount_of_maxima'] == 1)
	selectionMG = np.where(tdata_MG['amount_of_maxima'] == 1)
	mutuals = tdata_NN[selectionNN]
	bridged = tdata_MG[selectionMG]
	# plt.hist(bridged['PA'],bins=180/5);plt.show() # these sources are peaked around 90
	print ('Mutuals: %i' %len(mutuals))
	print ('Bridged: %i' %len(bridged))
	print ('\n')

	print ('Keep due to >=2 flux maxima')
	selectionNN = np.where(tdata_NN['amount_of_maxima'] > 1)
	selectionMG = np.where(tdata_MG['amount_of_maxima'] > 1)
	mutuals = tdata_NN[selectionNN]
	bridged = tdata_MG[selectionMG]
	print ('Mutuals: %i' %len(mutuals))
	print ('Bridged: %i' %len(bridged))
	print ('\n')
	
	
	print ('Reject due to >2 lobe ratio')
	selectionNN = np.where( (tdata_NN['amount_of_maxima'] > 1) 
		& ( (tdata_NN['lobe_ratio'] > 2) | (tdata_NN['lobe_ratio'] < 1./2 ) ) )
	selectionMG = np.where( (tdata_MG['amount_of_maxima'] > 1) 
		& ( (tdata_MG['lobe_ratio'] > 2) | (tdata_MG['lobe_ratio'] < 1./2 ) ) )
	mutuals = tdata_NN[selectionNN]
	bridged = tdata_MG[selectionMG]
	print ('Mutuals: %i' %len(mutuals))
	print ('Bridged: %i' %len(bridged))
	print ('\n')

	print ('Keep due to <= 2 lobe ratio')
	selectionNN = np.where( (tdata_NN['amount_of_maxima'] > 1) & (tdata_NN['lobe_ratio'] <= 2)
							& (tdata_NN['lobe_ratio'] >= 1./2 ) )
	selectionMG = np.where( (tdata_MG['amount_of_maxima'] > 1) & (tdata_MG['lobe_ratio'] <= 2)
							& (tdata_MG['lobe_ratio'] >= 1./2 ) )
	mutuals = tdata_NN[selectionNN]
	bridged = tdata_MG[selectionMG]
	print ('Mutuals: %i' %len(mutuals))
	print ('Bridged: %i' %len(bridged))
	print ('\n')
	
	print ('Reject due to orientation error > cutoff')
	selectionNN = np.where( (tdata_NN['amount_of_maxima'] > 1) 
							& (tdata_NN['lobe_ratio'] <= 2)  
							& (tdata_NN['lobe_ratio'] >= 1./2 )
							& (tdata_NN['classification'] != 'Small err') )
	selectionMG = np.where( (tdata_MG['amount_of_maxima'] > 1) 
							& (tdata_MG['lobe_ratio'] <= 2) 
							& (tdata_MG['lobe_ratio'] >= 1./2 )
							& (tdata_MG['classification'] != 'Small err') )
	mutuals = tdata_NN[selectionNN]
	bridged = tdata_MG[selectionMG]
	print ('Mutuals: %i' %len(mutuals))
	print ('Bridged: %i' %len(bridged))
	print ('\n')
	
	print ('Keep due to orientation error <= cutoff')
	selectionNN = np.where( (tdata_NN['amount_of_maxima'] > 1) 
							& (tdata_NN['lobe_ratio'] <= 2)
							& (tdata_NN['lobe_ratio'] >= 1./2 ) 
							& (tdata_NN['classification'] == 'Small err') )
	selectionMG = np.where( (tdata_MG['amount_of_maxima'] > 1)
							& (tdata_MG['lobe_ratio'] <= 2) 
							& (tdata_MG['lobe_ratio'] >= 1./2 )
							& (tdata_MG['classification'] == 'Small err') )
	mutuals = tdata_NN[selectionNN]
	bridged = tdata_MG[selectionMG]
	print ('Mutuals: %i' %len(mutuals))
	print ('Bridged: %i' %len(bridged))
	print ('\n')


	print ('Thats biggest_selection.fits: %i'%( len(mutuals) + len(bridged) ))

print_tree_counts()
	