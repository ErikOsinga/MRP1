# run from the environment kerastf
# source activate kerastf
import sys
sys.path.insert(0, '/data1/osinga/miniconda')
import numpy as np
from astropy.table import Table, join, vstack
from astropy.io import fits
from ionotomo.utils import gaussian_process as gp  
from ionotomo.utils.gaussian_process import SquaredExponential

def select_all_interesting():
	'''
	Function to select all sources in the leaves of the tree, so with err < 15
	'''

	dataNN = fits.open('/data1/osinga/data/NN/try_4/all_NN_sources.fits')
	dataNN = Table(dataNN[1].data)
	dataMG = fits.open('/data1/osinga/data/all_multiple_gaussians2.fits')
	dataMG = Table(dataMG[1].data)

	# calculate the cutoff value for the NN, since it wasnt well defined in the source_filtering
	Min = dataNN['Min']/60. # convert to arcmin
	nn_dist = dataNN['new_NN_distance(arcmin)']
	cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg

	selectionNN = np.where( (dataNN['amount_of_maxima'] > 1) & (dataNN['lobe_ratio'] < 2) &
		(dataNN['lobe_ratio'] > 1./2) & (dataNN['err_orientation'] < cutoff) & 
		(dataNN['classification'] != 'no_hope') )
	selectionMG = np.where( (dataMG['amount_of_maxima'] > 1) & (dataMG['lobe_ratio'] < 2) &
		(dataMG['lobe_ratio'] > 1./2) & (dataMG['classification'] == 'Small err') )

	selectionNN = dataNN[selectionNN]
	selectionMG = dataMG[selectionMG]

	return selectionNN, selectionMG

dataNN,dataMG = select_all_interesting()
tdata = vstack([dataNN,dataMG])

x = np.array([tdata['RA'].flatten(),tdata['DEC'].flatten()]).T
y = np.array(tdata['position_angle'])
sigma_y = y*0 + 5 

# length scale guess of 5 
k1 = gp.SquaredExponential(2,l=5)
k2 = gp.Diagonal(2)
k=k1+k2

print 'starting for loop'
for b in [[1e-5,1],[1,5],[5,10],[10,20],[20,40]]:
	k1.set_hyperparams_bounds(b,name='l')
	mask = (y < 150) * (y > 30) 
	k.hyperparams = gp.level2_solve(x[mask,:],y[mask],sigma_y[mask],k,n_random_start=1)
	k1.set_hyperparams_bounds([1e-5,5],name='sigma')
	k2.set_hyperparams_bounds([1e-5,5],name='sigma')    
	print (k)
	print (gp.log_mar_like(k.hyperparams,x,y,sigma_y,k))    
