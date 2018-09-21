import numpy as np 
import sys
sys.path.insert(0, '/data1/osinga/anaconda2/')
import numpy as np 
import math
from matplotlib import pyplot as plt

from astropy.io import fits
from astropy.table import Table, join, vstack

from utils import load_in, deal_with_overlap, distanceOnSphere

from scipy.spatial import cKDTree

from astropy.cosmology import Planck15
from astropy import units as u


fake_lines_x = [(0,1),(1,2),(2,3)]
fake_lines_y = [(0,1),(1,2),(2,3)]

fake_redshift = np.asarray([0,0.4,1.3])

# Have to normalize the redshifts for the colormap (since that takes values from 0.0 to 1.0)
fake_redshift_normalized = fake_redshift/np.max(fake_redshift)

colors = plt.cm.ScalarMappable(cmap='Reds',norm=plt.Normalize(vmin=fake_redshift.min(),vmax=fake_redshift.max()))
colors._A = [] # to prevent matplotlib from crying about something
cmap = colors.get_cmap()

for i in range(len(fake_lines_x)):
	plt.plot(fake_lines_x[i],fake_lines_y[i],c=cmap(fake_redshift_normalized[i]))


plt.colorbar(colors)
plt.show()