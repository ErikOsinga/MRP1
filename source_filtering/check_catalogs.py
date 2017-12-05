import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii


# catalog2 = '/data1/osinga/data/all_multiple_gaussians.fits'
# data2 =	Table(data2[1].data)
# data2 = fits.open(catalog2)


catalog1 = '/data1/osinga/downloads/hatfield_multiple_20x20_ID19_prototype_6_0.csv'

catalog_check = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/P7Hetdex11cat.srl.fits'


data1 = ascii.read(catalog1,format='csv')
data_check = fits.open(catalog_check)
data_check = Table(data_check[1].data)


def check(data1,data2):
	for i in range (0,len(data1)):
		for j in range(0,len(data2)):
			if data1['Source_id'][i] == data2['Source_Name'][j]:
				print 'Match at i = ' , i , ' j = ', j 


check(data1,data_check) 