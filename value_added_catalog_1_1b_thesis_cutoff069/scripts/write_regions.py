import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 
import math

from astropy.io import fits
from astropy.table import Table, join, vstack
from astropy.nddata.utils import extract_array
from astropy.wcs import WCS

import matplotlib.pyplot as plt

from utils import load_in, deal_with_overlap, convert_deg, tableMosaic_to_full

import pyfits as pf
import pylab as pl


def write_regions_from_table(tdata):
	"""
	Given a table with RA and DEC, will produce a region file that
	can be loaded into ds9 that just produces small circles at these RA and DEC
	"""

	with open('./TEMP.reg', "w") as f:
		RA = tdata['RA']
		DEC = tdata['DEC']

		f.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
""")
		for i in range(0,len(tdata)):
			# e.g. syntax: circle(12:05:33.094,+53:27:19.411,30.000")
			f.write( ('circle(%s,%s,30.000")\n')%(convert_deg(RA[i],DEC[i])) )
	



tdata = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/LOFAR_HBA_T1_DR1_merge_ID_optical_v1.1b.fits'
tdata = Table(fits.open(tdata)[1].data)

mosaicid = 'P4Hetdex16' #P4Hetdex16, cutoff at 8 chars
tdata_mosaic = tdata[tdata['Mosaic_ID'] == mosaicid[:8]] 
tdata_mosaic2 = tdata[tdata['Mosaic_ID'] == tableMosaic_to_full(mosaicid[:8]) ]
print len(tdata_mosaic)
print len(tdata_mosaic2)

tdata_mosaic = vstack([tdata_mosaic,tdata_mosaic2])

write_regions_from_table(tdata_mosaic)