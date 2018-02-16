import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.table import Table, join, vstack
from astropy.io import fits

'''
Python script used to cross all NearestNeighbours_efficient_spherical2.fits files to one big 
file, because these still have the information about the nearest neighbour (pairs are double in the file)

'''


FieldNames = ['P11Hetdex12', 'P173+55', 'P21', 'P8Hetdex', 'P30Hetdex06', 'P178+55', 
'P10Hetdex', 'P218+55', 'P34Hetdex06', 'P7Hetdex11', 'P12Hetdex11', 'P16Hetdex13', 
'P25Hetdex09', 'P6', 'P169+55', 'P187+55', 'P164+55', 'P4Hetdex16', 'P29Hetdex19', 'P35Hetdex10', 
'P3Hetdex16', 'P41Hetdex', 'P191+55', 'P26Hetdex03', 'P27Hetdex09', 'P14Hetdex04', 'P38Hetdex07', 
'P182+55', 'P33Hetdex08', 'P196+55', 'P37Hetdex15', 'P223+55', 'P200+55', 'P206+50', 'P210+47', 
'P205+55', 'P209+55', 'P42Hetdex07', 'P214+55', 'P211+50', 'P1Hetdex15', 'P206+52', 
'P15Hetdex13', 'P22Hetdex04', 'P19Hetdex17', 'P23Hetdex20', 'P18Hetdex03', 'P39Hetdex19', 'P223+52',
 'P221+47', 'P223+50', 'P219+52', 'P213+47', 'P225+47', 'P217+47', 'P227+50', 'P227+53', 'P219+50',
 ]

def load_catalogs(name_field):
	prefix = '/data1/osinga/data/NN/'
	name = name_field + 'NearestNeighbours_efficient_spherical2.fits' 

	try:
		catalog = fits.open(prefix+name)
		catalog = Table(catalog[1].data)
	except IOError:
		return 'No sources in this catalog: ' + name_field

	return catalog

# the first catalog is 'P11Hetdex12'
a = load_catalogs('P11Hetdex12')
print a
for name_field in FieldNames:
	# for all the other names
	print name_field
	if name_field != 'P11Hetdex12':
		# get the next catalog and stack them on top
		b = load_catalogs(name_field)
		if type(b) == str:
			print b
		else:
			a = vstack([a,b])


print a
a.write('/data1/osinga/data/NN/all_NearestNeighbours_efficient_spherical2.fits',overwrite=True) 
