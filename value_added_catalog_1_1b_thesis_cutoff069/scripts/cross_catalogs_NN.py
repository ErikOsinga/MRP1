import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.table import Table, join, vstack
from astropy.io import fits

'''
Python script used to cross all individual multiple gaussians in a field to one file that contains all
the multiple gaussians

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

def cross_catalogs(name_field):
	prefix = '../source_filtering/NN/'
	name = name_field + '_NN.fits' 

	try:
		results1 = fits.open(prefix+name)
		results1 = Table(results1[1].data)
	except IOError:
		return 'No sources in this catalog: ' + name_field

	catalog1 = fits.open('../source_filtering/NN/'+name_field+'NearestNeighbours_efficient_spherical2.fits')
	catalog1 = Table(catalog1[1].data)

	a = join(results1,catalog1,join_type='left')
	a.sort('Isl_id')
	return a


# the first catalog is 'P11Hetdex12'
a = cross_catalogs('P11Hetdex12')
print a
for name_field in FieldNames:
	# for all the other names
	print name_field
	if name_field != 'P11Hetdex12':
		# get the next catalog and stack them on top
		b = cross_catalogs(name_field)
		if type(b) == str:
			print b
		else:
			a = vstack([a,b])


print a
a.write('../source_filtering/NN/all_NN_sources.fits',overwrite=True) 
a.write('../source_filtering/all_NN_sources.fits',overwrite=True) 
