import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astroquery.simbad import Simbad 
import astropy.units as u
from astropy import coordinates



def query_objects():
	result_table = Simbad.query_object("m1")
	result_table.pprint(show_unit=True)


def query_coord():
	c = coordinates.SkyCoord("05h35m17.3s -05d23m28s")
	r = 5*u.arcminute
	result_table = Simbad.query_region(c,radius=r)
	result_table.pprint(show_unit=True, max_width=80, max_lines=5)


from astroquery.skyview import SkyView 
# SkyView.list_surveys()
#contains 'Optical:SDSS'
'''
'Optical:SDSS': [u'SDSSg',
                  u'SDSSi',
                  u'SDSSr',
                  u'SDSSu',
                  u'SDSSz',
                  u'SDSSdr7g',
                  u'SDSSdr7i',
                  u'SDSSdr7r',
                  u'SDSSdr7u',
                  u'SDSSdr7z'],
'''

#There are two essential methods: get_images searches for
#and downloads files, while get_image_list just searches for the files.
c = "11 43 06.00 53 50 46.01"
c2 = "175.7986916080435 53.84611451646336"
# print SkyView.get_image_list(position= c, survey=['SDSSg'])
# print 'backslash n \n\n'
print SkyView.get_image_list(position=c2,survey=['SDSSg','SDSSi','SDSSu','SDSSz'])



'''
Green (g) 4770
Near Infrared (i) 7625
Red (r) 6231
Ultraviolet (u) 3543
Infrared (z) 9134
and
dr7g
dr7i etc
'''