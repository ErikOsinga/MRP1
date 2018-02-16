import sys
sys.path.insert(0, '/data1/osinga/anaconda2')

import numpy as np 

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt

from utils import load_in, rotate_point


# file = '/data1/osinga/data/flux_bins2/flux_bins2_4.fits'  
file = '/data1/osinga/data/biggest_selection.fits'

tdata = fits.open(file)
tdata = Table(tdata[1].data)
RAs, DECs, position_angles, Majs, NN_dists = load_in(file,'RA','DEC',
											'position_angle','Maj','new_NN_distance(arcmin)')

def plot_position_angle(i):
	"""
	Plots the position angle for source (i) on the sky as a line with a radius proportional 
	to either the nearest neighbour distance or the major axis of a source.
	"""

	RA = RAs[i]
	DEC = DECs[i]
	position_angle = position_angles[i]
	Maj = Majs[i]
	NN_dist = NN_dists[i]

	if np.isnan(NN_dist): # MG source
		radius = 2*Maj/3600. # convert arcsec to deg
	else: # NN source
		radius = NN_dist/60. # convert arcmin to deg

	radius *= 2 # enhance radius by a factor

	origin = (RA,DEC) # (coordinates of the source)

	up_point = (RA,DEC+radius) # point above the origin (North)
	down_point = (RA,DEC-radius) # point below the origin (South)

	# rotate the line according to the position angle North through East (counterclockwise)
	x_rot_up,y_rot_up = rotate_point(origin,up_point,position_angle)
	x_rot_down,y_rot_down = rotate_point(origin,down_point,position_angle)

	x = x_rot_down
	y = y_rot_down
	dx = x_rot_up - x_rot_down
	dy = y_rot_up - y_rot_down

	# ax.arrow(x,y,dx,dy)

	plt.plot((x_rot_down,x_rot_up),(y_rot_down,y_rot_up),color='black',linewidth=0.2)

def hist_PA(tdata,normed=False):
	"""
	Makes a histogram of the position angles of a given table 'tdata' that contains position angles
	which have the key 'position_angle' 
	
	if normed=True, produces a normed histogram
	"""

	fig, ax = plt.subplots()
	binwidth=20
	nbins = 180/binwidth
	position_angles = tdata['position_angle']
	n, bins, patches = ax.hist(position_angles,bins=nbins,normed=normed,label='Postion Angle')

	plt.title('Histogram of the position angle | Number of sources: ' + str(len(tdata)) + 
	' | binwidth: ' + str(binwidth) )
	plt.xlabel('Position angle (deg)')
	if normed==True:
		plt.ylabel('Probability')
	else:
		plt.ylabel('Count')
	plt.legend()
	plt.show()	

def crossmatch(tdata1,tdata2):

	maxra = np.max(tdata1['RA'])
	minra = np.min(tdata1['RA'])
	maxdec = np.max(tdata1['DEC'])
	mindec = np.min(tdata1['DEC'])

	# Remove all the sources outside tdata1 scope.
	tdata2 = tdata2[ (tdata2['RA'] > minra) & (tdata2['RA'] < maxra) 
		& (tdata2['DEC'] > mindec) & (tdata2['DEC'] < maxdec)]

	def plotski():
		plt.scatter(tdata2['RA'],tdata2['DEC'],c='blue',alpha=0.5,label='FIRST')
		plt.scatter(tdata1['RA'],tdata1['DEC'],c='red',alpha=0.5,label='LOFAR')

		plt.legend()
		plt.xlabel('RA')
		plt.ylabel('DEC')
		plt.show()

	from astropy.coordinates import SkyCoord
	from astropy import units as u

	ra1 = np.asarray(tdata1['RA']) * u.deg
	dec1 = np.asarray(tdata1['DEC']) * u.deg
	ra2 = np.asarray(tdata2['RA']) * u.deg
	dec2 = np.asarray(tdata2['DEC']) * u.deg

	c1 = SkyCoord(ra1,dec1) # LOFAR
	c2 = SkyCoord(ra2,dec2) # FIRST

	idx, d2d, d3d = c2.match_to_catalog_sky(c1)

	'''
	# Now idx are indices into c1 that are the closest objects to each of the 
	# coordinates in c2
	# i.e. For every FIRST source, we have the coordinates of the closest lofar source
	# d2d are on sky distances between them, and d3d are 3d distances.
	'''
	# print 'Number of unique matches:', len(np.unique(idx))

	matches = c1[idx]

	lofar_match_first = idx[d2d < (0.1 * u.deg)] # index of lofar sources matched to first
	first_match_lofar = [d2d < (0.1*u.deg)] # True,False array of first sources matched to lofar

	# np.save('./lofar_match_first',lofar_match_first)
	# np.save('./first_match_lofar',first_match_lofar)	

	# matches = matches[d2d < (0.1 * u.deg)] # only grab sources with distance < 0.1 deg
	# ra2 = ra2[d2d < (0.1 * u.deg)]
	# dec2 = dec2[d2d < (0.1 * u.deg)]

	print 'Number of sources left:', len(matches)

	plt.scatter(np.asarray(ra2),np.asarray(dec2),c='blue',alpha=0.5,label='FIRST')
	plt.scatter(np.asarray(matches.ra),np.asarray(matches.dec),c='red',alpha=0.5,
		label='LOFAR MATCH')
	plt.legend()
	plt.xlabel('RA')
	plt.ylabel('DEC')
	plt.show()



if __name__ == '__main__':
	# hist_PA(tdata)

	# my_dpi = 96 # failed attempt at high res.
	# plt.figure(figsize=(1920/my_dpi,1080/my_dpi),dpi=my_dpi)


	filename1 = '/data1/osinga/data/biggest_selection.fits'
	filename2 = '/data1/osinga/data/omar_data/catalog_fixed.fits'
	tdata1 = Table(fits.open(filename1)[1].data)
	tdata2 = Table(fits.open(filename2)[1].data)


	crossmatch(tdata1,tdata2)

	''' Plotting all_PA
	plt.figure(figsize=(7,7))

	print 'Number of sources:', len(tdata)

	for i in range(len(tdata)):
		plot_position_angle(i)

	plt.xlim(160,240)
	plt.ylim(45,58)
	plt.title('Position angle distribution of ' + 'flux_bins2_4')
	plt.xlabel('Right Ascension (degrees)')
	plt.ylabel('Declination (degrees)')
	plt.savefig('/data1/osinga/figures/statistics/all_PA_flux_bins2_4.pdf', format='pdf')
	# plt.savefig('/data1/osinga/figures/all_PA.svg')
	# plt.show()

	'''