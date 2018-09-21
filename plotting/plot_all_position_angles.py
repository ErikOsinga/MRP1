import sys

import numpy as np 

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, '/data1/osinga/anaconda2')
from utils import load_in, rotate_point, FieldNames

sys.path.insert(0, '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts')
from general_statistics import (select_flux_bins1, select_size_bins1
		, select_flux_bins11, select_power_bins, select_physical_size_bins
		,select_flux_bins_cuts_biggest_selection)


all_mosaicIDs = ['P11Hetde',
 'P173+55',
 'P21',
 'P8Hetdex',
 'P30Hetde',
 'P178+55',
 'P10Hetde',
 'P218+55',
 'P34Hetde',
 'P7Hetdex',
 'P12Hetde',
 'P16Hetde',
 'P25Hetde',
 'P6',
 'P169+55',
 'P187+55',
 'P164+55',
 'P4Hetdex',
 'P29Hetde',
 'P35Hetde',
 'P3Hetdex',
 'P41Hetde',
 'P191+55',
 'P26Hetde',
 'P27Hetde',
 'P14Hetde',
 'P38Hetde',
 'P182+55',
 'P33Hetde',
 'P196+55',
 'P37Hetde',
 'P223+55',
 'P200+55',
 'P206+50',
 'P210+47',
 'P205+55',
 'P209+55',
 'P42Hetde',
 'P214+55',
 'P211+50',
 'P1Hetdex',
 'P206+52',
 'P15Hetde',
 'P22Hetde',
 'P19Hetde',
 'P23Hetde',
 'P18Hetde',
 'P39Hetde',
 'P223+52',
 'P221+47',
 'P223+50',
 'P219+52',
 'P213+47',
 'P225+47',
 'P217+47',
 'P227+50',
 'P227+53',
 'P219+50']

def load_in_table(tdata,*arg):
	x = (np.asarray(tdata[arg[0]]),)
	for i in range(1,len(arg)):
		x += (np.asarray(tdata[arg[i]]),)
	return x

#VA_biggest_selection
# file = '/data1/osinga/value_added_catalog1_1b/value_added_selection'

# VA value added compmatch
# file = '/data1/osinga/value_added_catalog1_1b/value_added_compmatch_plus_notround'

#VA_biggest_selection
# file = '/data1/osinga/value_added_catalog1_1b/value_added_selection_MG.fits'

# biggest selection 0.69
# file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/biggest_selection.fits'

# tdata, only MG sources and redshift sources
file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/value_added_selection_MG.fits'

# ############################################ VA selection MG only redshift sources, flux bin 3
tdata = Table(fits.open(file)[1].data)
# # With redshift
print ('Using redshift..') ## edited for not actually using z
z_available = np.invert(np.isnan(tdata['z_best']))
z_zero = tdata['z_best'] == 0#
# also remove sources with redshift 0, these dont have 3D positions
z_available = np.logical_xor(z_available,z_zero)
print ('Number of sources with available redshift:', np.sum(z_available))
tdata = tdata[z_available]
tdata_original = tdata # use redshift data for power bins and size bins !!!
# #########################################  VA selection MG only redshift sources, flux bin 3

try:
	tdata['RA_2'], tdata['DEC_2'] = tdata['RA'], tdata['DEC']
except:
	pass

tdata['final_PA'] = tdata['position_angle']

print ('Using flux bin 3')
tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']

# spectroscopic = np.where(tdata['z_source'] == 1)
# tdata = tdata[spectroscopic]
# tdata_original = tdata
# print ('Using spectroscopic redshift only')

# RAs, DECs, position_angles, Majs, NN_dists = load_in(file,'RA_2','DEC_2',
 											# 'final_PA','Maj_1','new_NN_distance(arcmin)')

# fluxbins11 = select_flux_bins11(tdata)
# RAs, DECs, position_angles, Majs, NN_dists = load_in_table(fluxbins11,'RA_2','DEC_2',
 											# 'final_PA','Maj_1','new_NN_distance(arcmin)')

# fluxbins = select_flux_bins1(tdata)
# tdata = fluxbins['0']
# RAs, DECs, position_angles, Majs, NN_dists = load_in_table(tdata,'RA_2','DEC_2',
#  											'final_PA','Maj_1','new_NN_distance(arcmin)')


def plot_position_angle(i,color):
	"""
	Plots the position angle for source (i) on the sky as a line with a radius proportional 
	to either the nearest neighbour distance or the major axis of a source.
	"""

	RA = tdata['RA_2'][i]
	DEC = tdata['DEC_2'][i]

	position_angle = tdata['final_PA'][i]
	NN_dist = tdata['new_NN_distance(arcmin)'][i]
	size = 160 # 80  #set fixed X*4 arcsec
	# size = tdata['size_thesis'][i]

	radius = size/3600. # convert arcsec to deg

	radius *= 4 # enhance radius by a factor 

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

	if color:
		# make colorbar with the redshift data
		plt.plot((x_rot_down,x_rot_up),(y_rot_down,y_rot_up),c=color
			,linewidth=0.4)
	else:
		plt.plot((x_rot_down,x_rot_up),(y_rot_down,y_rot_up),linewidth=0.4,c='k')

def hist_PA(tdata,normed=False):
	"""
	Makes a histogram of the position angles of a given table 'tdata' that contains position angles
	which have the key 'final_PA' 
	
	if normed=True, produces a normed histogram
	"""

	fig, ax = plt.subplots()
	binwidth=5
	nbins = 180/binwidth
	position_angles = tdata['final_PA']
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
	color = False
	# hist_PA(tdata)

	# my_dpi = 96 # failed attempt at high res.
	# plt.figure(figsize=(1920/my_dpi,1080/my_dpi),dpi=my_dpi)
	

	# Plotting all_PA
	f = plt.figure(figsize=(7,7))

	f.set_size_inches(11,4)

	# hist_PA(tdata)

	redshift = False

	if redshift:
		print 'Number of sources:', len(tdata)
		where_z = np.invert(np.isnan(tdata['z_best']))
		print ('Number of sources that have redshift:',np.sum(where_z))


	# tdata = select_size_bins1(tdata)['1']

	if redshift:
		where_z = np.invert(np.isnan(tdata['z_best']))
		tdata = tdata[where_z]
		print 'Using only redshift sources:', len(tdata)
		z_best = tdata['z_best']

		# tdata = select_power_bins(tdata)['0']
		# print ('Power bin .. 0', len(tdata))
		# z_best = tdata['z_best']


		tdata = tdata[z_best < 1.0]
		z_best = tdata['z_best']
		print ('Using only z < 1.0 sources', len(tdata))


		# We need normalized (0,1) redshift values to make the colormap work
		# since colormaps take values between 0 and 1
		z_best_normalized = z_best/z_best.max()
		color = True
		colors = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=z_best.min(),vmax=z_best.max()))
		colors._A = [] # create a fake array of a mappable so pyplot doesnt cry errors
		cmap = colors.get_cmap()
	for i in range(len(tdata)):
		# j = np.where(tdata['Mosaic_ID_2'][i].strip() == np.asarray(all_mosaicIDs))[0][0]
		if redshift:
			color = cmap(z_best_normalized[i])
		plot_position_angle(i,color=color)

	print ('Number of sources in final plot: %i'%len(tdata))

	plt.xlim(160,250)#160,240
	plt.ylim(45,58)
	print ('Position angle distribution of ' + file[38:] + '')
	# plt.title('Position angle distribution of ' + file[38:] + '')
	plt.xlabel('Right Ascension (degrees)')
	plt.ylabel('Declination (degrees)')
	plt.gca().set_aspect('equal', adjustable='box')

	if redshift:
		cb = plt.colorbar(colors)
		cb.set_label('redshift')
	

	# put spectro sources in same plot
	spectroscopic = np.where(tdata['z_source'] == 1)
	tdata = tdata[spectroscopic]
	tdata_original = tdata
	print ('Using spectroscopic redshift only')

	for i in range(len(tdata)):
		# j = np.where(tdata['Mosaic_ID_2'][i].strip() == np.asarray(all_mosaicIDs))[0][0]
		color = 'r'
		plot_position_angle(i,color=color)


	# for the legend
	plt.plot((0,0),(1,1),c='r',linewidth=0.4,label='Spectroscopic-z')
	plt.plot((0,0),(1,1),c='k',linewidth=0.4,label='Photometric-z')
	plt.legend()
	
	# Bound by the strongest alignment signal sources
	plt.xlim(210,230)#160,240
	plt.ylim(45,51.5)

	# plt.savefig('/data1/osinga/figures/statistics/all_PA_flux_bins2_4.pdf', format='pdf')
	# plt.savefig('/data1/osinga/figures/all_PA.svg')
	plt.show()

