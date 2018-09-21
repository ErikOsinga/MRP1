import sys

import numpy as np 

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt

sys.path.insert(0, '/data1/osinga/anaconda2')
from utils import FieldNames, rotate_point

sys.path.insert(0, '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts')
from general_statistics import select_flux_bins_cuts_biggest_selection


def use_only_redshift(tdata):
	print ('Using redshift..') 
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]
	return tdata

def plot_all_position_angles():

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

		z_best = tdata['z_best'][i]

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

	file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/value_added_selection_MG.fits'
	tdata = Table(fits.open(file)[1].data)

	# With redshift
	tdata = use_only_redshift(tdata)
	tdata_original = tdata

	tdata['final_PA'] = tdata['position_angle']
	print ('Using flux bin 3')
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']

	f = plt.figure(figsize=(7,7))

	for i in range(len(tdata)):
		plot_position_angle(i,color=False)

	print ('Number of sources in final plot: %i'%len(tdata))

	plt.xlim(160,250)#160,240
	plt.ylim(45,58)
	print ('Position angle distribution of ' + file[38:] + '')
	# plt.title('Position angle distribution of ' + file[38:] + '')
	# plt.xlabel('Right Ascension (degrees)',fontsize=14)
	# plt.ylabel('Declination (degrees)',fontsize=14)
	plt.gca().set_aspect('equal', adjustable='box')

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

	# plt.show()

def plot_LSS(filament_data,redshift=0.69):
	"""
	Plots LSS at a certain redshift, increments of 0.005
	starting at 0.05, ending at 0.695
	"""

	where_redshift = np.where(filament_data[:,2] == redshift)
	filament_data = filament_data[where_redshift]

	# select  160 < RA < 250
	where_ra = np.where( (filament_data[:,0] > 160) & (filament_data[:,0] < 240) )
	filament_data = filament_data[where_ra]

	# select 45 < DEC < 58
	where_dec = np.where( (filament_data[:,1] > 46) & (filament_data[:,1] < 58))
	filament_data = filament_data[where_dec]

	filament_RA = filament_data[:,0]
	filament_DEC = filament_data[:,1]

	plt.scatter(filament_RA,filament_DEC,s=4,label='z=%.3f'%redshift)

def plot_params(xlabel,ylabel,title):
	ax = plt.gca() 
	ax.tick_params(labelsize=12)
	ax.set_xlabel(xlabel,fontsize=14)
	ax.set_ylabel(ylabel,fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()
	plt.show()

def analyze_bounded_sources():
	file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/september/filament_maps/value_added_selection_MG_flux3_redshiftsources_bounded.fits'
	tdata = Table(fits.open(file)[1].data)

	print ('Median redshift: %f'%np.median(tdata['z_best']))
	spectroscopic = np.where(tdata['z_source'] == 1)
	tdata_spec = tdata[spectroscopic]

	plt.hist(tdata['z_best'],label='All data',bins=89)
	plt.hist(tdata_spec['z_best'],label='Spectroscopic',bins=89)
	plot_params('redshift','counts','')

# # first row is header: "RA" "dec" "z_low" "density" "H" "UM" "v_ra" "v_dec"
filament_data = np.loadtxt('./dr12_FMaps_full.txt',delimiter=' ',skiprows=1) 

plot_all_position_angles()
plot_LSS(filament_data,redshift=0.68)
plot_LSS(filament_data,redshift=0.685)
plot_LSS(filament_data,redshift=0.69)
plot_LSS(filament_data,redshift=0.695)	
plot_params('RA','DEC','')




# analyze_bounded_sources()