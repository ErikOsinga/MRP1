import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 
import matplotlib.pyplot as plt 
import astropy
import pyregion
import aplpy
import pylab as pl

'''
http://aplpy.readthedocs.io/en/stable/quickstart.html
'''

prefix = '/data1/osinga/downloads/tutorial/'

gc = aplpy.FITSFigure(prefix+'fits/2MASS_k.fits')

def easy_showing():
	# gc.show_grayscale()
	gc.show_colorscale(cmap='gist_heat')
	pl.show()

def three_color_image():
	# has to have exactly the same dimensions and the pixels from the PNG file
	# have to match those from the FITS file.
	gc.show_rgb(prefix+'graphics/2MASS_arcsinh_color.png')
	gc.tick_labels.set_font(size='small')
	# there are a number of arguments that can be passed to show_contour() to control
	# the appearance of the contours as well as the number of levels to show
	# for more information see the show_contour() documentation
	gc.show_contour(prefix+'fits/mips_24micron.fits', colors='white')
	gc.add_grid()
	# gc.remove_grid()

def overplot_positions():
	data = np.loadtxt(prefix+'data/yso_wcs_only.txt')
	ra, dec = data[:, 0], data[:, 1]
	#for more information see the show_makers() documentation
	gc.show_markers(ra,dec,edgecolor='green',facecolor='none', marker='o', s=10,alpha=0.5)

	def layers():
		print gc.list_layers()
		# you can use remove_layer(), hide_layer() and show_layer() to manipulate these
		# you can also specify the layer=name argument to show_contour() or show_markers()
		# this forces aplpy to name the layer you are creating or replace an existing layer
		gc.show_markers(ra, dec, layer='marker_set_1', edgecolor='red',
		                facecolor='none', marker='o', s=10, alpha=0.5)	

	layers()
	gc.recenter(convert_hh_mm_ss(17,44,12),-29,radius=.3)
	# gc.show_ellipses(convert_hh_mm_ss(17,44,12),-29,0.03*3600,0.03*3600)
	gc.show_markers(convert_hh_mm_ss(17,44,12),-29)
	gc.show_arrows(convert_hh_mm_ss(17,44,12),-29,0.03,0.03)

def convert_hh_mm_ss(hh,mm,ss):
	'''
	converts hh_mm_ss to degrees
	'''
	return ( (360./24.)* (hh + (mm/60.) + ss/3600.) ) 
	
def saving_file():
	gc.save(prefix+'myfirstplot.png')

# easy_showing()
# three_color_image()
# overplot_positions()
# # saving_file()
# plt.show()
# pl.show()


def astropy_wcs():
	'''
	Use in conjunction with /data1/osinga/scripts/create_cutout.py
	'''
	from astropy.wcs import WCS
	from astropy.io import fits
	from astropy.utils.data import get_pkg_data_filename

	# filename = get_pkg_data_filename('/data1/osinga/figures/thesis/irregular_shape.fits')
	filename = '/data1/osinga/figures/thesis/5707.fits'
	hdu = fits.open(filename)[0]
	wcs = WCS(hdu.header)

	plt.subplot(projection=wcs)
	plt.imshow(hdu.data, origin='lower')
	# plt.grid(color='white', ls='solid')
	# plt.xlabel('Galactic Longitude')
	# plt.ylabel('Galactic Latitude')
	plt.show()


astropy_wcs()

# print convert_hh_mm_ss(17,44,12)

