import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import os 

import numpy as np 
import math

from astropy.io import fits
from astropy.table import Table, join, vstack
from astropy.nddata.utils import extract_array

from scipy.ndimage import map_coordinates
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt

from utils import (angular_dispersion_vectorized_n, distanceOnSphere, load_in
					, rotate_point, PositionAngle)

import pyfits as pf
import pywcs as pw
import pylab as pl

# If you don't NEED LaTeX, don't use it.
# plt.rc('text', usetex=True)
# You can actually use latex commands without this line!!
# such as plt.title(r'5$^\circ$') TRY IT !


def plot_source_WCS(fitsfile):
	from astropy.wcs import WCS
	from astropy.io import fits

	hdu = fits.open(fitsfile)[0]
	wcs = WCS(hdu.header)
	ax1 = plt.subplot(111, projection=wcs)

	lon = ax1.coords[0]
	lat = ax1.coords[1]

	lon.set_ticks(exclude_overlapping=True)
	# lon.set_major_formatter('d.dd')

	ax1.imshow(hdu.data,origin='lower',cmap='gray')

	ax1.set_xlabel('Right ascension (J2000)',fontsize=12)
	ax1.set_ylabel('Declination (J2000)',fontsize=12)
	# plt.suptitle('Use this if you have multiple subplots')
	plt.title('Use this if you have 1 subplot')

	# plt.savefig('./'+head['OBJECT']+'src_'+str(i)+'.pdf') 		
	plt.show()
	plt.clf()
	plt.close()

def plot_black_lines_default():
	plt.rc('font', family='serif')
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(1, 1, 1)

	x = np.linspace(1., 8., 30)
	ax.plot(x, x ** 1.5, color='k', ls='solid')
	ax.plot(x, 20/x, color='0.50', ls='dashed')
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Temperature (K)')

	plt.grid()

def legend_handles_labels():
	#adjusting handles and labels
	handles, labels = ax.get_legend_handles_labels()
	
	handles = reversed(handles)
	labels = reversed(labels)
	
	ax.legend(handles, labels,fontsize=14,loc='upper right')

def plot_histogram():
	# make nice publication histograms
	plt.hist(all_sizes,bins=100,histtype=u'step',color='black')

	# make a vertical line in it
	plt.axvline(x=xposition,ymin=0,ymax=1, color = 'r', label='best orientation')
	# ymin=0, ymax=1 means bottom to top of the plot

	# Use fancy algorithms to determine the bins
	from astropy import visualization
	visualization.hist(position_angles,bins='scott',histtype=u'step',color='black')

def figsize():
	# getting
	plt.rcParams.get('figure.figsize')
	# setting
	plt.figure(figsize=(6.4,4.8))

def fontsizes():
	ax = plt.gca() 
	# or 
	fig, ax = plt.subplots()

	ax.tick_params(labelsize=12)

	ax.set_xlabel(r'$n_n$',fontsize=14)
	ax.set_ylabel(r'$\log_{10}$ SL',fontsize=14)
	ax.legend(fontsize=14)
	plt.title(title,fontsize=16)
	plt.tight_layout()

def colors():
	# setting the default color cycle to different options
	plt.rcParams['axes.prop_cycle'] = plt.rcParams['axes.prop_cycle'][1:]

	# get a list of default colors
	prop_cycle = plt.rcParamsDefault['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']