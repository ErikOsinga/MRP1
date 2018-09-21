import sys, os
sys.path.insert(0,'/data1/osinga/anaconda2')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Ellipse
import pyfits as pf
import aplpy
import pylab as pl
import pywcs as pw
from astropy.io import fits
from astropy.table import Table
import difflib
import math
from numdisplay_zscale import zscale

sys.path.insert(0,'/data1/osinga/anaconda2')
from utils import (angular_dispersion_vectorized_n, distanceOnSphere, load_in
					, rotate_point, PositionAngle, deal_with_overlap, deal_with_overlap_2, FieldNames, tableMosaic_to_full)
from utils_orientation import find_orientationNN, find_orientationMG

sys.path.insert(0,'/data1/osinga/scripts')
from quick_plot import plot_source_finding_latest_catalog


# parameters for the zscale
MAX_REJECT = 0.5 
MIN_NPIXELS = 5 
GOOD_PIXEL = 0 
BAD_PIXEL = 1 
KREJ = 2.5 
MAX_ITERATIONS = 5 

FieldNames = [
'P11Hetdex12', 'P173+55', 'P21', 'P8Hetdex', 'P30Hetdex06', 'P178+55', 
'P10Hetdex', 'P218+55', 'P34Hetdex06', 'P7Hetdex11', 'P12Hetdex11', 'P16Hetdex13', 
'P25Hetdex09', 'P6', 'P169+55', 'P187+55', 'P164+55', 'P4Hetdex16', 'P29Hetdex19', 'P35Hetdex10', 
'P3Hetdex16', 'P41Hetdex', 'P191+55', 'P26Hetdex03', 'P27Hetdex09', 'P14Hetdex04', 'P38Hetdex07', 
'P182+55', 'P33Hetdex08', 'P196+55', 'P37Hetdex15', 'P223+55', 'P200+55', 'P206+50', 'P210+47', 
'P205+55', 'P209+55', 'P42Hetdex07', 'P214+55', 'P211+50', 'P1Hetdex15', 'P206+52',
'P15Hetdex13', 'P22Hetdex04', 'P19Hetdex17', 'P23Hetdex20', 'P18Hetdex03', 'P39Hetdex19', 'P223+52',
 'P221+47', 'P223+50', 'P219+52', 'P213+47', 'P225+47', 'P217+47', 'P227+50', 'P227+53', 'P219+50'
 ]

no_sources = 0

# the class that contains the functions for the buttons
class Index(object):
	def single(self,event):
		global no_sources
		no_sources += 1
		print 'no. of sources done: ', no_sources
		F.write(str(Source_Names[i])+','+str(i)+',NN')
		F.write('\n')
		plt.close()

	def double(self,event):
		global no_sources
		no_sources += 1
		print 'no. of sources done: ', no_sources
		F.write(str(Source_Names[i])+','+str(i)+',MG')
		F.write('\n')
		plt.close()
	
	def unclassified(self,event):
		global no_sources
		no_sources += 1
		print 'no. of sources done: ', no_sources
		F.write(str(Source_Names[i])+','+str(i)+',unclassified')
		F.write('\n')
		plt.close()

def load_in(nnpath,*arg):
	'''
	Is used to load in columns from nnpath Table,
	*arg is a tuple with all the arguments (columns) given after nnpath
	returns the columns as a tuple of numpy arrays
	'''

	nn1 = fits.open(nnpath)
	nn1 = Table(nn1[1].data)

	x = (np.asarray(nn1[arg[0]]),)
	for i in range (1,len(arg)):
		x += (np.asarray(nn1[arg[i]]),)

	return x
		
def postage2(directory,i,ra,dec,nn_ra=None,nn_dec=None,maj=None,min=None,pa=None):
	'''
	Making the cutouts with plt.imshow
	'''
	# fig, ax = plt.subplots()

	# plt.subplots_adjust(bottom=0.2)

	# for NN excluded by also being MG
	plot_source_finding_latest_catalog(True,mosaicid,tdata['Source_Name'][i],plot=True)

	
	callback = Index()
	ax1 = plt.axes([0.2, 0.05, 0.1, 0.075])
	ax2 = plt.axes([0.5, 0.05, 0.1, 0.075])
	ax3 = plt.axes([0.8, 0.05, 0.1, 0.075])
	button1 = Button(ax1,'NN')
	button1.on_clicked(callback.single)
	button2 = Button(ax2,'MG')
	button2.on_clicked(callback.double)
	button3 = Button(ax3,'Unclassified')
	button3.on_clicked(callback.unclassified)
	
	plt.show()		

def postage3(directory,i,ra,dec,nn_ra=None,nn_dec=None):
	'''
	Making the cutouts with aplpy.FITSFigure
	'''
	image = '/net/reusel/data1/osinga/button_classification/'+directory+'/source'+str(i)+'.fits'
	gc = aplpy.FITSFigure(image)
	hdulist = fits.open(image)
	data = hdulist[0].data

	z1,z2 = zscale(data)

	plt.subplots_adjust(bottom=0.2)


	gc.show_markers(ra,dec,edgecolor='green',facecolor='none', marker='o', s=105,alpha=0.5)
	if nn_ra:
		gc.show_markers(nn_ra,nn_dec,edgecolor='green',facecolor='none', marker='o', s=105,alpha=0.5)
		plt.title('Nearest Neighbour sources')
	else:
		plt.title('Multiple Gaussian source')

	callback = Index()
	ax1 = plt.axes([0.2, 0.05, 0.1, 0.075])
	ax2 = plt.axes([0.5, 0.05, 0.1, 0.075])
	ax3 = plt.axes([0.8, 0.05, 0.1, 0.075])
	button1 = Button(ax1,'Single')
	button1.on_clicked(callback.single)
	button2 = Button(ax2,'Double')
	button2.on_clicked(callback.double)
	button3 = Button(ax3,'Unclassified')
	button3.on_clicked(callback.unclassified)

	gc.show_grayscale(vmin=z1,vmax=z2)
	gc.add_grid()
	pl.show()
	gc.close()

argc = len(sys.argv)
if argc == 2:
	number_of_classifications = int(sys.argv[1])
elif argc > 2:
		raise ValueError("Please provide only one argument")
else:
	number_of_classifications = 100

print '---------------------------------------------'
print "This program will show %d sources, of which the first half will be nearest neighbour sources and the second half sources made up of multiple gaussians." % number_of_classifications
print "\nFor the first half, two dots will indicate the two sources that are being considered."
print "The question is if the source in the middle may be part of a double with the other source or is a single source."
print "\nFor the second half, the PYBDSF ellipse will indicate the source, since we are only considering one source at a time."
print "If it is neither a single nor a double or you are unsure at any point, just press the 'Unclassified' button."
print "\nThe program will output two files in the current directory: '", os.getcwd(), "' named classification_NN.csv and classification_MG.csv"
print "\nBe careful not to run the program again after the classification is done, since it will overwrite the files."
print "\nThanks a lot for your help!"





directory = 'nutteloze variable'


# nearest neighbour classification

############## where the NN source was identified in the MG Table as well

file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/NN_excluded_by_also_being_MG_1.fits'
tdata = Table(fits.open(file)[1].data)
tdata['Source_Name_1'] = tdata['Source_Name']
tdata['Source_Name'] = tdata['Source_Name_1']
NN = np.invert(np.isnan(tdata['new_NN_RA']))
iis = tdata['Source_Name']
tdata['PA_1'] = tdata['PA']

#same thing, but global
Source_Names, RAs, DECs, Mosaic_IDs, NN_RAs, NN_DECs = load_in('/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/NN_excluded_by_also_being_MG_1.fits','Source_Name','RA','DEC','Mosaic_ID','new_NN_RA','new_NN_DEC')

F = open('./classification_NN_excluded_by_MG_1.csv','w')
F.write('Source_Name,source_index,classification')
F.write('\n')

# Have to do separately the sources that have a NN in the MG table and 
# sources that are themselves in the MG table

count = 0
for i in iis:
	i = np.where(tdata['Source_Name'] == i)[0][0]
	mosaicid = tdata['Mosaic_ID'][i]
	count += 1 

	print 'PA_1: %f, position_angle: %f' %(tdata['PA_1'][i],tdata['position_angle'][i])
	print (count)
	mosaicid = tableMosaic_to_full(tdata['Mosaic_ID'][i])
	

	postage2(directory,i,RAs[i],DECs[i],NN_RAs[i],NN_DECs[i])

F.close()

plt.figure()
plt.title('MOVING ON TO LOOKING AT THE NEIGHBOUR SOURCE NOW')
plt.show()

############## where the nearest neighbour of the NN source was identified in the MG Table as well

file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/NN_excluded_by_also_being_MG_2.fits'
tdata = Table(fits.open(file)[1].data)
tdata['Source_Name_1'] = tdata['Source_Name']
tdata['Source_Name'] = tdata['Source_Name_1']
NN = np.invert(np.isnan(tdata['new_NN_RA']))
iis = tdata['Source_Name']
tdata['PA_1'] = tdata['PA']

#same thing, but global
Source_Names, RAs, DECs, Mosaic_IDs, NN_RAs, NN_DECs = load_in('/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/NN_excluded_by_also_being_MG_2.fits','Source_Name','RA','DEC','Mosaic_ID','new_NN_RA','new_NN_DEC')
F = open('./classification_NN_excluded_by_MG_2.csv','w')
F.write('Source_Name,source_index,classification')
F.write('\n')

# Have to do separately the sources that have a NN in the MG table and 
# sources that are themselves in the MG table

count = 0
for i in iis:
	i = np.where(tdata['Source_Name'] == i)[0][0]
	mosaicid = tdata['Mosaic_ID'][i]
	count += 1 

	print 'PA_1: %f, position_angle: %f' %(tdata['PA_1'][i],tdata['position_angle'][i])
	print (count)
	mosaicid = tableMosaic_to_full(tdata['Mosaic_ID'][i])
	

	postage2(directory,i,RAs[i],DECs[i],NN_RAs[i],NN_DECs[i])

F.close()