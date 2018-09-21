import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.io import fits
from astropy.table import Table, join
import pyfits as pf
import pywcs as pw
import pylab as pl
import matplotlib.pyplot as plt
import os
import math

file = 'P173+55'


prefix = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'
filename1 = prefix + file + '/mosaic.fits'
# hdu_list = fits.open(filename1)
# image_data = hdu_list[0].data 
# header = hdu_list[0].header
# hdu_list.close()

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


def find_orientation(i,fitsim,ra,dec, nn_ra, nn_dec, nn_dist, maj, s = (3/60.),plot=False,head=None,hdulist=None):
	'''	
	Finds the orientation of NN

	To run this function for all sources, use setup_find_orientation_NN() instead.

	A big part of this function is a re-run of the postage function,
	since we need to make a new postage basically, but this time as an array.

	Arguments: 
	i : the new_Index of the source

	fitsim: the postage created earlier 

	ra, dec: the ra and dec of the source

	Maj: the major axis of the source // TO-BE

	s: the width of the image, default 3 arcmin, because it has to be a tiny bit
	lower than the postage created earlier (width 4 arcmin) or NaNs will appear in the image

	head and hdulist, the header and hdulist if a postage hasn't been created before.
	(This is so it doesn't open every time in the loop but is opened before the loop.)

	Output: 
	i, max_angle, len(err_indices)
	Which are: the new_Index of the source, the best angle of the orientation, and
	the amount of orientations that have avg flux value > 80% of the peak avg flux value along the line
	
	If plot=True, produces the plots of the best orientation and the Flux vs Orientation as well


	'''
	################## BEGIN Postage Function #############
	if not head :		
		head = pf.getheader(fitsim)
		hdulist = pf.open(fitsim)

	# for NN take the projected middle as the RA and DEC
	ra,dec = (ra+nn_ra)/2. , (dec+nn_dec)/2.

	# Parse the WCS keywords in the primary HDU
	wcs = pw.WCS(hdulist[0].header)
	# Some pixel coordinates of interest.
	skycrd = np.array([ra,dec])
	skycrd = np.array([[ra,dec,0,0]], np.float_)
	# Convert pixel coordinates to world coordinates
	# The second argument is "origin" -- in this case we're declaring we
	# have 1-based (Fortran-like) coordinates.
	pixel = wcs.wcs_sky2pix(skycrd, 1)
	# Some pixel coordinates of interest.
	x = pixel[0][0]
	y = pixel[0][1]
	pixsize = abs(wcs.wcs.cdelt[0])
	if pl.isnan(s):
	    s = 25.
	N = (s/pixsize)
	# print 'x=%.5f, y=%.5f, N=%i' %(x,y,N)
	ximgsize = head.get('NAXIS1')
	yimgsize = head.get('NAXIS2')
	if x ==0:
	    x = ximgsize/2
	if y ==0:
	    y = yimgsize/2
	offcentre = False
	# subimage limits: check if runs over edges
	xlim1 =  x - (N/2)
	if(xlim1<1):
	    xlim1=1
	    offcentre=True
	xlim2 =  x + (N/2)
	if(xlim2>ximgsize):
	    xlim2=ximgsize
	    offcentre=True
	ylim1 =  y - (N/2)
	if(ylim1<1):
	    ylim1=1
	    offcentre=True
	ylim2 =  y + (N/2)
	if(ylim2>yimgsize):
	    offcentre=True
	    ylim2=yimgsize

	xl = int(xlim1)
	yl = int(ylim1)
	xu = int(xlim2)
	yu = int(ylim2)
	################## END Postage Function #############

	# extract the data array instead of making a postage stamp
	from astropy.nddata.utils import extract_array
	data_array = extract_array(hdulist[0].data,(yu-yl,xu-xl),(y,x))

	# use a radius for the line that is the NN distance, 
	# but with 4 pixels more added to the radius
	# to make sure we do capture the whole source
	radius = nn_dist/2. * 40 #arcmin --> image units
	radius = int(radius) + 4 # is chosen arbitrarily

	# in the P173+55 1 arcmin = 40 image units, should check if this is true everywhere, for FieldNames it is.

	# check if there are no NaNs in the cutout image, if there are then make smaller cutout
	if True in (np.isnan(data_array)):
		if s < (2/60.):
			print "No hope left for this source "
			return i, 0.0, 100.5,100.5,'no_hope',100.5

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			try: 
				return find_orientation(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,maj,s=(2*radius+2)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			try:
				return find_orientation(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,maj,s=(2*radius+4)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5


	from scipy.ndimage import interpolation
	from scipy.ndimage import map_coordinates
	#the center of the image is at the halfway point -1 for using array-index
	xcenter = np.shape(data_array)[0]/2-1 # pixel coordinates
	ycenter = np.shape(data_array)[1]/2-1# pixel coordinates
	
	# make a line with 'num' points and radius = radius
	x0, y0 = xcenter-radius,ycenter 
	x1, y1 = xcenter+radius,ycenter
	num = 1000
	x, y = np.linspace(x0,x1,num), np.linspace(y0,y1,num)

	# the final orientation will be max_angle
	max_angle = 0
	max_value = 0
	# flux values for 0 to 179 degrees of rotation (which is also convenietly their index)
	all_values = []
	for angle in range (0,180):
		# using spline order 3 interpolation to rotate the data by 0-180 deg
		data_array2 = interpolation.rotate(data_array,angle*1.,reshape=False)
		# extract the values along the line, 
		zi = map_coordinates(data_array2, np.vstack((y,x)),prefilter=False)
		# calc the mean flux
		meanie = np.sum(zi)
		if meanie > max_value:
			max_value = meanie
			max_angle = angle
		all_values.append(meanie)
			
	# calculate all orientiations for which the average flux lies within
	# 80 per cent of the peak average flux
	err_orientations = np.where(all_values > (0.8 * max_value))[0]
	
	# find the cutoff value, dependent on the distance, should think about wheter i want to use Maj
	cutoff = 2*np.arctan((maj/60.)/nn_dist) * 180 / np.pi # convert rad to deg
	if len(err_orientations) > cutoff:
		classification = 'Large err'
	else:
		classification = 'Small err'

	# then find the amount of maxima and the lobe_ratio
	from scipy.signal import argrelextrema
	data_array2 = interpolation.rotate(data_array,max_angle,reshape=False)
	zi = map_coordinates(data_array2, np.vstack((y,x)),prefilter=False)
	# find the local maxima in the flux along the line
	indx_extrema = argrelextrema(zi, np.greater)

	if zi[0] > zi[1] :
		# then there is a maximum (outside of the line range) , namely the first point
		new_indx_extrema = (np.append(indx_extrema,0),)
		del indx_extrema
		indx_extrema = new_indx_extrema
		del new_indx_extrema

	if zi[len(zi)-1] > zi[len(zi)-2] :
		# then there is a maximum (outside of the line range), namely the last point
		new_indx_extrema = (np.append(indx_extrema,len(zi)-1),)
		del indx_extrema
		indx_extrema = new_indx_extrema
		del new_indx_extrema

	amount_of_maxima = len(indx_extrema[0])
	# calculate the flux ratio of the lobes
	lobe_ratio_bool = False
	lobe_ratio = 0.0 # in case there is only 1 maximum 
	if amount_of_maxima > 1:
		if amount_of_maxima == 2:
			lobe_ratio = (zi[indx_extrema][0] / zi[indx_extrema][1])
		else:
			# more maxima, so the lobe_ratio is defined as the ratio between the brightest lobes
			indx_maximum_extrema =  np.flip(np.argsort(zi[indx_extrema]),0)[:2] #argsort sorts ascending so flip makes it descending
			indx_maximum_extrema = indx_extrema[0][indx_maximum_extrema] 
			lobe_ratio = (zi[indx_maximum_extrema[0]] / zi[indx_maximum_extrema][1])


	if plot:
		# the plot of the rotated source and the flux along the line
		fig, axes = plt.subplots(nrows=2,figsize=(12,12))
		axes[0].imshow(data_array2,origin='lower')
		axes[0].plot([x0, x1], [y0, y1], 'r-',alpha=0.3)
		axes[1].plot(zi)
		Fratio = 10.
		if amount_of_maxima > 1:
			if ( (1./Fratio) < lobe_ratio < (Fratio) ):
				lobe_ratio_bool = True
			
		plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + '\n Best orientation = ' + str(max_angle) + ' degrees | err_orientation: '+ str(len(err_orientations)) +
			'\nlobe ratio ' + str(lobe_ratio_bool) + ' | extrema: ' +str(indx_extrema)  + ' | nn_dist: ' + str(nn_dist)
			+ '\n lobe ratio: ' + str(lobe_ratio))

		# # saving the figures to seperate directories
		# if amount_of_maxima == 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'.png')

		# elif amount_of_maxima > 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'.png') 

		# else:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'.png') 		


		plt.savefig('/data1/osinga/figures/speed_up_test/'+head['OBJECT']+'src_'+str(i)+'.png') 		

		plt.clf()
		plt.close()


		# the plot of the flux vs orientation
		plt.plot(all_values, label='all orientations')
		plt.scatter(err_orientations, np.array(all_values)[err_orientations], color= 'y',label='0.8 fraction')
		plt.axvline(x=max_angle,ymin=0,ymax=1, color = 'r', label='best orientation')
		plt.title('Best orientation for Source ' + str(i) + '\nClassification: '+classification + ' | Cutoff: '+str(cutoff)+ ' | Error: ' + str(len(err_orientations)))
		plt.ylabel('Average flux (arbitrary units)')
		plt.xlabel('orientation (degrees)')
		plt.legend()
		plt.xlim(0,180)
		# # saving the figures to seperate directories
		# if amount_of_maxima == 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		# elif amount_of_maxima > 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		# else:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 

		plt.savefig('/data1/osinga/figures/speed_up_test/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 

		plt.clf()
		plt.close()
	return i, max_angle, len(err_orientations), amount_of_maxima , classification, lobe_ratio


def rotate_efficient(origin,point,angle):
	'''
	Rotate a point counterclockwise by a given angle around a given origin

	Angle should be given in degrees
	
	'''

	angle = angle * np.pi / 180. # convert degrees to radians

	ox,oy = origin
	px,py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py-oy)
	qy = oy + math.sin(angle) * (px - ox) - math.cos(angle) * (py-oy)
	return qx,qy


def find_orientation_efficient(i,fitsim,ra,dec, nn_ra, nn_dec, nn_dist, maj, s = (3/60.),plot=False,head=None,hdulist=None):
	'''	
	Finds the orientation of NN

	To run this function for all sources, use setup_find_orientation_NN() instead.

	A big part of this function is a re-run of the postage function,
	since we need to make a new postage basically, but this time as an array.

	Arguments: 
	i : the new_Index of the source

	fitsim: the postage created earlier 

	ra, dec: the ra and dec of the source

	Maj: the major axis of the source // TO-BE

	s: the width of the image, default 3 arcmin, because it has to be a tiny bit
	lower than the postage created earlier (width 4 arcmin) or NaNs will appear in the image

	head and hdulist, the header and hdulist if a postage hasn't been created before.
	(This is so it doesn't open every time in the loop but is opened before the loop.)

	Output: 
	i, max_angle, len(err_indices)
	Which are: the new_Index of the source, the best angle of the orientation, and
	the amount of orientations that have avg flux value > 80% of the peak avg flux value along the line
	
	If plot=True, produces the plots of the best orientation and the Flux vs Orientation as well


	'''
	################## BEGIN Postage Function #############
	if not head :		
		head = pf.getheader(fitsim)
		hdulist = pf.open(fitsim)

	# for NN take the projected middle as the RA and DEC
	ra,dec = (ra+nn_ra)/2. , (dec+nn_dec)/2.

	# Parse the WCS keywords in the primary HDU
	wcs = pw.WCS(hdulist[0].header)
	# Some pixel coordinates of interest.
	skycrd = np.array([ra,dec])
	skycrd = np.array([[ra,dec,0,0]], np.float_)
	# Convert pixel coordinates to world coordinates
	# The second argument is "origin" -- in this case we're declaring we
	# have 1-based (Fortran-like) coordinates.
	pixel = wcs.wcs_sky2pix(skycrd, 1)
	# Some pixel coordinates of interest.
	x = pixel[0][0]
	y = pixel[0][1]
	pixsize = abs(wcs.wcs.cdelt[0])
	if pl.isnan(s):
	    s = 25.
	N = (s/pixsize)
	# print 'x=%.5f, y=%.5f, N=%i' %(x,y,N)
	ximgsize = head.get('NAXIS1')
	yimgsize = head.get('NAXIS2')
	if x ==0:
	    x = ximgsize/2
	if y ==0:
	    y = yimgsize/2
	offcentre = False
	# subimage limits: check if runs over edges
	xlim1 =  x - (N/2)
	if(xlim1<1):
	    xlim1=1
	    offcentre=True
	xlim2 =  x + (N/2)
	if(xlim2>ximgsize):
	    xlim2=ximgsize
	    offcentre=True
	ylim1 =  y - (N/2)
	if(ylim1<1):
	    ylim1=1
	    offcentre=True
	ylim2 =  y + (N/2)
	if(ylim2>yimgsize):
	    offcentre=True
	    ylim2=yimgsize

	xl = int(xlim1)
	yl = int(ylim1)
	xu = int(xlim2)
	yu = int(ylim2)
	################## END Postage Function #############

	# extract the data array instead of making a postage stamp
	from astropy.nddata.utils import extract_array
	data_array = extract_array(hdulist[0].data,(yu-yl,xu-xl),(y,x))

	# use a radius for the line that is the NN distance, 
	# but with 4 pixels more added to the radius
	# to make sure we do capture the whole source
	radius = nn_dist/2. * 40 #arcmin --> image units
	radius = int(radius) + 4 # is chosen arbitrarily

	# in the P173+55 1 arcmin = 40 image units, should check if this is true everywhere, for FieldNames it is.

	# check if there are no NaNs in the cutout image, if there are then make smaller cutout
	if True in (np.isnan(data_array)):
		if s < (2/60.):
			print "No hope left for this source "
			return i, 0.0, 100.5,100.5,'no_hope',100.5

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			try: 
				return find_orientation(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,maj,s=(2*radius+2)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			try:
				return find_orientation(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,maj,s=(2*radius+4)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5


	from scipy.ndimage import map_coordinates
	#the center of the image is at the halfway point -1 for using array-index
	xcenter = np.shape(data_array)[0]/2-1 # pixel coordinates
	ycenter = np.shape(data_array)[1]/2-1# pixel coordinates
	
	# make a line with 'num' points and radius = radius
	x0, y0 = xcenter-radius,ycenter 
	x1, y1 = xcenter+radius,ycenter
	num = 1000
	x, y = np.linspace(x0,x1,num), np.linspace(y0,y1,num)


	# the final orientation will be max_angle
	max_angle = 0
	max_value = 0
	# flux values for 0 to 179 degrees of rotation (which is also convenietly their index)
	all_values = []
	for angle in range (0,180):
		# rotate line on the left side of the center
		x0_rot, y0_rot = rotate_efficient((xcenter,ycenter),(x0,y0),angle)
		# rotate line on the right side of the center
		x1_rot,y1_rot = rotate_efficient((xcenter,ycenter),(x1,y1),angle)

		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 

		# extract the values along the line, 
		zi = map_coordinates(data_array, np.vstack((y_rot,x_rot)),prefilter=True)
		# calc the mean flux
		meanie = np.sum(zi)
		if meanie > max_value:
			max_value = meanie
			max_angle = angle
		all_values.append(meanie)
			
	# calculate all orientiations for which the average flux lies within
	# 80 per cent of the peak average flux
	err_orientations = np.where(all_values > (0.8 * max_value))[0]
	
	# find the cutoff value, dependent on the distance, should think about wheter i want to use Maj
	cutoff = 2*np.arctan((maj/60.)/nn_dist) * 180 / np.pi # convert rad to deg
	if len(err_orientations) > cutoff:
		classification = 'Large err'
	else:
		classification = 'Small err'

	# then find the amount of maxima and the lobe_ratio
	from scipy.signal import argrelextrema
	# rotate line on the left side of the center
	x0_rot, y0_rot = rotate_efficient((xcenter,ycenter),(x0,y0),max_angle)
	# rotate line on the right side of the center
	x1_rot,y1_rot = rotate_efficient((xcenter,ycenter),(x1,y1),max_angle)
	x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 

	zi = map_coordinates(data_array, np.vstack((y_rot,x_rot)),prefilter=True)
	# find the local maxima in the flux along the line
	indx_extrema = argrelextrema(zi, np.greater)

	if zi[0] > zi[1] :
		# then there is a maximum (outside of the line range) , namely the first point
		new_indx_extrema = (np.append(indx_extrema,0),)
		del indx_extrema
		indx_extrema = new_indx_extrema
		del new_indx_extrema

	if zi[len(zi)-1] > zi[len(zi)-2] :
		# then there is a maximum (outside of the line range), namely the last point
		new_indx_extrema = (np.append(indx_extrema,len(zi)-1),)
		del indx_extrema
		indx_extrema = new_indx_extrema
		del new_indx_extrema

	amount_of_maxima = len(indx_extrema[0])
	# calculate the flux ratio of the lobes
	lobe_ratio_bool = False
	lobe_ratio = 0.0 # in case there is only 1 maximum 
	if amount_of_maxima > 1:
		if amount_of_maxima == 2:
			lobe_ratio = (zi[indx_extrema][0] / zi[indx_extrema][1])
		else:
			# more maxima, so the lobe_ratio is defined as the ratio between the brightest lobes
			indx_maximum_extrema =  np.flip(np.argsort(zi[indx_extrema]),0)[:2] #argsort sorts ascending so flip makes it descending
			indx_maximum_extrema = indx_extrema[0][indx_maximum_extrema] 
			lobe_ratio = (zi[indx_maximum_extrema[0]] / zi[indx_maximum_extrema][1])


	if plot:
		# the plot of the rotated source and the flux along the line
		fig, axes = plt.subplots(nrows=2,figsize=(12,12))
		axes[0].imshow(data_array,origin='lower')
		axes[0].plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'r-',alpha=0.3)
		axes[1].plot(zi)
		Fratio = 10.
		if amount_of_maxima > 1:
			if ( (1./Fratio) < lobe_ratio < (Fratio) ):
				lobe_ratio_bool = True
			
		plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + '\n Best orientation = ' + str(max_angle) + ' degrees | err_orientation: '+ str(len(err_orientations)) +
			'\nlobe ratio ' + str(lobe_ratio_bool) + ' | extrema: ' +str(indx_extrema)  + ' | nn_dist: ' + str(nn_dist)
			+ '\n lobe ratio: ' + str(lobe_ratio))

		# saving the figures to seperate directories
		# if amount_of_maxima == 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'.png')

		# elif amount_of_maxima > 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'.png') 

		# else:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'.png') 		
		plt.savefig('/data1/osinga/figures/speed_up_test/'+head['OBJECT']+'src_'+str(i)+'_2.png') 		
		plt.clf()
		plt.close()


		# the plot of the flux vs orientation
		plt.plot(all_values, label='all orientations')
		plt.scatter(err_orientations, np.array(all_values)[err_orientations], color= 'y',label='0.8 fraction')
		plt.axvline(x=max_angle,ymin=0,ymax=1, color = 'r', label='best orientation')
		plt.title('Best orientation for Source ' + str(i) + '\nClassification: '+classification + ' | Cutoff: '+str(cutoff)+ ' | Error: ' + str(len(err_orientations)))
		plt.ylabel('Average flux (arbitrary units)')
		plt.xlabel('orientation (degrees)')
		plt.legend()
		plt.xlim(0,180)
		# # saving the figures to seperate directories
		# if amount_of_maxima == 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		# elif amount_of_maxima > 2:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		# else:
		# 	# plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 

		plt.savefig('/data1/osinga/figures/speed_up_test/'+head['OBJECT']+'src_'+str(i)+'_orientation2.png') 
		plt.clf()
		plt.close()
	return i, max_angle, len(err_orientations), amount_of_maxima , classification, lobe_ratio



def proof():
	dinges = np.zeros((100,100))
	dinges += 1
	xcenter = np.shape(dinges)[0]/2-1 # pixel coordinates
	ycenter = np.shape(dinges)[1]/2-1# pixel coordinates

	radius = 5
	x0, y0 = xcenter-radius,ycenter 
	x1, y1 = xcenter+radius,ycenter
	num = 1000
	x, y = np.linspace(x0,x1,num), np.linspace(y0,y1,num)

	# left side
	x0_rot, y0_rot = rotate_efficient((xcenter,ycenter),(x0,y0),45)
	# right side
	x1_rot,y1_rot = rotate_efficient((xcenter,ycenter),(x1,y1),45)

	x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 



	from scipy.ndimage import interpolation
	from scipy.ndimage import map_coordinates
	plt.imshow(interpolation.rotate(dinges,10,reshape=False),origin='lower')

	plt.plot((x),(y))
	plt.plot(x_rot,y_rot)
	plt.show()

# P6 
# i = 268
# orientation = 50
# err_orientation = 49
# amount_of_maxima = 2
# ra = 165.66440439125128
# dec = 48.43041765453008
# nn_ra = 165.661659977957
# nn_dec = 48.43259357031313

source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+'P6'+'/mosaic.fits'
head = pf.getheader(source)
hdulist = pf.open(source)

ra,dec,nn_ra,nn_dec,nn_dist,maj = load_in('/data1/osinga/data/NN/all_NN_sources.fits','RA','DEC','new_NN_RA','new_NN_DEC','new_NN_distance(arcmin)','Maj')
index = 3115 # index of the source i want to use as test-case in the all_NN_sources.fits file.
index_start = 2974
index_end = 3180

# proof()
for index in range(index_start,index_end):
	results1 = find_orientation(index,'useless',ra[index],dec[index],nn_ra[index],nn_dec[index],nn_dist[index],maj[index],plot=True,head=head,hdulist=hdulist)
	print results1
	# i, max_angle, len(err_orientations), amount_of_maxima , classification, lobe_ratio


	results2 = find_orientation_efficient(index,'useless',ra[index],dec[index],nn_ra[index],nn_dec[index],nn_dist[index],maj[index],plot=True,head=head,hdulist=hdulist)
	print results2