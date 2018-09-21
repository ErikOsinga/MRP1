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

def postage(NN,fitsim,postfits,ra,dec, s,nn_ra = None,nn_dec = None):
	'''
	Makes a postage 'postfits' from the entire fitsim at RA=ra and DEC=dec with radius s (degrees)
	Creates both a 'postfits.fits' cutout as well as a 'postfits.png' image for quick viewing

	If NN_RA and NN_DEC are provided it shows an arrow from the source to the nearest neighbour.
	'''
	import os 

	# for NN take the projected middle as the RA and DEC
	if NN:
		ra,dec = (ra+nn_ra)/2. , (dec+nn_dec)/2.


	head = pf.getheader(fitsim)

	hdulist = pf.open(fitsim)
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
	# pixsize = 0.000416666666666666 # this is what it should give for P173+55 
	# but it returns 0 for pixsize, and pixel etc
	if pl.isnan(s):
	    s = 25.
	N = (s/pixsize)
	print 'x=%.5f, y=%.5f, N=%i' %(x,y,N)

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
	print 'postage stamp is %i x %i pixels' %(xu-xl,yu-yl)


	from astropy.nddata.utils import extract_array

	# plt.imshow(extract_array(hdulist[0].data,(yu-yl,xu-xl),(y,x)),origin='lower')
	# plt.savefig(postfits+'.png')
	# plt.show()


	# make fits cutout
	inps = fitsim + '[%0.0f:%0.0f,%0.0f:%0.0f]' %(xl,xu,yl,yu)
	if os.path.isfile(postfits): os.system('rm '+postfits)
	os.system('fitscopy %s %s' %(inps,postfits) )
	print  'fitscopy %s %s' %(inps,postfits) 

	# # make a png cutout from the fits cutout with an arrow pointing to the NN
	# if NN_RA:
	# 	gc = aplpy.FITSFigure(postfits)
	# 	gc.show_grayscale()
	# 	gc.add_grid()
	# 	gc.show_arrows(ra,dec,NN_RA-ra,NN_DEC-dec)
	# 	# gc.show_regions(prefix+file+'cat.srl.reg') #very slow
	# 	# pl.show()
	# 	gc.save(postfits+'.png')
	# 	gc.close()

	return postfits

def extr_array(NN,fitsim,ra,dec, nn_ra=None, nn_dec=None, nn_dist=None,maj=None, s = (3/60.),head=None,hdulist=None):
	"""
	Produces a smaller image from the entire fitsim, with dimension s x s. 
	around coordinates ra,dec. If head != None, then provide the head and hdulist
	of the image that the source is in, else provide fitsim.  

	(A big part of this function is a re-run of the postage function,
	since we need to make a new postage basically, but this time as an array.)

	Arguments:
	NN - Boolean indicating if it's an NN source or a MG source.
	fitsim -- Image (.fits) that the source is located in. 
			  Faster if there is a small cutout file.
	ra,dec -- Right ascension and declination of the source. Will be center of image
	nn_ra, nn_dec, nn_dist -- RA, DEC and distance of nearest neighbour source.
	maj -- major axis of the source. Used for classifying the error.
	s -- Dimension of the cutout image. Default 3 arcminutes.
	head -- If theres no fitsim file, provide the head and hdulist of the large file.

	Returns:
	data_array -- Numpy array containing the extracted cutout image.

	"""

	if not head :		
		head = pf.getheader(fitsim)
		hdulist = pf.open(fitsim)

	# for NN take the projected middle as the RA and DEC
	if NN:
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

	# extract the data array instead of making a postage stamp
	data_array = extract_array(hdulist[0].data,(yu-yl,xu-xl),(y,x))

	return data_array

def find_lobes(data_array,positions,angle,hdulist,num=1000):
	# then find the amount of maxima and the lobe_ratio

	xcenter,ycenter = positions['xcenter'], positions['ycenter']
	x0,y0 = positions['x0'], positions['y0']
	x1,y1 = positions['x1'], positions['y1']


	# rotate line on the left side of the center
	x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),angle)
	# rotate line on the right side of the center
	x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),angle)
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
	position_angle = 1000.0 # in case there is only 1 maximum
	wcs = pw.WCS(hdulist[0].header)
	if amount_of_maxima > 1:
		if amount_of_maxima == 2:
			lobe_ratio = (zi[indx_extrema][0] / zi[indx_extrema][1])
			# calculate the position angle
			lobe1 = np.array([[x_rot[indx_extrema][0],y_rot[indx_extrema][0],0,0]])
			lobe2 = np.array([[x_rot[indx_extrema][1],y_rot[indx_extrema][1],0,0]])
			lobe1 = wcs.wcs_pix2sky(lobe1,1)
			lobe2 = wcs.wcs_pix2sky(lobe2,1)
			position_angle = PositionAngle(lobe1[0][0],lobe1[0][1],lobe2[0][0],lobe2[0][1])
			if position_angle < 0:
				position_angle += 180.

		else:
			# more maxima, so the lobe_ratio is defined as the ratio between the brightest lobes
			indx_maximum_extrema =  np.flip(np.argsort(zi[indx_extrema]),0)[:2] #argsort sorts ascending so flip makes it descending
			indx_maximum_extrema = indx_extrema[0][indx_maximum_extrema] 
			lobe_ratio = (zi[indx_maximum_extrema[0]] / zi[indx_maximum_extrema][1])

			#find the RA and DEC of the two brightest lobes
			lobe1 = np.array([[x_rot[indx_maximum_extrema][0],y_rot[indx_maximum_extrema][0],0,0]])
			lobe2 = np.array([[x_rot[indx_maximum_extrema][1],y_rot[indx_maximum_extrema][1],0,0]])
			lobe1 = wcs.wcs_pix2sky(lobe1,1)
			lobe2 = wcs.wcs_pix2sky(lobe2,1)
			# then calculate the position angle
			position_angle = PositionAngle(lobe1[0][0],lobe1[0][1],lobe2[0][0],lobe2[0][1])
			if position_angle < 0:
				position_angle += 180.
	else:
		raise ValueError("This function only works on sources with >1 maximum")

	lobe1 = lobe1[0][:2]
	lobe2 = lobe2[0][:2]

	return position_angle, lobe_ratio, lobe1,lobe2, indx_extrema

def flux_along_line(data_array,radius,cutoff,hdulist,head,i):
	"""
	Find the Position angle with the flux along a line method (Better: Gaussian fit.)
	Position angle is defined as the angle between the brightest lobes.

	Arguments:
	data_array -- The Numpy array that contains the source
	radius -- The radius of the line in pixel units.
	cutoff -- The cutoff value for a 'Large' or 'Small' error.
	hdulist -- Necessary to get the coordinates
	head -- only for plotting the title of the field
	i -- source index

	Returns:
	position_angle -- The position angle
	err_orientations -- The angles that produce 80% of the flux
	classification -- 'Large' or 'Small' error
	max_angle -- The orientation that produces the maximal flux (assumes source
																is symmetric...)
	lobe_ratio -- Flux ratio between the lobes.

	"""

	# Parse the WCS keywords in the primary HDU
	wcs = pw.WCS(hdulist[0].header)

	#the center of the image is at the halfway point -1 for using array-index
	xcenter = np.shape(data_array)[0]/2-1 # pixel coordinates
	ycenter = np.shape(data_array)[1]/2-1# pixel coordinates
	
	# define the start and end points of the line with 'num' points and radius = radius
	x0, y0 = xcenter-radius,ycenter 
	x1, y1 = xcenter+radius,ycenter
	num = 1000

	# the final orientation will be max_angle
	max_angle = 0
	max_value = 0
	# flux values for 0 to 179 degrees of rotation (which is also convenietly their index)
	all_values = []
	for angle in range (0,180):
		# rotate line on the left side of the center
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),angle)
		# rotate line on the right side of the center
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),angle)
		# the rotated line
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
	all_cutoff_values = np.asarray(all_values)[err_orientations]
	minima = np.argsort(all_cutoff_values)

	minimum1 = minima[0]
	minimum2 = minima[1]

	# Orientation can vary between the two flux minima. 
	min_orientation = err_orientations[minimum1]
	max_orientation = err_orientations[minimum2]

	if len(err_orientations) > cutoff:
		classification = 'Large err'
	else:
		classification = 'Small err'

	positions = dict()
	positions['xcenter'], positions['ycenter'] = xcenter, ycenter
	positions['x0'], positions['y0'] = x0, y0
	positions['x1'], positions['y1'] = x1, y1

	plot = False
	if plot:
		# the plot of the flux vs orientation
		# fig = plt.figure(figsize=(6.8,8))
		fig = plt.figure(figsize=(7.8,8))
		plt.plot(all_values, label='Probed orientations',color='k')
		# plt.axvline(x=max_angle,ymin=0,ymax=1, color = 'r', label='best orientation')
		plt.scatter(max_angle,all_values[max_angle],marker='v',color='k',label='Best fit orientation')
		# plt.scatter(err_orientations, np.array(all_values)[err_orientations], color= 'y',label='0.8 fraction')
		plt.axvline(min_orientation,ymin=0,ymax=1,color='k'
			,ls='dashed',label='0.8 flux fraction')
		plt.axvline(max_orientation,0,1,color='k'
			,ls='dashed')
		plt.title('Field: ' + head['OBJECT'] +' | Source ' + str(i) + ' | Best orientation %i '%max_angle+r'$^\circ$' #+ '\nClassification: '+classification 
			+ '\nCutoff: %i'%(cutoff)+ r'$^\circ$' + ' | Error: ' + str(len(err_orientations))+r'$^\circ$')
		plt.ylabel('Average flux (arbitrary units)',fontsize=12)
		plt.xlabel('orientation (degrees)',fontsize=12)
		plt.legend()
		plt.xlim(0,180)
		plt.show()

	return (max_angle, err_orientations, min_orientation, max_orientation, classification, positions)

def find_orientationNN(i,fitsim,ra,dec, nn_ra, nn_dec, nn_dist,Min, nn_Min, s = (3/60.),plot=False,head=None,hdulist=None):
	'''	
	Finds the orientation of NN

	To run this function for all sources, use setup_find_orientation_NN() instead.

	Arguments: 
	i -- the new_Index of the source
	fitsim -- the postage created earlier 
	ra, dec -- the ra and dec of the source
	Min -- the semiminor axis of the source 
	nn_Min -- the semiminor axis of the nearest neighbour 
	s: the width of the image, default 3 arcmin, because it has to be a tiny bit
		lower than the postage created earlier (width 4 arcmin) or NaNs will appear in the image
	head and hdulist, the header and hdulist if a postage hasn't been created before.
	(This is so it doesn't open every time in the loop but is opened before the loop.)

	Returns: 
	i -- new_Index of the source
	max_angle -- Angle of the flux line for which flux is maximized.
	len(err_orientations) -- Amount of degrees spread in the angle of the flux line
	len(indx_extrema[0]) -- Amount of extrema along the line
	classification -- 'Large' or 'Small' error, based on cutoff value
	lobe_ratio -- The flux ratio between the two brightest lobes
	position_angle -- Position angle between the two brightest lobes 
	error - Array containing difference in [RA,DEC] between the 80% cutoff value and the maximum flux

	If plot=True, produces the plots of the best orientation and the Flux vs Orientation as well

	'''

	wcs = pw.WCS(hdulist[0].header)

	data_array = extr_array(True,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,None,s=s
							,head=head,hdulist=hdulist)

	postfits = './temp.fits'
	print ('Creating temporary ./temp.fits file')
	postage(True,fitsim,postfits,ra,dec,s=s,nn_ra=nn_ra,nn_dec=nn_dec)

	# print ('Creating fits cutout')
	# postage(True,fitsim,'../figures/NN_excluded_by_also_being_MG/'+head['OBJECT']+'src_'+str(i)
	# 	,ra,dec,s=s,nn_ra=nn_ra,nn_dec=nn_dec)


	# use a radius for the line that is the NN distance, 
	# but with 4 pixels more added to the radius
	# to make sure we do capture the whole source
	radius = nn_dist/2. * 40 #arcmin --> image units
	radius = int(radius) + 4 # is chosen arbitrarily

	# in the P173+55 1 arcmin = 40 image units
	# this is true for field in FieldNames.

	# check if there are no NaNs in the cutout image, 
	# if there are then make smaller cutout
	if True in (np.isnan(data_array)):
		if s < (2/60.):
			print "No hope left for this source "
			return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			try: 
				return find_orientationNN(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,Min,nn_Min
					,s=(2*radius+2)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			try:
				return find_orientationNN(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,Min,nn_Min
					,s=(2*radius+4)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

	# find the cutoff value, dependent on the distance, should think about whether i want to use Maj
	# cutoff = 2*np.arctan((maj/60.)/nn_dist) * 180 / np.pi # convert rad to deg
	cutoff = (2*np.arctan((Min/60.)/(nn_dist/2)) + 2*np.arctan((nn_Min/60.)/(nn_dist/2)) ) /2.   * 180 / np.pi # convert rad to deg


	# find the max orientation of the flux along a line and '80% flux error'
	(max_angle, err_orientations, min_orientation
		, max_orientation, classification, positions) = flux_along_line(data_array,radius,cutoff,hdulist,head,i)

	# then find the amount of maxima and the lobe_ratio
	position_angle, lobe_ratio, lobe1, lobe2, indx_extrema = find_lobes(data_array,positions,max_angle,hdulist)

	# to make sure the lobes aren't switched
	if (min_orientation < 90 and max_orientation > 90):
		max_orientation -= 180 
	elif (min_orientation > 90 and max_orientation < 90):
		min_orientation -= 180

	original_position_angle = position_angle

	try: 
		lobe1_min,lobe2_min = find_lobes(data_array,positions,min_orientation,hdulist)[2:4]
		error = lobe1 - lobe1_min
	except ValueError:
		print 'tja1'
	try:
		lobe1_min,lobe2_min = find_lobes(data_array,positions,max_orientation,hdulist)[2:4]
		error = lobe1 - lobe1_min
	except ValueError:
		print 'tja2'

	xcenter,ycenter = positions['xcenter'], positions['ycenter']
	x0,y0 = positions['x0'], positions['y0']
	x1,y1 = positions['x1'], positions['y1']
	num = 1000

	if plot:
		# the plot of the rotated source and the flux along the line
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),max_angle)
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),max_angle)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		zi = map_coordinates(data_array, np.vstack((y_rot,x_rot)),prefilter=True)
		

		from astropy.wcs import WCS
		from astropy.io import fits
		from astropy.visualization.wcsaxes import SphericalCircle
		from astropy import units as u

		hdu = fits.open(postfits)[0]
		wcs = WCS(hdu.header)
		# fig = plt.figure(figsize=(6.8,8))
		fig = plt.figure(figsize=(7.8,8))

		ax1 = plt.subplot(211, projection=wcs)

		lon = ax1.coords[0]
		lat = ax1.coords[1]

		lon.set_ticks(exclude_overlapping=True)

		# lon.set_major_formatter('d.dd')
		# lat.set_major_formatter('d.dd')

		# to plot a sphere at ra,dec 
		# ax1.scatter(ra, dec, transform=ax1.get_transform('fk5'), s=300,
  #          edgecolor='white', facecolor='none')

		
		ax1.imshow(hdu.data,origin='lower')#,cmap='gray')
		ax1.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'r-',alpha=0.3)
		ax2 = plt.subplot(212)
		ax2.plot(zi,color='k',ls='solid')

		print ('Removing temporary ./temp.fits file')
		os.system('rm ./temp.fits')

		# rotate line on the left side of the center with min angle (error lower bound)
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),min_orientation)
		# rotate line on the right side of the center with min angle
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),min_orientation)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		ax1.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'b-',alpha=0.6) # alpha=0.3

		# rotate line on the left side of the center with max angle (error upper bound)
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),max_orientation)
		# rotate line on the right side of the center with max angle
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),max_orientation)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		ax1.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'b-',alpha=0.6) # alpha=0.3

		ax1.set_xlabel('Right ascension (J2000)',fontsize=12)
		ax1.set_ylabel('Declination (J2000)',fontsize=12)
		ax2.set_ylabel('Flux (arbitrary units)',fontsize=12)
		ax2.set_xlabel('Points along line',fontsize=12)
		
		# plt.rc('text', usetex=True) # :( SEEMS THIS IS NOT EVEN NEEDED WHAAAT
		plt.suptitle('Field: ' + head['OBJECT'] + r' $\vert$ Source ' + str(i) + r' $\vert$ Best orientation ' + str(max_angle) +r'$^\circ$' 
			+ '\n Num maxima: ' +str(len(indx_extrema[0]))  + r' $\vert$ NN_dist: %.3f' % nn_dist + ' arcmin'
			+ '\n Lobe ratio: %.3f' % lobe_ratio + r' $\vert$ Position Angle: %i'  % position_angle +r'$^\circ$'
			+ r' $\vert$ Err orientation: %i' % len(err_orientations) + r'$^\circ$')

		''' without usetex=True 

		plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + ' | Line orientation ' + str(max_angle) +' degrees' 
			+ '\n Num maxima: ' +str(len(indx_extrema[0]))  + ' | NN_dist: %.3f' % nn_dist + ' arcmin'
			+ '\n Lobe ratio: %.3f' % lobe_ratio + ' | Position Angle: %.3f'  % position_angle
			+ ' | Err orientation: %.1f' % len(err_orientations))
		
		'''

		# plt.savefig('../figures/NN_excluded_by_orientation_cutoff/'+head['OBJECT']+'src_'+str(i)+'.png') 		
		# plt.savefig('../figures/NN_excluded_by_also_being_MG/'+head['OBJECT']+'src_'+str(i)+'.png') 		
		
		plt.show()
		# plt.clf()
		# plt.close()

	return i, max_angle, len(err_orientations), len(indx_extrema[0]) , classification, lobe_ratio, position_angle, error

def find_orientationMG(i,fitsim,ra,dec, Maj, Min, s = (3/60.),plot=False,head=None,hdulist=None):
	'''	
	Finds the orientation of multiple gaussian single sources 

	To run this function for all sources, use setup_find_orientation_multiple_gaussians() instead.

	A big part of this function is a re-run of the postage function,
	since we need to make a new postage basically, but this time as an array.

	Arguments: 
	i -- The new_Index of the source
	fitsim -- Postage created earlier (if exists)
	ra, dec -- Ra and dec of the source
	Maj -- Major axis of the source
	s -- Width of the image, default 3 arcmin, because it has to be a tiny bit
		lower than the postage created earlier (width 4 arcmin) or NaNs will appear in the image
	plot -- Whether to plot the results. 
	head and hdulist -- the header and hdulist if a postage hasn't been created before.
		(This is so it doesn't open every time in the loop but is opened before the loop.)

	Returns: 
	i -- new_Index of the source
	max_angle -- Angle of the flux line for which flux is maximized.
	len(err_orientations) -- Amount of degrees spread in the angle of the flux line
	len(indx_extrema[0]) -- Amount of extrema along the line
	classification -- 'Large' or 'Small' error, based on cutoff value
	lobe_ratio -- The flux ratio between the two brightest lobes
	position_angle -- Position angle between the two brightest lobes 
	error - Array containing difference in [RA,DEC] between the 80% cutoff value and the maximum flux

	If plot=True, produces the plots of the best orientation and the Flux vs Orientation as well

	'''

	wcs = pw.WCS(hdulist[0].header)
	
	data_array= extr_array(False,fitsim,ra,dec,s=s,head=head,hdulist=hdulist)

	postfits = './temp.fits'
	print ('Creating temporary ./temp.fits file')
	postage(False,fitsim,postfits,ra,dec,s=s)

	# use a radius for the line that is the semimajor axis, 
	# but with 2 pixels more added to the radius
	# to make sure we do capture the whole source
	radius = Maj / 60.  * 40. #arcsec -- > arcmin --> image units
	radius = int(radius) + 2 # is chosen arbitrarily

	# in the P173+55 1 arcmin = 40 image units
	# this is true for field in FieldNames.

	# check if there are no NaNs in the cutout image, if there are then make smaller cutout
	if True in (np.isnan(data_array)):
		if s < (2/60.):
			print "No hope left for this source "
			return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			try: 
				return find_orientationMG(i,fitsim,ra,dec,Maj,Min,s=(Maj*2./60./60.),plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			try:
				return find_orientationMG(i,fitsim,ra,dec,Maj,Min,s = (2/60.),plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5


	# find the cutoff value, dependent on the distance 
	cutoff = 2*np.arctan(Min/Maj) * 180 / np.pi # convert rad to deg

	# find the max orientation of the flux along a line and '80% flux error'
	(max_angle, err_orientations, min_orientation
		, max_orientation, classification, positions) = flux_along_line(data_array,radius,cutoff,hdulist,head,i)

	# then find the amount of maxima and the lobe_ratio
	position_angle, lobe_ratio, lobe1, lobe2, indx_extrema = find_lobes(data_array,positions,max_angle,hdulist)

	# to make sure the lobes aren't switched
	if (min_orientation < 90 and max_orientation > 90):
		max_orientation -= 180 
	elif (min_orientation > 90 and max_orientation < 90):
		min_orientation -= 180

	original_position_angle = position_angle

	try: 
		lobe1_min,lobe2_min = find_lobes(data_array,positions,min_orientation,hdulist)[2:4]
		error = lobe1 - lobe1_min
	except ValueError:
		print 'tja1'
	try:
		lobe1_min,lobe2_min = find_lobes(data_array,positions,max_orientation,hdulist)[2:4]
		error = lobe1 - lobe1_min
	except ValueError:
		print 'tja2'

	xcenter,ycenter = positions['xcenter'], positions['ycenter']
	x0,y0 = positions['x0'], positions['y0']
	x1,y1 = positions['x1'], positions['y1']
	num = 1000

	if plot:
		# the plot of the rotated source and the flux along the line
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),max_angle)
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),max_angle)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		zi = map_coordinates(data_array, np.vstack((y_rot,x_rot)),prefilter=True)

		from astropy.wcs import WCS
		from astropy.io import fits

		hdu = fits.open(postfits)[0]
		wcs = WCS(hdu.header)
		fig = plt.figure(figsize=(7.8,8))

		ax1 = plt.subplot(211, projection=wcs)

		lon = ax1.coords[0]
		lat = ax1.coords[1]

		lon.set_ticks(exclude_overlapping=True)

		ax1.imshow(hdu.data,origin='lower')
		ax1.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'r-',alpha=0.3)
		
		ax2 = plt.subplot(212)
		ax2.plot(zi)

		ax1.imshow(hdu.data,origin='lower')
		ax1.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'r-',alpha=0.3)
		ax2.plot(zi,color='k')

		print ('Removing temporary ./temp.fits file')
		os.system('rm ./temp.fits')

		# rotate line on the left side of the center with min angle
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),min_orientation)
		# rotate line on the right side of the center with min angle
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),min_orientation)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		ax1.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'b-',alpha=0.3)

		# rotate line on the left side of the center with max angle
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),max_orientation)
		# rotate line on the right side of the center with max angle
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),max_orientation)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		ax1.plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'b-',alpha=0.3)

		ax1.set_xlabel('Right ascension (J2000)')
		ax1.set_ylabel('Declination (J2000)')
		ax2.set_ylabel('Flux (arbitrary units)')
		ax2.set_xlabel('Points along line')
		'''	
	plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + ' | Line orientation ' + str(max_angle)
			+ '\n maxima: ' +str(indx_extrema[0])
			+ '\n lobe ratio: %.3f' % lobe_ratio + ' | Position Angle: %.3f'  % position_angle
			+ ' Err orientation: %.1f' % len(err_orientations))
		'''


		plt.suptitle('Field: ' + head['OBJECT'] + r' $\vert$ Source ' + str(i) + r' $\vert$ Best orientation ' + str(max_angle) +r'$^\circ$' 
			+ '\n Num maxima: ' +str(len(indx_extrema[0]))  
			+ '\n Lobe ratio: %.3f' % lobe_ratio + r' $\vert$ Position Angle: %i'  % position_angle +r'$^\circ$'
			+ r' $\vert$ Err orientation: %i' % len(err_orientations) + r'$^\circ$')



		plt.show()
		plt.clf()
		plt.close()

	return i, max_angle, len(err_orientations), len(indx_extrema[0]) , classification, lobe_ratio, position_angle, error



''' OLD PLOT FUNCTION WITHOUT WORLD COORDINATES (for MG):

	if plot:
		# the plot of the rotated source and the flux along the line
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),max_angle)
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),max_angle)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		zi = map_coordinates(data_array, np.vstack((y_rot,x_rot)),prefilter=True)
		fig, axes = plt.subplots(nrows=2,figsize=(12,12))
		axes[0].imshow(data_array,origin='lower')
		axes[0].plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'r-',alpha=0.3)
		axes[1].plot(zi)

		# rotate line on the left side of the center with min angle
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),min_orientation)
		# rotate line on the right side of the center with min angle
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),min_orientation)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		axes[0].plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'b-',alpha=0.3)

		# rotate line on the left side of the center with max angle
		x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),max_orientation)
		# rotate line on the right side of the center with max angle
		x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),max_orientation)
		x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 
		axes[0].plot([x0_rot, x1_rot], [y0_rot, y1_rot], 'b-',alpha=0.3)

		axes[0].set_xlabel('pixels')
		axes[0].set_ylabel('pixels')
		axes[1].set_ylabel('Flux (arbitrary units)')
		axes[1].set_xlabel('points')
		plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i)
			+ '\n extrema: ' +str(indx_extrema)
			+ '\n lobe ratio: %.3f' % lobe_ratio + ' | Position Angle: %.3f'  % position_angle
			+ ' Err orientation: %.1f' % len(err_orientations))

		plt.show()
		plt.clf()
		plt.close()

	return i, max_angle, len(err_orientations), len(indx_extrema[0]) , classification, lobe_ratio, position_angle, error


'''