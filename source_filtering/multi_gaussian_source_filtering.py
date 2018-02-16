import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
from scipy import constants as S
from astropy.table import Table, join
from scipy.stats import norm
import aplpy
import pylab as pl

import pyfits as pf
import pywcs as pw
import math
'''
First script that is used for filtering the multiple gaussian sources after PyBDSF

'''
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


def convert_hh_mm_ss(hh,mm,ss):
	'''
	converts hh_mm_ss to degrees
	'''
	return ( (360./24.)* (hh + (mm/60.) + ss/3600.) ) 

def PositionAngle(ra1,dec1,ra2,dec2):
	'''
	Given the positions (ra,dec) in degrees, returns the position angle of the 2nd source
	wrt to the first source in degrees. The position angle is measured North through East
	'''

	#convert degrees to radians
	ra1,dec1,ra2,dec2 = ra1 * math.pi / 180. , dec1 * math.pi / 180. , ra2 * math.pi / 180. , dec2 * math.pi / 180. 
	return (math.atan( (math.sin(ra2-ra1))/(
			math.cos(dec1)*math.tan(dec2)-math.sin(dec1)*math.cos(ra2-ra1))
				)* 180. / math.pi )# convert radians to degrees

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

def rotate_point(origin,point,angle):
	'''
	Rotate a point counterclockwise by a given angle around a given origin

	Angle should be given in degrees
	
	'''

	angle = angle * np.pi / 180. # convert degrees to radians

	ox,oy = origin
	px,py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py-oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py-oy)
	return qx,qy

def postage(fitsim,postfits,ra,dec, s,NN_RA = None,NN_DEC = None):
	'''
	Makes a postage 'postfits' from the entire fitsim at RA=ra and DEC=dec with radius s (degrees)
	Creates both a 'postfits.fits' cutout as well as a 'postfits.png' image for quick viewing

	If NN_RA and NN_DEC are provided it shows an arrow from the source to the nearest neighbour.
	'''
	import os 

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
	os.system( 'fitscopy %s %s' %(inps,postfits) )
	# print  'fitscopy %s %s' %(inps,postfits) 

	# make a png cutout from the fits cutout with an arrow pointing to the NN
	if NN_RA:
		gc = aplpy.FITSFigure(postfits)
		gc.show_grayscale()
		gc.add_grid()
		gc.show_arrows(ra,dec,NN_RA-ra,NN_DEC-dec)
		# gc.show_regions(prefix+file+'cat.srl.reg') #very slow
		# pl.show()
		gc.save(postfits+'.png')
		gc.close()




	return postfits

def make_all_postages(catalog_name):
	'''
	Makes all cutout images from a catalog, for example 
	make_all_postages('all_multiple_gaussians')

	Outputs these in /data1/osinga/figures/cutouts/catalog_name/
	Both as a .png and as a .fits file
	'''

	import difflib
	filename2 = '/data1/osinga/data/'+catalog_name+'2.fits' # // EDITED TO INCLUDE '2'
	Source_Name, RAs, DECs, orientation, err_orientation, Mosaic_ID,  = load_in(filename2,'Source_Name','RA','DEC','orientation (deg)','err_orientation','Mosaic_ID')
	no_sources = len(RAs)

	# initialise a list that will contain sources that are pointing to eachother (mutual NNs) so they dont have to be done twice
	for i in range (0,no_sources):
		# the mosaic that the source is in
		# have to use get_close_matches because strings > 8 chars are cutoff
		MosaicID = difflib.get_close_matches(Mosaic_ID[i],FieldNames,n=1)[0]
		# a check to see if difflib got the right string		
		trying = 1
		while MosaicID[:8] != Mosaic_ID[i]:
			trying +=1
			print Mosaic_ID[i]
			print 'best match: ', MosaicID
			print 'trying to find new match'
			MosaicID = difflib.get_close_matches(Mosaic_ID[i],FieldNames,n=trying)[trying-1]

		filename2 = prefix + MosaicID + '/mosaic.fits'
		postfits = '/data1/osinga/figures/cutouts/'+catalog_name+'2/source'+str(i)+'.fits' # // EDITED TO INCLUDE '2'
		postage(filename2,postfits,RAs[i],DECs[i],(4/60.))
		plt.imshow(fits.open(postfits)[0].data,origin='lower')
		plt.title('Field: '+MosaicID+' | Source Name: '+Source_Name[i]+'\n Best orientation = ' + str(orientation[i]) + ' degrees | error: '+ str(err_orientation[i]))
		plt.savefig('/data1/osinga/figures/cutouts/'+catalog_name+'2/source'+str(i)+'.png') # // EDITED TO INCLUDE '2'
		plt.clf()
		plt.close()

def find_orientation(i,fitsim,ra,dec, Maj, Min, s = (3/60.),plot=False,head=None,hdulist=None):
	'''	
	Finds the orientation of multiple gaussian single sources 

	To run this function for all sources, use setup_find_orientation_multiple_gaussians() instead.

	A big part of this function is a re-run of the postage function,
	since we need to make a new postage basically, but this time as an array.

	Arguments: 
	i : the new_Index of the source

	fitsim: the postage created earlier 

	ra, dec: the ra and dec of the source

	Maj: the major axis of the source

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

	# use a radius for the line that is the semimajor axis, 
	# but with 2 pixels more added to the radius
	# to make sure we do capture the whole source
	radius = Maj / 60.  * 40. #arcsec -- > arcmin --> image units
	radius = int(radius) + 2 # is chosen arbitrarily

	# in the P173+55 1 arcmin = 40 image units, should check if this is true everywhere, for FieldNames it is.

	# check if there are no NaNs in the cutout image, if there are then make smaller cutout
	if True in (np.isnan(data_array)):
		if s < (2/60.):
			print "No hope left for this source "
			return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			try: 
				return find_orientation(i,fitsim,ra,dec,Maj,Min,s=(Maj*2./60./60.),plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			try:
				return find_orientation(i,fitsim,ra,dec,Maj,Min,s = (2/60.),plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5


	from scipy.ndimage import map_coordinates
	#the center of the image is at the halfway point -1 for using array-index
	xcenter = np.shape(data_array)[0]/2-1 # pixel coordinates
	ycenter = np.shape(data_array)[1]/2-1# pixel coordinates
	
	# make a line with 'num' points and radius = radius
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
	
	# find the cutoff value, dependent on the distance 
	cutoff = 2*np.arctan(Min/Maj) * 180 / np.pi # convert rad to deg
	if len(err_orientations) > cutoff:
		classification = 'Large err'
	else:
		classification = 'Small err'

	# then find the amount of maxima, the lobe_ratio, and the Position Angle
	from scipy.signal import argrelextrema
	# rotate line on the left side of the center
	x0_rot, y0_rot = rotate_point((xcenter,ycenter),(x0,y0),max_angle)
	# rotate line on the right side of the center
	x1_rot,y1_rot = rotate_point((xcenter,ycenter),(x1,y1),max_angle)
	x_rot, y_rot = np.linspace(x0_rot,x1_rot,num), np.linspace(y0_rot,y1_rot,num) 

	zi = map_coordinates(data_array, np.vstack((y_rot,x_rot)),prefilter=True)
	# find the local maxima in the flux along the line
	indx_extrema = argrelextrema(zi, np.greater)

	if zi[0] > zi[1] :
		# then there is a maximum (outside of the line range) , namely the first point
		new_indx_extrema = (np.append(indx_extrema,0),)
		del indx_extrema #tuples are immutable
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
	position_angle = 0.0 # in case there is only 1 maximum
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
			
		plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + '\n Best orientation = ' + str(max_angle) + ' degrees | classification: '+ classification +
			'\nlobe ratio ' + str(lobe_ratio_bool) + ' | extrema: ' +str(indx_extrema) 
			+ '\n lobe ratio: ' + str(lobe_ratio) + ' | Position Angle: '  + str(position_angle))

		# saving the figures to seperate directories
		if amount_of_maxima == 2:
				plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'.png') # // EDITED TO P173+55
		elif amount_of_maxima > 2:
				plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'.png') # // EDITED TO P173+55
		else:
			plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'.png') # // EDITED TO P173+55			
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
		# saving the figures to seperate directories
		if amount_of_maxima == 2:
				plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') # // EDITED TO P173+55
		elif amount_of_maxima > 2:
				plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') # // EDITED TO P173+55
		else:
			plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') # // EDITED TO P173+55

		plt.clf()
		plt.close()
	return i, max_angle, len(err_orientations), amount_of_maxima , classification, lobe_ratio, position_angle #, err_position_angle

def setup_find_orientation_multiple_gaussians(file,SN=10):
	'''
	Find all M gaussian sources and their orientation etc, see find_orientation

	Input: 

	file = fieldname 
	SN =  Signal to noise ratio of the source

	Output:

	A .fits table file with the Source_Name, orientation and err_orientation
	A parameters.txt file with the SN and the amount of multiple gaussian sources < 15 error

	'''
	Source_Data = prefix+file+'cat.srl.fits'
	Source_Name, S_Code, Mosaic_ID = load_in(Source_Data,'Source_Name', 'S_Code', 'Mosaic_ID')
	RA, DEC, Maj, Min, Total_flux , E_Total_flux = load_in(Source_Data,'RA','DEC', 'Maj', 'Min', 'Total_flux', 'E_Total_flux')
	source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'

	multiple_gaussian_indices = (np.where(S_Code == 'M')[0])
	source_names = []
	orientation = []
	err_orientation = []
	amount_of_maxima = []
	classification = []
	lobe_ratio = []
	position_angle = []
	head = pf.getheader(source)
	hdulist = pf.open(source)
	for i in multiple_gaussian_indices:
		if ( (Total_flux[i] / E_Total_flux[i]) > SN):
			if ( (Maj[i] - 1) < Min[i]):
				print 'Field Name: ' + file + ' has a round source: ' + str(i)
			else:
				# important in this function to give the head and hdulist since we do not have little cutouts so this is faster
				r_index, r_orientation, r_err_orientation, r_amount_of_maxima, r_classification, r_lobe_ratio, r_position_angle = find_orientation(i,'useless',RA[i],DEC[i],Maj[i],Min[i],(3/60.),plot=True,head=head,hdulist=hdulist)
				source_names.append(Source_Name[i])
				orientation.append(r_orientation)
				err_orientation.append(r_err_orientation)
				amount_of_maxima.append(r_amount_of_maxima)
				classification.append(r_classification)
				lobe_ratio.append(r_lobe_ratio)
				position_angle.append(r_position_angle)
				
	Mosaic_ID = Mosaic_ID[:len(source_names)]
	if ( len(source_names) == 0 ):
		print 'Field Name ' + file + ' has no sources found'
		F = open('/data1/osinga/data/'+file+'_parameters2.txt','w') # // EDITED TO P173+55
		F.write('SN is ' +str(SN))
		F.write('\namount of multiple gaussian sources ' + str(len(source_names)))
		F.close()

	else:
		results = Table([source_names,orientation,err_orientation,Mosaic_ID,amount_of_maxima,classification,lobe_ratio,position_angle],names=('Source_Name','orientation (deg)','err_orientation','Mosaic_ID','amount_of_maxima','classification','lobe_ratio','position_angle'))
		results.write('/data1/osinga/data/'+file+'_multiple_gaussians2.fits',overwrite=True) # // EDITED TO P173+55
		F = open('/data1/osinga/data/'+file+'_parameters2.txt','w') # // EDITED TO P173+55
		F.write('SN is ' +str(SN))
		F.write('\namount of multiple gaussian sources ' + str(len(source_names)))
		F.close()


# ------ multiple gaussians stuff ----------- #

# for finding all the multiple_gaussians 
for name in FieldNames:
	print name
	setup_find_orientation_multiple_gaussians(name,SN=10)
 
# then cross-matching these catalogs to one single file
sys.path.insert(0, '/data1/osinga/scripts')
import cross_catalogs

# and then making all postages of these multiple gaussians (not neccesary anymore)
# make_all_postages('all_multiple_gaussians')

# ------------------------------------------- #



# ----- other ------------------------------- #

# for calculating the position angle using RA and DEC of the 2 lobes
# phi =  positionAngle(src5_RA,src5_DEC,src5_RA2,src5_DEC2)
# if phi < 0:
# 	print phi+180 
# else:
# 	print phi

# ------------------------------------------- #
