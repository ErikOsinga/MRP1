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

def distanceOnSphere(RAs1, Decs1, RAs2, Decs2):
	"""
	Credits: Martijn Oei, uses great-circle distance

	Return the distances on the sphere from the set of points '(RAs1, Decs1)' to the
	set of points '(RAs2, Decs2)' using the spherical law of cosines.

	It assumes that all inputs are given in degrees, and gives the output in degrees, too.

	Using 'numpy.clip(..., -1, 1)' is necessary to counteract the effect of numerical errors, that can sometimes
	incorrectly cause '...' to be slightly larger than 1 or slightly smaller than -1. This leads to NaNs in the arccosine.
	"""

	return np.degrees(np.arccos(np.clip(
	np.sin(np.radians(Decs1)) * np.sin(np.radians(Decs2)) +
	np.cos(np.radians(Decs1)) * np.cos(np.radians(Decs2)) *
	np.cos(np.radians(RAs1 - RAs2)), -1, 1)))

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
			return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			try: 
				return find_orientation(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,maj,s=(2*radius+2)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			try:
				return find_orientation(i,fitsim,ra,dec,nn_ra,nn_dec,nn_dist,maj,s=(2*radius+4)/40./60.,plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0, 100.5,100.5,'no_hope',100.5,100.5


	from scipy.ndimage import map_coordinates
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
	
	# find the cutoff value, dependent on the distance, should think about whether i want to use Maj
	cutoff = 2*np.arctan((maj/60.)/nn_dist) * 180 / np.pi # convert rad to deg
	if len(err_orientations) > cutoff:
		classification = 'Large err'
	else:
		classification = 'Small err'


	# then find the amount of maxima and the lobe_ratio
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
			
		plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + '\n Best orientation = ' + str(max_angle) + ' degrees | err_orientation: '+ str(len(err_orientations)) +
			'\nlobe ratio ' + str(lobe_ratio_bool) + ' | extrema: ' +str(indx_extrema)  + ' | nn_dist: ' + str(nn_dist)
			+ '\n lobe ratio: ' + str(lobe_ratio) + ' | Position Angle: '  + str(position_angle))

		# saving the figures to seperate directories
		if amount_of_maxima == 2:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+head['OBJECT']+'src_'+str(i)+'.png')

		elif amount_of_maxima > 2:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+head['OBJECT']+'src_'+str(i)+'.png') 

		else:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_4/1_maximum/'+head['OBJECT']+'src_'+str(i)+'.png') 		
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
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		elif amount_of_maxima > 2:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		else:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_4/1_maximum/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 

		plt.clf()
		plt.close()
	return i, max_angle, len(err_orientations), amount_of_maxima , classification, lobe_ratio, position_angle #, err_position_angle

def setup_find_orientation_NN(file,SN=10):
	'''
	Find all mutual NN sources and their orientation etc, see find_orientation

	Input: 

	file = fieldname 
	SN =  Signal to noise ratio cutoff of the source

	Output:

	A .fits table file with the Source_Name, orientation and err_orientation
	A parameters.txt file with the SN and the amount of multiple gaussian sources < 15 error

	'''
	prefix = '/data1/osinga/data/NN/'
	Source_Data = prefix+file+'NearestNeighbours_efficient_spherical2.fits'
	Source_Name, Mosaic_ID = load_in(Source_Data,'Source_Name', 'Mosaic_ID')
	RA, DEC, NN_RA, NN_DEC, NN_dist, Total_flux, E_Total_flux, new_NN_index, Maj = load_in(Source_Data,'RA','DEC','new_NN_RA','new_NN_DEC','new_NN_distance(arcmin)','Total_flux', 'E_Total_flux','new_NN_index','Maj')
	source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+file+'/mosaic.fits'

	source_names = []
	orientation = []
	err_orientation = []
	amount_of_maxima = []
	classification = []
	lobe_ratio = []
	position_angle = []
	# err_position_angle = []
	head = pf.getheader(source)
	hdulist = pf.open(source)
	# only mutual NN are considered, so don't have to be done twice
	unnecessary_indices = []
	for i in range(0,len(RA)): # i is basically the new_Index
		if i not in unnecessary_indices:
			# check if both the source and the neighbour statisfy the SN condition. # should be true for all sources.
			if ( ((Total_flux[i] / E_Total_flux[i]) > SN) and ( (Total_flux[new_NN_index[i]] / E_Total_flux[new_NN_index[i]]) > SN)  ):
				# check if your nearest neighbour has you as nearest neighbour as well.(Mutuals)
				if new_NN_index[new_NN_index[i]] == i:
					unnecessary_indices.append(new_NN_index[i]) # don't have to do the neighbour
					# important in this function to give the head and hdulist since we do not have little cutouts so this is faster
					r_index, r_orientation, r_err_orientation, r_amount_of_maxima, r_classification, r_lobe_ratio, r_position_angle = find_orientation(i,'useless',RA[i],DEC[i],NN_RA[i],NN_DEC[i],NN_dist[i],Maj[i],(3/60.),plot=True,head=head,hdulist=hdulist)
					source_names.append(Source_Name[i])
					orientation.append(r_orientation)
					err_orientation.append(r_err_orientation)
					amount_of_maxima.append(r_amount_of_maxima)
					classification.append(r_classification)
					lobe_ratio.append(r_lobe_ratio)
					position_angle.append(r_position_angle)

				else:
					print 'Not a mutual: ', head['OBJECT'], i

			else:
				print 'Thats weird, we cut this before, src: ', i
			
	Mosaic_ID = Mosaic_ID[:len(source_names)]
	if ( len(source_names) == 0 ):
		print 'Field Name ' + file + ' has no sources found'
		F = open('/data1/osinga/data/NN/try_4/'+file+'_parameters.txt','w') 
		F.write('SN is ' +str(SN))
		F.write('\namount of sources ' + str(len(source_names)))
		F.close()

	else:
		results = Table([source_names,orientation,err_orientation,Mosaic_ID,amount_of_maxima,classification,lobe_ratio,position_angle],names=('Source_Name','orientation (deg)','err_orientation','Mosaic_ID','amount_of_maxima','classification','lobe_ratio','position_angle'))
		results.write('/data1/osinga/data/NN/try_4/'+file+'_NN.fits',overwrite=True)
		F = open('/data1/osinga/data/NN/try_4/'+file+'_parameters.txt','w') 
		F.write('SN is ' +str(SN))
		F.write('\namount of sources ' + str(len(source_names)))
		F.close()

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


	# from astropy.nddata.utils import extract_array

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

def make_all_cutouts(SN,Fratio):
	'''
	Makes all cutout images with a Signal to Noise higher than SN and a Flux ratio 
	between the lobes of 'mutuals' (mutual NN's) higher than Fratio

	Outputs these in different categories (directories):
	- multiple_gaussians
	- mutuals  
		- Fratio_good 
		- Fratio_bad
	- singles

	Beware: this only finds the multiple_gaussians that have a NN below the cutoff, which is in this case what we want.
	since the other interesting multiple_gaussians are found in multi_gaussian_source_filtering.py

	'''
	filename2 = '/data1/osinga/data/'+file+'NearestNeighbours_efficient_spherical2.fits'
	RAs, DECs, NN_RAs, NN_DECs, new_Indices, new_NN_indices = load_in(filename2,'RA','DEC','NN_RA','NN_DEC','new_Index','new_NN_index')
	Total_flux, E_Total_flux, S_Code  = load_in(filename2,'Total_flux','E_Total_flux', 'S_Code')
	no_sources = len(RAs)

	# initialise a list that will contain sources that are pointing to eachother (mutual NNs) so they dont have to be done twice
	unnecessary_indices = []
	mutual_indices = []
	multiple_gaussian_indices = []
	amount_of_multi_gaussians = 0
	for i in range (0,no_sources):
		# check every time if the source does not already have an image
		if i not in unnecessary_indices:
			# check the S/N ratio parameter
			if Total_flux[i]/E_Total_flux[i] > SN:
				# check if a source consists of multiple gaussians
				if S_Code[i] == 'M':
					postage(filename1,'/data1/osinga/figures/cutouts/NN/multiple_gaussians/'+file+'src_'+str(i),RAs[i],DECs[i],(4/60.),NN_RAs[i],NN_DECs[i])
					amount_of_multi_gaussians += 1
					multiple_gaussian_indices.append(new_Indices[i])
				# check if your nearest neighbour points to you as well (mutuals)
				elif new_NN_indices[new_NN_indices[i]] == new_Indices[i]:
					# apply a flux ratio of maximally Fratio between the lobes
					if ( (1./Fratio) < (Total_flux[i]/Total_flux[new_NN_indices[i]]) < Fratio):
						unnecessary_indices.append(new_NN_indices[i])
						mutual_indices.append(new_Indices[i])
						# put it in a separate folder
						postage(filename1,'/data1/osinga/figures/cutouts/'+file+'/mutuals/Fratio_good/'+file+'source_'+str(i),RAs[i],DECs[i],(4/60.),NN_RAs[i],NN_DECs[i])
					else:
						postage(filename1,'/data1/osinga/figures/cutouts/'+file+'/mutuals/Fratio_bad/'+file+'source_'+str(i),RAs[i],DECs[i],(4/60.),NN_RAs[i],NN_DECs[i])
				else:
					postage(filename1,'/data1/osinga/figures/cutouts/'+file+'/singles/'+file+'source_'+str(i),RAs[i],DECs[i],(4/60.),NN_RAs[i],NN_DECs[i])
	
	print 'amount of mutual nearest neighbours with SN > ' + str(SN)+' and Fratio > '+ str(Fratio)+' is:  ' + str(len(unnecessary_indices))
	print 'amount of multiple gaussian sources ' + str(amount_of_multi_gaussians)
	print 'amount of sources: ' + str(no_sources)
	mutuals = Table([mutual_indices], names=(['mutual_indices']))
	mutuals.write('/data1/osinga/figures/cutouts/indices_of_mutuals.fits')
	multiples = Table([multiple_gaussian_indices],names=(['multiple_gaussian_indices']))
	multiples.write('/data1/osinga/figures/cutouts/indices_of_multiple_gaussians.fits')
	F = open('/data1/osinga/figures/cutouts/parameters.txt','w')
	F.write('SN is ' +str(SN)+ '  Fratio is ' + str(Fratio))
	F.write('amount of mutual nearest neighbours with SN > ' + str(SN)+' and Fratio > '+ str(Fratio)+' is:  ' + str(len(unnecessary_indices)))
	F.write('amount of multiple gaussian sources ' + str(amount_of_multi_gaussians))
	F.write('amount of sources: ' + str(no_sources))
	F.close()

def filter_NN(cutoff=1):
	'''
	Filters the first result of the nearest neighbour algorithm
	given the cutoff value in arcmin (default = 1 arcmin)
	So it finds all sources with a NN within 1 arcmin and computes the nearest neighbour again for these sources.

	Produces a file with singles and with doubles (determined by cutoff value)
	'''

	# read in the data from the first iteration
	filename2 = '/data1/osinga/data/NN/'+file+'NearestNeighbours_efficient_spherical1.fits'
	RAs, DECs, source_names, result_distance = load_in(filename2,'RA','DEC', 'Source_Name','NN_distance(arcmin)')
	# get the RAs and the DEcs of the double-sources
	doublesRA =  RAs[result_distance < cutoff]
	doublesDEC = DECs[result_distance < cutoff]
	doubles_source_names =  source_names[result_distance < cutoff]
	# again calculate nearest neighbour for the doubles
	nearest_neighbour_distance_efficient(doublesRA,doublesDEC,iteration=2,source_names=doubles_source_names,write=True)

	# write the singles to a file
	singles_source_names =  source_names[result_distance > cutoff]
	t2 = fits.open('/data1/osinga/data/NN/'+file+'NearestNeighbours_efficient_spherical1.fits')
	t2 = Table(t2[1].data)
	singles_source_names = Table([singles_source_names], names=('Source_Name',))
	# print singles_source_names
	t2 = join(singles_source_names,t2)
	t2.sort('Index')
	t2.write('/data1/osinga/data/NN/'+file+'NearestNeighbours_efficient_spherical2_singles.fits',overwrite=True)

def nearest_neighbour_distance_efficient(RAs,DECs,iteration,source_names,write=False,p=2):
	'''
	This function is faster than the other NN-function with the same functionality. 
	
	Input: RAs and DECs of the sources as a numpy array, the iteration and, if iteration = 2
	also a list of the source_names of the sources corresponding to RA and DEC
	Iteration is 1 for the first NN search and 2 for the NN search after the cutoff
	distance

	It computes the nearest neighbour using the Euclidian distance between the points converted
	to a cartesian system. When the nearest neighbour is found, it computes spherical distance.

	Writes to a file if write=True

	Output: three arrays, one with the distance, one with the RA and one with the DEC of the NN
	The distance is given in arcminutes
	'''
	
	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
	y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
	z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T

	#make a KDTree for quick NN searching	
	from scipy.spatial import cKDTree
	coordinates_tree = cKDTree(coordinates,leafsize=16)
	TheResult_distance = []
	TheResult_RA = []
	TheResult_DEC = []
	TheResult_index = []
	for i,item in enumerate(coordinates):
		'''
		Find 2nd closest neighbours, since the 1st is the point itself.

		coordinates_tree.query(item,k=2)[1][1] is the index of this second closest 
		neighbour.

		We then compute the spherical distance between the item and the 
		closest neighbour.
		'''
		# print coordinates_tree.query(item,k=2,p=2)
		index=coordinates_tree.query(item,k=2,p=2,n_jobs=-1)[1][1]
		nearestN = [RAs[index],DECs[index]]
		source = [RAs[i],DECs[i]]
		distance = distanceOnSphere(nearestN[0],nearestN[1],#RA,DEC coordinates of the nearest
								source[0],source[1])*60 #RA,DEC coordinates of the current item
		# print distance/60
		TheResult_distance.append(distance)	
		TheResult_RA.append(nearestN[0])
		TheResult_DEC.append(nearestN[1])
		TheResult_index.append(index)
	
	if write==True:
		#we need the source names seperately, so we load in cross_catalog = PYBSDF catalog
		if iteration == 1:
			cross_catalog = fits.open(prefix+file+'cat.srl.fits')
			cross_catalog = Table(cross_catalog[1].data)
			results =  Table([source_names,TheResult_distance,TheResult_RA,TheResult_DEC,TheResult_index], names=('Source_Name','NN_distance(arcmin)','NN_RA','NN_DEC','NN_index'))
			# for sorting
			results['Index'] = np.arange(len(TheResult_distance))	
			results = join(results,cross_catalog,keys='Source_Name')
			results.sort('Index')
			results.write('/data1/osinga/data/NN/'+file+'NearestNeighbours_efficient_spherical'+str(iteration)+'.fits',overwrite=True)

		elif iteration == 2:
			#the second iteration we need to keep in mind which sources we use
			#so we need the source names seperately
			cross_catalog = fits.open('/data1/osinga/data/NN/'+file+'NearestNeighbours_efficient_spherical1.fits')
			cross_catalog = Table(cross_catalog[1].data)
			results = Table([source_names,TheResult_distance,TheResult_RA,TheResult_DEC,TheResult_index], names=('Source_Name','new_NN_distance(arcmin)','new_NN_RA','new_NN_DEC','new_NN_index'))
			results['new_Index'] = np.arange(len(TheResult_distance)) # we need a new index since we throwing away some of the sources
			results = join(results,cross_catalog,keys='Source_Name')
			results.sort('new_Index')
			results.remove_columns(['NN_distance(arcmin)', 'NN_RA', 'NN_DEC', 'Index', 'NN_index'])
			results.write('/data1/osinga/data/NN/'+file+'NearestNeighbours_efficient_spherical'+str(iteration)+'.fits',overwrite=True)
		else:
			raise ValueError ("iteration %d is not implemented yet" % iteration)

		'''
		So the layout will now be:
		NN_dist RA DEC Index etc, which indicates the first iteration
		new_NN_dist new_NN_RA etc which indicates the second iteration
		So you can compare the NN_index with Index and new_NN_index with new_Index
		'''
	return TheResult_distance,TheResult_RA, TheResult_DEC



# ------ nearest neighbour stuff ----------- #




# the first NN searching
for file in FieldNames:
	RAs, DECs, S_Code, Source_Name = load_in(prefix+file+'cat.srl.fits','RA','DEC','S_Code','Source_Name')
	Total_flux, E_Total_flux = load_in(prefix+file+'cat.srl.fits','Total_flux','E_Total_flux')
	# Will only find the nearest neighbour for SN > 10.
	initial_length = len(RAs)
	RAs = RAs[(Total_flux / E_Total_flux) > 10.]
	DECs = DECs[(Total_flux / E_Total_flux) > 10.]
	Source_Name = Source_Name[(Total_flux / E_Total_flux) > 10.]
	print file
	print 'Sources excluded due to Signal to Noise: ', initial_length-len(RAs)
	nearest_neighbour_distance_efficient(RAs,DECs,iteration=1,source_names=Source_Name,write=True)
	# the filtering with NN_distance < 1 arcmin and recalculating the NN again for the cutoff
	filter_NN(cutoff=1)
	# then finding the orientation.
	setup_find_orientation_NN(file)
	
# finally stacking the catalogs to one big catalog
sys.path.insert(0, '/data1/osinga/scripts')
import cross_catalogs_NN



# for making all cutouts (not neccesary)
# make_all_cutouts(SN=10.0,Fratio=10.0)


# ------------------------------------------- #