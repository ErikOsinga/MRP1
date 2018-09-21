import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import constants as S
import pyregion
from astropy.table import Table, join
from scipy.stats import norm
import aplpy
import pylab as pl

import pyfits as pf
import pywcs as pw

'''
First script that is used for filtering the sources after PyBDSF

'''
file = 'P173+55'


prefix = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'
filename1 = prefix + file + '/mosaic.fits'
hdu_list = fits.open(filename1)
image_data = hdu_list[0].data 
header = hdu_list[0].header
hdu_list.close()

FieldNames = ['P11Hetdex12', 'P173+55', 'P21', 'P8Hetdex', 'P30Hetdex06', 'P178+55', 
'P10Hetdex', 'P218+55', 'P34Hetdex06', 'P7Hetdex11', 'P12Hetdex11', 'P16Hetdex13', 
'P25Hetdex09', 'P6', 'P169+55', 'P187+55', 'P164+55', 'P4Hetdex16', 'P29Hetdex19', 'P35Hetdex10', 
'P3Hetdex16', 'P41Hetdex', 'P191+55', 'P26Hetdex03', 'P27Hetdex09', 'P14Hetdex04', 'P38Hetdex07', 
'P182+55', 'P33Hetdex08', 'P196+55', 'P37Hetdex15', 'P223+55', 'P200+55', 'P206+50', 'P210+47', 
'P205+55', 'P209+55', 'P42Hetdex07', 'P214+55', 'P211+50', 'P1Hetdex15', 'P206+52', 
'P15Hetdex13', 'P22Hetdex04', 'P19Hetdex17', 'P23Hetdex20', 'P18Hetdex03', 'P39Hetdex19', 'P223+52',
 'P221+47', 'P223+50', 'P219+52', 'P213+47', 'P225+47', 'P217+47', 'P227+50', 'P227+53', 'P219+50',
 ]

def convert_hh_mm_ss(hh,mm,ss):
	'''
	converts hh_mm_ss to degrees
	'''
	return ( (360./24.)* (hh + (mm/60.) + ss/3600.) ) 

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

def positionAngle(ra1,dec1,ra2,dec2):
	'''
	Given the ra1, dec1, ra2 and dec2 in degrees
	It returns the position angle in degrees
	'''

	#convert to radians
	ra1 = ra1 * np.pi / 180. 
	dec1 = dec1 * np.pi / 180.
	ra2 = ra2 * np.pi / 180.
	dec2 = dec2 * np.pi / 180.
	
	result = np.arctan(
				( np.sin(ra1-ra2) )/
	( np.cos(dec1)*np.tan(dec2)-np.sin(dec1)*np.cos(ra1-ra2) )
			 )
	return result * 180 / np.pi

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

def plot_hist(normed=False,fit=False):
	'''
	Plots a histogram of the NN distance for every source
	Prints the mean and std.
	if normed=True, produces a normed histogram
	if fit=True, produces the best fit as well.
	'''
	filename2 = '/data1/osinga/data/'+file+'NearestNeighbours_efficient_spherical2.fits'
	# result_RA, result_DEC, result_distance = load_in(filename2,'NN_RA','NN_DEC','NN_distance(arcmin)')
	result_distance = load_in(filename2,'new_NN_distance(arcmin)')[0]
	fig, ax = plt.subplots()
    # histogram of the data
	n, bins, patches = ax.hist(result_distance,bins=100,normed=normed)

    # add a 'best fit' line by computing mean and stdev
	(mu,sigma) = norm.fit(result_distance)
	print mu,sigma
	if fit==True:
		y = np.max(n)*mlab.normpdf(bins, mu, sigma)
		ax.plot(bins, y, '--', label='Best fit')
	
	plt.title('Histogram of the angular distance between a source and its closest neighbour\n Number of sources: ' + str(len(result_distance)) + ' | field: ' + file)
	plt.xlabel('Angular distance (arcmin)')
	plt.ylabel('Number')
	plt.legend()
	plt.show()	

def select_criteria(number):
	'''
	simple function to check how many sources have err_orientation < number
	'''
	# old data path
	path = '/data1/osinga/figures/cutouts/P173+55/zzz_other_tries/SN10_Fratio_10/multiple_gaussians/elongated/results_.fits'
	new_Index,orientation,err_orientation,long_or_round = load_in(path,'new_Index','orientation (deg)','err_orientation','long_or_round')
	print (new_Index[np.where(err_orientation < number)])
	print len(new_Index[np.where(err_orientation < number)])

def nearest_neighbour_distance_efficient(RAs,DECs,iteration,source_names=False,write=False,p=3):
	'''
	This function is faster than the other NN-function with the same functionality. 
	
	Input: RAs and DECs of the sources as a numpy array, the iteration and, if iteration = 2
	also a list of the source_names of the sources corresponding to RA and DEC
	Iteration is 1 for the first NN search and 2 for the NN search after the cutoff
	distance

	It computes the nearest neighbour using spherical distance if p=3
	and the Euclidian distance if p=2. Writes to a file if write=True

	Output: three arrays, one with the distance, one with the RA and one with the DEC of the NN
	The distance is given in arcminutes
	'''
	
	#convert RAs and DECs to an array that has following layout: [[ra1,dec1],[ra2,dec2],etc]
	coordinates = np.vstack((RAs,DECs)).T

	#make a KDTree for quick NN searching	
	import ScipySpatialckdTree
	coordinates_tree = ScipySpatialckdTree.KDTree_plus(coordinates,leafsize=16)
	TheResult_distance = []
	TheResult_RA = []
	TheResult_DEC = []
	TheResult_index = []
	for item in coordinates:
		'''
		Find 2nd closest neighbours, since the 1st is the point itself.

		coordinates_tree.query(item,k=2)[1][1] is the index of this second closest 
		neighbour.

		We then compute the spherical distance between the item and the 
		closest neighbour.
		'''
		# print coordinates_tree.query(item,k=2,p=3)
		index=coordinates_tree.query(item,k=2,p=p)[1][1]
		nearestN = coordinates[index]
		distance = distanceOnSphere(nearestN[0],nearestN[1],#coordinates of the nearest
								item[0],item[1])*60 #coordinates of the current item
		# print distance/60
		TheResult_distance.append(distance)	
		TheResult_RA.append(nearestN[0])
		TheResult_DEC.append(nearestN[1])
		TheResult_index.append(index)
	
	if write==True:
		if type(source_names) == bool:
			#the first time we consider all sources so we do not need
			#the source names seperately
			if iteration == 1:
				t2 = fits.open(prefix+file+'cat.srl.fits')#open the fits file to get the source names, RA, DEC etc
				t2 = Table(t2[1].data)
				t2['NN_distance(arcmin)'] = TheResult_distance
				t2['NN_RA'] = TheResult_RA
				t2['NN_DEC'] = TheResult_DEC	
				t2['Index'] = np.arange(len(TheResult_distance))
				t2['NN_index'] = TheResult_index
			else:
				raise ValueError ("source_names not found with iteration %d" % iteration)
		elif iteration == 2:
			#the second iteration we need to keep in mind which sources we use
			#so we need the source names seperately
			t2 = fits.open('/data1/osinga/data/'+file+'NearestNeighbours_efficient_spherical1.fits')
			t2 = Table(t2[1].data)
			results = Table([source_names,TheResult_distance,TheResult_RA,TheResult_DEC,TheResult_index], names=('Source_Name','new_NN_distance(arcmin)','new_NN_RA','new_NN_DEC','new_NN_index'))
			results['new_Index'] = np.arange(len(TheResult_distance)) # we need a new index since we throwing away some of the sources
			t2 = join(results,t2,keys='Source_Name')
			t2.sort('new_Index')
		else:
			raise ValueError ("iteration %d is not implemented yet" % iteration)

		'''
		So the layout will now be:
		NN_dist RA DEC Index etc, which indicates the first iteration
		new_NN_dist new_NN_RA etc which indicates the second iteration
		So you can compare the NN_index with Index and new_NN_index with new_Index
		'''
		t2.write('/data1/osinga/data/'+file+'NearestNeighbours_efficient_spherical'+str(iteration)+'.fits',overwrite=True)
	return TheResult_distance,TheResult_RA, TheResult_DEC

def filter_NN(cutoff=1):
	'''
	Filters the first result of the nearest neighbour algorithm
	given the cutoff value in arcmin (default = 1 arcmin)
	So it finds all sources with a NN within 1 arcmin and computes the nearest neighbour again for these sources.

	Produces a file with singles and with doubles (determined by cutoff value)
	'''

	# read in the data from the first iteration
	filename2 = '/data1/osinga/data/'+file+'NearestNeighbours_efficient_spherical1.fits'
	RAs, DECs, source_names, result_distance = load_in(filename2,'RA','DEC', 'Source_Name','NN_distance(arcmin)')
	# get the RAs and the DEcs of the double-sources
	doublesRA =  RAs[result_distance < cutoff]
	doublesDEC = DECs[result_distance < cutoff]
	doubles_source_names =  source_names[result_distance < cutoff]
	# again calculate nearest neighbour for the doubles
	nearest_neighbour_distance_efficient(doublesRA,doublesDEC,iteration=2,source_names=doubles_source_names,write=True)

	# write the singles to a file
	singles_source_names =  source_names[result_distance > cutoff]
	t2 = fits.open('/data1/osinga/data/'+file+'NearestNeighbours_efficient_spherical1.fits')
	t2 = Table(t2[1].data)
	singles_source_names = Table([singles_source_names], names=('Source_Name',))
	# print singles_source_names
	t2 = join(singles_source_names,t2)
	t2.sort('Index')
	t2.write('/data1/osinga/data/'+file+'NearestNeighbours_efficient_spherical2_singles.fits',overwrite=True)

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

	Beware: this only finds the multiple_gaussians that have a NN below the cutoff. So this is not good.

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
					postage(filename1,'/data1/osinga/figures/cutouts/'+file+'/multiple_gaussians/'+file+'source_'+str(i),RAs[i],DECs[i],(4/60.),NN_RAs[i],NN_DECs[i])
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

def find_orientation(i,fitsim,ra,dec, Maj, s = (3/60.),plot=False,head=None,hdulist=None):
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

	# use a radius for the line that is the major axis, 
	# but with 2 pixels more added to the radius
	# to make sure we do capture the whole source
	radius = Maj / 60  * 40 #arcsec -- > arcmin --> image units
	radius = int(radius) + 2 # is chosen arbitrarily

	# in the P173+55 1 arcmin = 40 image units, should check if this is true everywhere
	if True in (np.isnan(data_array)):
		if s < (2/60.):
			print "No hope left for this source "
			return i, 0.0, 100.5

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			try: 
				return find_orientation(i,fitsim,ra,dec,Maj,s=(Maj*2/60/60),plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0 ,100.5

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			try:
				return find_orientation(i,fitsim,ra,dec, Maj, s = (2/60.),plot=plot,head=head,hdulist=hdulist)
			except RuntimeError: 
				print "No hope left for this source, "
				return i, 0.0 ,100.5

	from scipy.ndimage import interpolation
	from scipy.ndimage import map_coordinates
	#the center of the image is at the halfway point -1 for using array-index
	xcenter = np.shape(data_array)[0]/2-1 # pixel coordinates
	ycenter = np.shape(data_array)[0]/2-1# pixel coordinates
	
	# make a line with 'num' points and radius = radius
	x0, y0 = xcenter-radius,ycenter 
	x1, y1 = xcenter+radius,ycenter
	num = 1000
	x, y = np.linspace(x0,x1,num), np.linspace(y0,y1,num)

	# make a second and third line to have a linewidth of 3
	# x0_2, y0_2 = xcenter-radius,ycenter-1
	# x1_2, y1_2 = xcenter+radius,ycenter-1
	# x0_3, y0_3 = xcenter-radius,ycenter+1
	# x1_3, y1_3 = xcenter+radius,ycenter+1
	# x2, y2 = np.linspace(x0_2,x1_2,num), np.linspace(y0_2,y1_2,num)
	# x3, y3 = np.linspace(x0_3,x1_3,num), np.linspace(y0_3,y1_3,num)

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
		# zi2 = map_coordinates(data_array2, np.vstack((y2,x2)),prefilter=False)
		# zi3 = map_coordinates(data_array2, np.vstack((y3,x3)),prefilter=False)
		# calc the mean flux
		# zi = zi+zi2+zi3
		meanie = np.sum(zi)
		if meanie > max_value:
			max_value = meanie
			max_angle = angle
		all_values.append(meanie)
			
	# calculate all orientiations for which the average flux lies within
	# 80 per cent of the peak average flux
	err_orientations = np.where(all_values > (0.8 * max_value))[0]
	
	# print 'winner at : '
	# print max_angle, ' degrees,  average flux (random units): ' , max_value
	# print 'amount of orientations within 80 per cent : ', len(err_orientations) 

	if len(err_orientations) > 15:
		classification = 'Large err'
	else:
		classification = 'Small err'


	if plot:
		data_array2 = interpolation.rotate(data_array,max_angle,reshape=False)
		plt.imshow(data_array2,origin='lower')
		plt.plot([x0, x1], [y0, y1], 'r-',alpha=0.3)
		plt.plot([x0_2, x1_2], [y0_2, y1_2], 'r-',alpha=0.3)
		plt.plot([x0_3, x1_3], [y0_3, y1_3], 'r-',alpha=0.3)
		plt.title('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + '\n Best orientation = ' + str(max_angle) + ' degrees | classification: '+ classification)
		# plt.savefig('/data1/osinga/figures/test2_src'+str(i)+'.png')
		plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians2/elongated/'+head['OBJECT']+'src_'+str(i)+'.png') # // EDITED TO INCLUDE '2'
		plt.clf()
		plt.close()

		plt.plot(all_values, label='all orientations')
		plt.scatter(err_orientations, np.array(all_values)[err_orientations], color= 'y',label='0.8 fraction')
		plt.axvline(x=max_angle,ymin=0,ymax=1, color = 'r', label='best orientation')
		plt.title('Best orientation for Source ' + str(i) + '\nClassification: '+classification + ' | Error: ' + str(len(err_orientations)))
		plt.ylabel('Average flux (arbitrary units)')
		plt.xlabel('orientation (degrees)')
		plt.legend()
		plt.xlim(0,180)
		# plt.savefig('/data1/osinga/figures/test2_src'+str(i)+'_orientation.png')
		plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians2/elongated/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') # // EDITED TO INCLUDE '2'
		plt.clf()
		plt.close()
	return i, max_angle, len(err_orientations)

def setup_find_orientation_multiple_gaussians(file,SN=10):
	'''
	Find all M gaussian sources with err_orientation < 15

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
	head = pf.getheader(source)
	hdulist = pf.open(source)
	for i in multiple_gaussian_indices:
		if ( (Total_flux[i] / E_Total_flux[i]) > SN):
			if ( (Maj[i] - 1) < Min[i]):
				print 'Field Name: ' + file + ' has a round source: ' + str(i)
			else:
				# important in this function to give the head and hdulist since we do not have little cutouts so this is faster
				r_index, r_orientation, r_err_orientation = find_orientation(i,source+str(i),RA[i],DEC[i],Maj[i],(3/60.),plot=False,head=head,hdulist=hdulist)
				if r_err_orientation < 15:
					source_names.append(Source_Name[i])
					orientation.append(r_orientation)
					err_orientation.append(r_err_orientation)
	
	Mosaic_ID = Mosaic_ID[:len(source_names)]
	if ( len(source_names) == 0 ):
		print 'Field Name ' + file + ' has no sources found'
		F = open('/data1/osinga/data/'+file+'_parameters2.txt','w') # // EDITED TO INCLUDE '2'
		F.write('SN is ' +str(SN))
		F.write('\namount of multiple gaussian sources ' + str(len(source_names)))
		F.close()

	else:
		results = Table([source_names,orientation,err_orientation,Mosaic_ID],names=('Source_Name','orientation (deg)','err_orientation','Mosaic_ID'))
		results.write('/data1/osinga/data/'+file+'_multiple_gaussians2.fits',overwrite=True) # // EDITED TO INCLUDE '2'
		F = open('/data1/osinga/data/'+file+'_parameters2.txt','w') # // EDITED TO INCLUDE '2'
		F.write('SN is ' +str(SN))
		F.write('\namount of multiple gaussian sources ' + str(len(source_names)))
		F.close()

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

def all_multiple_gaussian_filtering():
	# extract all gaussians with a 'good'
	from astropy.io import ascii
	data = ascii.read('/data1/osinga/data/all_multiple_gaussians_filtering1.csv',format='csv')
	bad_column = np.asarray(data['Bad'])
	good_indx = np.where(bad_column == 0)
	good_data = data[good_indx]
	good_data.write('/data1/osinga/data/all_multiple_gaussians_filtering2.csv')

def all_multiple_gaussians_error1():
	'''
	For rerunning the find_orientation on those sources that have error1
	'''
	orientation, err_orientation, RA, DEC, Maj = load_in('/data1/osinga/data/all_multiple_gaussians1.fits','orientation (deg)','err_orientation','RA','DEC','Maj')
	idx = (np.where(err_orientation == 1))[0]
	print idx
	# for i in idx:
	if True:
		i = 25
		name = '/data1/osinga/figures/cutouts/all_multiple_gaussians/source'+str(i)+'.fits'
		find_orientation(i,name,RA[i],DEC[i],Maj[i],plot=True)

def stats():
	from astropy.io import ascii
	data = ascii.read('/data1/osinga/data/all_multiple_gaussians1_filtering_confidence.csv',format='csv')
	bad_column = np.asarray(data['error_1'])
	good_indx = np.where(bad_column == 1)
	new_data = data[good_indx]
	return data['Source'][good_indx]
	return new_data
	return new_data[np.where(new_data['confidence']==3)]






# ------ nearest neighbour stuff ----------- #

# the first NN searching
# RAs, DECs = load_in(prefix+file+'cat.srl.fits','RA','DEC')
# nearest_neighbour_distance_efficient(RAs,DECs,iteration=1,write=True)

# the filtering with NN_distance < 1 arcmin
# filter_NN(cutoff=1)

# for making all cutouts
# make_all_cutouts(SN=10.0,Fratio=10.0)

# finding the orientation of the sources
# setup_find_orientation()

# ------------------------------------------- #

# ------ multiple gaussians stuff ----------- #

# for finding all the multiple_gaussians < 15 degrees error
# for name in FieldNames:
# 	print name
# 	setup_find_orientation_multiple_gaussians(name,SN=10)
 
# then cross-matching these catalogs to one single file
# sys.path.insert(0, '/data1/osinga/scripts')
# import cross_catalogs

# and then making all postages of these multiple gaussians
# make_all_postages('all_multiple_gaussians')

# ------------------------------------------- #



# ----- other ------------------------------- #

# for plotting a histogram of NN distance
# plot_hist()

# show_aplpy(src2_RA,src2_DEC,src2_NN_RA,src2_NN_DEC)

# showing the sources
# show_sources()

# the old nearest_neighbour distance code that uses Euclidian + Spherical
# nearest_neighbour_distance_efficient_old(write=True)

# for calculating the position angle using RA and DEC of the 2 lobes
# phi =  positionAngle(src5_RA,src5_DEC,src5_RA2,src5_DEC2)
# if phi < 0:
# 	print phi+180 
# else:
# 	print phi

# ------------------------------------------- #
