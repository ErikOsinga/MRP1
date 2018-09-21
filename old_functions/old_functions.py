old_functions.py


# was used to test coloring regions within a certain distance
def distance_test():
	'''
	changes the color, to red, of sources within a distance of 40 image units ~ 1 arcmin
	'''
	for i in range (0,len(r3)):
		for j in range (0,len(r3)):
			if i != j:
				delta_x = r3[i].coord_list[0] - r3[j].coord_list[0]
				delta_y = r3[i].coord_list[1] - r3[j].coord_list[1]
				distance = np.sqrt(delta_x**2 + delta_y**2)
				if distance < 39.6:
					# print distance
					#only change the color to red
					r3[i].attr = (['source'],
								 {'color': 'red',
								  'delete': '1 ',
								  'edit': '1 ',
								  'fixed': '0 ',
								  'font': '"helvetica 10 normal"',
								  'highlite': '1 ',
								  'include': '1 ',
								  'move': '1 ',
								  'select': '1 '})
# was used to show the regions around the sources with pyregion, very slow.
def show_sources():
	'''
	Shows the regions around all sources
	'''

	regions = pyregion.open(prefix+file+'cat.srl.reg')
	# convert to image coordinates so the regions actually show up in the plot
	regions2 = regions.as_imagecoord(header)

	#defines a list of matplotlib.patches.Patch and other kinds of artists (usually Text)
	patch_list, artist_list = regions2.get_mpl_patches_texts()

	fig, ax = plt.subplots()
	ax.imshow(image_data, cmap = plt.cm.gray, origin = 'lower', vmin = -0.000246005, vmax = 0.000715727)
	for p in patch_list:
		ax.add_patch(p) #add the first patch to the plot, then the second, etc
	#ax.add_artist(artist_list[0]) #similar for the first artist
	print 'number of initial sources: ' + str(len(regions))
	# print 'sources found: ' + str(len(r3))
	# print 'sources excluded: ' + str(len(r)-len(r3))
	plt.show()
# postage-like function except it uses AplPY and it is much slower
def show_aplpy(RA,DEC,NN_RA,NN_DEC,radius=2):
	'''
	Uses aplpy to show a cut-out around a given RA, DEC (degrees)
	with radius given in arcmin (2 arcmin as default)

	Is a pretty slow function, I recommend the function postage()
	'''
	radius = radius/60. #convert to degrees

	gc = aplpy.FITSFigure(filename1)
	gc.show_grayscale()
	# gc.show_colorscale(cmap='gist_heat')
	gc.recenter(RA,DEC,radius=radius)
	gc.add_grid()
	# gc.show_markers(RA,DEC,edgecolor='green',facecolor='none', marker='o', s=10,alpha=0.5)
	gc.show_arrows(RA,DEC,NN_RA-RA,NN_DEC-DEC)
	# gc.show_regions(prefix+file+'cat.srl.reg') #very slow
	pl.show()
	gc.close()
# used pyregions and double for-loop to calculate nearest neighbours
def nearest_neighbour_distance_pyregions(write=False):
	'''
	Returns the nearest neighbour for each source in arcminutes, if given the region list
	in degrees. Writes this to a file called <file>NearestNeighbours.fits

	This function is as slow as it gets as it requires r to be loaded in as a pyregions list.
	'''

	TheResult = [] 
	for i in range (0,len(r)):
		min_distance = 1e12
		for j in range(0,len(r)):
			if i != j:
				distance = distanceOnSphere(r[i].coord_list[0],r[i].coord_list[1],
					r[j].coord_list[0],r[j].coord_list[1])
				if distance < min_distance:#new nearest neighbour found
					min_distance = distance
		TheResult.append(min_distance)#after all the neighbours are checked, append the nearest

	TheResult=np.asarray(TheResult)*60 #converts to arcminutes

	if write==True:
		t2 = fits.open(prefix+file+'cat.srl.fits')#open the fits file to get the source names
		t2 = Table(t2[1].data)['Source_Name']
		t2 = Table([t2[:],TheResult], names = ('Source Name','NN_distance(arcmin)'))
		t2.write('/data1/osinga/data/'+file+'NearestNeighbours_pyregions.fits',overwrite=True)
	return TheResult
# used the fits table and a double for-loop to calculate nearest neighbours 
def nearest_neighbour_distance(write=False):
	'''

	Slow function to return the spherical nearest neighbours, but definitely foolproof.
	Can be used to verify faster functions

	Returns the nearest neighbour for each source in arcminutes, if given the region 
	list in degrees. Writes this to a file called <file>NearestNeighbours.fits

	This function is independent from pyregions, and uses the fits table
	instead, which opens faster.
	'''

	Source_Data = fits.open(prefix+file+'cat.srl.fits')
	Source_Data = Table(Source_Data[1].data)
	RAs = np.asarray(Source_Data['RA'])
	DECs = np.asarray(Source_Data['DEC'])
	TheResult_distance = []
	TheResult_index = [] 
	for i in range (0,len(RAs)):
		min_distance = 1e12
		for j in range(0,len(RAs)):
			if i != j:
				distance = distanceOnSphere(RAs[i],DECs[i],RAs[j],DECs[j])
				if distance < min_distance:#new nearest neighbour found
					min_distance = distance
					min_index = j
		TheResult_distance.append(min_distance)#after all the neighbours are checked, append the nearest
		TheResult_index.append(min_index)
	
	TheResult_distance=np.asarray(TheResult_distance)*60 #converts to arcminutes
	TheResult_index=np.asarray(TheResult_index)

	if write==True:
		t2 = fits.open(prefix+file+'cat.srl.fits')#open the fits file to get the source names
		t2 = Table(t2[1].data)['Source_Name']
		t2 = Table([t2[:],TheResult_distance,TheResult_index], names = ('Source Name','NN_distance(arcmin)','NN_index'))
		t2.write('/data1/osinga/data/'+file+'NearestNeighbours.fits',overwrite=True)
	return TheResult_distance,TheResult_index
# used the KDTree, but that only had Euclidian distances, so it went wrong about 1/8 of the time
def nearest_neighbour_distance_efficient_old(write=False):
    '''
    Still have to think about nearest neighbour with degrees.
    
    As it works now, it computes the nearest neighbour using !Euclidian! distance
    (So it does not take into account the spherical distance for this)

    But it does produce the spherical distance between the points
    after the nearest neighbour has been found.

    Returns two arrays, one with the distance and one with the index of the NN
    The distance is given in arcminutes

    It is very fast, but produces a different result from the other function. 
    '''

    from astropy.table import Table
    Source_Data = fits.open(prefix+file+'cat.srl.fits')
    Source_Data = Table(Source_Data[1].data)
    RAs = np.asarray(Source_Data['RA'])
    DECs = np.asarray(Source_Data['DEC'])
    
    #convert to an array that has following layout: [[ra1,dec1],[ra2,dec2],etc]
    coordinates = np.vstack((RAs,DECs)).T
    #make an cKDTree for quick NN searching
    import scipy.spatial
    coordinates_tree = scipy.spatial.cKDTree(coordinates,leafsize=16,balanced_tree=False)

    TheResult_distance = []
    TheResult_index = []
    for item in coordinates:
        '''
        Find 2nd closest neighbours, since the 1st is the point itself.

        coordinates_tree.query(item,k=2)[1][1] is the index of this second closest 
        neighbour.

        We then compute the spherical distance between the item and the 
        closest neighbour.
        '''

        index=coordinates_tree.query(item,k=2)[1][1]
        nearestN = coordinates[index]
        distance = distanceOnSphere(nearestN[0],nearestN[1],#coordinates of the nearest
                                item[0],item[1])*60 #coordinates of the current item
        TheResult_distance.append(distance) 
        TheResult_index.append(index)
    
    if write==True:
        t2 = fits.open(prefix+file+'cat.srl.fits')#open the fits file to get the source names
        t2 = Table(t2[1].data)['Source_Name']
        t2 = Table([t2[:],TheResult_distance,TheResult_index], names = ('Source Name','NN_distance(arcmin)','NN_index'))
        t2.write('/data1/osinga/data/'+file+'NearestNeighbours_efficient_old.fits',overwrite=True)
    return TheResult_distance,TheResult_index
# used to setup finding the orientation for all sources, but only worked for M_gaussians that have a NN < 1 arcmin

# the version that used purely distancetoSphere using my KDTreePlus module
# however, the cKDTree with appropriate conversion to cartisian coordinates is much better
def nearest_neighbour_distance_efficient(RAs,DECs,iteration,source_names,write=False,p=3):
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




def setup_find_orientation():
	'''
	This is a setup function for the find_orientation function
	It loads all the neccesary data, and produces a table file that
	contains the new_Index, the orientation and the number of
	orientations that have avg flux value > 80% of the best orientation avg flux value
	Also produces a column which the algorithm predicts elongated or round

	I forgot that it only loads in the Multiple Gaussians that have a NN < 1 arcmin..
	'''

	# load a list of all the indices of the multiple gaussians
	multiple_gaussian_indices = load_in('/data1/osinga/figures/cutouts/P173+55/indices_of_multiple_gaussians.fits','multiple_gaussian_indices')[0]
	# load a list of all the RA, DEC (degrees) and Major axis and Minor axis (in arcsec)
	RA, DEC, Maj, Min = load_in('/data1/osinga/data/P173+55NearestNeighbours_efficient_spherical2.fits','RA','DEC', 'Maj', 'Min')
	# simply the filename
	source = '/data1/osinga/figures/cutouts/P173+55/multiple_gaussians/P173+55source_'

	indices = []
	orientation = []
	err_orientation = []
	long_or_round = []
	for i in multiple_gaussian_indices:
		# check if the major axis lies within 1 arcsec of the minor axis
		if ( (Maj[i] - 1) < Min[i]):
			print 'round source: ' + str(i)
		else:
			print 'source number: ' +str(i)
			r_index, r_orientation, r_err_orientation = find_orientation(i,source+str(i),RA[i],DEC[i],Maj[i],(3/60.),plot=True)
			if r_err_orientation > 80:
				long_or_round.append('round')
			else:
				long_or_round.append('elongated')
			indices.append(r_index)
			orientation.append(r_orientation)
			err_orientation.append(r_err_orientation)

	
	results = Table([indices,orientation,err_orientation,long_or_round],names=('new_Index','orientation (deg)','err_orientation','long_or_round'))
	results.write('/data1/osinga/figures/cutouts/multiple_gaussians/elongated/results_'+file+'.fits',overwrite=True)

# not neccisarily an old function, but the find_orientation function before I tried it for a line of width 3 and before 19 oct
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
			return i, 0.0, 0.0

		elif s == (2/60.):
			print "Nan in the cutout image AGAIN ", head['OBJECT'], ' i = ' , i
			return find_orientation(i,fitsim,ra,dec,Maj,s=(Maj*2/60/60),plot=plot,head=head,hdulist=hdulist)

		else:
			print "NaN in the cutout image: ", head['OBJECT'], ' i = ' , i
			return find_orientation(i,fitsim,ra,dec, Maj, s = (2/60.),plot=plot,head=head,hdulist=hdulist)
		# raise ValueError("NaN in the cutout image")  
	
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
		plt.title('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + '\n Best orientation = ' + str(max_angle) + ' degrees | classification: '+ classification)
		plt.savefig('/data1/osinga/figures/test2_src'+str(i)+'.png')
		# plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians2/elongated/'+head['OBJECT']+'src_'+str(i)+'.png') # // EDITED TO INCLUDE '2'
		plt.show()
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
		plt.savefig('/data1/osinga/figures/test2_src'+str(i)+'_orientation.png')
		# plt.savefig('/data1/osinga/figures/cutouts/all_multiple_gaussians2/elongated/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') # // EDITED TO INCLUDE '2'
		plt.show()
		plt.clf()
		plt.close()
	return i, max_angle, len(err_orientations)


# This was used to make cutout images from the NN below the cutoff. but also to find multi_gaussians but it accidently had 
# the NN restriction on it, can still be used later for the NN cutoff cutouts though.
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


# the less efficient find_orientation function, it rotates the image instead of the line.
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
			
		plt.suptitle('Field: ' + head['OBJECT'] + ' | Source ' + str(i) + '\n Best orientation = ' + str(max_angle) + ' degrees | classification: '+ classification +
			'\nlobe ratio ' + str(lobe_ratio_bool) + ' | extrema: ' +str(indx_extrema)  + ' | nn_dist: ' + str(nn_dist)
			+ '\n lobe ratio: ' + str(lobe_ratio))

		# saving the figures to seperate directories
		if amount_of_maxima == 2:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'.png')

		elif amount_of_maxima > 2:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'.png') 

		else:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'.png') 		
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
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/2_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		elif amount_of_maxima > 2:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/more_maxima/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 
		else:
			plt.savefig('/data1/osinga/figures/cutouts/NN/try_3/1_maximum/'+head['OBJECT']+'src_'+str(i)+'_orientation.png') 

		plt.clf()
		plt.close()
	return i, max_angle, len(err_orientations), amount_of_maxima , classification, lobe_ratio


# Was the quickest way to calculate the dispersion for all n. Except that it used way too 
# much memory since we were doing it for all angles, instead of the Max_di formula
def angular_dispersion_vectorized_noob(tdata,n):
	'''
	Calculates and returns the Sn statistic for tdata
	Vectorized over n, starting at 200 down to 0 (included).
	i.e. calculate the Sn for every n from 0 to 81

	# n = number of sources closest to source i
	# N = number of sources
	
	Returns Sn, a (1xn) matrix containing S_1 to S_200

	'''

	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['PA'])
	angles = 180 # maximize dispersion for an angle between 0 and 180

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
	y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
	z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T
	
	#make a KDTree for quick NN searching	
	coordinates_tree = cKDTree(coordinates,leafsize=16)

	# for every source: find n closest neighbours, calculate max dispersion
	all_vectorized = [] # array of shape (N,180,n) which contains all angles used for all sources
	position_angles_array = np.zeros((N,n)) # array of shape (N,n) that contains position angles
	thetas = np.array(range(0,angles)).reshape(angles,1) # for checking angle that maximizes dispersion
	
	for i in range(N):
		index_NN = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1] # include source itself
		position_angles_array[i] = position_angles[index_NN] 
		all_vectorized.append(thetas - position_angles_array[i])

	all_vectorized = np.array(all_vectorized)

	assert all_vectorized.shape == (N,angles,n)

	x = np.radians(2*all_vectorized) # use numexpr to speed it up quite significantly
	cumsum_inner_products = np.cumsum( ne.evaluate('cos(x)'), axis=2) 

	assert cumsum_inner_products.shape == (N,angles,n)

	n_array = np.asarray(range(1,n+1)) # have to divide different elements by different n

	max_di = 1./n_array * np.max(cumsum_inner_products,axis=1) 
	max_theta = np.argmax(cumsum_inner_products,axis=1)
	
	assert max_di.shape == (N,n) # array of max_di for every source, for every n
	assert max_theta.shape == (N,n) # array of max_theta for every source, for every n

	Sn = 1./N * np.sum(max_di,axis=0) # array of shape (1xn) containing S_1 (nonsense)
										# to S_80
	return Sn

# Old way to monte carlo simulate. Bad.
def monte_carlo(tdata,n_sim=1000,totally_random=False):
	'''
	Make (default) 1000 random data sets and calculate the Sn statistic

	If totally_random, then generate new positions and position angles instead
	of shuffeling the position angles among the sources

	Better: Use the mc_ckd_parallel.py file.
	'''

	ra = np.asarray(tdata['RA'])
	dec = np.asarray(tdata['DEC'])
	pa = np.asarray(tdata['position_angle'])
	length = len(tdata)

	max_ra = np.max(ra)
	min_ra = np.min(ra)
	max_dec = np.max(dec)
	min_dec = np.min(dec)

	Sn_datasets = [] # a 1000x80 array containing 1000 simulations of 80 different S_n
	print 'Starting '+ str(n_sim) + ' Monte Carlo simulations for n = 0 to n = 80'
	for dataset in range(0,n_sim):
		if dataset % 100 == 0:
			print 'Simulation number: ' + str(dataset)
		rdata = Table()
		if totally_random:
			rdata['RA'] = (max_ra - min_ra) * np.random.random_sample(length) + min_ra
			rdata['DEC'] = (max_dec - min_dec) * np.random.random_sample(length) + min_dec
			rdata['position_angle'] = 180 * np.random.random_sample(length)
		else:
			np.random.shuffle(pa)
			rdata['RA'] = ra
			rdata['DEC'] = dec
			rdata['position_angle'] = pa

		Sn = angular_dispersion_vectorized_n(rdata)
		Sn_datasets.append(Sn)

	Sn_datasets = np.asarray(Sn_datasets)
	Result = Table()
	for i in range(80):
		Result['S_'+str(i)] = Sn_datasets[:,i]

	if totally_random:
		np.save('./TR_Sn_monte_carlo',Sn_datasets)
		Result.write('./TR_Sn_monte_carlo.fits',overwrite=True)
	else:
		np.save('./Sn_monte_carlo',Sn_datasets)
		Result.write('./Sn_monte_carlo.fits',overwrite=True)
