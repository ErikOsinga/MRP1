import numpy as np 
from astropy.io import fits
from astropy import wcs
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy import constants as S
import sys
sys.path.insert(0, '/data1/osinga/anaconda2')

prefix = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'
#P173+55
filename1 = prefix + '/P173+55/mosaic.fits'
hdu_list = fits.open(filename1)
# print hdu_list.info()
image_data = hdu_list[0].data 
header = hdu_list[0].header
# print type(image_data)
# print image_data.shape
hdu_list.close()

#doing stuff with the wcs module
def test1():
	w = wcs.WCS(filename1)
	plt.imshow(image_data, vmin=-3e-5, vmax=0.001, origin='lower')
	lon,lat = w.all_pix2world(30,40,0)
	print lon,lat
	fig = plt.figure()
	fig.add_subplot(111, projection = w)
	plt.imshow(image_data,vmin=-3e-5, vmax=0.001, origin='lower',cmap=plt.cm.viridis)
	plt.xlabel('RA')
	plt.ylabel('Dec')
	plt.show()
	print wcs.utils.wcs_to_celestial_frame(w)


def test_regions():
	'''
	loads in the regions from the region file and converts these to image coords,
	then plots them over the image.
	'''
	import pyregion
	region_name = prefix + 'P173+55cat.srl.reg'
	r = pyregion.open(region_name)
	r2 = r.as_imagecoord(header) #this uses image coord
	#below we use r instead of image coords
	print r[0].name 
	print 'right ascension, declination, semi-major axis, semi-minor axis, angle'
	print r[0].coord_format 
	print r[0].coord_list
	print r[0].comment
	print r[0].attr

	#defines a list of matplotlib.patches.Patch and other kinds of artists (usually Text)
	patch_list, artist_list = r2.get_mpl_patches_texts()
	fig, ax = plt.subplots()
	ax.imshow(image_data, cmap = plt.cm.gray, origin = 'lower', vmin = -0.000246005, vmax = 0.000715727)
	for p in patch_list:
		ax.add_patch(p) #add the first patch to the plot, then the second, etc
	#ax.add_artist(artist_list[0]) #similar for the first artist
	plt.show()



# test_regions()


def test_filter():
	'''
	could try testing what a restriction on the semi-major axis does
	First test has a restriction of 0.002, too low.
	Second test has a restriction of 0.003, looks good, still many single sources though
	perhaps try something with the ratio of semi to major axis. or exclude circles.
	Third test excludes circles on top of the second test
	Fourth test: 0.0165 deg is roughly an arcmin, roughly 39.6 image units
	,find sources close to eachother, mark them red.
	Interesting result: it looks at the distance between the centers so relatively
	close structures can still be marked green, see example1.pngP173+55_filter_example1.png

	'''

	import pyregion
	region_name = prefix + 'P173+55cat.srl.reg'
	#r = the region file
	r = pyregion.open(region_name)

	#note: below we print r (in degrees) instead of image coords
	print r[0].name 
	print 'right ascension, declination, semi-major axis, semi-minor axis, angle'
	print r[0].coord_list
	print r[0].coord_format 
	print r[0].comment
	print r[0].attr
	
	#filtered list
	def filter():
		r2 = r.as_imagecoord(header) #this converts to image coord
		r3 = []
		for i in range (0,len(r)):
			#test: semi-major axis > 0.003 degrees and no circular sources
			if r[i].name != 'circle':
				if r[i].coord_list[2] > 0.003:
					# print r[i].coord_list[2] #prints the semi-major axis
					r3.append(r2[i])#we append r2 since that has image coords
		return r3

	def nearest_neighbour_plot_efficient():
		TheResult = []
		import scipy
		Coords_array = np.array([r[i].coord_list[0:2] for i in range (0,len(r))])
		YourTreeName = scipy.spatial.cKDTree(Coords_array, leafsize = 100)
		for item in Coords_array:
			TheResult.append(YourTreeName.query(item,k=1))
			#TheResult will be a list of tuples with the dinstance and index
			#of the location of the point in Coords_array
		print TheResult


	def nearest_neighbour_plot():
		TheResult = [] #for each point the distance to closest neighbour
		for i in range (0,2):
			print 'Iteration number: ' + str(i)
			min_distance = 1e12
			for j in range(0,len(r)):
				if i != j:
					distance = distanceOnSphere(r[i].coord_list[0],r[i].coord_list[1],
						r[j].coord_list[0],r[j].coord_list[1])
					if distance < min_distance:#new nearest neighbour found
						min_distance = distance
			#after all the neighbours are checked, append the nearest
			TheResult.append(min_distance)

		TheResult=np.asarray(TheResult)
		# print TheResult
		fig, ax = plt.subplots()
		ax.hist(TheResult,bins=2)
		plt.show()

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

	#in case we use the filter function
	# r3 = pyregion.ShapeList( filter() )
	
	nearest_neighbour_plot()

	def distance_test():
		'''
		changes the color of sources within a distance of 40 image units ~ 1 arcmin
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

	#defines a list of matplotlib.patches.Patch and other kinds of artists (usually Text)
	patch_list, artist_list = r3.get_mpl_patches_texts()
	fig, ax = plt.subplots()
	ax.imshow(image_data, cmap = plt.cm.gray, origin = 'lower', vmin = -0.000246005, vmax = 0.000715727)
	for p in patch_list:
		ax.add_patch(p) #add the first patch to the plot, then the second, etc
	#ax.add_artist(artist_list[0]) #similar for the first artist
	print 'number of initial sources: ' + str(len(r))
	print 'sources found: ' + str(len(r3))
	print 'sources excluded: ' + str(len(r)-len(r3))
	plt.show()

test_filter()