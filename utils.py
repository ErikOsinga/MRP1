import numpy as np 
import math

from astropy.io import fits
from astropy.table import Table, join, vstack

from scipy.spatial import cKDTree

import numexpr as ne

import difflib

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
	"""
	Load in columns from Table nnpath

	Arguments:
	nnpath -- The Table (.fits) file.
	*arg -- The keys of the columns given in Table nnpath
	
	Returns:
	The columns as a tuple of numpy arrays
	"""

	nn1 = fits.open(nnpath)
	nn1 = Table(nn1[1].data)

	x = (np.asarray(nn1[arg[0]]),)
	for i in range (1,len(arg)):
		x += (np.asarray(nn1[arg[i]]),)

	return x

def rotate_point(origin,point,angle):
	"""
	Rotate a point counterclockwise by a given angle around a given origin
	
	Arguments:
	origin -- Tuple (X,Y) around which to rotate the point
	point -- Tuple (X,Y) 
	angle -- Counterclockwise angle of rotation given in DEGREES.

	Returns:
	(qx,qy) -- The location of the rotated point
	"""

	angle = angle * np.pi / 180. # convert degrees to radians

	ox,oy = origin
	px,py = point

	qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py-oy)
	qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py-oy)
	return qx,qy

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
	"""
	Given the positions (ra,dec) in degrees,
	calculates the position angle of the 2nd source wrt to the first source
	in degrees. The position angle is measured North through East

	Arguments:
	(ra,dec) -- Coordinates (in degrees) of the first and second source

	Returns: 
	  -- The position angle measured in degrees,

	"""

	#convert degrees to radians
	ra1,dec1,ra2,dec2 = ra1 * math.pi / 180. , dec1 * math.pi / 180. , ra2 * math.pi / 180. , dec2 * math.pi / 180. 
	return (math.atan( (math.sin(ra2-ra1))/(
			math.cos(dec1)*math.tan(dec2)-math.sin(dec1)*math.cos(ra2-ra1))
				)* 180. / math.pi )# convert radians to degrees

def inner_product(alpha1,alpha2):
	"""
	Returns the inner product of position angles alpha1 and alpha2
	The inner product is defined in Jain et al. 2004
	
	Arguments:
	alpha1,alpha2 -- Position angles given in degrees.

	Returns:
	 Inner product -- +1 indicates parallel -1 indicates perpendicular

	Using numpy is much quicker however.
	"""
	alpha1, alpha2 = math.radians(alpha1), math.radians(alpha2)
	return math.cos(2*(alpha1-alpha2))

def parallel_transport(RA_t,DEC_t,RA_s,DEC_s,PA_s):
	"""
	Parallel transports source(s) s at RA_s, DEC_s with position angle PA_s to target
	position t, with RA_t and DEC_t. See Jain et al. 2004

	Arguments:
	RA_t, DEC_t -- coordinates of the target in degrees
	RA_s, DEC_s -- numpy array with coordinates of the source(s) in degrees
	PA_s -- numpy array with position angle of the source(s) in degrees

	Returns:
	PA_transport -- numpy array with the parallel transported angle(s) PA_s to target position t
	 				in degrees

	"""

	PA_s = np.radians(PA_s)
	RA_s = np.radians(RA_s)
	DEC_s = np.radians(DEC_s)
	RA_t = np.radians(RA_t)
	DEC_t = np.radians(DEC_t)
	
	# define the radial unit vectors u_rs and u_rt
	x = np.cos(RA_s) * np.cos(DEC_s)
	y = np.sin(RA_s) * np.cos(DEC_s)
	z = np.sin(DEC_s)
	u_rs = -1 * np.array([x,y,z]) #pointing towards center of sphere 

	x = np.cos(RA_t) * np.cos(DEC_t)
	y = np.sin(RA_t) * np.cos(DEC_t)
	z = np.sin(DEC_t)
	u_rt = -1 * np.array([x,y,z]).reshape(1,3) #pointing towards center of sphere 

	xi_1 = np.arccos( (np.sin(DEC_t)*np.sin(RA_t - RA_s)) / (np.sqrt(1-(np.dot(u_rt,u_rs))**2 )) )[0]
	temp1 = ( (-1 * np.sin(DEC_s)*np.cos(DEC_t)+np.cos(DEC_s)*np.sin(DEC_t)*np.cos(RA_t-RA_s) ) 
				/ np.sqrt( 1 - (np.dot(u_rt,u_rs)**2)) )[0]
	xi_2 = np.arccos( (-1* np.sin(DEC_s)*np.sin(RA_s - RA_t)) / (np.sqrt(1-(np.dot(u_rt,u_rs))**2 )) )[0]
	temp2 = ( (-1 * np.sin(DEC_t)*np.cos(DEC_s)+np.cos(DEC_t)*np.sin(DEC_s)*np.cos(RA_t-RA_s) )
				 / np.sqrt( 1 - (np.dot(u_rt,u_rs)**2)) )[0]

	# Make sure that temp1 and temp2 are also never NaN
	assert np.sum(np.isnan(temp1)) == 0
	assert np.sum(np.isnan(temp2)) == 0

	# if vectors are too close NaN appears, use 'neglect parallel transport', it's super effective.
	wherenan1 = np.isnan(xi_1) 
	wherenan2 = np.isnan(xi_2)
	wherenan = np.logical_or(wherenan1,wherenan2)

	for j in range(len(wherenan)):
		if wherenan[j] == True:
			# if vectors are too close, parallel transport is negligible
			xi_1[j] = 0
			xi_2[j] = 0 

		if temp1[j] < 0:
			xi_1[j] *= -1
		if temp2[j] > 0:  		# REKEN DIT NOG FF NA
			xi_2[j] *= -1

	PA_transport = PA_s + xi_2 - xi_1 
	# fix it back into [0,pi] range if it went beyond it
	for j in range(len(RA_s)):
		if PA_transport[j] > np.pi:
			# print ('PT : Reducing angle by pi')
			PA_transport[j] -= np.pi
		if PA_transport[j] < 0:
			# print ('PT : Increasing angle by pi')
			PA_transport[j] += np.pi

	return np.degrees(PA_transport)
	
def angular_dispersion_vectorized_n(tdata,n,redshift=False):
	"""
	Calculates and returns the Sn statistic for tdata
	Vectorized over n, starting at n down to 1 (included).
	e.g. n=80: calculate the Sn for every n from 1 to 81
	
	Does not find the angle that maximizes the dispersion, which is why it is pretty fast.

	Arguments:
	tdata -- Astropy Table containing the sources.
	n -- Number of sources closest to source i (source i included)
	# N = number of sources in tdata
	
	Returns:
	Sn -- (1xn) matrix containing S_1 to S_n
	"""

	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['final_PA'])

	# hard fix for when n > amount of sources in a bin 
	if n > len(tdata):
		print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
		n = len(tdata)-1
		print ('Setting n=%i'%n)

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	if redshift:
		Z = tdata['z_best']
		'''
		H = 73450 # m/s/Mpc = 73.45 km/s/Mpc
		# but is actually unimportant since only relative distances are important
		from scipy.constants import c # m/s
		# assume flat Universe with q0 = 0.5 (see Hutsemekers 1998)
		# I think assuming q0 = 0.5 means flat universe
		r = 2.0*c/H * ( 1-(1+Z)**(-0.5) ) # comoving distance
		'''
		from astropy.cosmology import Planck15
		r = Planck15.comoving_distance(Z)
		x = r * np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
		y = r * np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
		z = r * np.sin(np.radians(DECs))	
	else:
		x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
		y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
		z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T
	
	#make a KDTree for quick NN searching	
	coordinates_tree = cKDTree(coordinates,leafsize=16)

	# for every source: find n closest neighbours, calculate max dispersion
	position_angles_array = np.zeros((N,n)) # array of shape (N,n) that contains position angles
	for i in range(N):
		index_NN = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1] # include source itself
		position_angles_array[i] = position_angles[index_NN] 

	position_angles_array = np.array(position_angles_array)

	assert position_angles_array.shape == (N,n)

	n_array = np.asarray(range(1,n+1)) # have to divide different elements by different n

	x = np.radians(2*position_angles_array) # use numexpr to speed it up quite significantly

	di_max = 1./n_array * ( (np.cumsum(ne.evaluate('cos(x)'),axis=1))**2 
				+ (np.cumsum(ne.evaluate('sin(x)'),axis=1))**2 )**0.5
	
	assert di_max.shape == (N,n) # array of max_di for every source, for every n

	Sn = 1./N * np.sum(di_max,axis=0) # array of shape (1xn) containing S_1 (nonsense)
										# to S_n
	return Sn

def angular_dispersion_vectorized_n_parallel_transport(tdata,n,redshift=False):
	"""
	Calculates and returns the Sn statistic for tdata
	Vectorized over n, starting at n down to 1 (included).
	Using parallel transport (PT) (Jain et al. 2004)
	e.g. n=80: calculate the Sn for every n from 1 to 81
	
	Does not find the angle that maximizes the dispersion, which is why it is pretty fast.

	Arguments:
	tdata -- Astropy Table containing the sources.
	n -- Number of sources closest to source i (source i included)
	# N = number of sources in tdata
	
	Returns:
	Sn -- (1xn) matrix containing S_1 to S_n
	"""

	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['final_PA'])

	# hard fix for when n > N
	if n > len(tdata):
		print ('n = %i, but this tdata only contains N=%i sources'%(n,len(tdata)))
		n = len(tdata)-1
		print ('Setting n=%i'%n)

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	if redshift:
		Z = tdata['z_best']
		'''
		H = 73450 # m/s/Mpc = 73.45 km/s/Mpc
		# but is actually unimportant since only relative distances are important
		from scipy.constants import c # m/s
		# assume flat Universe with q0 = 0.5 (see Hutsemekers 1998)
		# I think assuming q0 = 0.5 means flat universe
		r = 2.0*c/H * ( 1-(1+Z)**(-0.5) ) # comoving distance
		'''
		from astropy.cosmology import Planck15
		r = Planck15.comoving_distance(Z) #better to just use this calculator
		x = r * np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
		y = r * np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
		z = r * np.sin(np.radians(DECs))	
	else:
		x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
		y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
		z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T
	
	#make a KDTree for quick NN searching	
	coordinates_tree = cKDTree(coordinates,leafsize=16)

	# for every source: find n closest neighbours, calculate max dispersion using PT
	position_angles_array = np.zeros((N,n)) # array of shape (N,n) that contains position angles
	for i in range(N):
		index_NN = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1] # include source itself
		# print index_NN # if this gives an error we should check whether we have the right sources (redshift selection)
		# Transport every nearest neighbour to the current source i
		angles_transported = parallel_transport(RAs[i],DECs[i],RAs[index_NN[1:]],DECs[index_NN[1:]],position_angles[index_NN[1:]])
		# Then concatenate the transported angles to the current source position angle and store it
		position_angles_array[i] = np.concatenate(([position_angles[i]],angles_transported))

	position_angles_array = np.array(position_angles_array)

	assert position_angles_array.shape == (N,n)

	n_array = np.asarray(range(1,n+1)) # have to divide different elements by different n

	x = np.radians(2*position_angles_array) # use numexpr to speed it up quite significantly

	di_max = 1./n_array * ( (np.cumsum(ne.evaluate('cos(x)'),axis=1))**2 
				+ (np.cumsum(ne.evaluate('sin(x)'),axis=1))**2 )**0.5
	
	assert di_max.shape == (N,n) # array of max_di for every source, for every n

	Sn = 1./N * np.sum(di_max,axis=0) # array of shape (1xn) containing S_1 (nonsense)
										# to S_n
	return Sn

def select_on_size(tdata,cutoff=6):
	"""
	Outputs a table that only contains items below a given angular size cutoff
	Cutoff is given in arcseconds. Default = 6 arcseconds
	Selection is done differently based on the source:
		- MG: Semimajor axis has to be > cutoff/2
		- NN: Distance between NN has to be > cutoff
	"""

	cutoff_arcmin = cutoff/60.

	MG_index = np.isnan(tdata['new_NN_distance(arcmin)']) 
	NN_index = np.invert(MG_index)

	interestingMG = tdata[MG_index]['Maj'] > cutoff/2. #arcsec (/2 because Maj is semi-major axis)
	interestingNN = tdata[NN_index]['new_NN_distance(arcmin)'] > cutoff_arcmin

	tdataMG = tdata[MG_index][interestingMG]
	tdataNN = tdata[NN_index][interestingNN]

	tdata_interesting = vstack([tdataNN,tdataMG])

	print ('Selected data has ' + str(len(tdata_interesting)) + ' sources left')

	return tdata_interesting

def deal_with_overlap_2(tdata):
	"""
	Finds the non-unique sources the other way around as well. 
	If any source has a neighbour that is also classified as a legit MG source, it's found.
	
	"""

	count = 0
	all_names, all_indices, mosaic_ids = [], [], []
	MG_index = np.isnan(tdata['new_NN_distance(arcmin)']) 
	NN_index = np.invert(MG_index)

	NN_source_names = tdata[NN_index]['new_NN_Source_Name']
	source_names = tdata['Source_Name'] 
	drop_rows = []
	for i, name in enumerate(NN_source_names):
		# check which source has a neighbour that is also an MG source
		where = np.where(source_names == name)[0] 
		if len(where) > 0:
			# print name, where, np.asarray(tdata['Mosaic_ID'])[where]
			count +=1    
			all_names.append(name)
			all_indices.append(where[0])
			mosaic_ids.append(np.asarray(tdata['Mosaic_ID'])[where][0])
			drop_rows.append(i)

	print ('Number of double sources in Nearest Neighbours: %i' % count)
	# tdata[drop_rows].write('/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/NN_excluded_by_also_being_MG_2.fits')
	tdata.remove_rows(drop_rows)
	return all_names, all_indices, mosaic_ids, tdata

def deal_with_overlap(tdata,classified=False,classtable=None):
	"""
	If there are non-unique sources in tdata, remove the NN source from the data.

	Non-unique sources show up because we select first on NN and then on the MG 
	but this might cause overlap between the two..

	Uses deal_with_overlap2 as well.

	if classified=True, uses the table classtable to make the cut on the sources
	"""

	drop_rows = []

	if classified:
		for i in range(len(classtable)):
			# NN source index and MG source index, in order
			indices = np.where(tdata['Source_Name'] == classtable['Source_Name'][i])[0]
			
			if len(indices) == 2: # then the source is both in NN catalog and MG catalog
				NNindex = indices[0]
				MGindex = indices[1]
			elif len(indices) == 1:  # then the source's NN is in the MG catalog
				# so we should find it in the MG catalog by NN source name
				MG_index = np.where(tdata['new_NN_Source_Name'] == classtable['Source_Name'][i])[0]
			if classtable['classification'][i] == 'MG          ':
				# then we keep the source as MG source
				# thus we drop the NN entry
				drop_rows.append(NNindex)
			elif classtable['classification'][i] == 'NN          ':
				# then we keep the source as NN source
				# thus we drop the MG entry
				drop_rows.append(MGindex)
			elif classtable['classification'][i] == 'unclassified':
				# then we drop the source entirely
				# thus we drop both entries
				drop_rows.append(NNindex)
				drop_rows.append(MGindex)

			else:
				raise ValueError("This should not happen")
		
		# tdata[drop_rows].write('/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/NN_excluded_by_also_being_MG.fits')
		tdata.remove_rows(drop_rows)

	else:

		source_names, counts = np.unique(tdata['Source_Name'],return_counts=True)
		# array with the non-unique source names
		source_names = source_names[counts>1] 
		for i in range(len(source_names)):
			nonunique = np.where(tdata['Source_Name'] == source_names[i])
			n = nonunique[0][0] # NN source 
			m = nonunique[0][1] # MG source
			drop_rows.append(n)

		# tdata[drop_rows].write('/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/NN_excluded_by_also_being_MG_1.fits')
		print ('Number of double sources in 1st NN source: %i' % len(drop_rows))
		tdata.remove_rows(drop_rows)

		tdata = deal_with_overlap_2(tdata)[-1]

	return tdata

def tableMosaic_to_full(Mosaic_ID):
	'''
	Workaround to go from the Mosaic_ID in the table, to a full Mosaic id.
	Since the Mosaic_ID in the table is cutoff after 8 characters..
	'''
	Mosaic_ID = Mosaic_ID.strip()  # Strip because whitespace is added..
	MosaicID_full = difflib.get_close_matches(Mosaic_ID,FieldNames,n=1)[0]
	# check to see if difflib got the right string		
	trying = 1
	while MosaicID_full[:8] != Mosaic_ID: # Might have to change to [:8] as well
		trying +=1
		MosaicID_full = difflib.get_close_matches(Mosaic_ID,FieldNames,n=trying)[trying-1]

	return MosaicID_full

def convert_deg(RA,DEC):
	"""Converts degrees to sexagesimal string format
	RA is in hh:mm:sec, while DEC is in deg:arcmin:arcsec
	Returns 3 decimal points on the last digit
	"""
	
	hh = RA * 12/180
	mm = (hh - int(hh))*60
	ss = (mm - int(mm))*60
	RA = "%2i:%02i:%02.3f"%(hh,mm,ss)

	deg = DEC
	arcmin = (deg - int(deg))*60
	arcsec = (arcmin - int(arcmin))*60
	DEC = "%+i:%02i:%06.3f"%(deg,arcmin,arcsec)

	return RA,DEC


if __name__ == '__main__':
	print ('This script is not for execution')
