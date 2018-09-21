import sys
sys.path.insert(0, '/net/reusel/data1/osinga/modules/')
import numpy as np 
import math

from astropy.io import fits
# execute with python3
from astropy.table import Table, join, vstack, setdiff

from utils import load_in, deal_with_overlap, distanceOnSphere

from scipy.spatial import cKDTree

from astropy.cosmology import Planck15

from astropy import units as u

def NN_distance_final(tdata):
	"""
	For Table tdata, calculate the distance between all neighbours
	in arcminutes.
	"""
	RAs = tdata['RA_2']
	DECs = tdata['DEC_2']

	x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
	y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
	z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T

	coordinates_tree = cKDTree(coordinates,leafsize=16)
	TheResult_distance = []
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
		# distance in arcmin
		distance = distanceOnSphere(nearestN[0],nearestN[1],#RA,DEC coordinates of the nearest
								source[0],source[1])*60 #RA,DEC coordinates of the current item
		# print distance/60
		TheResult_distance.append(distance)	

	return TheResult_distance

# First join the catalogs on component name and preprocess based on distance
def join_catalogs_componentname(tdata,Write=False):
	comp = '../LOFAR_HBA_T1_DR1_merge_ID_v1.1b.comp.fits'
	comp = Table(fits.open(comp)[1].data)
	vacatalog = '../LOFAR_HBA_T1_DR1_merge_ID_optical_v1.1b.fits'
	vacatalog = Table(fits.open(vacatalog)[1].data)
	
	tdata['Match'] = tdata['Source_Name']
	comp['Match'] = comp['Component_Name']

	# 'Match' to match the source to the component and the rest to prevent double colnames.
	keys = ['Match','RA','E_RA','DEC','E_DEC','Peak_flux','E_Peak_flux', 'Total_flux',
	'E_Total_flux','Maj','E_Maj','Min','E_Min','PA','E_PA','Isl_rms','S_Code'] 

	result = join(tdata,comp,keys=keys)
	# Now we have two columns, Source_Name_1, which is the old one
	# and Source_Name_2 which is the new one.

	# first check if the new source names are unique
	names, counts = np.unique(result['Source_Name_2'],return_counts=True)
	Not_uniquename2 = names[np.where(counts>1)[0]]

	print ('Amount of matches on components: %i' % len(result))
	print ('Amount of non-unique source names_2: %i ' % np.sum(counts>1))
	# print 'Indices of these source_names: ' + str(np.where(counts>1)[0])
	# print 'Source names_2 of these sources:', Not_uniquename2

	# 76 entries are actually multiple components of the same source,
	# So we remove these by deleting the row corresponding to these components
	del result['Match']# Dont need the component name anymore
	del_rows = []
	for i in range(len(Not_uniquename2)):
		indices_double = np.where(result['Source_Name_2'] == Not_uniquename2[i])[0]
		# Loop over all the indices, but leave the last row in the Table
		#, because we do want to keep the source
		for j in range(len(indices_double)-1):
			del_rows.append(indices_double[j])
	result.remove_rows(del_rows)

	# Then match the result 'Source_Name_2', to the value added catalog.
	vacatalog['Source_Name_2'] = vacatalog['Source_Name']
	del vacatalog['Source_Name']
	result2 = join(result,vacatalog,keys='Source_Name_2')

	# Finally, check if we don't have double sources by imposing a 
	# minimum distance between to sources of 0.2 arcmin
	print ('Number of sources before distance check:',len(result2))
	distances = NN_distance_final(result2)
	sort = np.sort(distances)
	argsort = np.argsort(distances)
	remove = argsort[sort<0.2]
	print ('Number of sources to remove:', len(remove)/2.0)
	del_rows = []
	for i in range(0,len(remove),2):
		del_rows.append(i)
	result2.remove_rows(del_rows)
	print ('Number of sources after distance check :',len(result2))

	if Write:
		result2.write('../value_added_biggest_selection.fits',overwrite=True)

	return result2

# Then check if the NN sources are one or two component sources
def components(tdata):
	"""Investigate if the components match with my data
		by seeing how many component counts there are for every source"""

	# start by seeing how many components they match to source names.
	comp = '../LOFAR_HBA_T1_DR1_merge_ID_v1.1b.comp.fits'
	comp = Table(fits.open(comp)[1].data)
	comp['Source_Name_2'] = comp['Source_Name'] # For matching

	result = join(tdata,comp,keys='Source_Name_2',uniq_col_name='{col_name}_{table_name}_3')
	names, counts = np.unique(result['Source_Name_2'],return_counts=True)
	# component names now are the column: result['Component_Name_2_3']
	# print (len(comp), len(names), len(counts))
	indices =  np.where(counts > 1)
	# print (indices)
	multiple_comp_names = names[indices]

	# Should also check if we have NN when there is only 1 component
	num_matches = 0
	num_mg = 0 
	source_name1s = []
	source_names_correct = []
	for name in multiple_comp_names:
		current_index = np.where(result['Source_Name_2'] == name)
		compnames = result['Component_Name_2_3'][current_index] # Both components as in the VA
		comp1 = result['Source_Name_1'][current_index][0] # Component 1 
		comp2 = result['new_NN_Source_Name'][current_index][0] # Component 2

		if comp2 == 'N/A': # MG source
			num_mg +=1

		elif (comp1 in compnames and comp2 in compnames): # Both correct
			num_matches+=1
			source_names_correct.append(comp1)

		elif (comp1 in compnames) != (comp2 in compnames): # One wrong, one correct
			# print 'Half fout:', current_index
			# print compnames
			# print comp1, comp2
			source_name1s.append(comp1) # save the sourcenames that are wrong


	print ('Number of correct matches:',num_matches)
	print ('Number of MG sources:', num_mg)
	# print source_name1s
	# sourcenamesincorrect = Table()
	# sourcenamesincorrect['Source_Name_1'] = source_name1s
	# sourcenamesincorrect.write('/data1/osinga/value_added_catalog/2876_NOTmatched_sourcesnames.fits')
	
	# return the unique source names, how much times they appear and the (in)correct matches
	return names, counts, source_name1s, source_names_correct

def power(redshift, total_flux, alpha=-0.78):
	"""
	Calculates the power of a source(s) given the 
	redshift,
	integrated flux,
	and assumed average spectral index alpha
	# see https://arxiv.org/pdf/1609.00537.pdf
	"""

	# luminosity distance
	Dl = Planck15.luminosity_distance(redshift)
	Dl = Dl.to(u.kpc)

	# see (https://arxiv.org/pdf/0802.2770.pdf) 9: Appendix
	L = total_flux*4*np.pi*Dl**2 * (1+redshift)**(-1.0 * alpha - 1)
	print (L)

	return L

def physical_size(redshift, angular_size):
	'''
	Calculates the physical size of a source in Kpc

	'''


	# physical distance corresponding to 1 radian at redshift z
	size = Planck15.angular_diameter_distance(redshift)

	# to not adjust the actual size column
	angular_size = np.copy(angular_size)
	# angular_size is in arcsec, convert to radian
	angular_size /= 3600 # to degrees
	angular_size *= np.pi/180 # to radians

	# physical distance corresponding to angular distance
	size *= angular_size

	size = size.to(u.kpc)

	return size

def find_NN_properties(tdata,arg):
	"""
	Function to return a certain property of a Nearest Neighbour of a source

	tdata -- the table containing the data: Mosaic_ID_2 and 'new_NN_index'
	arg, a string with the requested property (e.g. 'Total_Flux')
	"""

	NN_sources = np.invert(np.isnan(tdata['new_NN_index']))
	tdata_NN = tdata[NN_sources]

	x = []

	mosaicid_prev = 'initializing'
	for i in range(0,len(tdata)):
		
		if NN_sources[i]:
			mosaicid = tdata['Mosaic_ID_2'][i]
			if mosaicid != mosaicid_prev: # only open new file if we need to
				NNfile = '../source_filtering/NN/'+mosaicid+'NearestNeighbours_efficient_spherical2.fits'
				NNfile = Table(fits.open(NNfile)[1].data)
				mosaicid_prev = mosaicid
			x.append(NNfile[arg][tdata['new_NN_index'][i]])
		else:
			x.append(np.nan)
	
	return x

def select_all_interesting():
	"""
	Function to select all sources in the leaves of the tree, so with err < 15 or lobe ratio < 2 etc

	"""
	dataNN = Table(fits.open('../source_filtering/all_NN_sources.fits')[1].data)
	dataMG = Table(fits.open('../source_filtering/all_multiple_gaussians.fits')[1].data)

	# NOT NEEDED ANYMROE FOR value_added_catalog_1_1b_thesis
	# # calculate the cutoff value for the NN, since it wasnt well defined in the source_filtering
	# Min = dataNN['Min']/60. # convert to arcmin
	# nn_dist = dataNN['new_NN_distance(arcmin)']
	# cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg

	selectionNN = np.where( (dataNN['amount_of_maxima'] > 1) & (dataNN['lobe_ratio'] <= 2) &
		(dataNN['lobe_ratio'] >= 1./2) & 
		(dataNN['classification'] == 'Small err') )
	selectionMG = np.where( (dataMG['amount_of_maxima'] > 1) & (dataMG['lobe_ratio'] <= 2) &
		(dataMG['lobe_ratio'] >= 1./2) & (dataMG['classification'] == 'Small err') )

	selectionNN = dataNN[selectionNN]
	selectionMG = dataMG[selectionMG]

	return selectionNN, selectionMG

if __name__ == '__main__':
	print ("The errors can be ignored, are because of NaN in the Tables.")
	print ("Astropy units doesn't like NaNs, but not worth the effort to work around this")
	
	# tdata_BS = Table(fits.open('../source_filtering/biggest_selection_latest.fits')[1].data)
	tdataNN, tdataMG = select_all_interesting() # select leaves of decision tree
	tdata_BS = vstack([tdataNN,tdataMG])
	print ('Length of biggest selection before dealing with overlap:', len(tdata_BS))	
	# have to write it out and read it again to prevent '--' from showing up in Table instead of NaN
	tdata_BS.write('../biggest_selection_with_overlap.fits',overwrite=True)
	tdata_BS = Table(fits.open('../biggest_selection_with_overlap.fits')[1].data)

	# fix the NN sources that are also MG sources and fix the nearest neighbour 
	# of NN sources that are also MG sources by visually classifying
	# the results are saved in classtable.
	classtable = '../NN_excluded_by_also_being_MG_classification.fits'
	classtable = Table(fits.open(classtable)[1].data)
	# classified=True makes it so it uses the visual classification to deal with overlap
	tdata_BS = deal_with_overlap(tdata_BS,classified=True,classtable=classtable)
	print ('Length of biggest selection after dealing with overlap:', len(tdata_BS))	
	tdata_BS.write('../source_filtering/biggest_selection_latest.fits',overwrite=True)

	# Get the value_added_biggest_selection 
	VA_bs = join_catalogs_componentname(tdata_BS,True)
	print ('Length of VA_bs:',len(VA_bs))

	# get the MG sources from VA_BS 
	VA_bs_MG = VA_bs[np.isnan(VA_bs['new_NN_RA'])]
	# MG sources: use PA_1
	VA_bs_MG['final_PA'] = VA_bs_MG['PA_1']
	# MG sources size; use MAJ_1 times two, since Maj is the SEMI-major axis
	VA_bs_MG['size'] = VA_bs_MG['Maj_1'] * 2
	# physical size
	VA_bs_MG['physical_size'] = physical_size(VA_bs_MG['z_best'],VA_bs_MG['size'])
	# power, use Total_Flux_1 since we assume it's a loner source
	VA_bs_MG['power'] = power(VA_bs_MG['z_best'], VA_bs_MG['Total_flux_1'])
	print ('Length of VA_bs_MG:',len(VA_bs_MG))

	# get the NN sources from VA_BS
	VA_bs_NN = VA_bs[np.invert(np.isnan(VA_bs['new_NN_RA']))]
	print ('Length of VA_bs_NN', len(VA_bs_NN))

	# get all sources that are correctly matched to components (+ MG sources)
	incorr_sourcenames, correct_sourcenames = components(VA_bs)[2:4]
	print ('Number of incorrect component matches:', len(incorr_sourcenames))
	corr_table = Table()
	corr_table['Source_Name_1'] = correct_sourcenames
	corr_table = join(corr_table,VA_bs,keys='Source_Name_1')
	# correctly matched components: use LGZ_PA
	corr_table['final_PA'] = corr_table['LGZ_PA']
	# correctly matched components size: use LGZ_size
	corr_table['size'] = corr_table['LGZ_Size']
	# physical size
	corr_table['physical_size'] = physical_size(corr_table['z_best'],corr_table['size'])
	# power, use Total_Flux_2, since we know it's a component source
	corr_table['power'] = power(corr_table['z_best'], corr_table['Total_flux_2'])
	VA_bs_NNmatch = vstack([VA_bs_MG,corr_table])
	print ('Number of correct component matches (+MG):', len(VA_bs_NNmatch) )
	
	# get the previous Table + the 'Not round' 'wrong' NN sources 
	# not round is defined as: (Maj-1)>Min
	# incorr_table = Table()
	# incorr_table['Source_Name_1'] = incorr_sourcenames
	NN_wrongcomp = setdiff(VA_bs_NN,corr_table,keys='Source_Name_1') # Get all wrong NN sources and save only the 1st component
	indx_notround = np.where((NN_wrongcomp['Maj_1'] - 1)> NN_wrongcomp['Min_1'])
	indx_round = np.where((NN_wrongcomp['Maj_1'] - 1) <= NN_wrongcomp['Min_1'])
	print ('Number of wrong component NN sources, but still useful because not round:', len(indx_notround[0]))
	notround_sources = NN_wrongcomp[indx_notround] 
	# not round sources: use PA_1
	notround_sources['final_PA'] = notround_sources['PA_1']
	# not round sources size: use Maj_1, times two for semi-major axis
	notround_sources['size'] = notround_sources['Maj_1'] * 2
	# physical size
	notround_sources['physical_size'] = physical_size(notround_sources['z_best'],notround_sources['size'])
	# not round sources power: use Total_Flux_1 since we assume its a loner source
	notround_sources['power'] = power(notround_sources['z_best'],notround_sources['Total_flux_1'])
	VA_bs_NN_match_plus_notround = vstack([VA_bs_NNmatch,notround_sources])
	print ('Number of correct comp matches + not round sources:',len(VA_bs_NN_match_plus_notround))

	# last table + the round sources should be biggest selection again, 
	# but this time with the final_PA column
	round_sources = NN_wrongcomp[indx_round] 
	# Round sources : use position_angle (pretend they are matched anyways)
	round_sources['final_PA'] = round_sources['position_angle']
	# Round sources size: use NN_distance (pretend they are matched anyways)
	round_sources['size'] = round_sources['new_NN_distance(arcmin)']*60 #convert to arcsec
	# physical size
	round_sources['physical_size'] = physical_size(round_sources['z_best'],round_sources['size'])
	# power is tricky here, we have to calculate the total flux of the nearest neighbours
	# and sum this with the total flux of the source for the actual total flux
	total_flux_NN = find_NN_properties(round_sources,'Total_flux')
	total_flux_round_sources = round_sources['Total_flux_1'] + total_flux_NN
	round_sources['power'] = power(round_sources['z_best'],total_flux_round_sources)
	VA_bs = vstack([VA_bs_NN_match_plus_notround,round_sources])
	print ('Again, the length of VA_bs:',len(VA_bs))

	# Scream and shout and Write it all out
	VA_bs.write('../value_added_selection.fits',overwrite=True)
	VA_bs_MG.write('../value_added_selection_MG.fits',overwrite=True)
	# VA_bs_NN.write('../value_added_selection_NN.fits',overwrite=True) # comment because doesnt make sense (since PA or position_angle or ??)
	VA_bs_NNmatch.write('../value_added_compmatch.fits',overwrite=True)
	VA_bs_NN_match_plus_notround.write('../value_added_compmatch_plus_notround.fits',overwrite=True)




