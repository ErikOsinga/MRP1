import sys
sys.path.insert(0, '/net/reusel/data1/osinga/anaconda2')
import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt

from utils import (angular_dispersion_vectorized_n, load_in, distanceOnSphere
	, deal_with_overlap, FieldNames)



def histedges_equalN(x, nbin):
	"""
 	Make nbin equal height bins
 	Call plt.hist(x, histedges_equalN(x,nbin))
	"""
	npt = len(x)
	return np.interp(np.linspace(0,npt,nbin+1),
					np.arange(npt),
					np.sort(x))

def histedges_equalN(x, nbin):
	"""
 	Make nbin equal height bins
 	Call plt.hist(x, histedges_equalN(x,nbin))
	"""
	npt = len(x)
	return np.interp(np.linspace(0,npt,nbin+1),
					np.arange(npt),
					np.sort(x))

def select_size_bins_cuts_biggest_selection(tdata,Write=False):
	"""
	Makes 4 size bins from the tdata, same bins as biggest_selection

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'3':bin5) -- 4 tables 
											containing the selected sources in each bin
	"""

	tdata_bs = Table(fits.open('../biggest_selection.fits')[1].data)
	try:
		size = np.asarray(tdata_bs['size_thesis']) 
	except KeyError:
		print ('using size: Maj*2 or NN_dist')
		MGwhere = np.isnan(tdata_bs['new_NN_RA'])
		NNwhere = np.invert(MGwhere)
		MGsources = tdata_bs[MGwhere]
		NNsources = tdata_bs[NNwhere]

		sizeNN = np.asarray(NNsources['new_NN_distance(arcmin)']) * 60 # to arcsec
		sizeMG = np.asarray(MGsources['Maj'])*2 # in arcsec

		size = np.concatenate((sizeNN,sizeMG))

	fig3 = plt.figure(3)
	ax = fig3.add_subplot(111)

	# get the size bins from biggest_selection
	n, bins, patches = ax.hist(size,histedges_equalN(size,4))
	plt.close(fig3)
	# print bins/60.
	print (n)
	# print ('Size bins (arcmin):',bins/60)

	try:
		size = np.asarray(tdata['size_thesis']) 
	except KeyError:
		print ('using size: Maj*2 or NN_dist')
		MGwhere = np.isnan(tdata['new_NN_RA'])
		NNwhere = np.invert(MGwhere)
		MGsources = tdata[MGwhere]
		NNsources = tdata[NNwhere]

		sizeNN = np.asarray(NNsources['new_NN_distance(arcmin)']) * 60 # to arcsec
		if len(MGsources) != 0:
			sizeMG = np.asarray(MGsources['Maj'])*2 # in arcsec
		else:
			print ('No MG sources found')
			sizeMG = np.asarray([])

		size = np.concatenate((sizeNN,sizeMG))
	
	a = dict()
	for i in range(len(bins)-1): 
		# select from current tdata with the bins from tdata_bs 
		a[str(i)] = tdata[(bins[i]<size)&(size<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./biggest_selection_SIZEssdfsdf%i.fits'%i,overwrite=True)
	return a

def select_flux_bins_cuts_biggest_selection(tdata,Write=False):
	"""
	Makes 4 flux bins from the tdata, equal freq.

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'3':bin5) -- 4 tables 
											containing the selected sources in each bin
	"""
	
	tdata_bs = Table(fits.open('../biggest_selection.fits')[1].data)
	
	# find the Peak flux of tdata_bs, based on isolated or connected lobe
	MGwhere = np.isnan(tdata_bs['new_NN_RA'])
	NNwhere = np.invert(MGwhere)
	MGsources = tdata_bs[MGwhere]
	NNsources = tdata_bs[NNwhere]
	fluxMG = np.asarray(MGsources['Peak_flux'])
	# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
	use_current = NNsources['Peak_flux'] > NNsources['new_NN_Peak_flux']
	fluxMG = np.asarray(MGsources['Peak_flux'])
	fluxNN = []
	for i in range(len(NNsources)):
		if use_current[i]:
			fluxNN.append(NNsources['Peak_flux'][i])
		else: # use NN peak flux
			fluxNN.append(NNsources['new_NN_Peak_flux'][i])
	fluxNN = np.asarray(fluxNN)
	assert len(fluxNN) == len(NNsources)

	# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
	flux = np.concatenate((fluxNN,fluxMG))

	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	# get flux bins from tdata_bs
	n, bins, patches = ax.hist(flux,histedges_equalN(flux,4))
	plt.close(fig2)
	print ('Flux bins:',bins)

	# find the Peak flux of tdata, based on isolated or connected lobe
	MGwhere = np.isnan(tdata['new_NN_RA'])
	NNwhere = np.invert(MGwhere)
	MGsources = tdata[MGwhere]
	NNsources = tdata[NNwhere]
	try:
		fluxMG = np.asarray(MGsources['Peak_flux_2'])
		# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
		use_current = NNsources['Peak_flux_2'] > NNsources['new_NN_Peak_flux']
		fluxNN = []
		for i in range(len(NNsources)):
			if use_current[i]:
				fluxNN.append(NNsources['Peak_flux_2'][i])
			else: # use NN peak flux
				fluxNN.append(NNsources['new_NN_Peak_flux'][i])
		fluxNN = np.asarray(fluxNN)
		assert len(fluxNN) == len(NNsources)
		print ('using Peak_flux_2')

	except KeyError:
		print ('using Peak_flux, so this is biggest_selection')
		fluxMG = np.asarray(MGsources['Peak_flux'])
		# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
		# Use as peak flux the peak flux of the brightest of the lobes for isolated lobes
		use_current = NNsources['Peak_flux'] > NNsources['new_NN_Peak_flux']
		fluxMG = np.asarray(MGsources['Peak_flux'])
		fluxNN = []
		for i in range(len(NNsources)):
			if use_current[i]:
				fluxNN.append(NNsources['Peak_flux'][i])
			else: # use NN peak flux
				fluxNN.append(NNsources['new_NN_Peak_flux'][i])
		fluxNN = np.asarray(fluxNN)
		assert len(fluxNN) == len(NNsources)

	flux = np.concatenate((fluxNN,fluxMG))

	a = dict()
	for i in range(len(bins)-1): 
		# select from tdata with the bins from tdata_bs 
		a[str(i)] = tdata[(bins[i]<flux)&(flux<bins[i+1])]		
		print ('Number in bin %i:'%i,len(a[str(i)]))

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selection_redshift_FLUX_%i.fits'%i,overwrite=True)
	return a

def select_power_bins_cuts_VA_selection(tdata,Write=False):
	"""
	Makes 4 size bins from the tdata, same bins as value_added_selection

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'3':bin5) -- 4 tables 
											containing the selected sources in each bin
	"""

	tdata_VA = Table(fits.open('../value_added_selection.fits')[1].data)
	
	# Take only the available redshift sources
	z_available = np.invert(np.isnan(tdata_VA['z_best']))
	z_zero = tdata_VA['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	tdata_VA = tdata_VA[z_available]

	power = np.asarray(tdata_VA['power_thesis']) 
		
	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	# get the power bins from value_added selection
	n, bins, patches = ax.hist(power,histedges_equalN(power,4))
	plt.close(fig2)
	print ('Power bins:')
	print (bins)
	print (n)

	power = np.asarray(tdata['power_thesis'])

	
	a = dict()
	for i in range(len(bins)-1): 
		# select from current data with bins from tdata_VA
		a[str(i)] = tdata[(bins[i]<power)&(power<bins[i+1])]		
		# print ('Number in bin %i:'%i,len(a[str(i)]))

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selection_power_VACUTS_%i.fits'%i,overwrite=True)
	return a

def select_physical_size_bins_cuts_VA_selection(tdata,Write=False):
	

	tdata_VA = Table(fits.open('../value_added_selection.fits')[1].data)
	
	# Take only the available redshift sources
	z_available = np.invert(np.isnan(tdata_VA['z_best']))
	z_zero = tdata_VA['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	tdata_VA = tdata_VA[z_available]

	physical_size = np.asarray(tdata_VA['physical_size_thesis']) 
	
	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	# get size bins from tdata_VA
	n, bins, patches = ax.hist(physical_size,histedges_equalN(physical_size,4))
	plt.close(fig2)
	print ('Physical size bins:')
	print (bins)
	print (n)
	
	physical_size = np.asarray(tdata['physical_size_thesis']) 

	a = dict()
	for i in range(len(bins)-1): 
		# select from tdata with the bins from tdata_bs
		a[str(i)] = tdata[(bins[i]<physical_size)&(physical_size<bins[i+1])]		
		# print ('Number in bin %i:'%i,len(a[str(i)]))

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selection_physical_size_VACUTS_%i.fits'%i,overwrite=True)
	return a

def select_flux_bins1(tdata,Write=False):
	"""
	Makes 4 flux bins from the tdata, equal freq.

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'3':bin5) -- 4 tables 
											containing the selected sources in each bin
	"""
	
	try:
		flux = np.asarray(tdata['Peak_flux_2']) 
	except KeyError:
		print ('using Peak_flux')
		flux = np.asarray(tdata['Peak_flux'])

	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	n, bins, patches = ax.hist(flux,histedges_equalN(flux,4))
	plt.close(fig2)
	print ('Flux bins:',bins)
	
	a = dict()
	for i in range(len(bins)-1): 
		a[str(i)] = tdata[(bins[i]<flux)&(flux<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selectionDFSDG_%i.fits'%i,overwrite=True)
	return a

def select_flux_bins11(tdata,Write=False):
	"""
	Selects a flux bin where all sources have Peak_flux > 11
	"""

	try:
		flux = np.asarray(tdata['Peak_flux_2']) 
	except KeyError:
		print ('using Peak_flux')
		flux = np.asarray(tdata['Peak_flux'])

	tdata = tdata[flux > 11]

	if Write:
		tdata.write('./biggest_selection_flux_bins11.fits')

	return tdata

def select_size_bins1(tdata,Write=False):
	"""
	Makes 4 size bins from the tdata, equal freq.


	[ 0.09576402  0.14808477  0.24311793  0.36977309  3.2558575 ] arcmin 

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'3':bin5) -- 4 tables 
											containing the selected sources in each bin
	"""
	try:
		size = np.asarray(tdata['size']) 
	except KeyError:
		print ('using size: Maj*2 or NN_dist')
		MGwhere = np.isnan(tdata['new_NN_RA'])
		NNwhere = np.invert(MGwhere)
		MGsources = tdata[MGwhere]
		NNsources = tdata[NNwhere]

		sizeNN = np.asarray(NNsources['new_NN_distance(arcmin)']) * 60 # to arcsec
		sizeMG = np.asarray(MGsources['Maj'])*2 # in arcsec

		size = np.concatenate((sizeNN,sizeMG))

	fig3 = plt.figure(3)
	ax = fig3.add_subplot(111)
	n, bins, patches = ax.hist(size,histedges_equalN(size,4))
	plt.close(fig3)
	# print bins/60.
	print (n)
	print ('Size bins (arcmin):',bins/60)

	
	a = dict()
	for i in range(len(bins)-1): 
		a[str(i)] = tdata[(bins[i]<size)&(size<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./biggest_selection_SIZE%i.fits'%i,overwrite=True)
	return a

def PA_vs_size(tdata):
	"""See how the size influences the PA distribution"""
	MG = np.isnan(tdata['new_NN_RA'])

	dataNN = tdata[np.invert(MG)]
	dataMG = tdata[MG]

	NNsize = np.asarray(dataNN['new_NN_distance(arcmin)'] * 60) # arcminutes to arcseconds
	MGsize = np.asarray(dataMG['Maj_1'] * 2) # in arcseconds
	sizes = np.append(NNsize,MGsize)
	
	def plotsize():
		plt.hist(sizes,bins=20)
		plt.xlabel('size (deg)')
		plt.ylabel('count')
		plt.show()

	def plotPA():
		for i in range(0,110,10):
			plt.title("Size > %i"%i)
			plt.hist(tdata['position_angle'][sizes > i],bins=180/5)
			plt.xlabel('PA (deg)')
			plt.ylabel('count')
			plt.show()

	plotPA()

def size_vs_PA(tdata):
	plt.title('Major axis size vs position angle')
	plt.ylabel('Major axis size (arcsec)')
	plt.xlabel('Position angle (degrees)')
	plt.scatter(tdata['position_angle'],tdata['Maj_1'])
	plt.show()

def NN_distance_final(tdata,Write=False):
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

	from scipy.spatial import cKDTree
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
		distance = distanceOnSphere(nearestN[0],nearestN[1],#RA,DEC coordinates of the nearest
								source[0],source[1])*60 #RA,DEC coordinates of the current item
		# print distance/60
		TheResult_distance.append(distance)	

	if Write:
		np.save('./distance_bs_VA',TheResult_distance)
	return TheResult_distance

def fix_more_single_sources():
	tdata_nomatch = '/data1/osinga/value_added_catalog/2876_sources_notmatched.fits'
	tdata_nomatch = Table(fits.open(tdata_nomatch)[1].data) 
	idx_notround = (np.where( (tdata_nomatch['Maj_1'] - 1) > tdata_nomatch['Min_1'])[0])

	not_round = tdata_nomatch[idx_notround]

	not_round.write('/data1/osinga/value_added_catalog/not_round_NN_sources.fits')

def all_sources_number_density():
	all_va = '/data1/osinga/value_added_catalog1_1b/LOFAR_HBA_T1_DR1_merge_ID_optical_v1.1b.fits'
	all_va = Table(fits.open(all_va)[1].data)

	z_indx = np.invert(np.isnan(all_va['z_best']))

	all_va_z = all_va[z_indx]
	all_va_no_z = all_va[np.invert(z_indx)]

	def plot_for_z(cutoff):
		data = all_va_z[all_va_z['z_best'] < cutoff]
		print 'Cutoff: %f | Number of sources: %i' %(cutoff,len(data))

		plt.scatter(data['RA'],data['DEC'],marker='.',linewidths=0.01)
		plt.title('Number density of all sources | z < ' + str(cutoff))
		plt.xlim(160,240)
		plt.ylim(45,58)
		plt.xlabel('Right Ascension (degrees)')
		plt.ylabel('Declination (degrees)')
		plt.gca().set_aspect('equal', adjustable='box')
		plt.show()
	
	plot_for_z(0.2)

def select_redshift_bins(tdata,bins=[0, 0.25, 0.5, 0.75, 1.0],Write=False):
	'''
	Function to make redshift bins from table tdata with redshift 'z_best'

	if bins = [0, 0.25, 0.5, 0.75, 1.0], it will produce the following bins:
	0.00 < z < 0.25
	0.25 < z < 0.50
	0.50 < z < 0.75
	0.75 < z < 1.00
	'''

	print ('Using the following bins: ', bins)

	where_z = np.invert(np.isnan(tdata['z_best']))
	tdata = tdata[where_z]
	z_best = tdata['z_best']

	z_bins = dict()
	for i in range(0,len(bins)-1):
		z_bins[i] = np.where( (z_best > bins[i]) & (z_best < bins[i+1]) )

	tdata_binned = []
	for key in z_bins:
		print ('Bin %i, number of sources: %i'%(key,len(tdata[z_bins[key]])))
		tdata_binned.append(tdata[z_bins[key]])

	if Write:	
		print ('Writing..')
		for i, data in enumerate(tdata_binned):
			print ('Bin %i, number of sources: %i'%(i,len(data)))
			data.write('./'+filename+'z_bin_%i.fits'%i)

	return tdata_binned

def select_power_bins(tdata,Write=False):
	power = np.asarray(tdata['power']) 
	
	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	n, bins, patches = ax.hist(power,histedges_equalN(power,4))
	plt.close(fig2)
	print 'Power bins:'
	print bins
	
	a = dict()
	for i in range(len(bins)-1): 
		a[str(i)] = tdata[(bins[i]<power)&(power<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selection_power_%i.fits'%i,overwrite=True)
	return a

def select_physical_size_bins(tdata,Write=False):
	physical_size = np.asarray(tdata['physical_size']) 
	
	fig2 = plt.figure(3)
	ax = fig2.add_subplot(111)
	n, bins, patches = ax.hist(physical_size,histedges_equalN(physical_size,4))
	plt.close(fig2)
	print 'Physical size bins:'
	print bins
	
	a = dict()
	for i in range(len(bins)-1): 
		a[str(i)] = tdata[(bins[i]<physical_size)&(physical_size<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			# a[str(i)].write('./biggest_selection_flux_bins1_'+str(i)+'.fits',overwrite=True)
			a[str(i)].write('./value_added_biggest_selection_physical_size_%i.fits'%i,overwrite=True)
	return a

def hist_power_or_size(tdata,which):
	"""
	Create histogram of the power or physical size distribution
	"""

	print ('amount of sources in tdata:', len(tdata))
	z_available = np.invert(np.isnan(tdata['z_best']))
	print ('amount of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]

	if which == 'power':
		plt.hist(tdata['power'])
		plt.xlabel('power in mJy*kpc^2')
		plt.ylabel('count')
		plt.show()

	else:
		plt.hist(tdata['physical_size'])
		plt.xlabel('physical size in kpc')
		plt.ylabel('count')
		plt.show()



if __name__ == '__main__':
	print 'Running general_statistics.py..'
	# all_sources_number_density()

	filename = 'value_added_selection'
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)

	hist_power_or_size(tdata,'size')

	# tdata_binned = select_redshift_bins(tdata,Write=True)

else: 
	print 'Importing..'
# PA_vs_size(tdata)
# tdataMG = Table(fits.open('/data1/osinga/value_added_catalog/value_added_biggest_selection_MG.fits')[1].data)
# size_vs_PA(tdataMG)

# tdata = Table(fits.open('/data1/osinga/value_added_catalog/bins/value_added_biggest_selection_MG_3.fits')[1].data)
# a = select_size_bins1(tdata,False)

# tdata = Table(fits.open('/net/reusel/data1/osinga/value_added_catalog/value_added_biggest_selection.fits')[1].data)
# select_flux_bins1(tdata,True)
# select_size_bins1(tdata,True)
# select_flux_bins11(tdata,True)




# dist = NN_distance_final(tdata)



# MG_index = np.isnan(tdata['new_NN_distance(arcmin)'])
# tdata = tdata[MG_index]
# select_flux_bins1(tdata,True)

# Select 4 flux bins.
# select_flux_bins1(tdata,Write=False)

# select > 11 flux
# tdata = select_flux_bins11(tdata,False)


# tdata = Table(fits.open('../source_filtering/biggest_selection_latest.fits')[1].data)
# select_flux_bins11(tdata,Write=True)
