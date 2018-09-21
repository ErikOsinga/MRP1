import sys

import numpy as np 
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

from scipy.stats import norm
from scipy.spatial import cKDTree

from astropy.io import fits
from astropy.table import Table, join, vstack

sys.path.insert(0,'/data1/osinga/anaconda2')
from utils import angular_dispersion_vectorized_n, distanceOnSphere

from general_statistics import select_flux_bins1, select_size_bins1, select_flux_bins11


def tick_function(X):
	return ["%.2f" % z for z in X]

def plot_additional_info(tdata,filename,redshift):
	# some additional info
	if 'size' in filename and not redshift: # Only have to plot this once
		plt.plot((0,0),(0,0),c='white',label='Min size %i arsec'%np.min(tdata['size']))
		plt.plot((0,0),(0,0),c='white',label='Max size %i arcsec'%np.max(tdata['size']))
	if 'flux' in filename and not redshift: # Only have to plot this once
		plt.plot((0,0),(0,0),c='white',label='Min flux %i mJy'%np.min(tdata['Peak_flux_2']))
		plt.plot((0,0),(0,0),c='white',label='Max flux %i mJy'%np.max(tdata['Peak_flux_2']))

	plt.legend()


def Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,ending_n=180,load=True,redshift=False):
	"""
	Make a plot of the SL (Significance level) statistic vs n.
	Makes only the upper plot, unlike statistics_mc.py
	"""

	starting_n = 1 # In the data file the S_1 is the first column
	n_range = range(0,ending_n) # index 0 corresponds to Sn_1 

	all_sn = []
	all_sn_mc = []
	all_sl = [] # (length n list containing SL_1 to SL_80)
	all_std = [] # contains the standard deviations of the MC simulations
	N = len(tdata)
	jain_sigma = (0.33/N)**0.5

	# Whether to calculate or load the data.
	if not load:
		for n in n_range:
			# print 'Now doing n = ', n+starting_n , '...'  
			Sn_mc_n = np.asarray(Sn_mc['S_'+str(n+starting_n)])
			av_Sn_mc_n = np.mean(Sn_mc_n)
			sigma = np.std(Sn_mc_n)

			Sn_data_n = Sn_data[n]
			# print Sn_data_n, av_Sn_mc_n
			SL = 1 - norm.cdf(   (Sn_data_n - av_Sn_mc_n) / (sigma)   )
			all_sn.append(Sn_data_n)
			all_sl.append(SL)
			all_sn_mc.append(av_Sn_mc_n)
			all_std.append(sigma)
		
		Results = Table()
		Results['n'] = range(starting_n,ending_n+1)
		Results['Sn_data'] = all_sn
		Results['SL'] = all_sl
		Results['Sn_mc'] = all_sn_mc
		Results['angular_radius'] = angular_radius
		Results['sigma_S_n'] = all_std
		Results['jain_sigma'] = [jain_sigma]

	else: #(load == True)
		Results = Table(fits.open('./data/Statistics_'+filename+'_results.fits')[1].data)
		all_sl = Results['SL']

	if redshift:
		# add comma because a tuple is always returned
		line, = plt.plot(Results['n'],np.log(all_sl),c='r',label='redshift')
	else:
		line, = plt.plot(Results['n'],np.log(all_sl),c='b',label='no redshift')

	plt.legend()

	# print np.argmin(all_sl)
	if redshift:
		plt.title('SL vs n for '+filename+'\n\n')
		plt.ylabel('Log_e SL')
		plt.xlim(2,180)
		plt.ylim(-4,0)
		plt.xlabel('n')

	if angular_radius is not None and redshift:
		ax1 = plt.gca()
		ax2 = plt.twiny()
		ax2.set_xlabel('angular_radius (degrees)')
		xticks = ax1.get_xticks()
		ax2.set_xticks(xticks)
		print (np.asarray(xticks,dtype='int'))
		xticks2 = np.append(0,angular_radius)[np.asarray(xticks,dtype='int')]
		ax2.set_xticklabels(tick_function(xticks2))
	plt.subplots_adjust(top=0.850)
	# plt.gca().set_aspect('equal', adjustable='datalim')

	return line

def angular_radius_vs_n(tdata,filename,n=180,starting_n=2):
	"""
	Calculate the angular separation vs the amount of neighbours n
	Does not make a plot, unlike the function in statistics_mc.py
	"""

	try: # if saved, no need to calculate again.
		median = np.load('./data/angular_separation_'+filename+'.npy')
	except IOError: 
		print ('calculating angular_separation..')
		n_range = range(starting_n,n+1) 

		N = len(tdata)
		RAs = np.asarray(tdata['RA'])
		DECs = np.asarray(tdata['DEC'])
		position_angles = np.asarray(tdata['final_PA'])

		#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
		x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
		y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
		z = np.sin(np.radians(DECs))
		coordinates = np.vstack((x,y,z)).T
		
		#make a KDTree for quick NN searching	
		coordinates_tree = cKDTree(coordinates,leafsize=16)

		# for every source: find n closest neighbours, calculate median dist to furthest
		furthest_nearestN = np.zeros((N,n),dtype='int') # array of shape (N,n) that contains index of furthest_nearestN
		for i in range(N):
			indices = coordinates_tree.query(coordinates[i],k=n,p=2)[1] # include source itself
			furthest_nearestN[i] = indices
			# source = [RAs[i],DECs[i]]

		temp = np.arange(0,N).reshape(N,1) - np.zeros(n,dtype='int') # Array e.g.
			# to make sure we calc the distance between				#[[0,0,0,0],
			# the current source and the neighbours					#[[1,1,1,1]]
		distance = distanceOnSphere(RAs[temp],DECs[temp]
							,RAs[furthest_nearestN],DECs[furthest_nearestN])
		
		median = np.median(distance,axis=0)
		# std = np.std(distance,axis=0)

		assert median.shape == (n,) # array of the median of (the distance to the furthest 
									# neighbour n for all sources), for every n. 

		np.save('./data/angular_separation_'+filename,median)

	return median

def statistics(filename,tdata,load=True,redshift=False):
	Sn_mc = Table(fits.open('./data/Sn_monte_carlo_'+filename+'.fits')[1].data)

	# calculate angular radius for number of nn (does not include redshift obviously)
	angular_radius = angular_radius_vs_n(tdata,filename,n)
	if redshift: # same number of sources, so only have to plot this once
		plt.plot((-10,-10),(-10,-10),c='white',label='Num_sources %i'%len(tdata))
	Sn_data = angular_dispersion_vectorized_n(tdata,n,redshift)
	line = Sn_vs_n(tdata,Sn_mc,Sn_data,filename,angular_radius,n,load=load,redshift=redshift)
	plot_additional_info(tdata,filename,redshift)

	return line

n = 180 

def take_redshift_sources(tdata):
	z_available = np.invert(np.isnan(tdata['z_best']))
	return tdata[z_available]

def merge_figures_z_and_no_z(load=True):
	"""
	Function to merge all individual SL plots, but plot the redshift as a red line
	and the non-redshift as a blue line in the same plot!
	"""
	print ("""Script only makes sense if the redshift and no redshift have the same 
		amount of sources""")

	def produce_subset_figures(starting_i,filename,load=True):
		'''
		Produces an array of 2x5 figures starting at starting_i, corresponding
		to a subset with a certain filename
		'''

		# Load the subset
		print (filename)
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		tdata['RA'] = tdata['RA_2']
		tdata['DEC'] = tdata['DEC_2']
		filename_original = filename
		tdata_original = tdata
		
		# Use only the sources that actually have a redshift
		tdata = take_redshift_sources(tdata)

		# Make the first figure of the subset
		i = 1 + starting_i
		fig.add_subplot(rows,columns,i)
		# for no redshift 
		filename_original_no_z = filename_original + '_only_redshift_sources_' 
		statistics(filename_original_no_z,tdata,load=load,redshift=False)
		
		# and for redshift
		filename += '_redshift_'
		filename_original_redshift = filename
		statistics(filename_original_redshift,tdata,load=load,redshift=True)
		
		# Make the size bins
		sizebins = select_size_bins1(tdata_original)
		for key in sorted(sizebins):
			sizebin_current = sizebins[key]

			i += 1 # increment the plot
			fig.add_subplot(rows,columns,i)
			filename = filename_original_no_z + 'size%s'%key
			statistics(filename,sizebin_current,load=load,redshift=False)

			sizebin_current = take_redshift_sources(sizebin_current)
			filename = filename_original_redshift + 'size%s'%key
			statistics(filename,sizebin_current,load=load,redshift=True)

		# Make the flux 11 mJy cutoff
		i += 1
		fig.add_subplot(rows,columns,i)
		fluxbins11 = select_flux_bins11(tdata_original)
		filename = filename_original_no_z + 'flux11'
		statistics(filename,fluxbins11,load=load,redshift=False)
		
		fluxbins11 = take_redshift_sources(fluxbins11)
		filename = filename_original_redshift + 'flux11'
		statistics(filename,fluxbins11,load=load,redshift=True)


		# Make the highest flux bins
		fluxbins = select_flux_bins1(tdata_original)
		for key in sorted(fluxbins):
			fluxbin_current = fluxbins[key]
			filename = filename_original_no_z + 'flux%s'%key

			i += 1 
			fig.add_subplot(rows,columns,i)
			line_noz = statistics(filename,fluxbin_current,load=load,redshift=False)
		
			fluxbin_current = take_redshift_sources(fluxbin_current)
			filename = filename_original_redshift + 'flux%s'%key
			linewithz = statistics(filename,fluxbin_current,load=load,redshift=True)

		return linewithz, line_noz


	# amount of columns and rows of plots
	columns = 5
	rows = 8
	my_dpi = 96 # failed attempt at high res.
	# fig = plt.figure(figsize=(1920/my_dpi,1080/my_dpi))
	fig = plt.figure(figsize=(40,30)) # Big figures resolve the problem of small individual figs
	
	# First subset
	filename = 'value_added_selection'
	produce_subset_figures(0,filename,load=load)
	
	# Second subset
	filename = 'value_added_selection_MG'
	produce_subset_figures(10,filename,load=load)

	# Third subset
	filename = 'value_added_compmatch'
	produce_subset_figures(20,filename,load=load)

	# Fourth subset
	filename = 'value_added_compmatch_plus_notround'
	linewithz, line_noz = produce_subset_figures(30,filename,load=load)
	
	# plt.figlegend((linewithz,line_noz),('SL with redshift','SL without redshift'),loc='best' )
	
	plt.tight_layout()
	# plt.show()
	print ("Saving figure..")
	plt.savefig('./test.png')
	print ("Figure saved as ./test.png")


# merge_figures(load=True,redshift=False)
merge_figures_z_and_no_z(load=True)


