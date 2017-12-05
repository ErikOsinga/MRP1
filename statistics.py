import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 

from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import constants as S
from astropy.table import Table, join, vstack
from astropy.io import ascii
from shutil import copy2
from scipy.stats import norm
import matplotlib.mlab as mlab

import seaborn as sns
sns.set()
plt.rc('text', usetex=True)

# import treecorr
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
	filename2 = '/data1/osinga/data/NN/'+file+'NearestNeighbours_efficient_spherical2.fits'
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

def stats_multi_gauss():
	'''
	Simple function to check columns and lay restrictions on them
	'''
	# data = ascii.read('/data1/osinga/data/all_multiple_gaussians1_filtering_confidence.csv',format='csv')
	data = fits.open('/data1/osinga/data/all_multiple_gaussians2.fits')
	data = Table(data[1].data)
	# data = data[ data['classification'] == 'Small err']
	# print data
	# print np.where( (0.5 <= (data['lobe_ratio'])) & ( (data['lobe_ratio']) <= 2.0 ))
	# print len(np.where( (data['amount_of_maxima'] > 2) & (data['classification'] == 'Small err' ) & (data['lobe_ratio'] > 1./2.) & (data['lobe_ratio'] > 2. ) )  [0])
	print len((np.where( (data['amount_of_maxima'] > 2) & (data['err_orientation'] < 15)    )[0]))
# stats()

def stats_NN():
	'''
	Used to print the amount of sources in every branch of the decision tree
	'''
	# data = ascii.read('/data1/osinga/data/all_multiple_gaussians1_filtering_confidence.csv',format='csv')

	def tree():
		data = fits.open('/data1/osinga/data/NN/try_4/all_NN_sources.fits')
		data = Table(data[1].data)
		data2 = fits.open('/data1/osinga/data/NN/'+'P11Hetdex12'+'NearestNeighbours_efficient_spherical2_mutuals.fits')
		data2 = Table(data2[1].data)
		
		single_gauss = 0
		one_multi_gauss = 0
		two_multi_gauss = 0
		c_gauss = 0
		single_gauss_1maxima = 0
		single_gauss_2maxima = 0
		single_gauss_moremaxima = 0
		lobe_ratio_small_two_maxima = 0
		lobe_ratio_big_two_maxima = 0
		lobe_ratio_small_more_maxima = 0
		lobe_ratio_big_more_maxima = 0
		err_orientation_small_two_maxima = 0

		one_multi_gauss_1maxima = 0
		one_multi_gauss_2maxima = 0
		two_multi_gauss_1maxima = 0
		two_multi_gauss_2maxima = 0
		more_multi_gauss_moremaxima = 0

		err_orientation_total = 0 
		count_total = 0

		prev_mosaic_ID = data['Mosaic_ID'][0]
		mosaic_index = 0
		for i in range(0,len(data)):
			NN_index = data['new_NN_index'][i]
			Mosaic_ID = data['Mosaic_ID'][i]
			if Mosaic_ID != prev_mosaic_ID: # next mosaic in FieldNames
				prev_mosaic_ID = data['Mosaic_ID'][i]
				mosaic_index +=1
				fieldname = FieldNames[mosaic_index]
				print i,fieldname
				data2 = fits.open('/data1/osinga/data/NN/'+fieldname+'NearestNeighbours_efficient_spherical2_mutuals.fits')
				data2 = Table(data2[1].data)


			# ----------------------- #
			# 2 single gauss branch
			if data['S_Code'][i] == 'S' and data2['S_Code'][NN_index] == 'S':
				single_gauss += 1

				# 1 maximum branch
				if data['amount_of_maxima'][i] == 1:
					single_gauss_1maxima +=1

				# 2 maxima branch
				elif data['amount_of_maxima'][i] == 2:
					single_gauss_2maxima +=1
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						lobe_ratio_small_two_maxima +=1
						err_orientation_total += data['err_orientation'][i]
						count_total += 1
					else:
						lobe_ratio_big_two_maxima +=1

				# more_maxima branch
				elif data['amount_of_maxima'][i] > 2:
					single_gauss_moremaxima +=1
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						lobe_ratio_small_more_maxima +=1
					else:
						lobe_ratio_big_more_maxima +=1


			# ----------------------- #
			# one multi gauss branch
			elif ( (data['S_Code'][i] == 'S' and data2['S_Code'][NN_index] == 'M') or 
					(data['S_Code'][i] == 'M' and data2['S_Code'][NN_index] == 'S' ) ):
				one_multi_gauss +=1

			
			# ----------------------- #
			# 2 multi gauss branch
			elif ( (data['S_Code'][i] == 'M' and data2['S_Code'][NN_index] == 'M') ):
				two_multi_gauss +=1
			

			# ----------------------- #
			# >=1 C gauss branch
			else:
				c_gauss +=1

		print single_gauss, one_multi_gauss,two_multi_gauss


	# tree()

	# print '-----------------------------------------------------------|---------------------------------------------------'
	# print '------------------------|----------------------------------|----------------------------------|----------------------------------'
	# print '                        |                                  |                                  |'
	# print '                2single_gauss                        >=1 multi gaus                       >=1 c gauss              '
	# print '                       ',2,'                                ',1,'                                ',1
	# print '                        |                                  |                                  |                         '
	# print '             |----------|--------|                |--------|-----------|            |----------|---------|           '
	# print '             |          |        |                |                    |            |          |         |    '
	# print '           2 max      1 max     >=2 max         2 multi             1 multi        1 max     2 max     >= 2 max         '
	# print '                                                                                                                    '
	# print '             |          |        |                |                    |                       |         '
	# print '             |          |        |                |                    |                       |         '
	# print '          |--|--|             |--|--|      |------|-----|       |------|------|         |------|------|     '
	# print '       lobe<2 lobe>2        lobe<2 lobe>2 2max   1max >=2      2max   2max   >=2      lobe<2         lobe>2'         
	# print '                                                                                                                             ' 
	# print '          |                   |            |            |       |      |      |         |             | '
	# print '      |---|---|           |---|---|     |--|--|      |--|--| |--|--| |-|-|  |-|-|    |--|--|'

	def count_mutuals():
		for field in FieldNames:
			# open the file with the mutuals and singles
			data2 = fits.open('/data1/osinga/data/NN/'+field+'NearestNeighbours_efficient_spherical2_mutuals.fits')
			data2 = Table(data[1].data)
			for i in range (0,len(data2)):
				NN_index = data2['new_NN_index'][i]
				# check if your nearest neighbour has you as nearest neighbour as well.
				if data2['new_NN_index'][NN_index] == i:
					count +=1
		print count/2 # might be counting double


	# print len((np.where( (data['S_Code'] == 'S' ) 
		    # )[0]))
# stats_NN()

def copy_cut_multi_gauss():
	'''
	For copying all the sources that make a defined cut, see the tree in my notebook.
	''' 

	two_maxima_check = 0 # for counting the amount of sources that make the cut
	more_maxima_check = 0 

	for name in FieldNames:
		print name
		# multiple gaussians with their properties
		data = fits.open('/data1/osinga/data/'+name+'_multiple_gaussians2.fits')
		data = Table(data[1].data)
		# original PYBSDF catalog.
		crossdata = fits.open('/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+name+'cat.srl.fits')
		crossdata = Table(crossdata[1].data)
		for i in range(0,len(data)):

			# 2 maxima branch
			if data['amount_of_maxima'][i] == 2:
				if data['classification'][i] == 'Small err':
					if ( 1./2. < (data['lobe_ratio'][i]) < 2. ):
						index = np.where(crossdata['Source_Name'] == data['Source_Name'][i])[0][0] # gives the index of the source in the original data
																									# which is the index used in the filename
						if crossdata['Min'][index] < 15: # Minor axis smaller than 15
							two_maxima_check +=1
							copy2('/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/2_maxima/small_err/'+name+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/selected2/2_maxima/small/')
						else: # Minor axis bigger than 15
							copy2('/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/2_maxima/small_err/'+name+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/selected2/2_maxima/big')

			# more maxima branch
			elif data['amount_of_maxima'][i] > 2:
				index = np.where(crossdata['Source_Name'] == data['Source_Name'][i])[0][0] # gives the index of the source in the original data
																							# which is the index used in the filename
				if data['err_orientation'][i] < 15:
					# instant inclusion, check Fratio
					if data['lobe_ratio'][i] < 3:
						copy2('/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/more_maxima/small_err/'+name+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/selected2/more_maxima/instant/fratio_3')
					else:
						copy2('/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/more_maxima/small_err/'+name+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/selected2/more_maxima/instant/fratio_bigger')

					more_maxima_check +=1
				elif data['classification'][i] == 'Small err':
					if ( 1./2. < (data['lobe_ratio'][i]) < 2. ):
						if crossdata['Min'][index] < 10: # Minor axis smaller than 10						
							more_maxima_check +=1
							copy2('/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/more_maxima/small_err/'+name+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/selected2/more_maxima/small/')
						else: # Minor axis bigger than 10
							copy2('/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/more_maxima/small_err/'+name+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/all_multiple_gaussians2/angle_distance/selected2/more_maxima/big/')


	return two_maxima_check, more_maxima_check

def copy_cut_NN():
	'''
	For copying all the sources that make a defined cut, see the tree in my notebook.
	''' 

	# all the source pairs
	data = fits.open('/data1/osinga/data/NN/try_4/all_NN_sources.fits')
	data = Table(data[1].data)
	# all the sources, including singles and mutuals
	data2 = fits.open('/data1/osinga/data/NN/'+'P11Hetdex12'+'NearestNeighbours_efficient_spherical2_mutuals.fits')
	data2 = Table(data2[1].data)
	
	# counters
	single_gauss = 0
	one_multi_gauss = 0
	two_multi_gauss = 0
	c_gauss = 0
	single_gauss_1maxima = 0
	single_gauss_2maxima = 0
	single_gauss_moremaxima = 0
	lobe_ratio_small_two_maxima = 0
	lobe_ratio_big_two_maxima = 0
	lobe_ratio_small_more_maxima = 0
	lobe_ratio_big_more_maxima = 0
	err_orientation_small_two_maxima = 0

	one_multi_gauss_1maxima = 0
	one_multi_gauss_2maxima = 0
	one_multi_gauss_moremaxima = 0
	two_multi_gauss_1maxima = 0
	two_multi_gauss_2maxima = 0
	two_multi_gauss_moremaxima = 0


	# to keep track of the Mosaic ID
	prev_mosaic_ID = data['Mosaic_ID'][0]
	mosaic_index = 0
	fieldname = 'P11Hetdex12'
	for i in range(0,len(data)):
		# check if there was a NaN in the cutout image, then the source has no orientation or hope
		if data['classification'][i] == 'no_hope':
			print 'no hope for source number : ',i
		else:
			index = data['new_Index'][i]
			NN_index = data['new_NN_index'][i]
			Mosaic_ID = data['Mosaic_ID'][i]
			if Mosaic_ID != prev_mosaic_ID: # next mosaic in FieldNames
				prev_mosaic_ID = data['Mosaic_ID'][i]
				mosaic_index +=1
				fieldname = FieldNames[mosaic_index]
				print i,fieldname
				data2 = fits.open('/data1/osinga/data/NN/'+fieldname+'NearestNeighbours_efficient_spherical2_mutuals.fits')
				data2 = Table(data2[1].data)


			# ----------------------- #
			# 2 single gauss branch
			if data['S_Code'][i] == 'S' and data2['S_Code'][NN_index] == 'S':
				single_gauss += 1

				# 1 maximum branch
				if data['amount_of_maxima'][i] == 1:
					single_gauss_1maxima +=1
					copy2('/data1/osinga/figures/cutouts/NN/try_4/1_maximum/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_single_gauss/1_max/')

				# 2 maxima branch
				elif data['amount_of_maxima'][i] == 2:
					single_gauss_2maxima +=1
					# lobe ratio branch
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						lobe_ratio_small_two_maxima +=1
						Min = data['Min'][i]/60. # convert to arcmin
						nn_dist = data['new_NN_distance(arcmin)'][i] 
						cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg

						# err_orientation branch
						if data['err_orientation'][i] < cutoff:
							err_orientation_small_two_maxima += 1
							copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_single_gauss/2_max/lobe_ratio_small/err_orientation_small/')
						else:
							copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_single_gauss/2_max/lobe_ratio_small/err_orientation_big/')

					else:
						lobe_ratio_big_two_maxima +=1
						copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_single_gauss/2_max/lobe_ratio_big/')


				# more_maxima branch
				elif data['amount_of_maxima'][i] > 2:
					single_gauss_moremaxima +=1
					# lobe ratio branch
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						lobe_ratio_small_more_maxima +=1
						Min = data['Min'][i]/60. # convert to arcmin
						nn_dist = data['new_NN_distance(arcmin)'][i] 
						cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg

						# err_orientation branch
						if data['err_orientation'][i] < cutoff:
							err_orientation_small_two_maxima += 1
							copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_single_gauss/more_max/lobe_ratio_small/err_orientation_small/')
						else:
							copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_single_gauss/more_max/lobe_ratio_small/err_orientation_big/')


					else:
						lobe_ratio_big_more_maxima +=1
						copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_single_gauss/more_max/lobe_ratio_big/')



			# ----------------------- #
			# one multi gauss branch
			elif ( (data['S_Code'][i] == 'S' and data2['S_Code'][NN_index] == 'M') or 
					(data['S_Code'][i] == 'M' and data2['S_Code'][NN_index] == 'S' ) ):
				one_multi_gauss +=1
				# 1 maximum branch
				if data['amount_of_maxima'][i] == 1:
					one_multi_gauss_1maxima +=1
					copy2('/data1/osinga/figures/cutouts/NN/try_4/1_maximum/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/1_multi_gauss/1_max/')

				# 2 maxima branch
				elif data['amount_of_maxima'][i] == 2:
					one_multi_gauss_2maxima +=1
					# lobe ratio branch
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						Min = data['Min'][i]/60. # convert to arcmin
						nn_dist = data['new_NN_distance(arcmin)'][i] 
						cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg
						# err_orientation branch
						if data['err_orientation'][i] < cutoff:
							err_orientation_small_two_maxima += 1
							copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/1_multi_gauss/2_max/lobe_ratio_small/err_orientation_small/')
						else:
							copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/1_multi_gauss/2_max/lobe_ratio_small/err_orientation_big/')

					else:
						lobe_ratio_big_two_maxima +=1
						copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/1_multi_gauss/2_max/lobe_ratio_big/')


				# more_maxima branch
				elif data['amount_of_maxima'][i] > 2:
					one_multi_gauss_moremaxima +=1
					# lobe ratio branch
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						Min = data['Min'][i]/60. # convert to arcmin
						nn_dist = data['new_NN_distance(arcmin)'][i] 
						cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg
						# err_orientation branch
						if data['err_orientation'][i] < cutoff:
							err_orientation_small_two_maxima += 1
							copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/1_multi_gauss/more_max/lobe_ratio_small/err_orientation_small/')
						else:
							copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/1_multi_gauss/more_max/lobe_ratio_small/err_orientation_big/')

					else:
						lobe_ratio_big_two_maxima +=1
						copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/1_multi_gauss/more_max/lobe_ratio_big/')

				else:
					print i


			
			# ----------------------- #
			# 2 multi gauss branch
			elif ( (data['S_Code'][i] == 'M' and data2['S_Code'][NN_index] == 'M') ):
				two_multi_gauss +=1

				# 1 maximum branch
				if data['amount_of_maxima'][i] == 1:
					two_multi_gauss_1maxima +=1
					copy2('/data1/osinga/figures/cutouts/NN/try_4/1_maximum/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_multi_gauss/1_max/')

				# 2 maxima branch
				elif data['amount_of_maxima'][i] == 2:
					two_multi_gauss_2maxima +=1
					# lobe ratio branch
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						Min = data['Min'][i]/60. # convert to arcmin
						nn_dist = data['new_NN_distance(arcmin)'][i] 
						cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg
						# err_orientation branch
						if data['err_orientation'][i] < cutoff:
							err_orientation_small_two_maxima += 1
							copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_multi_gauss/2_max/lobe_ratio_small/err_orientation_small/')
						else:
							copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_multi_gauss/2_max/lobe_ratio_small/err_orientation_big/')

					else:
						lobe_ratio_big_two_maxima +=1
						copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_multi_gauss/2_max/lobe_ratio_big/')


				# more_maxima branch
				elif data['amount_of_maxima'][i] > 2:
					two_multi_gauss_moremaxima +=1
					# lobe ratio branch
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						Min = data['Min'][i]/60. # convert to arcmin
						nn_dist = data['new_NN_distance(arcmin)'][i] 
						cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg
						# err_orientation branch
						if data['err_orientation'][i] < cutoff:
							err_orientation_small_two_maxima += 1
							copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_multi_gauss/more_max/lobe_ratio_small/err_orientation_small/')
						else:
							copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_multi_gauss/more_max/lobe_ratio_small/err_orientation_big/')

					else:
						lobe_ratio_big_two_maxima +=1
						copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/2_multi_gauss/more_max/lobe_ratio_big/')

				else:
					print i

			# ----------------------- #
			# >=1 C gauss branch
			else:
				c_gauss +=1
				amount_of_maxima = data['amount_of_maxima'][i]
				# more maxima branch
				if amount_of_maxima > 2:
					copy2('/data1/osinga/figures/cutouts/NN/try_4/more_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/c_gauss/more_max/')
					
				# 2 maxima branch
				elif amount_of_maxima == 2:
					# lobe ratio branch
					if ( 1./2. <= data['lobe_ratio'][i] <= 2.):
						copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/c_gauss/2_max/lobe_ratio_small/')
					else:
						copy2('/data1/osinga/figures/cutouts/NN/try_4/2_maxima/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/c_gauss/2_max/lobe_ratio_big/')
				# 1 maximum branch
				else :
					copy2('/data1/osinga/figures/cutouts/NN/try_4/1_maximum/'+fieldname+'src_'+str(index)+'.png','/data1/osinga/figures/cutouts/NN/try_4/selected/c_gauss/1_max/')


		
	print single_gauss, one_multi_gauss,two_multi_gauss, c_gauss
	print '\n'
	print single_gauss_1maxima, single_gauss_2maxima, single_gauss_moremaxima
	print two_multi_gauss_2maxima, two_multi_gauss_1maxima, two_multi_gauss_moremaxima
	print one_multi_gauss_2maxima, one_multi_gauss_1maxima, one_multi_gauss_moremaxima
	print '\n'

	print (single_gauss,one_multi_gauss,two_multi_gauss,c_gauss)
	print (single_gauss_1maxima,single_gauss_2maxima,single_gauss_moremaxima,one_multi_gauss_1maxima,one_multi_gauss_2maxima,one_multi_gauss_moremaxima,
		two_multi_gauss_1maxima,two_multi_gauss_2maxima,two_multi_gauss_moremaxima)
	print (lobe_ratio_small_two_maxima,lobe_ratio_big_two_maxima,lobe_ratio_small_more_maxima,lobe_ratio_big_more_maxima)

def select_all_interesting():
	'''
	Function to select all sources in the leaves of the tree, so with err < 15 or lobe ratio < 2 etc
	'''

	dataNN = fits.open('/data1/osinga/data/NN/try_4/all_NN_sources.fits')
	dataNN = Table(dataNN[1].data)
	dataMG = fits.open('/data1/osinga/data/all_multiple_gaussians2.fits')
	dataMG = Table(dataMG[1].data)

	# calculate the cutoff value for the NN, since it wasnt well defined in the source_filtering
	Min = dataNN['Min']/60. # convert to arcmin
	nn_dist = dataNN['new_NN_distance(arcmin)']
	cutoff = 2*np.arctan(Min/nn_dist) * 180 / np.pi # convert rad to deg

	selectionNN = np.where( (dataNN['amount_of_maxima'] > 1) & (dataNN['lobe_ratio'] < 2) &
		(dataNN['lobe_ratio'] > 1./2) & (dataNN['err_orientation'] < cutoff) & 
		(dataNN['classification'] != 'no_hope') )
	selectionMG = np.where( (dataMG['amount_of_maxima'] > 1) & (dataMG['lobe_ratio'] < 2) &
		(dataMG['lobe_ratio'] > 1./2) & (dataMG['classification'] == 'Small err') )

	selectionNN = dataNN[selectionNN]
	selectionMG = dataMG[selectionMG]

	return selectionNN, selectionMG

def hist_PA(tdata,normed=False,fit=False):
	'''
	Makes a histogram of the position angles of a given table 'tdata' that contains position angles
	which have the key 'position_angle' 
	Prints the mean and std. of the data
	
	if normed=True, produces a normed histogram
	if fit=True, produces the best (gaussian) fit as well.
	'''

	fig, ax = plt.subplots()
	binwidth=20
	nbins = 180/binwidth
	position_angles = tdata['position_angle']
	n, bins, patches = ax.hist(position_angles,bins=nbins,normed=normed,label='Postion Angle')
	# add a normal distribution fit line by computing mean and stdev
	(mu,sigma) = norm.fit(position_angles)
	print 'Mean of the data: {}  \nStandard deviation of the data {} '.format(mu,sigma)
	if fit==True:
		if normed == True:
			y = mlab.normpdf(bins, mu, sigma)
			ax.plot(bins, y, '--', label='Best fit')
		else:
			raise ValueError ("If fit=True please provide normed=True")

	plt.title('Histogram of the position angle | Number of sources: ' + str(len(tdata)) + 
	' | binwidth: ' + str(binwidth) )
	plt.xlabel('Position angle (deg)')
	if normed==True:
		plt.ylabel('Probability')
	else:
		plt.ylabel('Count')
	plt.legend()
	plt.show()	

def uniformCDF(x):
	a, b = 0., 180.
	if (0 <= x.any() <= 180):
		return ( (x-a) / (b-a) )
	return 'error'

def KuiperTest(tdata):
	'''
	Calculates the Kuiper statistic for a distribution of position angles 
	versus the uniform distribution. 
	'''
	n = len(tdata)

	X = tdata['position_angle'] # X = [x1,x2,...,xn]
	# should use cumulative probability for X 
	i = np.arange(n) + 1 # i = (1, 2, ...., n)
	
	# the uniform CDF	
	X = np.arange(180)
	z = uniformCDF(X)
	plt.plot(X,z)

	binwidth=1
	nbins = 180/binwidth
	position_angles = tdata['position_angle']
	# for normalising the histogram, weighing each bin with the number of values
	# weights = np.asarray(np.ones_like(position_angles)/float(len(position_angles)))
	n, bins, patches = plt.hist(position_angles,bins=nbins,normed=True,
		label='Postion Angle')#, weights = weights)
	pdf = n
	cdf = np.cumsum(pdf)
	# print cdf
	plt.clf()

	D_plus = np.max(cdf-z)
	D_minus = np.max(z-cdf)
	V = D_plus + D_minus
	print V

	plt.plot(X,z,label='Uniform distribution')
	plt.plot(cdf, label='Data') 
	plt.title('Comparison of CDFs, kuiper statistic: ' +str(V))
	plt.xlabel('Position angle (degrees)')
	plt.ylabel('Probability')
	plt.legend()
	plt.show()

def angular_size(tdataNN,tdataMG):
	'''
	Makes a histogram of the angular size distribution of sources
	'''
	NNsize = np.asarray(tdataNN['new_NN_distance(arcmin)'] * 60) # arcminutes to arcseconds
	MGsize = np.asarray(tdataMG['Maj'] * 2) # in arcseconds
	all_size = np.append(NNsize,MGsize)

	used_data = all_size

	binwidth =  1 # arcsecond
	nbins = 70/binwidth 
	n, bins, patches = plt.hist(used_data,bins=nbins,normed=False,
		label='Data')#, weights = weights)

	plt.title('Angular size distribution | no. of sources: ' +str(len(used_data)))
	plt.xlabel('Angular size (arcsec)')
	plt.ylabel('Counts')

	# plt.legend()
	plt.show()

	print np.median(n)

def correlation_funct(tdata,kappa=False):
	'''
	Calculates the Correlation Function using TreeCorr
	If kappa=True uses k-k correlation (with k = cos(2*PA))
	Otherwise, uses g-g correlation.
	'''

	# don't forget degrees to radians 
	angles = np.radians(np.asarray(tdata['position_angle']))
	g1 = np.cos(2 * angles) 
	g2 = np.sin(2 * angles)
	nbins = 20
	nsources = len(tdata)

	if kappa:
		k = np.cos(2*angles)
		cat = treecorr.Catalog(ra=tdata['RA'],dec=tdata['DEC'],k=k,ra_units='deg',dec_units='deg')
		kk = treecorr.KKCorrelation(nbins=nbins,min_sep=0.1,max_sep=10,sep_units='deg')
		kk.process(cat)
		npairs = kk.npairs
		print npairs
		print kk.meanr
		print kk.xi
		plt.title(r'Correlation "k-k", Nbins = ' +str(nbins) + ' Nsources = ' + str(nsources) )
		plt.errorbar(kk.meanr,kk.xi)
		plt.plot((0,max(kk.meanr)),(0,0),'r--')
		plt.xlabel('mean r (degrees)')
		plt.ylabel('Correlation')
		# plt.ylim(-0.004,0.008)
		plt.xticks(np.arange(0,max(kk.meanr),1))
		plt.show()

	else: # gamma
		cat = treecorr.Catalog(ra=tdata['RA'],dec=tdata['DEC'],g1=g1,g2=g2,ra_units='deg',dec_units='deg')
		kk = treecorr.GGCorrelation(nbins=nbins,min_sep=0.1,max_sep=10,sep_units='deg')
		kk.process(cat)
		npairs = kk.npairs

		print npairs
		print kk.meanr
		print kk.xip
		plt.title(r'Correlation "$\xi_{tt}$", Nbins = ' +str(nbins) + ' Nsources = ' + str(nsources) )
		plt.errorbar(kk.meanr,kk.xip)
		plt.plot((0,max(kk.meanr)),(0,0),'r--')
		plt.xlabel('mean r (degrees)')
		plt.ylabel('Correlation')
		# plt.ylim(-0.004,0.008)
		plt.xticks(np.arange(0,max(kk.meanr),1))
		plt.show()


		print '----'
		print npairs
		print kk.meanr
		print kk.xip
		plt.title(r'Correlation "$\xi_{\times \times}$", Nbins = ' +str(nbins) + ' Nsources = ' + str(nsources) )
		plt.errorbar(kk.meanr,kk.xim)
		plt.plot((0,max(kk.meanr)),(0,0),'r--')
		plt.xlabel('mean r (degrees)')
		plt.ylabel('Correlation')
		# plt.ylim(-0.004,0.008)
		plt.xticks(np.arange(0,max(kk.meanr),1))
		plt.show()

def random_data(tdata):
	'''
	Makes random data with the same no. of sources in the same area
	Better: just shuffle the array of position angles
	'''
	maxra = np.max(tdata['RA'])
	minra = np.min(tdata['RA'])
	maxdec = np.max(tdata['DEC'])
	mindec = np.min(tdata['DEC'])
	minpa = np.min(tdata['position_angle'])
	maxpa = np.max(tdata['position_angle'])
	length = len(tdata)

	#maybe numpy random shuffle is better

	rdata=Table()
	rdata['RA'] = np.random.randint(minra,maxra,length)
	rdata['DEC'] = np.random.randint(mindec,maxdec,length)
	rdata['position_angle'] = np.random.randint(minpa,maxpa,length)
	return rdata

def inner_product(alpha1,alpha2):
	'''
	Returns the inner product of position angles alpha1 and alpha2
	The inner product is defined in Jain et al. 2014
	
	Assumes input is given in degrees
	+1 indicates parallel -1 indicates perpendicular
	'''
	alpha1, alpha2 = math.radians(alpha1), math.radians(alpha2)
	return math.cos(2*(alpha1-alpha2))

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

def angular_dispersion(tdata,n=20):
	'''
	Calculates and returns the Sn statistic for tdata
	with number of sources n closest to source i
	
	# n = number of sources closest to source i
	# N = number of sources
	
	'''
	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['position_angle'])

	#convert RAs and DECs to an array that has following layout: [[ra1,dec1],[ra2,dec2],etc]
	coordinates = np.vstack((RAs,DECs)).T
	#make a KDTree for quick NN searching	
	import ScipySpatialckdTree
	coordinates_tree = ScipySpatialckdTree.KDTree_plus(coordinates,leafsize=16)

	d_max = 0
	i_max = 0
	# for i-th source (item) find n closest neighbours
	di_max_set = []
	thetha_max_set = []
	for i, item in enumerate(coordinates):
		# stores the indices of the n nearest neighbours as an array (including source i)
		# e.g. the first source has closest n=2 neighbours:  ' array([0, 3918, 3921]) '
		indices=coordinates_tree.query(item,k=n+1,p=3)[1]
		#coordinates of the n closest neighbours
		nearestN = coordinates[indices]
		nearestRAs = nearestN[:,0]
		nearestDECs = nearestN[:,1]
		# distance in arcmin
		# distance = distanceOnSphere(nearestRAs,nearestDECs,#coordinates of the nearest
		# 						item[0],item[1])*60 #coordinates of the current item
		max_di = 0
		max_theta =  0
		# find the angle for which the dispersion is maximized
		for theta in range(0,180):
			sum_inner_product = 0
			for j in indices:
				sum_inner_product += inner_product(theta,position_angles[j])
			dispersion = 1./n * sum_inner_product
			# should maximalize d_i for theta
			if dispersion > max_di:
				max_di = dispersion
				max_theta = theta

		thetha_max_set.append(max_theta)
		di_max_set.append(max_di)

		# sum_inner_product_cos = 0
		# sum_inner_product_sin = 0
		# for j in indices:
		# 	sum_inner_product_cos += math.cos(2*math.radians(position_angles[j]))
		# 	sum_inner_product_sin += math.sin(2*math.radians(position_angles[j]))
		# di_max = np.abs(max_di - 1./n * (sum_inner_product_cos**2 + sum_inner_product_sin**2)**(1./2))

	# Sn measures the average position angle dispersion of the sets containing every source
	# and its n neighbours. If the condition N>>n>>1 is statisfied then Sn is expected to be
	# normally distributed (Contigiani et al.) 
	Sn = 1./N * np.sum(np.asarray(di_max_set))

	#from scipy.stats import norm
	#Sn_mc = fits.open('/data1/osinga/data/monte_carlo_Sn.fits') # n = 20
	#Sn_mc = Table(Sn_mc[1].data)
	#sigma = (0.33/N)**0.5
	#SL = 1 - norm.cdf(   (Sn - np.average(np.asarray(Sn_mc['Sn']))) / sigma   )
	# SL = 1 - cdfnorm( (Sn - Snmc) / sigma )

	return Sn # , SL

def monte_carlo(tdata,totally_random=True):
	'''
	Make 1000 random data sets and calculate the Sn statistic

	If totally_random, then generate new positions and position angles instead
	of shuffeling the position angles among the sources
	'''
	ra = np.asarray(tdata['RA'])
	dec = np.asarray(tdata['DEC'])
	pa = np.asarray(tdata['position_angle'])
	length = len(tdata)

	max_ra = np.max(ra)
	min_ra = np.min(ra)
	max_dec = np.max(dec)
	min_dec = np.min(dec)

	for n in range(15,81):
		Sn_datasets = []
		print 'Starting 1000 monte carlo simulations with n = '+ str(n) + '..'
		for dataset in range(0,1000):
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

			Sn = angular_dispersion(rdata,n=n)
			Sn_datasets.append(Sn)



		Sn_datasets = np.asarray(Sn_datasets)
		temp = Table()
		temp['Sn'] = Sn_datasets
		if totally_random:
			temp.write('./TR_Sn_monte_carlo_n='+str(n)+'.fits',overwrite=True)
		else:
			temp.write('./Sn_monte_carlo_n='+str(n)+'.fits',overwrite=True)

		# print np.average(Sn_datasets)

def Sn_vs_n(tdata):
	path = '/data1/osinga/data/monte_carlo/results/'
	n_range = range(30,81)
	from scipy.stats import norm
	all_sn = []
	all_sn_mc = []
	all_sl = []
	N = len(tdata)
	sigma = (0.33/N)**0.5
	for n in n_range:
		print 'Now doing n = ', n , '...'
		Sn_montecarlo = fits.open(path+'Sn_monte_carlo_n='+str(n)+'.fits')
		Sn_montecarlo = Table(Sn_montecarlo[1].data)
		Sn_montecarlo = np.average(np.asarray(Sn_montecarlo['Sn']))
		Sn_data = angular_dispersion(tdata,n=n)
		SL = 1 - norm.cdf(   (Sn_data - Sn_montecarlo) / (sigma)   )
		all_sn.append(Sn_data)
		all_sl.append(SL)
		all_sn_mc.append(Sn_montecarlo)
	
	Results = Table()
	Results['n'] = n_range
	Results['Sn_data'] = all_sn
	Results['SL'] = all_sl
	Results['Sn_mc'] = all_sn_mc
	Results.write('/data1/osinga/data/Statistics_results.fits',overwrite=True)
	
	plt.plot(n_range,np.log10(all_sl))
	plt.ylabel('Log SL')
	plt.xlabel('n')
	plt.savefig('/data1/osinga/figures/SL_vs_n.png')
	

def alot_of_histograms(tdata):
	n_range = range(30,81)
	path = '/data1/osinga/data/monte_carlo/results/'
	Sn_data = fits.open('/data1/osinga/data/Statistics_results.fits')
	Sn_data = Table(Sn_data[1].data)
	for n in n_range:
		print n
		Sn_montecarlo = fits.open(path+'Sn_monte_carlo_n='+str(n)+'.fits')
		Sn_montecarlo = Table(Sn_montecarlo[1].data)
		Sn_montecarlo = (np.asarray(Sn_montecarlo['Sn']))
		# Sn_data = angular_dispersion(tdata,n=n)
		Sn = Sn_data['Sn_data'][Sn_data['n'] == n]
		plt.hist(Sn_montecarlo)
		plt.axvline(Sn)
		plt.savefig('/data1/osinga/figures/statistics/Sn_histograms/n='+str(n)+'.png')
		plt.close()



dataNN,dataMG = select_all_interesting()
tdata = vstack([dataNN,dataMG])
alot_of_histograms(tdata)



#print angular_dispersion(tdata)





# monte_carlo(tdata)
# correlation_funct(random_data(tdata),kappa=False)

# angular_size(dataNN,dataMG)


# KuiperTest(tdata)
# hist_PA(tdata,normed=False,fit=False)

# print copy_cut_NN()
# stats_NN()

# print copy_cut_multi_gauss()



# for plotting a histogram of NN distance
# plot_hist()





