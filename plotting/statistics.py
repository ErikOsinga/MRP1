import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table, join, vstack

from scipy import constants as S
from scipy.spatial import cKDTree
from scipy.stats import norm

from shutil import copy2

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# import seaborn as sns
# sns.set()
# plt.rc('text', usetex=True)

from utils import angular_dispersion_vectorized_n, load_in, distanceOnSphere
from utils_orientation import find_orientationNN, find_orientationMG

import difflib

import pyfits as pf

# import treecorr


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
	"""
	Function to select all sources in the leaves of the tree, so with err < 15 or lobe ratio < 2 etc
	"""

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
	# (mu,sigma) = norm.fit(position_angles)
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
		label='Postion Angle')#, weights = weights) # I believe its not necessary
													# when binwidth = 1
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

def histedges_equalN(x, nbin):
	"""
 	Make nbin equal height bins
 	Call plt.hist(x, histedges_equalN(x,nbin))
	"""
	npt = len(x)
	return np.interp(np.linspace(0,npt,nbin+1),
					np.arange(npt),
					np.sort(x))

def flux_hist(tdata):
	"""
	Makes a histogram of the flux distribution of the sources
	"""

	peak_flux = tdata['Peak_flux']
	print plt.hist(peak_flux,histedges_equalN(peak_flux,3))

	plt.title('Flux distribution | no. of sources: ' + str(len(tdata)))
	plt.xlabel('Peak flux (Jy/beam)')
	plt.ylabel('Count')
	plt.show()

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

def angular_dispersion(tdata,n=20):
	'''
	Calculates and returns the Sn statistic for tdata
	with number of sources n closest to source i
	
	# n = number of sources closest to source i, including itself
	# e.g. n=5 implies 4 nearest neighbours
	# N = number of sources

	Returns Sn, as a float.

	DEPRECATED> NEEDS FIXING. REMOVE ANGLE FROM CALCULATION 
							AND REMOVE SOME MATRICES ALLTOGETHER
	
	'''
	N = len(tdata)
	RAs = np.asarray(tdata['RA'])
	DECs = np.asarray(tdata['DEC'])
	position_angles = np.asarray(tdata['position_angle'])
	angles = 180 # maximize dispersion for an angle between 0 and 180

	#convert RAs and DECs to an array that has following layout: [[x1,y1,z1],[x2,y2,z2],etc]
	x = np.cos(np.radians(RAs)) * np.cos(np.radians(DECs))
	y = np.sin(np.radians(RAs)) * np.cos(np.radians(DECs))
	z = np.sin(np.radians(DECs))
	coordinates = np.vstack((x,y,z)).T

	#make a KDTree for quick NN searching	
	from scipy.spatial import cKDTree
	coordinates_tree = cKDTree(coordinates,leafsize=16)
	
	# for every source: find n closest neighbours, calculate max dispersion
	all_vectorized = [] # array of shape (N,180,n) which contains all angles used for all sources
	position_angles_array = np.zeros((N,n)) # array of shape (N,n) that contains position angles
	thetas = np.array(range(0,angles)).reshape(angles,1) # for checking angle that maximizes dispersion
	for i in range(N):
		index_NN = coordinates_tree.query(coordinates[i],k=n,p=2,n_jobs=-1)[1]
		position_angles_array[i] = position_angles[index_NN] 
		all_vectorized.append(thetas - position_angles_array[i])
	all_vectorized = np.array(all_vectorized)

	assert all_vectorized.shape == (N,angles,n)

	sum_inner_products = np.sum( np.cos(np.radians(2*all_vectorized)), axis=2)

	assert sum_inner_products.shape == (N,angles)

	max_di = 1./n * np.max(sum_inner_products,axis=1) 
	max_theta = np.argmax(sum_inner_products,axis=1)

	assert max_di.shape == (N,) # array of max_di for every source
	assert max_theta.shape == (N,) # array of max_theta for every source

	Sn = 1./N * np.sum(max_di)

	''' This was to check the d_i max function, it produces the same d_i max
		But you do not know the 'mean angle' '''
	# sum_inner_product_cos = 0
	# sum_inner_product_sin = 0
	# for j in indices:
	# 	sum_inner_product_cos += math.cos(2*math.radians(position_angles[j]))
	# 	sum_inner_product_sin += math.sin(2*math.radians(position_angles[j]))
	# di_max = np.abs(max_di - 1./n * (sum_inner_product_cos**2 + sum_inner_product_sin**2)**(1./2))

	return Sn

def Sn_vs_n(tdata):
	print 'See quick_plot.py'

def alot_of_histograms(tdata):
	'''
	Calculate the histograms for n = 30 to n = 80 for the Monte Carlo simulations
	'''

	n_range = range(30,81)
	path = '/net/zegge/data1/osinga/montecarlo_TR/'
	Sn_data = fits.open('/data1/osinga/data/Statistics_results_TR.fits')
	Sn_data = Table(Sn_data[1].data)
	for n in n_range:
		print n
		Sn_montecarlo = fits.open(path+'TR_Sn_monte_carlo_n='+str(n)+'.fits')
		Sn_montecarlo = Table(Sn_montecarlo[1].data)
		Sn_montecarlo = (np.asarray(Sn_montecarlo['Sn']))
		# Sn_data = angular_dispersion(tdata,n=n)
		Sn = Sn_data['Sn_data'][Sn_data['n'] == n]
		plt.hist(Sn_montecarlo)
		plt.axvline(Sn)
		plt.savefig('/data1/osinga/figures/statistics/Sn_histograms/TR/n='+str(n)+'.png')
		plt.close()

def angular_radius_vs_n(tdata):
	print 'see quick_plot.py'

def select_size_bins(dataNN,dataMG):
	"""
	Makes three size bins from the table tdata, defined as follows

	Bin small: 0 to 20 arcsec.
	Bin medium: 20 to 50 arcsec.
	Bin large: 50+ arcsec.

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	small, medium, large -- Three tables containing the selected sources
	"""
	
	NNsize = np.asarray(dataNN['new_NN_distance(arcmin)'] * 60) # arcminutes to arcseconds
	MGsize = np.asarray(dataMG['Maj'] * 2) # in arcseconds
	# all_sizes = np.append(NNsize,MGsize)

	small = vstack([dataNN[NNsize < 20],dataMG[MGsize < 20]])
	medium = vstack([dataNN[(20<NNsize)&(NNsize<50)],dataMG[(20<MGsize)&(MGsize<50)]])
	large = vstack([dataNN[NNsize>50],dataMG[MGsize>50]])

	assert len(small)+len(medium)+len(large) == len(dataNN) + len(dataMG)

	return small, medium, large
	
def select_size_bins2(dataNN,dataMG,Write=False):
	"""
	Makes 5 size bins from the table tdata, equal frequency.

	[   1.73503268   25.24703724   33.08347987   42.40452224   54.21193253
  	390.70290048]

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'4':bin5) -- 5 tables 
											containing the selected sources in each bin
	"""

	tdata = vstack([dataNN,dataMG])
	NNsize = np.asarray(dataNN['new_NN_distance(arcmin)'] * 60) # arcminutes to arcseconds
	MGsize = np.asarray(dataMG['Maj'] * 2) # in arcseconds
	all_sizes = np.append(NNsize,MGsize)

	n, bins, patches = plt.hist(all_sizes,histedges_equalN(all_sizes,5))
	print bins

	a = dict() #a['0'] is the first bin, etc.
	for i in range(len(bins)-1):
		a[str(i)] = tdata[(bins[i]<all_sizes)&(all_sizes<bins[i+1])]


	if Write:
		for i in range(len(a)):
			a[str(i)].write('./size_bins2_'+str(i)+'.fits',overwrite=True)

	return a

def select_flux_bins2(tdata,Write=False):
	"""
	Makes 5 flux bins from the table tdata, equal frequency.

	[  7.22352897e-02   8.83232569e-01   1.61426015e+00   3.27119503e+00
 	  1.13179333e+01   4.82002807e+03]

	Arguments:
	tdata -- Astropy Table containing the data

	Returns:
	A dictionary {'0':bin1, ...,'4':bin5) -- 5 tables 
											containing the selected sources in each bin
	"""

	peak_flux = tdata['Peak_flux']
	n , bins , patches = plt.hist(peak_flux,histedges_equalN(peak_flux,5))
	print bins

	a = dict() #a['0'] is the first bin, etc.
	for i in range(len(bins)-1):
		a[str(i)] = tdata[(bins[i]<peak_flux)&(peak_flux<bins[i+1])]

	if Write:	
		for i in range(len(a)):
			a[str(i)].write('./flux_bins2_'+str(i)+'.fits',overwrite=True)

	return a
	
def select_size_bins3(dataMG,Write=False):
	"""
	Makes 5 size bins from the dataMG, equal freq.

	Arguments:
	dataMG -- Astropy Table containing the data for the MG sources

	Returns:
	A dictionary {'0':bin1, ...,'4':bin5) -- 5 tables 
											containing the selected sources in each bin
	"""
	MGsize = np.asarray(dataMG['Maj'] * 2) # in arcseconds
	
	n, bins, patches = plt.hist(MGsize,histedges_equalN(MGsize,5))
	print bins
	
	a = dict()
	for i in range(len(bins)-1): 
		a[str(i)] = dataMG[(bins[i]<MGsize)&(MGsize<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			a[str(i)].write('./size_bins3_'+str(i)+'.fits')

	return a

def select_flux_bins3(dataMG,Write=False):
	"""
	Makes 5 flux bins from the dataMG, equal freq.

	Arguments:
	dataMG -- Astropy Table containing the data for the MG sources

	Returns:
	A dictionary {'0':bin1, ...,'4':bin5) -- 5 tables 
											containing the selected sources in each bin
	"""
	MGflux = np.asarray(dataMG['Peak_flux']) 
	
	n, bins, patches = plt.hist(MGflux,histedges_equalN(MGflux,5))
	print bins
	
	a = dict()
	for i in range(len(bins)-1): 
		a[str(i)] = dataMG[(bins[i]<MGflux)&(MGflux<bins[i+1])]		

	if Write:
		for i in range(len(a)):
			a[str(i)].write('./flux_bins3_'+str(i)+'.fits')

	return a

def show_overlap(tdata):
	"""
	Function to show overlapping sources due to selection from NN and then 
	selection from MG, some MG sources might be also in NN.
	"""

	source_names, counts = np.unique(tdata['Source_Name'],return_counts=True)
	# array with the non-unique source names
	source_names = source_names[counts>1] 

	Source_Data = '/data1/osinga/data/biggest_selection.fits'
	Source_Name, Mosaic_ID = load_in(Source_Data,'Source_Name', 'Mosaic_ID')
	RA, DEC, NN_RA, NN_DEC, NN_dist, Total_flux, E_Total_flux, new_NN_index, Maj, Min = (
		load_in(Source_Data,'RA','DEC','new_NN_RA','new_NN_DEC','new_NN_distance(arcmin)'
			,'Total_flux', 'E_Total_flux','new_NN_index','Maj', 'Min') )
	

	for i in range(len(source_names)):
		nonunique = np.where(tdata['Source_Name'] == source_names[i])
		i = nonunique[0][0] # NN source 
		j = nonunique[0][1] # MG source

		# workaround since the string is cutoff after 8 characters...
		MosaicID = difflib.get_close_matches(Mosaic_ID[i],FieldNames,n=1)[0]
		# check to see if difflib got the right string		
		trying = 1
		while MosaicID[:8] != Mosaic_ID[i]:
			trying +=1
			MosaicID = difflib.get_close_matches(Mosaic_ID[i],FieldNames,n=trying)[trying-1]

		source = '/disks/paradata/shimwell/LoTSS-DR1/mosaic-April2017/all-made-maps/mosaics/'+MosaicID+'/mosaic.fits'
		head = pf.getheader(source)
		hdulist = pf.open(source)
		print Source_Name[i]
		print Source_Name[j]

		find_orientationNN(i,'',RA[i],DEC[i],NN_RA[i],NN_DEC[i],NN_dist[i],Maj[i],(3/60.),plot=True,head=head,hdulist=hdulist)

		find_orientationMG(j,'',RA[j],DEC[j],Maj[j],Min[j],(3/60.),plot=True,head=head,hdulist=hdulist)

def deal_with_overlap(tdata):
	"""
	If there are non-unique sources in tdata, remove the NN source from the data.

	Non-unique sources show up because we select first on NN and then on the MG 
	but this might cause overlap between the two..
	"""

	source_names, counts = np.unique(tdata['Source_Name'],return_counts=True)
	# array with the non-unique source names
	source_names = source_names[counts>1] 
	drop_rows = []
	for i in range(len(source_names)):
		nonunique = np.where(tdata['Source_Name'] == source_names[i])
		i = nonunique[0][0] # NN source 
		j = nonunique[0][1] # MG source
		drop_rows.append(i)

	tdata.remove_rows(drop_rows)

	return tdata

def make_bins():
	"""
	Calls the select_size_bins2 and select_flux_bins2 and produces the .fits files
	Deals with overlap. 
	"""
	dataNN,dataMG = select_all_interesting()
	tdata = vstack([dataNN,dataMG])

	a = select_size_bins2(dataNN,dataMG,Write=True)
	a = select_flux_bins2(tdata,Write=True)

	for i in range(5):
		size_data = Table(fits.open('./size_bins2_'+str(i)+'.fits')[1].data)
		flux_data = Table(fits.open('./flux_bins2_'+str(i)+'.fits')[1].data)

		size_data = deal_with_overlap(size_data)
		flux_data = deal_with_overlap(flux_data)

		size_data.write('./size_bins2_'+str(i)+'.fits',overwrite=True)
		flux_data.write('./flux_bins2_'+str(i)+'.fits',overwrite=True)




dataNN,dataMG = select_all_interesting()
tdata = vstack([dataNN,dataMG])

# file = '/data1/osinga/data/flux_bins2/flux_bins2_4.fits'  #'/data1/osinga/data/biggest_selection.fits'
# fluxdata = Table(fits.open(file)[1].data)

tdata2 = deal_with_overlap(tdata)

# make_bins()

# select_size_bins3(dataMG,)
# select_flux_bins3(dataMG,)

# KuiperTest(dataMG)

# flux_hist(tdata)


# small, medium, large = select_size_bins(dataNN,dataMG)
# small.write('./small_selection.fits')
# medium.write('./medium_selection.fits')
# large.write('./large_selection.fits')


# print angular_dispersion(tdata)
# alot_of_histograms(tdata)

# sdata = select_on_size(tdata,cutoff=30)





# angular_size(dataNN,dataMG)

# hist_PA(tdata,normed=False,fit=False)

# print copy_cut_NN()
# stats_NN()

# print copy_cut_multi_gauss()

# for plotting a histogram of NN distance
# plot_hist()