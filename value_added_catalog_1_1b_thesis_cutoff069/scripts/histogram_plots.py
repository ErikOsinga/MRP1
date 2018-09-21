import sys
sys.path.insert(0, '/net/reusel/data1/osinga/anaconda2')
import numpy as np 
import numexpr as ne
import math

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

from utils import (angular_dispersion_vectorized_n, load_in, distanceOnSphere
	, deal_with_overlap, FieldNames, parallel_transport)

from general_statistics import (select_power_bins_cuts_VA_selection, select_physical_size_bins_cuts_VA_selection
	, select_flux_bins_cuts_biggest_selection,select_size_bins_cuts_biggest_selection)

from astropy import visualization

def kuiper_FPP(D,N):
    """Compute the false positive probability for the Kuiper statistic.
    Uses the set of four formulas described in Paltani 2004; they report 
    the resulting function never underestimates the false positive probability 
    but can be a bit high in the N=40..50 range. (They quote a factor 1.5 at 
    the 1e-7 level.
    Parameters
    ----------
    D : float
        The Kuiper test score.
    N : float
        The effective sample size.
    Returns
    -------
    fpp : float
        The probability of a score this large arising from the null hypothesis.
    Reference
    ---------
    Paltani, S., "Searching for periods in X-ray observations using 
    Kuiper's test. Application to the ROSAT PSPC archive", Astronomy and
    Astrophysics, v.240, p.789-790, 2004.
    """

    from numpy import copy, sort, amax, arange, exp, sqrt, abs, floor, searchsorted
    import itertools


    if D<0. or D>2.:
        raise ValueError("Must have 0<=D<=2 by definition of the Kuiper test")

    if D<2./N:
        return 1. - factorial(N)*(D-1./N)**(N-1)
    elif D<3./N:
        k = -(N*D-1.)/2.
        r = sqrt(k**2 - (N*D-2.)/2.)
        a, b = -k+r, -k-r
        return 1. - factorial(N-1)*(b**(N-1.)*(1.-a)-a**(N-1.)*(1.-b))/float(N)**(N-2)*(b-a)
    elif (D>0.5 and N%2==0) or (D>(N-1.)/(2.*N) and N%2==1):
        def T(t):
            y = D+t/float(N)
            return y**(t-3)*(y**3*N-y**2*t*(3.-2./N)/N-t*(t-1)*(t-2)/float(N)**2)
        s = 0.
        # NOTE: the upper limit of this sum is taken from Stephens 1965
        for t in xrange(int(floor(N*(1-D)))+1):
            term = T(t)*comb(N,t)*(1-D-t/float(N))**(N-t-1)
            s += term
        return s
    else:
        z = D*sqrt(N) 
        S1 = 0.
        term_eps = 1e-12
        abs_eps = 1e-100
        for m in itertools.count(1):
            T1 = 2.*(4.*m**2*z**2-1.)*exp(-2.*m**2*z**2)
            so = S1
            S1 += T1
            if abs(S1-so)/(abs(S1)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        S2 = 0.
        for m in itertools.count(1):
            T2 = m**2*(4.*m**2*z**2-3.)*exp(-2*m**2*z**2)
            so = S2
            S2 += T2
            if abs(S2-so)/(abs(S2)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        return S1 - 8*D/(3.*sqrt(N))*S2

def uniformCDF(x):
	a, b = 0., 180.
	if (0 <= x.any() <= 180):
		return ( (x-a) / (b-a) )
	return 'error'

def KuiperTest(tdata,PT=False):
	'''
	Calculates the Kuiper statistic for a distribution of position angles 
	versus the uniform distribution. 
	'''
	N = len(tdata)

	position_angles = tdata['position_angle'] 
	try:
		RAs, DECs = tdata['RA'], tdata['DEC']
	except KeyError:
		RAs, DECs = tdata['RA_2'], tdata['DEC_2']

	if PT:
		i = 0
		position_angles_PT = parallel_transport(RAs[i],DECs[i],RAs[1:],DECs[1:],position_angles[1:])
		position_angles = np.concatenate(([position_angles[i]],position_angles_PT))

	# should use cumulative probability for X 
	i = np.arange(N) + 1 # i = (1, 2, ...., n)
	
	# the uniform CDF	
	X = np.arange(180)
	z = uniformCDF(X)
	plt.plot(X,z)

	binwidth=1
	nbins = 180/binwidth
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

	FPP = kuiper_FPP(V,N)

	print ('False positive probability:' + str(FPP))

	plt.plot(X,z,label='Uniform distribution')
	plt.plot(cdf, label='Data') 
	plt.title('Comparison of CDFs, kuiper statistic: ' +str(V) + '\n FPP: %.5f'%FPP)
	plt.xlabel('Position angle (degrees)')
	plt.ylabel('Probability')
	plt.legend()
	plt.tight_layout()
	plt.show()

def do_PT(tdata):
	"""
	Transport every source position angle to the first source

	"""
	
	position_angles = tdata['position_angle']
	try:
		RAs, DECs = tdata['RA'], tdata['DEC']
	except KeyError: # not biggest_selection.py
		RAs, DECs = tdata['RA_2'], tdata['DEC_2']
	i = 0
	if not i == 0: raise ValueError('Only works for i = 0')
	print ('Transporting every source to source %i'%i)
	position_angles_PT = parallel_transport(RAs[i],DECs[i],RAs[1:],DECs[1:],position_angles[1:])
	position_angles = np.concatenate(([position_angles[i]],position_angles_PT))

	return position_angles

def initial_subsample_hist(PT=False):
	'''histogram of the position angles of initial subsample'''
	# tdata = '../biggest_selection.fits'
	# tdata = '../value_added_selection.fits'
	tdata = '../value_added_selection_MG.fits'
	# tdata = '../value_added_selection_NN.fits'
	tdata = Table(fits.open(tdata)[1].data)

	if PT:
		position_angles = do_PT(tdata)
	else:
		position_angles = tdata['position_angle']


	plt.hist(position_angles,bins=180/5,histtype=u'step',color='black')
	plt.ylabel('Counts',fontsize=12)
	plt.xlabel('Position angle (degrees)',fontsize=12)
	plt.xlim(-8.9863954872984522,188.91420232532226)
	plt.ylim(0,250)
	plt.tight_layout()
	plt.show()

def initial_subsample_hist_AP(PT=False):
	'''histogram of the position angles of initial subsample using Astropy'''
	tdata = '../biggest_selection.fits'
	tdata = Table(fits.open(tdata)[1].data)

	if PT:
		position_angles = do_PT(tdata)
	else:
		position_angles = tdata['position_angle']


	visualization.hist(position_angles,bins='scott',histtype=u'step',color='black')

	# plt.hist(position_angles,bins=180/5,histtype=u'step',color='black')
	plt.ylabel('Counts',fontsize=12)
	plt.xlabel('Position angle (degrees)',fontsize=12)
	plt.xlim(-8.9863954872984522,188.91420232532226)
	# plt.ylim(0,250)
	plt.tight_layout()
	plt.show()

def all_subsamples_hist(PT=False,redshift=False):
	'''Overlay of all the subsamples and the distribution of their position angles'''
	
	prop_cycle = plt.rcParamsDefault['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	# Initial sample = blue; 
	color_initial_sample = colors[0]
	color_value_added_subset = colors[1]
	color_connected_lobes = colors[2]
	color_isolated_lobes = colors[3]
	color_concluding_subset = colors[4]

	tdata = '../biggest_selection.fits'
	tdata = Table(fits.open(tdata)[1].data)

	bins = 180/5
	# bins = 180/5
	fig = plt.figure(figsize=(8,6.5))

	if PT:
		position_angles = do_PT(tdata)
	else:
		position_angles = tdata['position_angle']

	if not redshift: plt.hist(position_angles,bins=bins,label='Initial sample',color=color_initial_sample)
	# else: tmp = plt.hist([2,1,3],label='Initial sample'); del tmp # for the right color subsamples

	tdata = '../value_added_selection.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	if PT:
		position_angles = do_PT(tdata)
	else:
		position_angles = tdata['position_angle']
	
	plt.hist(position_angles,bins=bins,label='Value-added subset',color=color_value_added_subset)
	

	tdata = '../value_added_compmatch.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	if PT:
		position_angles = do_PT(tdata)
	else:
		position_angles = tdata['position_angle']
	plt.hist(position_angles,bins=bins,label='Concluding subset',color=color_concluding_subset)


	tdata = '../value_added_selection_MG.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	if PT:
		position_angles = do_PT(tdata)
	else:
		position_angles = tdata['position_angle']
	plt.hist(position_angles,bins=bins,label='Connected lobes subset',color=color_connected_lobes)
	
	tdata = '../value_added_selection_NN.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!
	
	if PT:
		position_angles = do_PT(tdata)
	else:
		position_angles = tdata['position_angle']
	
	plt.hist(position_angles,bins=bins,label='Isolated lobes subset',color=color_isolated_lobes)
	

	plt.ylabel('Counts',fontsize=14)
	plt.xlabel('Position angle (degrees)',fontsize=14)
	plt.xlim(-8.9863954872984522,188.91420232532226)
	if not redshift: plt.ylim(0,300)
	else: plt.ylim(0,150)
	plt.legend(fontsize=14)

	plt.tick_params(labelsize=12)
	plt.tight_layout()
	plt.show()

def SN_mc(filename):
	'''
	Plot of histogram of SN_mc (full sample)
	and the data point from data sample
	'''

	if filename == 'biggest_selection':
		# raw SN-mc data
		loc = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/data/Sn_monte_carlo_PT_full_sample_biggest_selection_PAcorrect.fits'
		# SN-data SN-mc(average) and SL
		loc2 = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/full_sample/biggest_selection_PAcorrect_statistics_full_sample.fits'
	elif filename == 'value_added_selection':
		loc = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/data/Sn_monte_carlo_PT_full_sample_value_added_selection_PAcorrect.fits'
		loc2 = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/full_sample/value_added_selection_PAcorrect_statistics_full_sample.fits'
	elif filename == 'value_added_selection_MG':
		loc = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/data/Sn_monte_carlo_PT_full_sample_value_added_selection_MG_PAcorrect.fits'
		loc2 = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/full_sample/value_added_selection_MG_PAcorrect_statistics_full_sample.fits'
	elif filename == 'value_added_compmatch':
		loc = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/data/Sn_monte_carlo_PT_full_sample_value_added_compmatch_PAcorrect.fits'
		loc2 = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/full_sample/value_added_compmatch_PAcorrect_statistics_full_sample.fits'
	elif filename == 'value_added_selection_NN':
		loc = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/data/Sn_monte_carlo_full_sample_value_added_selection_NN_PAcorrect.fits'
		loc2 = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/scripts/full_sample/value_added_selection_NN_PAcorrect_statistics_full_sample.fits'

	tdata = Table(fits.open(loc)[1].data)
	SN_mc = np.asarray(tdata['S_N'])
	sigma = np.std(SN_mc)
	averageSN = np.average(SN_mc)

	tsn_data = Table(fits.open(loc2)[1].data)
	sn_data = tsn_data['Sn_data']
	SL = tsn_data['SL']

	# plt.axvline(x=averageSN,ymin=0,ymax=1, label='Mean')
	plt.hist(SN_mc,bins=100,histtype=u'step',color='black',label='Simulations')
	plt.axvline(x=sn_data,ymin=0,ymax=1, label='Data',ls='dashed',color='k')

	# plt.legend(fontsize=14)
	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	handles = reversed(handles)
	labels = reversed(labels)
	ax.legend(handles, labels,fontsize=14,loc='upper right')

	print ('SN_MC = %f, SN_data = %f, Sigma = %f, SL = %f'%(averageSN,sn_data,sigma,SL))

	plt.xlabel(r'$S_N$',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	
	plt.tight_layout()
	plt.show()

def source_counts(filename):
	''' find how many sources have redshift, returns a boolean array corresponding to indices '''
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)

	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))

	return z_available

def KSTEST(filename,redshift):
	from scipy import stats 
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	position_angles = np.asarray(tdata['position_angle'])

	statistic, p_value = stats.kstest(position_angles,stats.uniform(loc=0.0, scale=180.0).cdf)

	print ('KS test: ', filename)
	print ('Statistic: %f | p value (per cent): %f'%(statistic,p_value*100))

	return p_value*100 # in percent

def print_parameters(tdata,filename):
	'''helper function for subsample_parameters'''
	MG_index = np.isnan(tdata['new_NN_distance(arcmin)']) 
	NN_index = np.invert(MG_index)
	print ('Sample: %s'%filename)
	print ('Number of matched isolated lobes: %i'%np.sum(NN_index))
	print ('Number of connected lobes: %i'%np.sum(NN_index))
	print ('\n')

def subsample_parameters(filename,equal_width=False):
	''' print the number of matched and isolated lobes, and the distribution of PA
		whether it is random according to the KStest etc. '''

	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	filename_original = filename
	tdata_original = tdata

	print_parameters(tdata,filename)

	if equal_width: 
		fluxbins = select_flux_bins1(tdata_original)
	else:
		fluxbins = select_flux_bins_cuts_biggest_selection(tdata_original)

	for key in sorted(fluxbins):
		tdata = fluxbins[key]
		filename = filename_original + 'flux%s'%key

		print_parameters(tdata,filename)

	if equal_width:
		sizebins = select_size_bins1(tdata_original)
	else:
		sizebins = select_size_bins_cuts_biggest_selection(tdata_original)

	for key in sorted(sizebins):
		tdata = sizebins[key]
		filename = filename_original + 'size%s'%key

		print_parameters(tdata,filename)

def plot_hist(filename,normed=False,fit=False):
	'''
	Plots a histogram of the SN montecarlo
	Prints the mean and std.
	if normed=True, produces a normed histogram
	if fit=True, tries to fit sigma=sqrt(0.33/N)
	'''
	from scipy.stats import norm
	import matplotlib.mlab as mlab

	parallel_transport = True
	if parallel_transport:
		Sn_mc = Table(fits.open('/net/reusel/data1/osinga/value_added_catalog_1_1b_thesis/omar_data/Sn_monte_carlo_PT'+filename+'.fits')[1].data)
	else: 
		Sn_mc = Table(fits.open('/net/reusel/data1/osinga/value_added_catalog_1_1b_thesis/omar_data/Sn_monte_carlo_'+filename+'.fits')[1].data)
	
	Sn_mc_35 = np.asarray(Sn_mc['S_35']) 
	
	fig, ax = plt.subplots()
    # histogram of the data
	n, bins, patches = ax.hist(Sn_mc_35,bins=100,normed=normed)

	print ('According to numpy: mean=%f, std=%f'%(np.mean(Sn_mc_35),np.std(Sn_mc_35)))
    # add a 'best fit' line by computing mean and stdev
	(mu,sigma) = norm.fit(Sn_mc_35)
	print ('According to norm fit: mean=%f, std=%f'%(mu,sigma))

	N = 30059
	# add a best fit line by mean and stdev = 0.33/N
	mu_1, sigma_1 = np.mean(Sn_mc_35), (0.33/N)**0.5
	print ('According to Jain formulae: mean=%f, std=%f'%(mu_1,sigma_1))

	if fit==True:
		y = mlab.normpdf(bins, mu, sigma)
		print y
		ax.plot(bins, y, '--', label='Best fit normal distribution')
		y = mlab.normpdf(bins, mu_1, sigma_1)
		ax.plot(bins,y, '--', label='Best fit Jain formulae')
	
	plt.title('Histogram of simulated S_35 values')
	plt.xlabel('S_35')
	plt.ylabel('')
	plt.legend()
	plt.show()	


def get_distance(angle1,angle2):
	"""
	To get distance on a scale where [0,180) confines all angles. 
	So 179 and 2 have a difference of 3 degrees
	"""

	delta = abs(angle1-angle2) 
	if delta > 90:
		if angle1 > angle2:
			delta =  (angle2 + 180) - angle1
		else: 
			delta = (angle1 + 180) - angle2

	return abs(delta)

def plot_PA_vs_position_angle():
	'''
	For the fidelity of the position angles:

	Compare PA_1 (PyBDSF PA) to position_angle (my algorithm)
	for the multi-gaussian sources. 

	'''
	filename = 'value_added_selection_MG'
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)

	PA_1 = tdata['PA_1']
	position_angle = tdata['position_angle']

	cutoff = 10 # degrees

	count_agree = 0
	count_disagree = 0
	for i in range(len(tdata)):
		difference = get_distance(PA_1[i],position_angle[i])
		if difference > cutoff:
			count_disagree +=1
		else:
			count_agree +=1

	print ('Number of PA agreeing within %i degrees: %i'%(cutoff,count_agree))
	print ('Number of PA disagreeing within %i degrees: %i'%(cutoff,count_disagree))
	print ('Percentage agreeing: %i/%i = %f percent'%(count_agree,len(tdata),(count_agree/float(len(tdata))*100)))

	plt.hist(PA_1,bins=180/5,histtype=u'step',label='PyBDSF',alpha=0.5)
	plt.hist(position_angle,bins=180/5,histtype=u'step',label='This study',alpha=0.5)

	plt.legend(fontsize=14)
	plt.xlabel(r'Position angle (degrees)',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	plt.tight_layout()

	plt.show()



	# filename = 'value_added_compmatch'
	# print (filename)
	# tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	# correct_match = tdata[np.invert(np.isnan(tdata['new_NN_RA']))]

	# PA_2 = correct_match['LGZ_PA']
	# position_angle = correct_match['position_angle']

	# plt.hist(PA_2,bins=180/5,histtype=u'step',label='LGZ',alpha=0.5)
	# plt.hist(position_angle,bins=180/5,histtype=u'step',label='This study',alpha=0.5)

	# plt.legend(fontsize=14)
	# plt.xlabel(r'Position angle (degrees)',fontsize=14)
	# plt.ylabel('Counts',fontsize=14)
	# plt.tight_layout()

	# plt.show()

def plot_E_PA_vs_E_position_angle():
	'''
	For the fidelity of the E_position angles:

	Compare E_PA_1 (PyBDSF PA) to  (my algorithm)
	for the multi-gaussian sources. 

	'''
	filename = 'value_added_selection_MG'
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)

	E_PA_1 = tdata['E_PA_1']
	E_position_angle = tdata['err_orientation']

	cutoff = 5 # degrees

	count_agree = 0
	count_disagree = 0
	for i in range(len(tdata)):
		difference = get_distance(E_PA_1[i],E_position_angle[i])
		if difference > cutoff:
			count_disagree +=1
		else:
			count_agree +=1

	# print ('Number of E_PA agreeing within %i degrees: %i'%(cutoff,count_agree))
	# print ('Number of E_PA disagreeing within %i degrees: %i'%(cutoff,count_disagree))
	# print ('Percentage agreeing: %i/%i = %f percent'%(count_agree,len(tdata),(count_agree/float(len(tdata))*100)))

	print ('Number of E_PA < 5 degrees: %i'%(np.sum(E_PA_1 < 5)))
	print ('Which is %i/%i = %f'%(np.sum(E_PA_1 < 5),len(tdata),np.sum(E_PA_1 < 5)/float(len(tdata))))

	# plt.hist(E_PA_1,bins=20,histtype=u'step')#,label='PyBDSF',alpha=0.5)
	# plt.hist(E_position_angle,bins=180/5,histtype=u'step',label='This study',alpha=0.5)

	# Use fancy algorithms to determine the bins
	from astropy import visualization
	visualization.hist(E_PA_1,bins='scott',histtype=u'step',color='black')

	# plt.yscale('log')
	# plt.legend(fontsize=14)
	plt.xlabel(r'1$\sigma$ error (degrees)',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	plt.tight_layout()

	plt.show()

def plot_size_of_horizontal_connected_lobes():
	filename = 'value_added_selection_MG'
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)

	size = tdata['size_thesis']/60.

	# Use fancy algorithms to determine the bins
	from astropy import visualization
	visualization.hist(size,bins='scott',histtype=u'step',color='black',label='All sources')

	where = np.where( (tdata['position_angle'] < 100) & (tdata['position_angle'] > 80) ) 
	size_vertical = size[where]
	visualization.hist(size_vertical,bins='scott',histtype=u'step',label='80<PA<100')


	# plt.yscale('log')
	plt.legend(fontsize=14)
	plt.xlabel(r'Size (arcmin)',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	plt.tight_layout()

	plt.show()

def plot_size_of_vertical_isolated_lobes():
	filename = 'value_added_selection_NN'
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)

	size = tdata['size_thesis']/60.

	# Use fancy algorithms to determine the bins
	from astropy import visualization
	visualization.hist(size,bins='scott',histtype=u'step',color='black',label='All sources')

	where = np.where( (tdata['position_angle'] < 100) & (tdata['position_angle'] > 80) ) 
	size_vertical = size[where]
	visualization.hist(size_vertical,bins='scott',histtype=u'step',label='80<PA<100')


	# plt.yscale('log')
	plt.legend(fontsize=14,loc='upper left')
	plt.xlabel(r'Size (arcmin)',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	plt.tight_layout()

	plt.show()

def find_NN_properties(tdata,arg):
	"""
	Function to return a certain property of a Nearest Neighbour of a source

	tdata -- the table containing the data: Mosaic_ID_2 and 'new_NN_index'
	arg, a string with the requested property (e.g. 'Total_Flux')
	"""

	NN_sources = np.invert(np.isnan(tdata['new_NN_RA']))
	tdata_NN = tdata[NN_sources]

	x = []

	mosaicid_prev = 'initializing'
	for i in range(0,len(tdata)):
		if NN_sources[i]:
			try:
				mosaicid = tdata['Mosaic_ID_2'][i]
			except KeyError: # No Mosaic_ID_2 because no matches yet, so have to use this
				mosaicid = tableMosaic_to_full(tdata['Mosaic_ID'][i])


			if mosaicid != mosaicid_prev: # only open new file if we need to
				NNfile = '../source_filtering/NN/'+mosaicid+'NearestNeighbours_efficient_spherical2.fits'
				NNfile = Table(fits.open(NNfile)[1].data)
				mosaicid_prev = mosaicid
			x.append(NNfile[arg][tdata['new_NN_index'][i]])
		else:
			x.append(np.nan)
	
	return x

def plot_all_sources_MG_PA():
	tdata = Table(fits.open('/data1/osinga/all_MG_sources/all_mg_sources.fits')[1].data)

	PA_1 = tdata['PA_1']

	plt.hist(PA_1,bins=180/5,histtype=u'step',color='k',label='All sources')
	
	# plt.hist(PA_1,bins=180/5,histtype=u'step',label='All sources')

	print ("Remember that Maj is the semi-major axis")
	# PA_3 = PA_1[tdata['Maj_1'] > 15/2. ]

	# plt.hist(PA_3,bins=180/5,histtype=u'step',label='Maj > 15"')

	PA_4 = PA_1[tdata['Maj_1'] > 23/2. ]

	plt.hist(PA_4,bins=180/5,histtype=u'step',label='Maj > 23"')

	cutoff = 15/2. # 15 arcsec major axis = 7/2. arcsec semi-major axis
	PA_2 = PA_1[tdata['Maj_1'] < cutoff]

	plt.hist(PA_2,bins=180/5,histtype=u'step',label='Maj < %i"'%(cutoff*2),alpha=0.5)

	# PA_5 = PA_1[tdata['Maj_1'] < 3.5]

	# plt.hist(PA_2,bins=180/5,histtype=u'step',label='Maj < %i"'%(7),alpha=0.5)

	# plt.hist(PA_5)
	# plt.yscale('log')

	plt.legend(fontsize=14,loc='upper left')
	plt.xlabel(r'Position angle (degrees)',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	plt.tight_layout()
	# plt.ylim(0,175)

	plt.show()


	# #which sources have PA pref 90
	# tdata = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/source_filtering/all_multiple_gaussians.fits'
	# tdata = Table(fits.open(tdata)[1].data)

	# deg90 = tdata[ (tdata['PA'] > 75) & (tdata['PA'] < 125)  ]


	# string = 'Maj'
	# plt.hist(tdata[string],label='All data',bins=180/5,alpha=0.5)
	# plt.hist(deg90[string],label='Sources around 90',bins=180/5,alpha=0.5)
	# plt.legend()
	# # plt.yscale('log')
	# # plt.hist(tdata['PA'])
	# # plt.xlim(-10,10)
	# plt.show()

	# plt.show()

def matched_sources_redshift():

	filename = 'value_added_selection_NN'
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)
	z_available = np.invert(np.isnan(tdata['z_best']))
	tdata = tdata[z_available]

	spectro_where = np.where(tdata['z_source'] == 1)
	spec = tdata[spectro_where]

	position_angle = tdata['position_angle']
	# Use fancy algorithms to determine the bins
	from astropy import visualization
	visualization.hist(position_angle,bins='scott',histtype=u'step',color='black',label='All sources')
	visualization.hist(spec['position_angle'],bins='scott',histtype=u'step',label='Spectro')


	# plt.yscale('log')
	plt.legend(fontsize=14,loc='upper left')
	plt.xlabel(r'Position angle (degrees)',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	plt.tight_layout()
	plt.title('NN only redshift sources')

	plt.show()

def plot_VA_MG_redshift_dist():
	filename = 'value_added_selection_MG'
	print (filename)
	tdata = Table(fits.open('../%s.fits'%filename)[1].data)

	print tdata['z_best_source']
	# With redshift
	print ('Using redshift..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]
	tdata_original = tdata # use redshift data for power bins and size bins !!!
	#########################################  VA selection MG only redshift sources, flux bin 3

	# tdata['RA_2'], tdata['DEC_2'] = tdata['RA'], tdata['DEC']
	tdata['final_PA'] = tdata['position_angle']

	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']

	spectroscopic = np.where(tdata['z_best_source'] == 1)
	photometric = np.where(tdata['z_best_source'] == 0)
	tdata_spec = tdata[spectroscopic]
	tdata_phot = tdata[photometric]
	print (' flux bin 3 VA MG')

	print (len(tdata_spec), len(tdata_phot), len(tdata))

	# Use fancy algorithms to determine the bins
	from astropy import visualization
	# visualization.hist(tdata_phot['z_best'],bins='scott',histtype=u'step',label='Photometric')
	# visualization.hist(tdata_spec['z_best'],bins='scott',histtype=u'step',label='Spectroscopic')
	
	plt.hist(tdata_phot['z_best'],histtype=u'step',label='Photometric')
	plt.hist(tdata_spec['z_best'],histtype=u'step',label='Spectroscopic')
	

	# plt.yscale('log')
	plt.legend(fontsize=14)
	plt.xlabel(r'Redshift',fontsize=14)
	plt.ylabel('Counts',fontsize=14)
	plt.tight_layout()

	plt.show()

def threeD_binned_samples_sourcenumbers_table():

	F = open('./3dbinsubsets.csv','w')
	F.write('Subset,Number of sources,')
	F.write('\n')
	F.write(',Power bin 1,Flux bin 3, Angular size bin 0')
	F.write('\n')
	
	
	def helper_func(filename):
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		# Take only the available redshift sources
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		tdata = tdata[z_available]

		powerbin1 = select_power_bins_cuts_VA_selection(tdata)['1']
		powerbin1 = len(powerbin1)
		
		fluxbin3 = select_flux_bins_cuts_biggest_selection(tdata)['3']
		fluxbin3 = len(fluxbin3)
		print (fluxbin3)


		sizebin0 = select_size_bins_cuts_biggest_selection(tdata)['0']
		sizebin0 = len(sizebin0)

		return powerbin1, fluxbin3, sizebin0

	# YES NEEDED
	filename = 'value_added_selection'
	powerbin1, fluxbin3, sizebin0 = helper_func(filename)
	F.write('Value-added,%i,%i,%i'%(powerbin1,fluxbin3,sizebin0))
	F.write('\n')

	filename = 'value_added_selection_MG'
	powerbin1, fluxbin3, sizebin0 = helper_func(filename)
	F.write('Connected lobes,%i,%i,%i'%(powerbin1,fluxbin3,sizebin0))
	F.write('\n')

	filename = 'value_added_selection_NN'
	powerbin1, fluxbin3, sizebin0 = helper_func(filename)
	F.write('Isolated lobes,%i,%i,%i'%(powerbin1,fluxbin3,sizebin0))
	F.write('\n')

	filename = 'value_added_compmatch'
	powerbin1, fluxbin3, sizebin0 = helper_func(filename)
	F.write('Concluding,%i,%i,%i'%(powerbin1,fluxbin3,sizebin0))
	F.write('\n')

	F.close()

def how_many_spectroscopic():
	def helper_func(filename):
		tdata = Table(fits.open('../%s.fits'%filename)[1].data)
		# Take only the available redshift sources
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		tdata = tdata[z_available]

		spectro_where = np.where(tdata['z_source'] == 1)
		tdata = tdata[spectro_where]
		
		fluxbin3 = select_flux_bins_cuts_biggest_selection(tdata)['3']
		fluxbin3 = len(fluxbin3)

		return fluxbin3

	
	print ('Number of spectroscopic redshifts in flux bin 3:')
	filename = 'value_added_selection'
	fluxbin3 = helper_func(filename)

	print (filename, fluxbin3)

	filename = 'value_added_selection_MG'
 	fluxbin3 = helper_func(filename)
	
	print (filename, fluxbin3)

def dispersion_vectorized_n(tdata,n,redshift=False):
	"""
	Calculates and returns the dispersion for tdata
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

	return di_max

def di_vs_right_ascension():
	n = 60
	n = 140

	file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/value_added_selection_MG.fits'

	# ############################################ VA selection MG only redshift sources, flux bin 3
	tdata = Table(fits.open(file)[1].data)
	# # With redshift
	print ('Using redshift..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]
	tdata_original = tdata # use redshift data for power bins and size bins !!!
	# #########################################  VA selection MG only redshift sources, flux bin 3
	tdata['final_PA'] = tdata['position_angle']
	print ('Using flux bin 3')
	tdata['RA'], tdata['DEC'] = tdata['RA_2'], tdata['DEC_2']
	# tdata = only redshift sources flux bin 3
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']

	# array of N,n for all sources N, with nearest neighbour n
	di_max = dispersion_vectorized_n(tdata,n,redshift=False)

	# index 0 = 1
	di_max_n = di_max[:,n-1]

	plt.scatter(tdata['RA'],di_max_n,c='k',s=10)
	plt.xlabel('Righ ascension (degrees)',fontsize=14)
	plt.ylabel('Dispersion',fontsize=14)
	# ax.legend(fontsize=14)
	# plt.title(title,fontsize=16)
	plt.tight_layout()
	plt.show()

def di_vs_right_ascension_declination():
	n = 60
	n = 140

	file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/value_added_selection_MG.fits'

	# ############################################ VA selection MG only redshift sources, flux bin 3
	tdata = Table(fits.open(file)[1].data)
	# # With redshift
	print ('Using redshift..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]
	tdata_original = tdata # use redshift data for power bins and size bins !!!
	# #########################################  VA selection MG only redshift sources, flux bin 3
	tdata['final_PA'] = tdata['position_angle']
	print ('Using flux bin 3')
	tdata['RA'], tdata['DEC'] = tdata['RA_2'], tdata['DEC_2']
	# tdata = only redshift sources flux bin 3
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']

	# array of N,n for all sources N, with nearest neighbour n
	di_max = dispersion_vectorized_n(tdata,n,redshift=False)

	# index 0 = 1
	di_max_n = di_max[:,n-1]

	ra, dec = tdata['RA'], tdata['DEC']

	if n == 140:
		cutoff = 0.17
	else:
		cutoff = 0.25

	less_than_25 = np.where(di_max_n < cutoff)
	more_than_25 = np.where(di_max_n > cutoff)

	plt.scatter(ra[less_than_25],dec[less_than_25],c='k',s=10,label=r'$d_i$<%.2f'%cutoff)
	plt.scatter(ra[more_than_25],dec[more_than_25],c='r',s=10,label=r'$d_i$>%.2f'%cutoff)
	plt.xlabel('Righ ascension (degrees)',fontsize=14)
	plt.ylabel('Declination (degrees)',fontsize=14)
	# ax.legend(fontsize=14)
	# plt.title(title,fontsize=16)
	plt.tight_layout()
	plt.ylim(45.058947246972785,59)
	plt.legend(loc='upper left')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()


def investigate_strong_alignment_sources():
	n = 60
	n = 140

	file = '/data1/osinga/value_added_catalog_1_1b_thesis_cutoff069/value_added_selection_MG.fits'

	# ############################################ VA selection MG only redshift sources, flux bin 3
	tdata = Table(fits.open(file)[1].data)
	# # With redshift
	print ('Using redshift..') ## edited for not actually using z
	z_available = np.invert(np.isnan(tdata['z_best']))
	z_zero = tdata['z_best'] == 0#
	# also remove sources with redshift 0, these dont have 3D positions
	z_available = np.logical_xor(z_available,z_zero)
	print ('Number of sources with available redshift:', np.sum(z_available))
	tdata = tdata[z_available]
	tdata_original = tdata # use redshift data for power bins and size bins !!!
	# #########################################  VA selection MG only redshift sources, flux bin 3
	tdata['final_PA'] = tdata['position_angle']
	print ('Using flux bin 3')
	tdata['RA'], tdata['DEC'] = tdata['RA_2'], tdata['DEC_2']
	# tdata = only redshift sources flux bin 3
	tdata = select_flux_bins_cuts_biggest_selection(tdata)['3']

	ra, dec = tdata['RA'], tdata['DEC']
	pa = tdata['final_PA']

	where = np.where( (tdata['RA'] < 230) & (tdata['RA'] > 210) 
		& (tdata['DEC'] < 50) & (tdata['DEC'] > 45) )


	tdata_region = tdata[where]

	plt.hist(pa[where],bins=180/5)
	plt.ylabel('Counts',fontsize=14)
	plt.xlabel('Position angle (degrees)',fontsize=14)
	# ax.legend(fontsize=14)
	# plt.title(title,fontsize=16)
	plt.tight_layout()
	plt.show()

def all_subsamples_hist_size(redshift=True):
	'''Overlay of all the subsamples and the distribution of their size'''
	
	prop_cycle = plt.rcParamsDefault['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	# bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	bins=10

	# Initial sample = blue; 
	color_initial_sample = colors[0]
	color_value_added_subset = colors[1]
	color_connected_lobes = colors[2]
	color_isolated_lobes = colors[3]
	color_concluding_subset = colors[4]

	tdata = '../value_added_selection.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	sizes = tdata['size_thesis']
	
	# plt.hist(sizes,bins=bins,label='Value-added subset',color=color_value_added_subset,histtype=u'step')
	

	tdata = '../value_added_compmatch.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	sizes = tdata['size_thesis']

	# plt.hist(sizes,bins=bins,label='Concluding subset',color=color_concluding_subset,histtype=u'step')


	tdata = '../value_added_selection_MG.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!

	sizes = tdata['size_thesis']
	print ('Minimum size MG sources:',np.min(sizes))
	print ('Minimum semi-major axis MG sources:',np.min(tdata['Maj_1']))

	plt.hist(sizes,bins=bins,label='Connected lobes subset',color=color_connected_lobes,histtype=u'step')
	
	tdata = '../value_added_selection_NN.fits'
	print (tdata)
	tdata = Table(fits.open(tdata)[1].data)

	if redshift:
		z_available = np.invert(np.isnan(tdata['z_best']))
		z_zero = tdata['z_best'] == 0#
		# also remove sources with redshift 0, these dont have 3D positions
		z_available = np.logical_xor(z_available,z_zero)
		print ('Number of sources with available redshift:', np.sum(z_available))
		tdata = tdata[z_available] # use redshift data for power bins and size bins !!!
	
	position_angles = tdata['position_angle']
	sizes = tdata['size_thesis']
	# plt.hist(sizes,bins=bins,label='Isolated lobes subset',color=color_isolated_lobes,histtype=u'step')
	

	plt.ylabel('Counts',fontsize=14)
	plt.xlabel('Angular size (arc seconds)',fontsize=14)

	# plt.xlim(0,20)
	# if not redshift: plt.ylim(0,300)
	# else: plt.ylim(0,150)
	plt.legend(fontsize=14)
	# plt.yscale('log')

	plt.tick_params(labelsize=12)
	plt.tight_layout()
	plt.show()

# plot_PA_vs_position_angle()
# plot_E_PA_vs_E_position_angle()
# plot_size_of_horizontal_connected_lobes()
# plot_size_of_vertical_isolated_lobes()
plot_all_sources_MG_PA()
# plot_VA_MG_redshift_dist()
# matched_sources_redshift()

# all_subsamples_hist_size(redshift=False)

# threeD_binned_samples_sourcenumbers_table()
# how_many_spectroscopic()
# di_vs_right_ascension()
# di_vs_right_ascension_declination()
# investigate_strong_alignment_sources()
# initial_subsample_hist(PT=False)


# initial_subsample_hist_AP(PT=False)

# all_subsamples_hist(PT=False,redshift=False)
# all_subsamples_hist(PT=True,redshift=False)
# all_subsamples_hist(PT=False,redshift=True)

# SN_mc('biggest_selection')
# SN_mc('value_added_selection')
# SN_mc('value_added_selection_MG')
# SN_mc('value_added_compmatch')
# SN_mc('value_added_selection_NN')

# tdata = '../biggest_selection.fits'
# tdata = '../value_added_selection_MG.fits'
# tdata = Table(fits.open(tdata)[1].data)
# print ('Kuiper test:')
# KuiperTest(tdata,PT=True)

# source_counts('value_added_selection')
# source_counts('value_added_selection_MG')
# source_counts('value_added_compmatch')

# KSTEST('biggest_selection')
# KSTEST('value_added_selection',redshift=True)
# KSTEST('value_added_selection_MG',redshift=True)
# KSTEST('value_added_selection_NN',redshift=True)
# KSTEST('value_added_compmatch',redshift=True)

# subsample_parameters('value_added_selection')

# plot_hist('FIRSTdata',True,True)