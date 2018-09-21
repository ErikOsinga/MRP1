import sys
sys.path.insert(0, '/net/reusel/data1/osinga/anaconda2')
import numpy as np 

from astropy.io import fits
from astropy.table import Table, join, vstack

import matplotlib.pyplot as plt

from astropy.cosmology import Planck15
from astropy import units as u

FieldNames = ['P11Hetdex12', 'P173+55', 'P21', 'P8Hetdex', 'P30Hetdex06', 'P178+55', 
'P10Hetdex', 'P218+55', 'P34Hetdex06', 'P7Hetdex11', 'P12Hetdex11', 'P16Hetdex13', 
'P25Hetdex09', 'P6', 'P169+55', 'P187+55', 'P164+55', 'P4Hetdex16', 'P29Hetdex19', 'P35Hetdex10', 
'P3Hetdex16', 'P41Hetdex', 'P191+55', 'P26Hetdex03', 'P27Hetdex09', 'P14Hetdex04', 'P38Hetdex07', 
'P182+55', 'P33Hetdex08', 'P196+55', 'P37Hetdex15', 'P223+55', 'P200+55', 'P206+50', 'P210+47', 
'P205+55', 'P209+55', 'P42Hetdex07', 'P214+55', 'P211+50', 'P1Hetdex15', 'P206+52', 
'P15Hetdex13', 'P22Hetdex04', 'P19Hetdex17', 'P23Hetdex20', 'P18Hetdex03', 'P39Hetdex19', 'P223+52',
 'P221+47', 'P223+50', 'P219+52', 'P213+47', 'P225+47', 'P217+47', 'P227+50', 'P227+53', 'P219+50',
 ]


def pdf_scott_tout(N):
	"""
	Plot the PDF of any nearest neighbour to a point on the celestial sphere
	being at distance theta when the total amount of points is N

	according to the paper by Scott and Tout 1989
	"""

	N = 1.*N

	theta_domain = np.linspace(0,0.00407243492) # 14 arcmin
	# theta_domain = np.linspace(0,np.pi) # pi

	pdf = (N-1)/(2**(N-1)) * np.sin(theta_domain)*(1+np.cos(theta_domain))**(N-2)

	plt.plot(theta_domain, pdf, label='PDF scott tout , N = %i'%N)


def pdf_martijn_oei(N):
	"""
	Plot the PDF of any nearest neighbour to a point on the celestial sphere
	being at distance theta when the total amount of points is N

	according to the thesis by oei 
	"""
	N = 1.*N


	theta_domain = np.linspace(0,0.00407243492) # 14 arcmin
	# theta_domain = np.linspace(0,np.pi) # pi

	pdf = (N+1.)/2. * np.sin(theta_domain) * ((np.cos(theta_domain)+1)/2.)**(N)
	plt.plot(theta_domain, pdf, label='PDF oei , N = %i'%N)

def pdf_poisson(N):
	N = 1.*N
	
	theta_domain = np.linspace(0,0.00407243492) # 14 arcmin
	# theta_domain = np.linspace(0,np.pi) # pi
	# theta_domain = np.linspace(0,0.000290888209) # 1 arcmin


	pdf = 2*np.pi*N/(4*np.pi) * theta_domain * np.exp(-1*np.pi*N/(4*np.pi)*theta_domain**2)
	# plt.plot(theta_domain, pdf, label='PDF poisson , N = %i'%N)

						# conversion to arcmin
	plt.plot(theta_domain*3437.74677, pdf/3437.74677, label='PDF, N = %i'%(N/1e6) + r' $\cdot 10^6$')




def plot_NN_distance_distribution():
	'''
	Plot distribution of NN angular distance for all sources with SN>10
	no cutoff applied yet.
	'''

	def retreive_NN_distance(name_field):
		prefix = '../source_filtering/NN/'

		catalog1 = fits.open('../source_filtering/NN/'+name_field+'NearestNeighbours_efficient_spherical1.fits')
		catalog1 = Table(catalog1[1].data)

		a = catalog1
		return a

	# the first catalog is 'P11Hetdex12'
	a = retreive_NN_distance('P11Hetdex12')
	for name_field in FieldNames:
		# for all the other names
		print name_field
		if name_field != 'P11Hetdex12':
			# get the next catalog and stack them on top
			b = retreive_NN_distance(name_field)
			a = vstack([a,b])

	a['NN_distance(arcmin)']	

	plt.hist(a['NN_distance(arcmin)'],normed=True,bins=300,histtype=u'step',color='black'
		,label='Observed')
	plt.xlabel('Angular distance to NN (arcmin)',fontsize=12)
	plt.ylabel(r'Probability density (arcmin$^{-1}$)',fontsize=12)


	return a

a = plot_NN_distance_distribution()
pdf_poisson(9.9e6)
plt.legend()
plt.show()



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

def plot_size_VA_catalog():

	tdata = '/data1/osinga/value_added_catalog_1_1b_thesis/SN_10_value_added_sources.fits'
	tdata = Table(fits.open(tdata)[1].data)
	tdata = tdata[np.invert(np.isnan(tdata['z_best']))]
	z_best = tdata['z_best']

	all_sizes = []
	for i in range(len(tdata)):
		angular_size = 2*tdata['Maj_1'][i]
		if np.isnan(angular_size): angular_size = tdata['LGZ_size'][i]

		if not (z_best[i] == 0 or z_best[i] < 0):
			size = physical_size(z_best[i],angular_size).take(0)
			all_sizes.append(size)

	all_sizes = np.asarray(all_sizes)

	plt.hist(all_sizes,bins=100,histtype=u'step',color='black')
	# plt.hist(all_sizes,bins=int(num_bins),histtype=u'step',color='black')

	plt.xlabel('Radio source size (kpc)',fontsize=12)
	plt.ylabel('Counts',fontsize=12)
	plt.show()



def angular_distance_vs_redshift(size):
	"""
	Plots the anuglar distance vs redshift function as a function of the physical
	size of the radio source
	""" 
	
	tdata = '/data1/osinga/value_added_catalog_1_1b_thesis/SN_10_value_added_sources.fits'
	tdata = Table(fits.open(tdata)[1].data)
	tdata = tdata[np.invert(np.isnan(tdata['z_best']))]
	z_best = tdata['z_best']

	print ('redshift:')
	print ('min: %f, max: %f, mean %f, median: %f' %(np.min(z_best)
						,np.max(z_best),np.mean(z_best),np.median(z_best)))
	domain_z = np.linspace(0,np.max(z_best),5000)

	arcsec_per_kpc = Planck15.arcsec_per_kpc_proper(domain_z)
	
	# arcsec_per_kpc = Planck15.arcsec_per_kpc_comoving(domain_z)


	# plt.plot(domain_z, arcsec_per_kpc,label='1 kpc')
	plt.plot(domain_z, 10*arcsec_per_kpc,label='10 kpc')
	plt.plot(domain_z, 100*arcsec_per_kpc,label='100 kpc')
	plt.plot(domain_z, 200*arcsec_per_kpc,label='200 kpc')
	plt.plot(domain_z, 300*arcsec_per_kpc,label='300 kpc')
	plt.plot(domain_z, 400*arcsec_per_kpc,label='400 kpc')
	plt.hlines(41.4,0,np.max(z_best),label="0.69' cutoff")
	plt.ylabel('Angular size (arcseconds)',fontsize=12)
	plt.ylim(0,150)
	plt.xlim(0,np.max(z_best))
	plt.xlabel('Redshift',fontsize=12)
	plt.legend(title='Radio source physical size')
	plt.grid()
	plt.show()

def plot_z_dist():

	tdata = '/data1/osinga/value_added_catalog_1_1b_thesis/SN_10_value_added_sources.fits'
	tdata = Table(fits.open(tdata)[1].data)
	tdata = tdata[np.invert(np.isnan(tdata['z_best']))]
	z_best = tdata['z_best']

	all_z = []
	for i in range(len(tdata)):
		if not (z_best[i] == 0 or z_best[i] < 0):
			all_z.append(z_best[i])

	all_z = np.asarray(all_z)

	plt.hist(all_z,bins=100,histtype=u'step',color='black')
	# plt.hist(all_sizes,bins=int(num_bins),histtype=u'step',color='black')

	plt.xlabel('Redshift',fontsize=12)
	plt.ylabel('Counts',fontsize=12)
	plt.show()

# plot_size_VA_catalog()
# angular_distance_vs_redshift('useless_variable')
# plot_z_dist()