import numpy as np 

from astropy.cosmology import Planck15
from astropy import units as u

F = open('./table_other_studies.csv','w')

F.write('Name,Frequency,RMS(mJ/beam),SNR,Source density(deg$^{-2}$),Resolution(arcsec),Study area(deg$^2$),Significance(per cent),Scale(deg),Redshift(median),Physical scale($h^{-1}$Mpc)')
F.write('\n')
# RMS noise in mJ beam


# Me, connected lobes subset , highest flux
name = 'This study'
frequency = '150 MHz'
RMS = '0.071' #mJ/beam
SNR = '10'
Source_density = 879/424. # 2.07311321 per square degree # Only > 9.41 mJy flux sources
														# that have a redshift measurement
													# and are connected lobes
										# 1549/424 = 3.65330189 connected lobes >9.41 mJy
Resolution = '6x6'
area = '424' # square degrees
SL = '$<$0.1'#\%
Redshift = '0.69' # 0.68848131918907163 = median redshift of the 879 connected lobes >9.41 flux sources
Scale = '5' #degrees
Phys_scale_planck15 = ( (Planck15.arcsec_per_kpc_comoving(0.69) )**-1 * (5*u.degree).to(u.arcsec) ).to(u.Mpc) 
h = Planck15.H0 / (100 * u.km/u.s/u.Mpc)
Phys_scale = Phys_scale_planck15 * h
print (Phys_scale)
Phys_scale = '%i'%Phys_scale.to(u.Mpc).value #$h^{-1}$ Mpc
# Mpc
F.write('%s,%s,%s,%s,%.1f,%s,%s,%s,%s,%s,%s'%(name,frequency,RMS,SNR,Source_density,Resolution,area,SL,Scale,Redshift,Phys_scale)  )
F.write('\n')


# Me_again, but now the value_added_subset, 
#  For n = 500:
# log10 SL data : -2.381
# log10 SL upper bound: -2.897
# lgo10 SL lower bound: -1.922
name = 'This study'
frequency = '150 MHz'
RMS = '0.071' #mJ/beam
SNR = '10'
Source_density = 947/424. # 2.07311321 per square degree # Only > 9.41 mJy flux sources
														# that have a redshift measurement
													# and are connected lobes
										# 1549/424 = 3.65330189 connected lobes >9.41 mJy
Resolution = '6x6'
area = '424' # square degrees
SL = '$<$1.3'#\%
Redshift = '0.54' # 0.53807181119918823 = median redshift of the 947 connected lobes >9.41 flux sources
Scale = '10' #degrees
Phys_scale_planck15 = ( (Planck15.arcsec_per_kpc_comoving(0.54) )**-1 * (10*u.degree).to(u.arcsec) ).to(u.Mpc) 
h = Planck15.H0 / (100 * u.km/u.s/u.Mpc)
Phys_scale = Phys_scale_planck15 * h
print (Phys_scale)
Phys_scale = '%i'%Phys_scale.to(u.Mpc).value #$h^{-1}$ Mpc
# Mpc
F.write('%s,%s,%s,%s,%.1f,%s,%s,%s,%s,%s,%s'%(name,frequency,RMS,SNR,Source_density,Resolution,area,SL,Scale,Redshift,Phys_scale)  )
F.write('\n')


#contigiani
name = 'Contigiani'
frequency = '1.4 GHz'
RMS = '0.15' #mJ/beam
SNR = '10' 
Source_density = ((30059/7000.)) #4.29 per square degree 
Resolution = '5x5' # arcseconds
area = '7000' # square degrees
SL = '$<$2'#\%
Redshift = '0.47'
Scale = '1.5' # degrees
Phys_scale = '19-38' # $h^{-1}$ Mpc
F.write('%s,%s,%s,%s,%.1f,%s,%s,%s,%s,%s,%s'%(name,frequency,RMS,SNR,Source_density,Resolution,area,SL,Scale,Redshift,Phys_scale)  )
F.write('\n')


name = 'Taylor'
frequency = '612 MHz'
RMS = '0.01' #mJ/beam
SNR = '3'
Source_density = 64/1.2 # 54.17 per square degree
Resolution = '6.1x5.1' # arcseconds
area = '1.2' # square degrees
SL = '$<$0.1'#\%
Redshift = '1' # 3 samples: z < 0.5, 0.5<z<1.0 and z>1: 12, 11 and 10 objects
Scale = '1' #degrees
Phys_scale = '$>$20'#h^{-1}$ Mpc
F.write('%s,%s,%s,%s,%.1f,%s,%s,%s,%s,%s,%s'%(name,frequency,RMS,SNR,Source_density,Resolution,area,SL,Scale,Redshift,Phys_scale)  )
F.write('\n')
# p < 0.01 to reject uniformity of the whole sample and all subsamples
# z < 0.5 shows alignment at 0.8 degrees
# z > 1.0 shows alignment from 0.4-1.2 degrees

# Results from Ripleys K and twopoint correlation shows 3.06-16.35 h^-1 Mpc comoving scales at z=1
# Result from variogram shows 50-75 h^-1 Mpc at z= 1

F.close()




# Should probably make a diffent table for the radio polarization studies.
# They have no RMS, SNR, or resolution, but do have polarized flux 
# Or make no table, and just mention it in the text

name = 'Pelgrims' #http://adsabs.harvard.edu/abs/2015MNRAS.450.4161P
frequency = '8.4GHz'
RMS = ''
SNR = ''
Source_density = '%s'%(str(4155/1.2)) #  per square degree
										# 1531 with spectroscopic redshifts
Resolution = ''
area = ''
SL = ''
Scale = '20' #degrees
Redshift = '>1' # 
Phys_scale = '$ >20 h^-1$ Mpc'

''' From the paper:
Alignment more pronounced in 2D than in 3D. Mean PAs are multiples of 45 deg
But one class of object does show alignments and clustering (alignment?) is consistent with
those found at optical wavelength

'''

name = 'Pelgrims' # http://adsabs.harvard.edu/abs/2016A%26A...590A..53P


Scale = 'With their LQG'
Redshift = '1-1.8' # 

# See the paper.






'''
# This study finds only 6 AGN are aligned out of 78 selected and matched radio lobes
# The statistics are questionable however, so we dont mention this
name = 'Bansal'
frequency = ''
RMS = '%s'%(str(150/1000.)) # 150 microJansky to mJ
SNR = ''
Source_density = '%s'str(6/19.) # 0.315789474 per square degree
Resolution = '5.6x7.4' # arcseconds
area = '19' # degrees
SL = '$<0.01$'
Phys_scale = '$ 8-10 h^-1$ Mpc'
'''