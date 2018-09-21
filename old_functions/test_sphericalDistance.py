import sys
sys.path.insert(0, '/data1/osinga/anaconda2')
import numpy as np 




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


'''
This is the test of the implementation of the
distaceOnsphere into the scipyspatialckdTree 

'''

a = np.array([175.85474414716302,53.70654260165704])
b = np.array([
[175.82993904758683,54.052400361551946],
[175.81642070615968,53.84952801110464],
[175.83718240740995,54.24480505739465],
[175.8325265803577,54.11319667129783],	
[175.83380784051545,54.24073564009313],	
[175.83452918657517,54.27958957701325],	
[175.83084527831383,54.25643297691266],
[175.8109413037666,53.89886554921686],
[175.7986916080435,53.84611451646336],	
[175.83947629493076,54.496654194665005],
[175.79533860263874,53.70964089165918]
])

RAs = b[:,0]
DECs = b[:,1]

print distanceOnSphere(175.85474414716302,53.70654260165704,
						RAs,DECs)*60