from multiprocessing import Pool
from multiprocessing import cpu_count

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

FieldNames2 = [
'P11Hetdex12', 'P173+55', 'P21', 'P8Hetdex','P30Hetdex06', 'P178+55', 
'P10Hetdex', 'P218+55'


]

def setup_orientation(file):
	for i in range(0,10000):
		for k in range(0,100):
			if i == 1534:
				print file






if __name__ == '__main__':
	print cpu_count()
	p = Pool(4)
	print (p.map(setup_orientation, FieldNames))

	# print '___________________\n\n'
	# for file in FieldNames:
	# 	print setup_orientation(file)






