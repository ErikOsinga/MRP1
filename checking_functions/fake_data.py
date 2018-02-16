fake_data.py

Functions that shouldnt be found in the source_filtering.py code
but are used to test 


# data_array = [[10,4,5,6,7,8,9,10,11,12,13,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],
# [10,4,5,6,7,8,9,10,11,12,13,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],
# [2,4,5,6,7,8,9,10,11,12,np.nan,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],
# [2,4,5,6,7,8,9,10,11,12,13,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],
# [2,4,5,6,7,8,9,10,11,12,13,14,15],[2,4,5,6,7,8,9,10,11,12,13,14,15],
# ]

def fake_data():
	data_array = np.zeros((30,30))
	print np.shape(data_array)
	data_array[13] = np.array([0,0,0,0,0,0,0,0,0,0,0,3,4,2,0,3,4,2,0,0,0,0,0,0,0,0,0,0,0,0])+2
	data_array[14] = np.array([0,0,0,0,0,0,0,0,0,0,0,5,8,4,0,4,8,2,0,0,0,0,0,0,0,0,0,0,0,0])+2
	data_array[15] = np.array([0,0,0,0,0,0,0,0,0,0,0,3,4,2,0,2,4,1,0,0,0,0,0,0,0,0,0,0,0,0])+2

	from random import randint
	for i in range(2,28):
		if i not in (13,14,15):
			for j in range(0,30):
				data_array[i][j] = randint(0,2)
		else:
			for j in range(0,10):
				data_array[i][j] = randint(0,2)
			for j in range(19,30):
				data_array[i][j] = randint(0,2)
	return fake_data



src1_RA = 175.81642070615968
src1_DEC = 53.84952801110464
src1_NN_RA = 175.7986916080435
src1_NN_DEC = 53.84611451646336

src2_RA = 175.8371824
src2_DEC = 54.24480506
src2_NN_RA = 175.83380784051545
src2_NN_DEC = 54.24073564009313


src5_RA = 175.7986916080435
src5_DEC = 53.84611451646336


src0_RA = 175.81642070615968
src0_DEC = 53.84952801110464
src0_NN_RA = 175.7986916080435
src0_NN_DEC = 53.84611451646336

src5_RA = 175.8024162
src5_DEC = 53.84631247
src5_RA2 = 175.7942829
src5_DEC2 = 53.84533179




# plot the image and the flux below it
fig, axes = plt.subplots(nrows=1)#figsize=(10,10))
		axes[0].imshow(data_array2,origin='lower')
		axes[0].plot([x0, x1], [y0, y1], 'r-',alpha=0.3)
		axes[0].axis('image')
		axes[1].plot(zi)
		plt.suptitle(fitsim + '\n Winning orientation = ' + str(max_angle) + ' degrees')
		plt.savefig('/data1/osinga/figures/cutouts/multiple_gaussians/elongated/src_'+str(i)+'.png')
		