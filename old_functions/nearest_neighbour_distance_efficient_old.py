def nearest_neighbour_distance_efficient(write=False):
    '''
    Still have to think about nearest neighbour with degrees.
    
    As it works now, it computes the nearest neighbour using !Euclidian! distance
    (So it does not take into account the spherical distance for this)

    But it does produce the spherical distance between the points
    after the nearest neighbour has been found.

    Returns two arrays, one with the distance and one with the index of the NN
    The distance is given in arcminutes

    It is very fast, but produces a different result from the other function. 
    '''

    from astropy.table import Table
    Source_Data = fits.open(prefix+file+'cat.srl.fits')
    Source_Data = Table(Source_Data[1].data)
    RAs = np.asarray(Source_Data['RA'])
    DECs = np.asarray(Source_Data['DEC'])
    
    #convert to an array that has following layout: [[ra1,dec1],[ra2,dec2],etc]
    coordinates = np.vstack((RAs,DECs)).T
    #make an cKDTree for quick NN searching
    import scipy.spatial
    coordinates_tree = scipy.spatial.cKDTree(coordinates,leafsize=16,balanced_tree=False)

    TheResult_distance = []
    TheResult_index = []
    for item in coordinates:
        '''
        Find 2nd closest neighbours, since the 1st is the point itself.

        coordinates_tree.query(item,k=2)[1][1] is the index of this second closest 
        neighbour.

        We then compute the spherical distance between the item and the 
        closest neighbour.
        '''

        index=coordinates_tree.query(item,k=2)[1][1]
        nearestN = coordinates[index]
        distance = distanceOnSphere(nearestN[0],nearestN[1],#coordinates of the nearest
                                item[0],item[1])*60 #coordinates of the current item
        TheResult_distance.append(distance) 
        TheResult_index.append(index)
    
    if write==True:
        t2 = fits.open(prefix+file+'cat.srl.fits')#open the fits file to get the source names
        t2 = Table(t2[1].data)['Source_Name']
        t2 = Table([t2[:],TheResult_distance,TheResult_index], names = ('Source Name','NN_distance(arcmin)','NN_index'))
        t2.write('/data1/osinga/data/'+file+'NearestNeighbours_efficient.fits',overwrite=True)
    return TheResult_distance,TheResult_index
