import numpy as np
from astropy.table import Table

CHIP1XLEN = CHIP2XLEN = 4096 
CHIP1YLEN = CHIP2YLEN = 2048

# Read in the data
path = '/Users/dkossakowski/Desktop/Data/'
datafil = 'wfc_f606w_r5.lflat'
data = np.genfromtxt(path + datafil)

# Create an Astropy table
# tab[starID][0]: starID; ...[2]: chip#; ...[3]: x pixel; ...[4]: y pixel; 
# ...[5]: magnitude; ...[6]: magnitude error; rest: dummy
names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr', 'd1', 'd2', 'd3']
types = [int, int, int, np.float64, np.float64, np.float64, np.float64,
         float, float, float]
tab = Table(data, names=names, dtype=types)

# Build up the indicies for each star in the table and store in a dictionary
# keys: star ID 
# values: indicies of the rows in tab corresponding to star ID
# Ex: tab[starindexdict[starID] gives the rows corresponding to the star ID
starindexdict = {}
for row in tab[1000:2000]:
    try:
        starindexdict[row['id']]
    except KeyError:
        starindexdict[row['id']] = np.where(tab['id'] == row['id'])[0]

def remove_stars(stararr, somedict, min_num_obs = 3):
    """ 
    *** Helper function for the extract_data function ***
    Purpose
    -------
    Removes the stars that do not have the min_num_obs observations
    
    Parameters
    ----------
    stararr:        The array of all the star IDs
    somedict:       The dictionary where the keys are the star IDs and the values
                    depend on which dictionary is passed in
    min_num_obs:    The minimum number of observations required for a star to have
    
    Returns
    -------
    somedict:       The same dictionary that was passed in but now filtered 
    NOTES
    *this is slow if we are iterating through every dict, I want to instead find 
    the stars that we should remove and then remove them from the other dicts
    (but idk how to and i will figure it out later)
    *change the name of somedict lol
    """
    for star in stararr:
        if len(somedict[star]) <= 3:
            somedict.pop(star)
    return somedict
    
def extract_data(indexdict, datatab):
    '''
    Purpose
    -------
    Extract all the data (star ID, x and y pixels, magnitude, and magnitude errors)
    
    Parameters
    ----------
    indexdict:      This is a dictionary where the keys are the star IDs and the
                    values are the row indexes of datatab that correspond to the
                    star ID; 
                    -- indexdict[starID] gives the index array of where starID is
    datatab:        Astropy table that was created from a file; for each row of 
                    table, the items are: row[0] = starID, row[1] = file number
                    row[2] = chip number, row[3] = x pixel, row[4] =y pixel, 
                    row[5] = magnitude, row[6] = magnitude error, 
                    row[7:10] = dummy
                    -- datatab[indexdict[starID]] gives the table rows of those 
                    indicies
    Returns
    -------
    [stararr, xdict, ydict, mdict, merrdict]
    stararr:        An array of all the star IDs (will be useful later on when
                    iterating over all the stars)
    xdict:          A dictionary where the keys are the star IDs and the values
                    are the x pixel values corresponding to that star ID
    ydict:          A dictionary where the keys are the star IDs and the values
                    are the y pixel values corresponding to that star ID 
    mdict:          A dictionary where the keys are the star IDs and the values
                    are the magnitude values corresponding to that star ID 
    merrdict:       A dictionary where the keys are the star IDs and the values
                    are the magnitude errors corresponding to that star ID        
    '''
    xdict = {}
    ydict = {}
    mdict = {}
    merrdict = {}
    for starID in indexdict:
        xarr = np.array([])
        yarr = np.array([])
        marr = np.array([])
        merrarr = np.array([])
        # Go through the indicies of a certain star in order to extract:
        # x pixel values, y pixel values, magnitude values, and magnitude errors
        # Ex: star id = 16382, values: [ 1006, 73149] for indexdict[starid]
        # 'i' will iterate as index 1006 and 73149 in the data table and each
        #   parameter will be added to an array which is then going to be the 
        #   value for a specified dictionary (xdict, ydict, etc.)
        for i in indexdict[starID]:
            chip = datatab[i][2]
            xpixel = datatab[i][3]
            ypixel = datatab[i][4]
            mag = datatab[i][5]
            magerr = datatab[i][6]
            xarr = np.append(xarr, xpixel)
            yarr = np.append(yarr, ypixel) if chip == 2 else np.append(yarr, ypixel + CHIP2YLEN)
            marr = np.append(marr, mag)
            merrarr = np.append(merrarr, magerr)
        xdict[starID] = xarr
        ydict[starID] = yarr
        mdict[starID] = marr
        merrdict[starID] = merrarr
    stararr = indexdict.keys()
    xdict = remove_stars(stararr, xdict, 3)
    ydict = remove_stars(stararr, ydict, 3)
    mdict = remove_stars(stararr, mdict, 3)
    merrdict = remove_stars(stararr, merrdict, 3)
    return [stararr, xdict, ydict, mdict, merrdict]

stararr, xdict, ydict, mdict, merrdict = extract_data(starindexdict, tab)


    

    
    
