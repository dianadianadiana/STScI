import numpy as np
from astropy.table import Table
import time

start_time = time.time()

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
#     keys: star ID 
#     values: indicies of the rows in tab corresponding to star ID
# Ex: tab[starindexdict[starID] gives the rows corresponding to the star ID
starindexdict = {}
for row in tab[:]:
    try:
        starindexdict[row['id']]
    except KeyError:
        starindexdict[row['id']] = np.where(tab['id'] == row['id'])[0]
    
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
    [stardict, xdict, ydict, mdict, merrdict] -- all as were in the files; they
        will be filtered through
    stardict:       A dictionary where the keys are the star IDs and the values
                    are also the star IDs -- useful in iterating through the stars
    xdict:          A dictionary where the keys are the star IDs and the values
                    are the x pixel values corresponding to that star ID
    ydict:          A dictionary where the keys are the star IDs and the values
                    are the y pixel values corresponding to that star ID 
    mdict:          A dictionary where the keys are the star IDs and the values
                    are the magnitude values corresponding to that star ID 
    merrdict:       A dictionary where the keys are the star IDs and the values
                    are the magnitude errors corresponding to that star ID        
    '''
    
    stardict = {}
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
        stardict[starID] = starID
        xdict[starID] = xarr
        ydict[starID] = yarr
        mdict[starID] = marr
        merrdict[starID] = merrarr
    return [stardict, xdict, ydict, mdict, merrdict]
    
def remove_stars(dicts, min_num_obs = 3):
    """            *** Filter function ***
    Purpose
    -------
    Removes the stars with less than a min num of observations from each dictionary 
    
    Parameters
    ----------
    dicts:          An array of dictionaries where the keys are the star IDs and 
                    the values depend on each dictionary; each dictionary will 
                    be filtered; ASSUME that the 0th dictionary is the star 
                    dictionary
    min_num_obs:    The minimum number of observations required for a star to have
    
    Returns
    -------
    dicts:          The filtered dictionaries that were passed in
    removestarlist: The list of stars that need to be removed since they don't
                    have enough observations -- returned just in case we want
                    to know how many were filtered out
    """
    stararr = dicts[0].keys() # ASSUME that stardict is the 0th element
    otherdict = dicts[1]
    removestarlist = [star for star in stararr if len(otherdict[star]) <= min_num_obs]
    for star in removestarlist:
        for somedict in dicts:
            somedict.pop(star)
    return dicts, removestarlist
    
def sigmaclip(a, low = 3, high = 3):
    # copied exactly from scipy.stats.sigmaclip with some variation to keep 
    # account for the index/indicies that is/are being removed
    c = np.asarray(a).ravel()
    remove_arr = np.array([]) #indicies that have been removed
    delta = 1
    while delta:
        c_std = c.std()
        c_mean = c.mean()
        size = c.size
        critlower = c_mean - c_std*low
        critupper = c_mean + c_std*high
        removetemp = np.where(c < critlower) and np.where(c > critupper)
        remove_arr = np.append(remove_arr, removetemp)
        c = np.delete(c, removetemp)
        delta = size-c.size
    return c, remove_arr
    
def sigmaclip_dict(stardict, xdict, ydict, mdict, merrdict, low = 3, high = 3):
    for starID in stardict:
        mdict[starID], remove_arr = sigmaclip(mdict[starID], low, high)
        xdict[starID] = np.delete(xdict[starID], remove_arr)
        ydict[starID] = np.delete(ydict[starID], remove_arr)
        merrdict[starID] = np.delete(merrdict[starID], remove_arr)
    return [stardict, xdict, ydict, mdict, merrdict]

##########################################################
# In this module, we should have by the end:
# stardict                 -- holds the star IDs, 
# xdict, ydict             -- holds the x and y pixel values for a star
# mdict, merrdict          -- holds the magnitudes and associated magnitude errors 
# mabsdict, mabserrdict    -- holds the absolute value for the magnitude and its error
# Note: all dictionaries have array elements except for mabsdict and mabserrdict,
#   which are just floats

# Get all the data nicely from the file/table into dictionaries
stardict, xdict, ydict, mdict, merrdict = extract_data(starindexdict, tab)   
# Filter the stars that don't have enough observations 
[stardict, xdict, ydict, mdict, merrdict], removestarlist = remove_stars([stardict, xdict, ydict, mdict, merrdict], 3)
# Look at each star and make sure that observations would not mess up a fit
#   by sigma clipping them
low, high = 3, 3
stardict, xdict, ydict, mdict, merrdict = sigmaclip_dict(stardict, xdict, ydict, mdict, merrdict, low, high)
# Create dictionaries for the absolute magnitude of each star and the error for
#   that absolute magnitude
# Absolute magnitude will just be the mean of all the magnitudes for that star
# The error is the the quadratic error ex. (e1^2 + e2^2 + .. + eN^2)^(1/2) / N
mabsdict = {}
mabserrdict = {}
for star in stardict:
    mabsdict[star] = np.mean(mdict[star])
    mabserrdict[star] = np.sqrt(np.sum(merrdict[star]**2)) / len(merrdict[star])
print "%s seconds" % (time.time() - start_time) # For everything above to run ~ 30 seconds
##########################################################   

def extract_out(stardict, xdict, ydict, mdict, merrdict, mabsdict, mabserrdict): 
    """
    Purpose
    -------
    Everything is in a dictionary and this function extracts all the information
    into arrays so that the info can be 'easier' to use
    
    Parameters
    ----------
    stardict:                holds the star IDs, 
    xdict, ydict:            holds the x and y pixel values for a star
    mdict, merrdict:         holds the magnitudes and associated magnitude errors 
    mabsdict, mabserrdict:   holds the absolute value for the magnitude and its error 
    
    Note: all of them have array values except for mabsdict and mabserrdict
    
    Returns
    -------
    [xall, yall, delmall, delmerrall, mall, merrall, mabsall, mabserrall]
    xall, yall:              all of the x and y pixel points
    delmall, delmerrall:     all of the respective delta magnitudes and their errors
    mall, merrall:           all of the observed magnitudes and their errors
    mabsall, abserrall:      all of the absolute magnitudes and their errors
    
    So if looking at certain index i, xall[i], yall[i], delmall[i], .. etc. all
    correspond to that point of information, which can be easier to use/to index
    """
    xall = np.array([])
    yall = np.array([])
    mall = np.array([])
    merrall = np.array([])
    mabsall = np.array([])
    mabserrall = np.array([])
    
    delmall = np.array([])
    delmerrall = np.array([])
    for star in stardict:
        xall = np.append(xall, [x for x in xdict[star]])
        yall = np.append(yall, [y for y in ydict[star]])
        mall = np.append(mall, [m for m in mdict[star]])
        merrall = np.append(merrall, [merr for merr in merrdict[star]])
        
        mabs, mabserr = mabsdict[star], mabserrdict[star]
        mabsall = np.append(mabsall, np.ones(len(mdict[star])) * mabs)
        mabserrall = np.append(mabserrall, np.ones(len(merrdict[star])) * mabserr)
        
        delmarr = mdict[star] - mabs # array of delta mag
        delmall = np.append(delmall, [delm for delm in delmarr])
        delmerrall = np.append(delmerrall, [np.sqrt(merr**2 + mabserr**2) for merr in merrdict[star]])
    return [xall, yall, delmall, mall, merrall, mabsall, mabserrall, delmerrall] 
    
#xall, yall, delmall, delmerrall, mall, merrall, mabsall, mabserrall = extract_out(stardict, xdict, ydict, mdict, merrdict, mabsdict, mabserrdict)

def printnicely(dictionary):
    for key in dictionary:
        print key, dictionary[key]