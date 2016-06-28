import numpy as np
from astropy.table import Table, Column
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
tab.remove_columns(['d1','d2','d3']) # remove the dummy columns  
tab.sort(['id'])                     # sort the table by starID
starIDarr = np.unique(tab['id'])     # collect all the star IDs


#####################################################
################ Filter Functions ###################
#####################################################
def remove_stars_tab(tab, starIDarr, min_num_obs = 4):
    """            *** Filter function ***
    Purpose
    -------
    Removes the stars with less than a min num of observations from the table
    
    Parameters
    ----------
    tab:                The Astropy table with all the information
    starIDarr:          The array of unique star IDs in the table
    min_num_obs:        The minimum number of observations required for a star 
                        to have (default = 4)
    
    Returns
    -------
    tab:                The filtered table after deleting all the stars with not
                        enough observations
    starIDarr:          The filtered array after deleting all the stars with not
                        enough observations
    removestarlist:     The list of stars that need to be removed since they don't
                        have enough observations -- returned just in case we want
                        to know how many were filtered out
    """
    # get an index list of the stars to remove in starIDarr
    removestarindexlist = [i for i in range(len(starIDarr)) if len(np.where(tab['id'] == starIDarr[i])[0]) < min_num_obs]
    # get a name list of the stars to remove (to use for the table)
    removestarlist = [star for star in starIDarr[removestarindexlist]]
    # remove the stars from the starIDarr 
    starIDarr = np.delete(starIDarr, removestarindexlist)
    #removetabindicies = np.asarray([np.where(tab['id'] == removestar)[0] for removestar in removestarlist])
    # had to do everything below since the above statement wouldn't unravel nicely..
    removetabindicies = np.array([])
    for removestar in removestarlist:
        removetabindicies = np.append(removetabindicies, np.where(tab['id'] == removestar)[0])
    removetabindicies = map(int, removetabindicies) # need to make removing indicies ints 
    tab.remove_rows(removetabindicies)
    return tab, starIDarr, removestarlist

def sigmaclip(c, low = 3, high = 3):
    # copied exactly from scipy.stats.sigmaclip with some variation to keep 
    # account for the index(es) that is(are) being removed
    c = np.asarray(c).ravel()
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

def sigmaclip_tab(tab, starIDarr, low = 3, high = 3):
    """
    Purpose
    -------
    To get rid of any observations that are not within a low sigma and high simga
    
    Paramters
    ---------
    tab:                The Astropy table with all the information
    starIDarr:          The array of unique star IDs in the table
    low:                The bottom cutoff (low sigma); default is 3
    high:               The top cutoff (high sigma); default is 3
        
    Returns
    -------
    tab:                The updated Astropy table with obscure observations removed
    starIDarr:          The array with all the star IDs; should not be modified
                        but returned for consistency
    """
    removetabindices = np.array([])
    for star in starIDarr:
        starindexes = np.where(tab['id'] == star)[0]
        currmags = tab[starindexes]['mag']
        currmags, remove_arr = sigmaclip(currmags, low, high)
        remove_arr = map(int, remove_arr)
        removetabindicies = np.append(removetabindices, starindexes[remove_arr])
    removetabindicies = map(int, removetabindicies)
    tab.remove_rows(removetabindicies)
    return tab, starIDarr
#####################################################
#####################################################
#####################################################


#####################################################
################# Apply Filters #####################
#####################################################
print 'Len of tab and starIDarr before anything'
print len(tab)
print len(starIDarr) 
print 'Len of tab and starIDarr after 1st min_num_obs'
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 3)
print len(tab)
print len(starIDarr) 
tab, starIDarr = sigmaclip_tab(tab, starIDarr[:], low = 3, high = 3)
print 'Len of tab and starIDarr after sigmaclipping'
print len(tab)
print len(starIDarr)
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 3)
print 'Len of tab and starIDarr after 2nd min_num_obs'
print len(tab)
print len(starIDarr)
#####################################################
#####################################################
#####################################################


#####################################################
########### Find the Absolute Magnitude #############
#####################################################
def make_absmag(tab, starIDarr):
    """
    Purpose
    -------
    To create two columns to store the absolute magnitude and its error
    
    Paramters
    ---------
    tab:                The Astropy table with all the information
    starIDarr:          The array of unique star IDs in the table
        
    Returns
    -------
    tab:                The updated Astropy table with abs mag and its error
    starIDarr:          The array with all the star IDs; should not be modified
                        but returned for consistency
    
    """
    # Create two new columns
    filler = np.arange(len(tab))
    c1 = Column(data = filler, name = 'abs mag')
    c2 = Column(data = filler, name = 'abs magerr')
    tab.add_column(c1)
    tab.add_column(c2)

    for star in starIDarr:
        starindexes = np.where(tab['id'] == star)[0]    # the indexes in the tab of where the star is
        currmags = tab[starindexes]['mag']              # the current magnitudes (type = <class 'astropy.table.column.Column'>)
        currmagerr = tab[starindexes]['magerr']         # the current magnitude errors (type = class <'astropy.table.column.Column'>)
        absmag = np.mean(currmags)                      # the absolute magnitude
        
        for index in starindexes:                       # input the abs mag and abs magerr
            tab[index]['abs mag'] = absmag
            tab[index]['abs magerr'] = np.sqrt(np.sum(currmagerr**2)) / len(currmagerr)        
    return tab, starIDarr

tab, starIDarr = make_absmag(tab, starIDarr)
print "%s seconds" % (time.time() - start_time) # For everything to run ~ 111 seconds

#####################################################
#####################################################
#####################################################


#####################################################
######### Extract data into dictionaries ############
##################### OPTIONAL ######################
#####################################################
    
def extract_data(starIDarr, datatab):
    '''
    Purpose
    -------
    Extract all the data (star ID, x and y pixels, magnitude, and magnitude errors)
    
    Parameters
    ----------
    starIDarr:      This is the array of star IDs
    datatab:        Astropy table that was created from a file

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
    absmdict:       A dictionary where the keys are the star IDs and the values
                    are the absolute magnitude corresponding to that star ID 
    absmerrdict:    A dictionary where the keys are the star IDs and the values
                    are the absolute magnitude errors corresponding to that star ID 
    '''
    
    stardict = {}
    xdict = {}
    ydict = {}
    mdict = {}
    merrdict = {}
    absmdict = {}
    absmerrdict = {}
    for starID in starIDarr:
        starindexes = np.where(tab['id'] == starID)[0]
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
        for i in starindexes:
            chip = datatab[i]['chip']
            xpixel = datatab[i]['x']
            ypixel = datatab[i]['y']
            mag = datatab[i]['mag']
            magerr = datatab[i]['magerr']
            xarr = np.append(xarr, xpixel)
            yarr = np.append(yarr, ypixel) if chip == 2 else np.append(yarr, ypixel + CHIP2YLEN)
            marr = np.append(marr, mag)
            merrarr = np.append(merrarr, magerr)
        stardict[starID] = starID
        xdict[starID] = xarr
        ydict[starID] = yarr
        mdict[starID] = marr
        merrdict[starID] = merrarr
        absmdict[starID] = datatab[starindexes[0]]['abs mag']
        absmerrdict[starID] = datatab[starindexes[0]]['abs magerr']
    return [stardict, xdict, ydict, mdict, merrdict, absmdict, absmerrdict]
    
#stardict, xdict, ydict, mdict, merrdict, absmdict, absmerrdict = extract_data(starIDarr, tab)

#####################################################
#####################################################
#####################################################