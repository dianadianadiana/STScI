# global imports
import numpy as np
from astropy.table import Table, Column
import time

start_time = time.time()

CHIP1XLEN = CHIP2XLEN = 4096 
CHIP1YLEN = CHIP2YLEN = 2048

#####################################################
################ Filter Functions ###################
#####################################################
def remove_stars_tab(tab, min_num_obs = 4):
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
    starIDarr = np.unique(tab['id'])
    removestarlist = [star for star in starIDarr if len(np.where(tab['id'] == star)[0]) < min_num_obs] # Get a list of the stars to remove
    removetabindicies = np.array([])
    for removestar in removestarlist:
        removetabindicies = np.append(removetabindicies, np.where(tab['id'] == removestar)[0])
    removetabindicies = map(int, removetabindicies) # need to make removing indicies ints 
    tab.remove_rows(removetabindicies)
    return tab

def remove_certain_star(tab, star_names):
    '''           *** Filter function ***
    Purpose
    -------
    Remove any stars in star_names from tab
    
    Parameters
    ----------
    tab:                The Astropy table with all the information
    star_names:         The list or array of the star names to remove
    
    Return
    ------
    tab:                The filtered table after deleteing the stars in star_names
    '''
    removetabindicies = np.array([])
    for removestar in star_names:
        removetabindicies = np.append(removetabindicies, np.where(tab['id'] == removestar)[0])
    removetabindicies = map(int, removetabindicies) # need to make removing indicies ints 
    tab.remove_rows(removetabindicies)
    return tab

def sigmaclip(z, low = 3, high = 3, num = 5):
    """           
    Purpose
    -------
    Applies sigma clipping to an array
    
    Parameters
    ----------
    z:                  The array that will be sigma clipped
    low:                The lower bound of the sigma clip (Default = 3)
    high:               The higher bound of the sigma clip (Default = 3)
    num:                The maximum number of times the sigma clipping will iterate
    
    Returns
    -------
    remove_arr:         An array of the indexes that have been sigmaclipped
    
    * So if you want to get rid of those values in z; 
    do z = np.delete(z, remove_arr)
    * Copied exactly from scipy.stats.sigmaclip with some variation to keep
    account for the index(es) that is (are) being removed
    """
    c = np.asarray(z).ravel()           # this will be changing
    c1 = np.copy(c)                     # the very original array
    delta = 1
    removevalues = np.array([])
    count = 0
    while delta and count < num:
        c_std = c.std()
        c_mean = c.mean()
        size = c.size
        critlower = c_mean - c_std*low
        critupper = c_mean + c_std*high
        removetemp = np.where(c < critlower)[0]
        removetemp = np.append(removetemp, np.where(c > critupper)[0])
        removevalues = np.append(removevalues, c[removetemp])
        c = np.delete(c, removetemp)
        delta = size - c.size
        count += 1
    removevalues = np.unique(removevalues)
    remove_arr = np.array([])
    for val2remove in removevalues:
        remove_arr = np.append(remove_arr, np.where(c1 == val2remove)[0])
    remove_arr = map(int, remove_arr)
    return remove_arr

def sigmaclip_starflux(tab, low = 3, high = 3):
    """
    Purpose
    -------
    To remove any observations for each star that are not within a low sigma 
    and high simga (Ex. a star has flux values [24,24.5,25,25,25,50] --> the 
    observation with 50 will be removed from the table
    
    Paramters
    ---------
    tab:                The Astropy table with all the information
    low:                The bottom cutoff (low sigma); default is 3
    high:               The top cutoff (high sigma); default is 3
        
    Returns
    -------
    tab:                The updated Astropy table with obscure observations removed
    """
    
    removetabindices = np.array([])
    starIDarr = np.unique(tab['id'])
    for star in starIDarr:
        starindexes = np.where(tab['id'] == star)[0]
        currfluxes = tab[starindexes]['flux']
        remove_arr = sigmaclip(currfluxes, low, high)
        removetabindicies = np.append(removetabindices, starindexes[remove_arr])
    removetabindicies = map(int, removetabindicies)
    tab.remove_rows(removetabindicies)
    return tab

def sigmaclip_delflux(tab, low = 3, high = 3):
    '''
    Purpose
    -------
    To remove any observations in the data set as a whole whose delta flux is 
    not within a certain sigma
    
    Paramters
    ---------
    tab:                The Astropy table with all the information
    low:                The bottom cutoff (low sigma); default is 3
    high:               The top cutoff (high sigma); default is 3
        
    Returns
    -------
    tab:                The updated Astropy table with obscure observations removed
    '''
    delfarr = (tab['flux'] - tab['avgflux']) / tab['avgflux']   # normalized flux
    delfarr = np.asarray(delfarr)
    # sigma clipping the delta fluxes
    remove_arr = sigmaclip(delfarr, low, high)
    tab.remove_rows(remove_arr)
    return tab
    
def bin_filter(tab, xpixelarr, ypixelarr, xbin, ybin, low = 3, high = 3):
    '''
    Purpose
    -------
    
    Parameters
    ----------
    tab:                The Astropy table with all the information
    xpixelarr:          Array of the ranging values of pixels in the x direction (1D)
    ypixelarr:          Array of the ranging values of pixels in the y direction (1D)
    xbin:               Number of bins in the x direction
    ybin:               Number of bins in the y direction
    low:                The bottom cutoff (low sigma); default is 3
    high:               The top cutoff (high sigma); default is 3
    
    Returns
    -------
    [tab, zz, zztabindex]
    tab:                The updated Astropy table with observations removed if
                        didn't fit well in a bin
    zz:                 2D array of size xbin * ybin -- the final one -- where the 
                        averages of each bin are taken, and if there was nothing in 
                        a bin, the average is set to 0
    zzdel:             2D array of size xbin * ybin -- in each bin, there 
                        is an array of the delta fluxes -- this is returned 
                        in case we want to see the values of delta flux in a bin
    zztabindex:         2D array of size xbin * ybin -- in each bin, there 
                        is an array of the indexes that correspond to tab (equals
                        None if no array) -- this is returned in case we want 
                        to know which indexes of the tab belong in a bin
    '''
    # Initialize an empty 2D array for the binning;
    # Create xbinarr and ybinarr as the (lengths of xbin and ybin, respectively);
    #     to make up the bins
    # Find dx and dy to help later with binning x+dx and y+dy
    # zz is a 2D array that can be used for imshow
    zz = np.array([np.array([None for i in range(np.int(xbin))]) for j in range(np.int(ybin))])
    zzdel = np.copy(zz)
    zztabindex = np.copy(zz)
    xbin, ybin = np.double(xbin), np.double(ybin)
    xbinarr = np.linspace(np.min(xpixelarr), np.max(xpixelarr), xbin, endpoint = False)
    ybinarr = np.linspace(np.min(ypixelarr), np.max(ypixelarr), ybin, endpoint = False)
    dx, dy = xbinarr[1] - xbinarr[0], ybinarr[1] - ybinarr[0]

    # Take out all the information from the table
    xall = tab['x']
    yall = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    delmall = (tab['flux'] - tab['avgflux']) / tab['avgflux'] # normalized
    xall = np.asarray(xall)
    yall = np.asarray(yall)
    delmall = np.asarray(delmall)
    
    indexestoremove = np.array([])      # An array that will hold all the indexes 
                                        # that will need to be removed from tab
    for i, x in enumerate(xbinarr):
        for j, y in enumerate(ybinarr):
            inbin = np.where((xall >= x) & (xall < x + dx) & (yall >= y) & (yall < y + dy))[0]
            if len(inbin): # if inbin exists
                # ex. inbin = [1100, 1101, 1105] -- the indexes that correspond to tab
                #   remove_arr may be [0], so we add index 1100 to indextoremove
                #   and update inbin so that it's only [1101, 1105]
                # zztabindex will take in inbin
                # zzdel will take in the delta fluxes corresponding from tab[inbin]
                # zz will take the mean of the delta fluxes
                remove_arr = sigmaclip(delmall[inbin], low, high)
                indextoremove = np.append(indexestoremove, inbin[remove_arr])
                inbin = np.delete(inbin, remove_arr)
                zztabindex[i][j] = inbin
                zzdel = delmall[inbin]
                zz[i][j] = np.mean(delmall[inbin])
            else:
                zz[i][j] = 0
    indextoremove = map(int, indexestoremove)
    tab.remove_rows(indextoremove)
    return [tab, zz, zzdel, zztabindex]
#####################################################
#####################################################
#####################################################


#####################################################
######## Find the Real Magnitude and Fluxes #########
#####################################################
def convertmag2flux(mag, mag0 = 25, flux0 = 1):
    ''' Converts a magnitude to a flux 
    -- assume zero-point mag is 0 and flux is a constant '''
    return flux0 * 10**(.4*(mag0-mag))
    
def convertflux2mag(flux, mag0 = 25, flux0 = 1):
    ''' Converts a flux into a magnitude
    -- assume zero-point mag is 0 and flux is a constant ''' 
    return mag0 - 2.5 * np.log10(flux/flux0)
    
def make_avgflux(tab):
    # Create new columns for average flux and average flux error
    filler = np.arange(len(tab))
    c1 = Column(data = filler, name = 'avgflux',       dtype = np.float64)
    c2 = Column(data = filler, name = 'avgfluxerr',    dtype = np.float64)
    tab.add_column(c1)
    tab.add_column(c2)
    
    starIDarr = np.unique(tab['id'])
    for star in starIDarr:
        starindexes = np.where(tab['id'] == star)[0]    # the indexes in the tab of where the star is
        currfluxes = tab[starindexes]['flux']           # the current fluxes (type = class <'astropy.table.column.Column'>)
        currfluxerr = tab[starindexes]['fluxerr']       # the current flux errors (type = class <'astropy.table.column.Column'>)
        avgerror = lambda errarr: np.sqrt(np.sum(errarr**2)) / len(errarr)
        avgfluxerr = avgerror(currfluxerr)
        for i, index in enumerate(starindexes):         # input the average flux and its error
            tab[index]['avgflux'] = np.mean(currfluxes)
            tab[index]['avgfluxerr'] = avgfluxerr
    return tab
    
def make_avgmagandflux(tab):
    """
    Purpose
    -------
    To create six new columns to store the 
    (1) average magnitude               ['avgmag']
    (2) average magnitude error         ['avgmagerr'] 
    (3) flux                            ['flux']
    (4) flux error                      ['fluxerr']
    (5) average flux                    ['avgflux']
    (6) average flux error              ['avgfluxerr']
    The magnitude readings are converted into fluxes and then the average flux 
    is taken and converted to magnitude to make the 'real' magnitude.
    
    Paramters
    ---------
    tab:                The Astropy table with all the information
        
    Returns
    -------
    tab:                The updated Astropy table

    Notes
    -----
    1) Average magnitude is the converted magnitude of the average flux
    2) The error for avgmag is the the quadratic error ex. (e1^2 + e2^2 + .. + eN^2)^(1/2) / N
    3) The fluxes are just converted from the magnitudes
    """
    # Create new columns
    filler = np.arange(len(tab))
    c1 = Column(data = filler, name = 'avgmag',        dtype = np.float64)
    c2 = Column(data = filler, name = 'avgmagerr',     dtype = np.float64)
    c3 = Column(data = filler, name = 'flux',          dtype = np.float64)
    c4 = Column(data = filler, name = 'fluxerr',       dtype = np.float64)
    c5 = Column(data = filler, name = 'avgflux',       dtype = np.float64)
    c6 = Column(data = filler, name = 'avgfluxerr',    dtype = np.float64)
    tab.add_column(c1)
    tab.add_column(c2)
    tab.add_column(c3)
    tab.add_column(c4)
    tab.add_column(c5)
    tab.add_column(c6)
    
    starIDarr = np.unique(tab['id'])
    for star in starIDarr:
        starindexes = np.where(tab['id'] == star)[0]    # the indexes in the tab of where the star is
        currmags = np.array(tab[starindexes]['mag'])    # the current magnitudes 
        currmagerr = tab[starindexes]['magerr']         # the current magnitude errors (type = class <'astropy.table.column.Column'>)
        currfluxes = convertmag2flux(currmags)          # the current fluxes
        #currfluxerr = np.array([2.303*flux*magerr for flux, magerr in zip(currfluxes, currmagerr)]) # Using antilog error propagation
        currfluxerr = np.array([flux*magerr/1.086 for flux, magerr in zip(currfluxes, currmagerr)])
        # http://www.astro.wisc.edu/~mab/education/astro500/lectures/a500_lecture2_s13.pdf Slide #11
        
        avgmag = convertflux2mag(np.mean(currfluxes))   # the absolute magnitude
        avgerror = lambda errarr: np.sqrt(np.sum(errarr**2)) / len(errarr)
        avgmagerr = avgerror(currmagerr)
        avgfluxerr = avgerror(currfluxerr)
        for i, index in enumerate(starindexes):         # input the abs mag and abs magerr
            tab[index]['avgmag'] = avgmag
            tab[index]['avgmagerr'] = avgmagerr  
            tab[index]['flux'] = currfluxes[i]
            tab[index]['fluxerr'] = currfluxerr[i]
            tab[index]['avgflux'] = np.mean(currfluxes)
            tab[index]['avgfluxerr'] = avgfluxerr
    
    return tab
    
'''
tab = make_avgmagandflux(tab)
tab =  tab[np.where(tab['flux']/tab['fluxerr'] > 5)[0]] # S/N ratio for flux is greater than 5
tab, starIDarr = sigmaclip_delmagall(tab, starIDarr, low = 3, high = 3)
print 'Len of tab and starIDarr after sigmaclipping delta magnitudes'
print len(tab)
print len(starIDarr)
# Don't really need to do a bin filter as below
#xpixelarr, ypixelarr = np.arange(CHIP1XLEN), np.arange(CHIP1YLEN + CHIP2YLEN)
#xbin, ybin = 10, 10 
#tab, zz, zzdel, zztabindex = bin_filter(tab, xpixelarr, ypixelarr, xbin, ybin, low=3, high=3)
print "%s seconds for filtering the data" % (time.time() - start_time) # For everything to run ~ 111 seconds
'''

#####################################################
#####################################################
#####################################################


#####################################################
######### Extract data into dictionaries ############
##################### OPTIONAL ######################
#####################################################
    
def extract_data_dicts(starIDarr, datatab):
    '''
    Purpose
    -------
    Extract all the data (star ID, x and y pixels, magnitude, and magnitude errors)
    into dictionaries
    
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
        absmdict[starID] = datatab[starindexes[0]]['avgmag']
        absmerrdict[starID] = datatab[starindexes[0]]['avgmagerr']
    return [stardict, xdict, ydict, mdict, merrdict, absmdict, absmerrdict]

#stardict, xdict, ydict, mdict, merrdict, absmdict, absmerrdict = extract_data_dicts(starIDarr, tab)

#####################################################
#####################################################
#####################################################
   
##########
## Read in the data
#path = '/Users/dkossakowski/Desktop/Data/'
#datafil = 'wfc_f606w_r5.lflat'
#data = np.genfromtxt(path + datafil)
#
## Create an Astropy table
## tab[starID][0]: starID; ...[2]: chip#; ...[3]: x pixel; ...[4]: y pixel; 
## ...[5]: magnitude; ...[6]: magnitude error; rest: dummy
#names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr', 'd1', 'd2', 'd3']
#types = [int, int, int, np.float64, np.float64, np.float64, np.float64,
#         float, float, float]
#
#tab = Table(data, names=names, dtype=types)
#tab.remove_columns(['d1','d2','d3']) # remove the dummy columns  
#tab.sort(['id'])                     # sort the table by starID
##tab = tab[10000:20000]
#starIDarr = np.unique(tab['id'])     # collect all the star IDs
#########

#####################################################
################# Apply Filters #####################
#####################################################
'''
print 'Len of tab and starIDarr before anything'
print len(tab)
print len(starIDarr) 
tab =  tab[np.where((tab['mag'] <= 25) & (tab['mag'] >= 13))[0]] # (13,25)
print 'Len of tab after constraining the magnitudes'
print len(tab)
print 'Len of tab and starIDarr after 1st min_num_obs'
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)
print len(tab)
print len(starIDarr) 
tab, starIDarr = sigmaclip_starmag(tab, starIDarr, low = 3, high = 3)
print 'Len of tab and starIDarr after sigmaclipping each star'
print len(tab)
print len(starIDarr)
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)
print 'Len of tab and starIDarr after 2nd min_num_obs'
print len(tab)
print len(starIDarr)
'''
#####################################################
#####################################################
#####################################################