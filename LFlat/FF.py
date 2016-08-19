"""
Low-Frequency Flat (LFlat) Program using a Markov Chain Monte Carlo (MCMC) optimizer

:Author: Diana Kossakowski

:Organization: Space Telescope Science Institute

:History:
    * Aug 2016 Finished

:Helpful Links
    * emcee: http://dan.iel.fm/emcee/current/
    * emcee example: http://dan.iel.fm/emcee/current/user/line/
    
Examples
--------
To call from command line::
    python FF.py --datafile '/user/dkossakowski/sbc_f125lp_flux.phot' --chosenfunc 'poly' --n 3
    --nwalkers 100 --nsteps 700 --filmcmc '/user/dkossakowski/poly3_100_700_mcmc.txt'
    --filcoeff '/user/dkossakowski/poly3_100_700_coeff.txt' --figpath '/user/dkossakowski/poly3_100_700_'

"""
###########################################
################ Imports ##################
###########################################

# Global Imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
import time
import argparse

# Local Imports
import emcee

# Functions
from sympy import *
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as leg
from scipy import integrate

# Data
from astropy.table import Table, Column        

# Plotting
import corner                              # local import
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

###########################################
###########################################
###########################################


###########################################
######### Functions to Optimize ###########
###########################################

'''
Note for all functions: 
   * lambdify needs to take in a list and not array; and for a 2d
     polynomial, the constant term needs to be 1 + 0*x but lambdify
     makes it just 1, which becomes a problem when trying to fit the 
     function and hence why the 0th element turns into np.ones(len(x))
Returns for all functions:
    funclist -- A list of the different components of the function 
                (elements are Symbol types) -- useful for printing
    f        -- A function made from funclist and takes in two
                parameters, x and y
'''
def norder2dpoly(n):
    ''' 
    Purpose
    -------
    Create the 2D nth order polynomial
    
    How it works:
        a 2nd order can be grouped like: 1; x1y0 x0y1; x2y0 x1y1 x0y2
        (where x0 = x**0, x1 = x**1 and so on)
        So the degree of x starts at a certain number (currnum) and decreases
        by one, while the degree of y starts at 0 and increase by one until currnum
    Note: 
        * Just found out (2 weeks later after making this), that there 
        exists numpy.polynomial.polynomial.polyvander2d which does what I made
    '''
    x = Symbol('x')
    y = Symbol('y')
    funcarr = np.array([])
    for currnum in range(n+1):
        xi = currnum
        yi = 0
        while yi <= currnum:
            funcarr = np.append(funcarr, x**xi * y**yi)
            yi += 1
            xi -= 1
    funclist = funcarr.tolist()     # lambdify only takes in lists and not arrays
    f = lambdify((x, y), funclist)  # lambdify looks at 1 + 0*x as 1 and makes f[0] = 1
    return funclist, f              # return the list in case we want to look at how the function is
        
def norder2dcheb(nx, ny):
    ''' 
        Purpose
        -------
        Create the 2D nx th and ny th order Chebyshev polynomial using
        numpy.polynomial.chebyshev.chebvander2d(x, y, [nx, ny])
    '''
    x = Symbol('x')
    y = Symbol('y')
    funcarr = cheb.chebvander2d(x,y,[nx,ny])
    funcarr = funcarr[0]            # Because chebvander2d returns a 2d matrix
    funclist = funcarr.tolist()     # lambdify only takes in lists and not arrays
    f = lambdify((x, y), funclist)  # Note: lambdify looks at 1 as 1 and makes f[0] = 1 and not an array
    return funclist, f
    
def norder2dlegendre(nx, ny):
    ''' 
        Purpose
        -------
        Create the 2D nx th and ny th order Legendre polynomial using
        numpy.polynomial.legendre.legvander2d(x, y, [nx, ny])
    '''
    x = Symbol('x')
    y = Symbol('y')
    funcarr = leg.legvander2d(x,y,[nx,ny])
    funcarr = funcarr[0]            # Because chebvander2d returns a 2d matrix
    funclist = funcarr.tolist()     # lambdify only takes in lists and not arrays
    f = lambdify((x, y), funclist)  # Note: lambdify looks at 1 as 1 and makes f[0] = 1 and not an array
    return funclist, f
    
def get_function(chosenfunc, n):
    nx = ny = n
    if chosenfunc == 'poly':
        func2read, func2fit = norder2dpoly(n) 
    elif chosenfunc == 'cheb':
        func2read, func2fit = norder2dcheb(nx, ny)        # nx th and ny th order 2d Chebyshev Polynomial
    elif chosenfunc == 'leg':
        func2read, func2fit = norder2dlegendre(nx, ny) 
    return [func2read, func2fit]
    
########## Integration
def funcintegrate(x, y, coeff, chosenfunc, n):
    func2read, func2fit = get_function(chosenfunc, n)
    return np.sum(func2fit(x, y) * coeff)

def bounds_x():
    if SCALE2ONE:
        return [-1, 1]
    else:
        return [0, CHIPXLEN]
def bounds_y():
    if SCALE2ONE:
        return [-1, 1]
    else:
        return [0, CHIPYLEN]
def area():
    return np.sum(np.abs(bounds_y() + bounds_x()))
##########

###########################################
###########################################
########################################### 
    
###########################################
######### Creating the Table ##############
###########################################

def do_table(fil, names, types, removenames=[]):
    ''' Make the Astropy Table -- Assuming that we are given flux and fluxerr '''
    print '******************************************'
    print '*************** START TABLE **************'
    print '******************************************'
    data = np.genfromtxt(fil)
    tab = Table(data, names=names, dtype=types)
    tab.remove_columns(removenames)             # Remove columns that are not useful
    print 'The names of the columns in the Table:'
    print tab.colnames
    print '******************************************'
    print '**************** END TABLE ***************'
    print '******************************************\n'
    return tab

###########################################
###########################################
########################################### 


###########################################
########### Making the Filters ############
########################################### 

def remove_stars_tab(tab, min_num_obs=4):
    """            *** Filter function ***
    Purpose
    -------
    Removes the stars with less than a min num of observations from the table
    
    Parameters
    ----------
    tab:                The Astropy table with all the information
    min_num_obs:        The minimum number of observations required for a star 
                        to have (default = 4)
    
    Returns
    -------
    tab:                The filtered table after deleting all the stars with not
                        enough observations
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
    """           *** Filter function ***
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
        removetabindices = np.append(removetabindices, starindexes[remove_arr])
    removetabindices = map(int, removetabindices)
    tab.remove_rows(removetabindices)
    return tab

def sigmaclip_delflux(tab, low = 3, high = 3):
    '''           *** Filter function ***
    Purpose
    -------
    To remove any observations in the data set as a whole whose delta flux is not within a certain sigma
    
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
    
def make_avgflux(tab):
    ''' Create new columns for average flux and average flux error '''  
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
    
###########################################
###########################################
########################################### 


###########################################
########## Filtering the Table ############
###########################################

def do_filter(tab, min_num_obs, flux_ratio, low, high, remove_stars):
    # this is assuming that tab only has flux values (NO magnitude)
    '''
    Purpose
    -------
    To filter the Astropy Table
    
    Parameters
    ----------
    tab:            The Astropy Table with all the information
    min_num_obs:    The minimum number of observations required for each star
    flux_ratio:     The minimum flux ratio for each observation
    low / high:     The lower and upper sigma limits for sigma clipping
    remove_stars:   A list of specific stars that should be removed
        
    Return
    ------
    tab:            The filtered Astropy Table
    '''
    starIDarr = np.unique(tab['id'])                  # collect all the star IDs
    num_stars0 = np.double(len(starIDarr))
    num_obs0 = np.double(len(tab))
    print '******************************************'
    print '************** START FILTER **************'
    print '******************************************'
    print 'Initial number of observations:\t', len(tab) 
    
    tab = remove_certain_star(tab, remove_stars)      
    tab = remove_stars_tab(tab, min_num_obs)                         # Remove rows with less than min num of observations
    tab = make_avgflux(tab)                                          # Create columns for 'avgflux' and 'avgfluxerr'
    tab = sigmaclip_starflux(tab, low, high)                         # Sigmaclip the fluxes for each star
    tab = sigmaclip_delflux(tab, low, high)                          # Sigmaclip the delta fluxes as a whole
    tab = tab[np.where(tab['flux']/tab['fluxerr'] > flux_ratio)[0]]  # S/N ratio for flux is greater than flux_ratio

    print 'Number of observations after filtering:\t', len(tab)
    print 'Percent of observations kept:\t', len(tab)/num_obs0 * 100
    print 'Number of stars after filtering:\t', len(starIDarr)
    print 'Percent of stars kept:\t', len(starIDarr)/num_stars0 * 100
    print '******************************************'
    print '*************** END FILTER ***************'
    print '******************************************\n'
    return tab

###########################################
###########################################
###########################################

###########################################
########## Setting Up MCMC ################
###########################################

def lnprior(params):
    realparams = params
    if USE_F:
        realparams, lnf = params[:-1], params[-1]
    if -.1 < realparams[0] < .1:
        return 0.0
    return -np.inf
    
def chisqstar(starrows, params, func2fit):
    ''' 
    Purpose
    -------
    Compute the chi-square of one star
    
    Parameters
    ----------
    starrows:   The rows of the table corresponding to a certain star (Astropy Table subset)
    params:     The parameters corresponding to the function that is being fit
    func2fit:   The function that is being fit
    
    Return
    ------
    starsqsum:  The chi-square value for the star
    '''
    starvals = starrows['flux']
    starvalerrs = starrows['fluxerr']
    func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p)) + 1           # The 'delta' function
    if SCALE2ONE:
        deltas = [func(params, (row['x']-CHIPXLEN/2)/(CHIPXLEN/2), (row['y']-CHIPYLEN/2)/(CHIPYLEN/2)) for row in starrows]
    else:
        deltas = [func(params, row['x'], row['y']) for row in starrows]
    avgf = np.mean(starvals/deltas)                                              # Our 'expected' value for the Flux
    
    def get_sigmasq():
        if USE_F:
            return np.asarray((starvalerrs/deltas))**2 + np.exp(2*lnf)*avgf**2
        return np.asarray((starvalerrs/deltas))**2
            
    starsq = (starvals/deltas - avgf)**2 / get_sigmasq() + np.log(get_sigmasq()) # ignore the 2pi since that is just a constant for the chisq
    starsq = np.asarray(starsq)
    starsqsum = np.sum(starsq)
    return starsqsum   
       
def lnlike(params, tab, func2fit):
    realparams = params
    if USE_F:
        realparams, lnf = params[:-1], params[-1]
    starIDarr = np.unique(tab['id'])   
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    starsqsums = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]], realparams, func2fit) for star in starIDarr])     # an array of the sq of sums for each star
    totalsqsum = np.sum(starsqsums)
    return -0.5 * totalsqsum

def lnprob(params, tab, func2fit):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, tab, func2fit)

def get_pos(ndim, nwalkers, scale_factor, base_coeff):
    ''' Creates the initial tiny gaussian balls '''
    pos = [base_coeff + scale_factor*np.random.randn(ndim) for i in range(nwalkers)]
    def filter_pos(pos):
        # filters the pos by making sure they are within the prior
        remove_pos = np.array([])    
        for i, elem in enumerate(pos):
            lp = lnprior(elem)
            if not np.isfinite(lp):
                remove_pos = np.append(remove_pos, i)
        pos = np.delete(pos, remove_pos, axis = 0)
        return pos
    start_time = time.time()
    pos = filter_pos(pos)
    
    # the process below ensures that number of walkers equals the length of pos
    while len(pos) - nwalkers != 0 and len(pos) - nwalkers < 0:
        num = len(pos) - nwalkers
        newpos = [base_coeff + scale_factor*np.random.randn(ndim) for i in range(-1*num)]
        pos = np.append(pos, newpos, axis = 0)
        pos = filter_pos(pos)
        if len(pos) - nwalkers > 0:
            difference = len(pos) - nwalkers
            pos = pos[:-difference]
        if time.time()-start_time > 45.0:
            warnings.warn("Warning: Finding the intial MCMC walkers is taking too long")
    return pos
    
###########################################
###########################################
###########################################

###########################################
############## Doing MCMC #################
###########################################
    
def do_MCMC(tab, nsteps, nwalkers, chosenfunc, n, scale_factor, burnin, txtfil, mcmcfil):
    '''
    Purpose
    -------
    Do the Markov Chain Monte Carlo MCMC runs
    
    Parameters
    ----------
    tab:            The Astropy Table with all the information
    nsteps:         The number of steps each walker takes
    nwalkers:       The number of walkers (must be even)
    chosenfunc:     The functional form that is being fit
    n:              The order of the functional form
    scale_factor:   The size of the gaussian balls
    burnin:         The number of steps taken to run mcmc and then reset the sampler afterwards
    txtfil:         The text file of where each location of each walker for each step is stored
    mcmcfil:        The text file to store the final MCMC coefficients
    
    Returns
    -------
    [sampler, samples, valswerrs, mcmccoeff,  f, ndim]
    sampler:        The MCMC sampler (which contains all the information)
    samples:        The samples that is used for the analysis
    valswerrs:      A List or Array of tuples where the tuple is 3 elements; 
                    tuple[0]: mcmc coefficient, tuple[1]: top error, tuple[2]: bottom error
    mcmccoeff:      The final mcmc coefficients 
    f:              The fudge factor
    ndim:           The number of dimensions (used for later functions)
    '''
    print '******************************************'
    print '*************** START MCMC ***************'
    print '******************************************'
    
    print 'Number of walkers:', nwalkers
    print 'Number of steps:', nsteps
    
    ### Get the function
    func2read, func2fit = get_function(chosenfunc, n)
    print 'Function that is being fit:', chosenfunc + str(n)
    print func2read
    ###
    
    ### Reduce the Table so that it doesn't have unused Columns that take up memory/time
    tabreduced = np.copy(tab)               
    tabreduced = Table(tabreduced)
    tabreduced.remove_columns(['avgflux','avgfluxerr'])
    ###
    
    ### Set up the initial coefficients 
    if USE_F:
        initialcoeff = np.zeros(len(func2read)+1)
        initialcoeff[-1] = np.log(F)
    else:
        initialcoeff = np.zeros(len(func2read))
    ###
    
    ### Determine the initial locations of the walkers
    ndim = len(initialcoeff)
    start_time = time.time()
    pos = get_pos(ndim, nwalkers, scale_factor, base_coeff=initialcoeff)
    print 'Time it took to get the initial positions of the walkers:', time.time() - start_time, 'seconds'
    ### 
    
    start_time = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(tabreduced, func2fit,))
    
    if burnin:
        pos = sampler.run_mcmc(pos, burnin)[0]
        sampler.reset()

    if txtfil:
        writefil = txtfil 
        f = open(writefil, "w")
        f.write('#chosenfunc: ' + chosenfunc + str(n) + '\n')
        f.write('#nsteps: '   + str(nsteps)     + '\n')
        f.write('#nwalkers: ' + str(nwalkers)   + '\n')
        f.write('#ndim: '     + str(ndim)       + '\n')
        f.close()
        
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=True)):
        if i%20 == 0: print 'step #', i 
        ### put the Walker Path GIF here
        if txtfil:
            position = result[0]
            f = open(writefil, "a")
            f.write('#nstep ' + str(i) + '\n')
            for k in range(position.shape[0]):
                #f.write('{0:d} {1:s}\n'.format(k, "".join(str(position[k]))))
                for elem in position[k]:
                    f.write(str(elem) + ' ')
                f.write('\n')
            f.close()
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    
    print 'Time it took to do MCMC:'
    print time.time() - start_time, 'seconds'
    print (time.time() - start_time)/60., 'minutes'
    print (time.time() - start_time)/3600., 'hours'
    
    ### Get the true values
    if USE_F:
        samples[:, -1] = np.exp(samples[:, -1])
    valswerrs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    valswerrs = np.asarray(valswerrs)
    mcmccoeff = valswerrs.T[0]   
    mcmccoeff[0] += 1 # Because we defined p0 as p0-1 so we are just adding the 1 back in
    ###
    
    ###
    if USE_F:
        f = mcmccoeff[-1]
        mcmccoeff = mcmccoeff[:-1]
    else:
        f = 0
    ###
    
    ### Normalize
    int_resmcmc = integrate.nquad(funcintegrate, [bounds_x(), bounds_y()], args = (mcmccoeff,chosenfunc,n,))[0]  
    int_resmcmc /= area()
    mcmccoeff /= int_resmcmc
    valswerrs /= int_resmcmc
    print
    print 'MCMC Coefficients:'
    print mcmccoeff
    ###
    
    if mcmcfil:
        writefil = mcmcfil
        f = open(writefil, "w")
        f.close()
        
        # Make it easy to copy and paste
        for coeff in mcmccoeff:
            f = open(writefil, "a")
            f.write(str(coeff)+', ')
            f.close()
        # Format it just down the line
        f = open(writefil, "a")
        f.write('\n \n')
        f.close()
        for coeff in mcmccoeff:
            f = open(writefil, "a")
            f.write(str(coeff)+'\n')
            f.close()
        # Include the errors  
        f = open(writefil, "a")
        f.write('\n \n')
        f.close()
        for coeff in valswerrs:
            f = open(writefil, "a")
            f.write(str(coeff[0]) + ' +' + str(coeff[1]) + ' -' + str(coeff[2]) + '\n')
            f.close()
            
        # The function written out
        f = open(writefil, "a")
        f.write('\n \n')
        f.write(chosenfunc+str(n) + '\n')
        f.close()
        
        for char in func2read:
            f = open(writefil, "a")
            f.write(str(char)+'\n')
            f.close()
    
    print '******************************************'
    print '**************** END MCMC ****************'
    print '******************************************\n'
    return [sampler, samples, valswerrs, mcmccoeff,  f, ndim]

def get_samplerchain(sampler, start, end, ndim, nsteps):
    ''' Creates a new samples variable depending on which steps to start and end with '''
    if start < 0: start = 0
    if end > nsteps: end = nsteps
    samples = sampler.chain[:, start:end, :].reshape((-1, ndim))
    return samples
    
def get_truevalues(samples):
    ''' Takes the new samples and figures out the new MCMC coefficients '''
    if USE_F:
        samples[:, -1] = np.exp(samples[:, -1])
    valswerrs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    valswerrs = np.asarray(valswerrs)
    mcmccoeff = valswerrs.T[0]   
    mcmccoeff[0] += 1 # Because we defined p0 as p0-1 so we are just adding the 1 back in
    
    if USE_F:
        f = mcmccoeff[-1]
        mcmccoeff = mcmccoeff[:-1]
    else:
        f = 0
        
    int_resmcmc = integrate.nquad(funcintegrate, [bounds_x(), bounds_y()], args = (mcmccoeff,chosenfunc,n,))[0]  
    int_resmcmc /= area()
    mcmccoeff /= int_resmcmc
    valswerrs /= int_resmcmc
    return [mcmccoeff, valswerrs, f]
    
###########################################
###########################################
###########################################

###########################################
############ FITS File ####################
###########################################

def create_fits(fitsfil, chosenfunc, n, coeff):
    func2read, func2fit = get_function(chosenfunc, n)
    fits.writeto(fitsfil, convert2mesh(func2fit, coeff, xpixel=np.arange(CHIPXLEN), ypixel=np.arange(CHIPYLEN))[2], clobber=True)

###########################################
###########################################
###########################################


###########################################
############### Plotting ##################
###########################################

def plotwalkerpaths(sampler, coeff_num, start, end, nsteps):
    ''' 
    Purpose
    -------
    Return the plots of walker paths for your choice of num_subplots -- 
    Warning: if num_subplots is greater or equal to 5, then the figure gets too crowded.
    
    Parameters
    ----------
    sampler:        The MCMC sampler
    coeff_num:      An array or list of coefficients whose walker paths will be plotted
    start:          The beginning of the sampler chain
    end:            The end of the sampler chain
    nsteps:         The number of steps each walker takes
    
    Return
    ------
    fig:            The figure with subplots of the walker paths for the given coeff_num
    '''
    if start < 0: start = 0
    if end > nsteps: end = nsteps
    num_subplots = len(coeff_num)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    for axnum, coeff_i in zip(range(num_subplots), coeff_num):
        axes[axnum].plot(sampler.chain[:, start:end, coeff_i].T, color="k", alpha=0.4)
        axes[axnum].yaxis.set_major_locator(MaxNLocator(5))
        axes[axnum].set_ylabel('Coeff #' + str(coeff_i))
    fig.tight_layout(h_pad=0.0)
    return fig

def plotwalkerpathsmult(sampler, savefigloc, chosenfunc, n):
    ''' 
    Purpose
    -------
    Save the plots of the walker paths for ALL coefficients. 
    Each figure/plot will have 4 subplots.
    This function SAVES the figures.
    
    Parameters
    ----------
    sampler:        The MCMC sampler
    savefigloc:     The folder location of where the figures will be saved
    chosenfunc:     The name of the function that is being fit
    n:              The order of the 2d chosenfunc
    
    Return
    ------
    This function does not return anything but SAVES the figures to savefigloc
    '''
    func2read, func2fit = get_function(chosenfunc, n)
    num_plots = len(func2read)/4 + 1 if len(func2read)%4 != 0 else len(func2read)/4 
    for index, i in enumerate(np.arange(num_plots)*4):
        start = i
        end = i+4 
        if end > len(func2read): end = len(func2read)
        fig, axes = plt.subplots(4, 1, sharex=True)
        axes[3].set_xlabel('Number of Steps', fontsize=16)
        for axnum, coeffnum in zip(range(end-start), np.arange(start, end)):
            axes[axnum].plot(sampler.chain[:, :, coeffnum].T, color="k", alpha=0.4)
            axes[axnum].yaxis.set_major_locator(MaxNLocator(5))
            axes[axnum].set_ylabel('Coeff #' + str(axnum), fontsize=16)
        fig.tight_layout(h_pad=0.0)
        fig.savefig(savefigloc + 'walkerpath' + str(index+1) + '.png', dpi = 500)

def plottriangle(samples, func2read):
    ''' Returns the triangle plot given the samples '''
    labels = [str(i) + "th Coeff" for i in range(len(func2read))]
    labels[1] = "1st Coeff"
    labels[2] = "2nd Coeff"
    labels[3] = "3rd Coeff"
    fig = corner.corner(samples, labels=labels)
    return fig
       
##########
def convert2mesh(func2fit, coeff, xpixel, ypixel):
    ''' Creates a mesh using the function, the coefficients and the x and y pixel values '''
    if SCALE2ONE:
        xpixel = (xpixel - CHIPXLEN/2)/(CHIPXLEN/2)
        ypixel = (ypixel - CHIPYLEN/2)/(CHIPYLEN/2)
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) 
    fmesh = func2fit(xx, yy)
    coeff = np.asarray(coeff)
    fmesh[0] = np.ones(len(xpixel))
    zzfit = np.sum(fmesh * coeff, axis = 0)
    return [xx, yy, zzfit]
##########

def plotmesh(a, title = ''):
    ''' Returns the figure of the mesh fit plot '''
    X, Y, Z = a                   # a is the xx yy and zz (2d array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red')
    ax.set_title(title)
    plt.legend()
    return fig

def plotimg(img, title = '', fitplot = False):
    '''
    Purpose
    -------
    Return an imshow figure of the image
    
    Parameters
    ----------
    img:        The image (2d array)
    title:      The title of the figure: Default = ''
    fitplot:    True if the image is an LFlat -- ensures that the midpoint of the
                colorbar is 1
                False if the image is not an LFlat
    
    Return
    ------
    fig:        The figure with the imshow plot of the img
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('X Pixel', fontsize = 18);  ax.set_ylabel('Y Pixel', fontsize = 18)
    extent = (0, CHIPXLEN, 0, CHIPYLEN)    

    if fitplot:
        scale = np.max([np.max(img) - 1, 1 - np.min(img)])
        vmin = 1 - scale
        vmax = 1 + scale
    else:
        vmin = np.min(img)
        vmax = np.max(img)
    cax = ax.imshow(np.double(img), cmap = 'viridis', interpolation='nearest', \
                                    origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    if fitplot:
        c = np.linspace(1 - scale, 1 + scale, num = 9)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04, ticks = c)
        cbar.ax.set_yticklabels(c)  # vertically oriented colorbar
    else:
        fig.colorbar(cax, fraction=0.046, pad=0.04)
    return fig
    
###########################################
###########################################
###########################################


###########################################
################ Misc. ####################
###########################################
  
########### If we want to save information of table to a file 
########### Star ID, Fluxes, Average Flux, Standard Deviation of Flux, Flux Errors  
def tabinfo(tab, writefil):
    stars = np.unique(tab['id'])
    f = open(writefil, "w")
    f.close()
    for star in stars:
        tabstar = tab[np.where(tab['id']==star)[0]]
        f = open(writefil, "a")
        f.write('Star #' + str(star) + '\n')
        f.write('Fluxes: ' + str(np.asarray(tabstar['flux'])))
        f.write('\nAverage Flux: ' + str(np.mean(tabstar['flux'])))
        f.write('\nStd Flux: ' + str(np.std(tabstar['flux'])))
        f.write('\nFlux Errors: ' + str(np.asarray(tabstar['fluxerr'])))
        f.write('\n\n')
        f.close()
##########

########## If we want to see the evolution of chi-squared by varying ALL the coefficients
def minimizechi(tab, chosenfunc, n, initialcoeff, finalcoeff, num):
    func2read, func2fit = get_function(chosenfunc, n)
    diffarr = (finalcoeff - initialcoeff)/np.double(num)
    chisqarr = np.array([])
    for incr in range(num*4+1):
        currparams = initialcoeff + incr*diffarr
        currchisq = lnlike(currparams, tab, func2fit) * -2.
        chisqarr = np.append(chisqarr, currchisq)
    return [range(len(chisqarr)), chisqarr]
def plotminimizechi(tab, chosenfunc, n, initialcoeff, mcmccoeff, num=20):
    x, y = minimizechi(tab, chosenfunc, n, initialcoeff, mcmccoeff, num)
    fig = plt.figure(facecolor = 'white', figsize = (12,5))
    plt.title('Evolution of $\chi^2$', fontsize=18)
    plt.axhline(y = 10516.8849567, ls = '--', label = 'Sum of scatters with current L-Flat: ' + str(10516.88))
    plt.plot(x, y,'o')
    plt.plot(x[np.argmin(y)], y[np.argmin(y)], 'ko', label = 'Sum of scatters with new L-Flat: ' + str(np.min(y)))
    plt.ylabel('$\chi^2$', fontsize=18)
    labels = ['Initial Coeff.', 'MCMC Coeff.', 'Beyond Coeff.']
    plt.xticks([0,num*2,num*4], labels, rotation=10)
    plt.legend(loc=0)
    return fig   
##########

########## If we want to see the evolution of chi-squared by just varying ONE coefficient
########## The coefficient is varied by dx until it reaches the current_chisq * mult
def chisq_varyone(tab, chosenfunc, n, mcmccoeff, coeff_num, dx, mult):
    func2read, func2fit = get_function(chosenfunc, n)
    chisqorig = lnlike(mcmccoeff, tab, func2fit) * -2
    mcmccoefforig = np.copy(mcmccoeff)
    currchisq = chisqorig
    chisqleft = []
    k=0
    while currchisq <= mult * chisqorig:
        mcmccoeff[coeff_num] -= dx*k
        currchisq = lnlike(mcmccoeff, tab, func2fit) *-2
        chisqleft = chisqleft + [currchisq]
        k+=1
    xleft = range(len(chisqleft))
    
    currchisq = chisqorig
    chisqright = []
    mcmccoeff = np.copy(mcmccoefforig)
    k=1
    while currchisq <= mult * chisqorig and currchisq > 0:
        mcmccoeff[coeff_num] += dx*k
        currchisq = lnlike(mcmccoeff, tab, func2fit)*-2
        chisqright = chisqright + [currchisq]
        k+=1
    xright = range(len(chisqright)) + np.ones(len(chisqright))
    xright = xright.tolist()
        
    chisqleft.reverse()
    chisqarr = chisqleft + chisqright
    
    xleft = xleft * np.ones(len(xleft)) * -1
    xleft = xleft.tolist()
    xleft.reverse()
    x = xleft + xright
    
    mcmccoeff = np.copy(mcmccoefforig)
    return [x, chisqarr]

def plotchisq_varyone(tab, chosenfunc, n, mcmccoeff, coeff_num, dx, mult):
    fig = plt.figure()
    x, y = chisq_varyone(tab, chosenfunc, n, mcmccoeff, coeff_num, dx, mult)
    plt.plot(x, y, 'o')
    plt.title('Varying coeff: ' + str(coeff_num) + ' dx: ' + str(dx) + ' mult: ' + str(mult) +\
                '\nMinimum value: ' +str(np.min(y)) + ' at index ' + str(x[np.argmin(y)]) +\
                 ' ' + chosenfunc + str(n))
    plt.ylabel('$\chi^2$', fontsize=18)
    plt.xlabel('Number of dx', fontsize=18)
    return fig   
##########

##########
########## Binning function not copied here but in the ipython notebook
##########

###########################################
###########################################
###########################################
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Perform MCMC to find the coefficients for the LFlat")
    
    # Constants
    parser.add_argument("--chiplen", default=1024,
                        help="The length of the chip: Default = 1024")
    parser.add_argument("--bin", default=20,
                        help="The bin number; used for plotting the meshgrid: Default = 20")
    parser.add_argument("--usef", default=0.0, 
                        help="Input F if want to use the fudge factor: Default = 0.0: Should be between 0 and 1")
    
    # True False arguments
    parser.add_argument("-filt", "--filtertable", action = "store_false", default=True,
                        help="True if filtering the table: Default = True")
    parser.add_argument("-mcmc", "--mcmc", action = "store_false", default=True,
                        help="True if performing MCMC: Default = True")
    parser.add_argument("-s2one", "--scale2one", action = "store_false", default=True,
                        help="True if scaling from -1 to 1 (Preferred): Default = True")
    
    # Table arguments
    parser.add_argument("--datafile", 
                        help="The location of the datafile")  # PROBABLY SHOULD NOT BE OPTIONAL
    parser.add_argument("--names", default=['id', 'filenum', 'chip', 'x', 'y', 'flux', 'fluxerr'],
                        help="The names for the table -- Need to have 'id' 'x' 'y' 'flux' 'fluxerr'")
    parser.add_argument("--types", default=[int, int, int, np.float64, np.float64, np.float64, np.float64],
                        help="The types for the table")
    parser.add_argument("--remnames", default=['filenum', 'chip'],
                        help="The names to remove from the table")
                        
    # Filter Table aruments
    parser.add_argument("--min_num_obs", type=int, default=4, choices=range(10),
                        help="The minimum number of observations required for each star: Default = 4")
    parser.add_argument("--flux_ratio", type=int, default=5, 
                        help="The minimum flux signal to noise for an observation: Default = 5")
    parser.add_argument("--low", type=np.double, default=3.0,
                        help="The lower limit for the sigma clipping when filtering the table")
    parser.add_argument("--high", type=np.double, default=3.0,
                        help="The upper limit for the sigma clipping when filtering the table")
    parser.add_argument("--remove_stars", default = [],
                        help="A List of certain stars that should be removed from the data")                    
    # MCMC arguments
    parser.add_argument("--n", type=int, choices=range(8), default=1,
                        help="The nth order of the fit: Default = 1")
    parser.add_argument("--chosenfunc", type=str, choices=["poly", "cheb", "leg"], default="cheb",
                        help="The functional form: Default = 'cheb'")
    parser.add_argument("--nwalkers", type=int, default=10,
                        help="The number of walkers: Default = 10")
    parser.add_argument("--nsteps", type=int, default=10,
                        help="The number of steps each walker takes: Default = 10")
    parser.add_argument("--scale_factor", type=np.double, default=1e-1,
                        help="The shape of the initial tiny gaussian balls: Default = 1e-1")
    parser.add_argument("--burnin", type=int, default=0,
                        help="The number of steps MCMC should do and then start from there SOMETHING SOMETHING: Default = 0")    
    
    # Saving Data arguments
    parser.add_argument("--filmcmc", default ='',
                        help="Name of text file to save the MCMC coefficients")
    parser.add_argument("--filcoeff", default = '',
                        help="Name of text file to save the locations of each walker from each step")
    parser.add_argument("--figpath", default = '',
                        help="The folder path of where the figures will be saved")   
    parser.add_argument("--filfits", default = "",
                       help="Name of FITS file to save FITS file using MCMC coefficients")        
    args = parser.parse_args()
        
    CHIPXLEN = CHIPYLEN = args.chiplen
    XBIN = YBIN = args.bin
    XPIX = np.linspace(0, CHIPXLEN, XBIN)
    YPIX = np.linspace(0, CHIPYLEN, YBIN)
    SCALE2ONE = args.scale2one
    
    if args.usef:
        F = args.usef
        USE_F = True
    else:
        USE_F = False
    
    table = do_table(fil=args.datafile, names=args.names, types=args.types, removenames=args.remnames)

    if args.filtertable:
        table = do_filter(table,                        \
                        min_num_obs=args.min_num_obs,   \
                        flux_ratio=args.flux_ratio,     \
                        low=args.low, high=args.high,   \
                        remove_stars=args.remove_stars)
    
    if args.mcmc:
        sampler, samples, vaslwerrs, mcmccoeff, f, ndim = do_MCMC(table, 
                        nsteps=args.nsteps, nwalkers=args.nwalkers,         \
                        chosenfunc=args.chosenfunc, n=args.n,               \
                        scale_factor=args.scale_factor, burnin=args.burnin, \
                        txtfil=args.filcoeff, mcmcfil=args.filmcmc)
        if args.figpath:
            print '******************************************'
            print '************* START FIGURES **************'
            print '******************************************'
            func2read, func2fit = get_function(args.chosenfunc, args.n)
            
            plotmesh(convert2mesh(func2fit, coeff=mcmccoeff, \
                                    xpixel=XPIX, ypixel=YPIX), \
                    title = 'MCMC: ' + str(args.chosenfunc) + ' n = ' + str(args.n)).savefig(args.figpath+'meshgrid.png',dpi=500)
            plotwalkerpathsmult(sampler, args.figpath, args.chosenfunc, args.n)
            
            zzfitmcmc = convert2mesh(func2fit, coeff=mcmccoeff, xpixel=np.double(range(int(CHIPXLEN))), ypixel=np.double(range(int(CHIPYLEN))))[2]      # convert2mesh returns [xx, yy, zzfit]
            imgmcmc = plotimg(zzfitmcmc, title = 'MCMC: ' + args.chosenfunc + ' n = ' + str(args.n), fitplot = True)
            imgmcmc.savefig(args.figpath+'Lflat.png', dpi=500)
            plottriangle(samples, func2read).savefig(args.figpath+'triangle.png', dpi=700)

            print 'New files:'
            print 'The paths of the walkers ::: ' + args.figpath + 'walker*.png'
            print 'The triangle plot ::: ' + args.figpath + 'triangle.png'
            print 'The meshgrid of the Lflat ::: ' + args.figpath + 'meshgrid.png'
            print 'The Lflat (imshow) ::: ' + args.figpath + 'Lflat.png' 
            print '******************************************'
            print '************** END FIGURES ***************'
            print '******************************************\n'
        if args.filfits:
            create_fits(args.filfits, args.chosenfunc, args.n, mcmccoeff)
            print 'FITS file: ' + args.filfits