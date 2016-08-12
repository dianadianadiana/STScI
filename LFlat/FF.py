"""
LFlat Program using MCMC

:Author: Diana Kossakowski

:Organization: Space Telescope Science Institute

:History:
    * Aug 2016 Finished

Examples
--------
To call from command line::
    python FF.py

"""

# Global Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import warnings
import time
import argparse

from multiprocessing import Pool

# Local Imports
import emcee

# Functions
from sympy import *
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as leg
from scipy import integrate

# Data
from astropy.table import Table, Column

# Filter
# These functions are imported form DataInfoTab.py
from DataInfoTab import remove_stars_tab, convertmag2flux, convertflux2mag,\
                        make_avgmagandflux, sigmaclip_starmagflux,         \
                        sigmaclip_delmagdelflux                                        

# Plotting
import corner
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

## Creating constants to keep a "standard"
#CHIPXLEN =  CHIPYLEN = 1024.
#
#BIN_NUM = 2
#XBIN = YBIN = 10. * BIN_NUM
#XPIX = np.linspace(0,     CHIPXLEN,     XBIN)
#YPIX = np.linspace(0,     CHIPYLEN,     YBIN)
#
#SCALE2ONE = True
#SPACE_VAL = 'flux'                               # What space we are working in ('flux' or 'mag')
#USE_F = False
#F = 0.05
#Norder = 1
#Nx = Ny = Norder
#CHOSENFUNC = 'cheb'
#funcname = CHOSENFUNC + '2d'
#NWALKERS = 32
#NSTEPS = 100
#path = '/Users/dkossakowski/Desktop/Data/'
#path = '/user/dkossakowski/'
#datafil = 'sbc_f125lp.phot'

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
    funclist -- A list of the different components of the poly 
                (elements are Symbol types) useful for printing
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
    data = np.genfromtxt(fil)
    tab = Table(data, names=names, dtype=types)
    tab.remove_columns(removenames)
    return tab

###########################################
###########################################
########################################### 

###########################################
########## Filtering the Table ############
###########################################

def do_filter(tab, mag_constrain = [13,19], min_num_obs = 4, flux_ratio = 5):
    # this is assuming that tab['mag'] and tab['magerr'] exist
    #tab.sort(['id'])                                 # sort the table by starID
    starIDarr = np.unique(tab['id'])                  # collect all the star IDs
    num_stars0 = np.double(len(starIDarr))
    num_obs0 = np.double(len(tab))
    print '******************************************'
    print '************** START FILTER **************'
    print '******************************************'
    print 'Initial number of observations:\t', len(tab)                        
    tab =  tab[np.where((tab['mag'] <= mag_constrain[1]) & (tab['mag'] >= mag_constrain[0]))[0]]   # Constrain magnitudes (13,25)
    tab = tab[np.where(tab['magerr'] < 1.)[0]]
    tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs)                 # Remove rows with less than min num of observations
    tab, starIDarr = make_avgmagandflux(tab, starIDarr)                                            # Create columns ('avgmag', 'avgmagerr', 'flux', 'fluxerr', 'avgflux', 'avgfluxerr')
    tab, starIDarr = sigmaclip_starmagflux(tab, starIDarr, flux = True, mag = False,  \
                                            low = 3, high = 3)                                     # Sigmaclip the fluxes and/or magnitudes for each star
    tab, starIDarr = sigmaclip_delmagdelflux(tab, starIDarr, flux = True, mag = False,\
                                            low = 3, high = 3)                                     # Sigmaclip the delta magnitudes and/or delta fluxes
    tab =  tab[np.where(tab['flux']/tab['fluxerr'] > flux_ratio)[0]]                               # S/N ratio for flux is greater than 5
    print 'Number of observations after filtering:\t', len(tab)
    print 'Percent of observations kept:\t', len(tab)/num_obs0 * 100
    print 'Number of stars after filtering:\t', len(starIDarr)
    print 'Percent of stars kept:\t', len(starIDarr)/num_stars0
    print '******************************************'
    print '*************** END FILTER ***************'
    print '******************************************\n'
    return tab

###########################################
###########################################
###########################################


##########
def convert2mesh(func2fit, coeff, xpixel, ypixel):
    ''' Creates a mesh using the coefficients and x and y pixel values '''
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

def lnprior(params):
    realparams = params
    if USE_F:
        realparams, lnf = params[:-1], params[-1]
    if -.1 < realparams[0] < .1:
        return 0.0
    return -np.inf
    
########## Doing it with multiprocessing
#def chisqstar(inputs):
#    starrows, p = inputs
########## 
def chisqstar(starrows, params, func2fit):
    starvals = starrows['flux']
    starvalerrs = starrows['flux' + 'err']
    func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p)) + 1           # The 'delta' function
    if SCALE2ONE:
        fits = [func(params, (row['x']-CHIPXLEN/2)/(CHIPXLEN/2), (row['y']-CHIPYLEN/2)/(CHIPYLEN/2)) for row in starrows]
    else:
        fits = [func(params, row['x'], row['y']) for row in starrows]
    avgf = np.mean(starvals/fits)                                              # Our 'expected' value for the Flux
    def get_sigmasq():
        if USE_F:
            return np.asarray((starvalerrs/fits))**2 + np.exp(2*lnf)*avgf**2
        return np.asarray((starvalerrs/fits))**2
            
    starsq = (starvals/fits - avgf)**2 / get_sigmasq() + np.log(get_sigmasq()) # ignore the 2pi since that is just a constant for the chisq
    starsq = np.asarray(starsq)
    starsqsum = np.sum(starsq)
    return starsqsum   
       
def lnlike(params, tab, func2fit, num_cpu=4):
    realparams = params
    if USE_F:
        realparams, lnf = params[:-1], params[-1]
    starIDarr = np.unique(tab['id'])
    
    ########## Doing it with multiprocessing
    #runs = [(tab[np.where(tab['id'] == star)[0]], realparams) for star in starIDarr]
    #pool = Pool(processes=num_cpu)
    #results = pool.map_async(chisqstar, runs)
    #pool.close()
    #pool.join()
    #return -0.5 * np.sum(results.get())
    ########## 
    
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
    
def do_MCMC(tab, nsteps, nwalkers, scale_factor, chosenfunc, n, burnin, txtfil=''):

    print '******************************************'
    print '*************** START MCMC ***************'
    print '******************************************'
    
    print 'Number of walkers:', nwalkers
    print 'Number of steps:', nsteps
    
    ### Get the function
    func2read, func2fit = get_function(chosenfunc, n)
    print 'Function that is being fit:', func2read
    ###
    
    ### Reduce the Table so that it doesn't have unused Columns that take up memory/time
    tabreduced = np.copy(tab)               
    tabreduced = Table(tabreduced)
    tabreduced.remove_columns(['avgflux','avgfluxerr'])
    
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
        f.write('nsteps: '   + str(NSTEPS)     + '\n')
        f.write('nwalkers: ' + str(NWALKERS)   + '\n')
        f.write('ndim: '     + str(ndim)       + '\n')
        f.close()
        
    for i, result in enumerate(sampler.sample(pos, iterations=nsteps, storechain=True)):
        if i%20 == 0: print 'step #', i 
        if txtfil:
            position = result[0]
            f = open(writefil, "a")
            f.write('nstep #' + str(i) + '\n')
            for k in range(position.shape[0]):
                f.write('{0:4d} {1:s}\n'.format(k, " ".join(str(position[k]))))
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
    print
    print 'MCMC Coefficients:'
    print mcmccoeff
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
    ###
    
    print '******************************************'
    print '**************** END MCMC ****************'
    print '******************************************\n'
    return [sampler, samples, valswerrs, mcmccoeff,  f, ndim]

def get_samplerchain(sampler, start, end, ndim):
    if start < 0:
        start = 0
    if end > NSTEPS:
        end = NSTEPS
    samp = sampler.chain[:, start:end, :].reshape((-1, ndim))
    return samp

###########################################
############### Plotting ##################
###########################################

#def plotwalkerpaths(samp, num_subplots, start=0, end=NSTEPS):
#    ''' Plot the walker paths for your choice of num_subplots '''
#    if start < 0: start = 0
#    
#    fig, axes = plt.subplots(num_subplots, 1, sharex=True)#, figsize=(8, 9))
#    for axnum in range(num_subplots):
#        #axes[axnum].plot(samp[:,axnum].reshape(NSTEPS, NWALKERS, order='F'), color="k", alpha=.4)
#        axes[axnum].plot(sampler.chain[:, start:end, axnum].T, color="k", alpha=0.4)
#        axes[axnum].yaxis.set_major_locator(MaxNLocator(5))
#        axes[axnum].set_ylabel('Coeff #' + str(axnum))
#        axes[axnum].set_xticklabels(np.arange(start,end+1,10))
#    fig.tight_layout(h_pad=0.0)
#    return fig

def plottriangle(samp):
    fig = corner.corner(samp)
    return fig
       
def plotmesh(a, title = ''):
    ''' Returns the figure of the mesh fit plot '''
    X, Y, Z = a                   # a is the xx yy and zz (2d array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red')
    ax.set_title(title)
    plt.legend()
    return fig

finalfunc = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))     # The final flat

def apply_flat(rows, coeff, somethingchar):
    final = np.array([])
    for x,y,something in zip(rows['x'], rows['y'], rows[somethingchar]):
        if SCALE2ONE:
            x = (x - CHIPXLEN/2)/(CHIPXLEN/2)
            y = (y - CHIPYLEN/2)/(CHIPYLEN/2)
        final = np.append(final, something / finalfunc(coeff,x,y))
    return final
    
def simple3dplot(rows, something, title = ''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rows['x'],rows['y'], something)
    ax.set_title(title)
    plt.show(fig)
    
def simple3dplotindstars(tab, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(tab['id']))))
    for star, col in zip(np.unique(tab['id']), colors):
        starrows = tab[np.where(tab['id'] == star)[0]]
        ax.scatter(starrows['x'],starrows['y'], starrows['flux'], c = col)
    ax.set_title(title)
    plt.show(fig)
    
def simple3dplotindstarsafter(tab, coeff, title = ''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for star in np.unique(tab['id']):
        starrows = tab[np.where(tab['id'] == star)[0]]
        starfinal = apply_flat(starrows, coeff, 'flux')
        starfinalavg = np.mean(starfinal)
        ax.scatter(starrows['x'], starrows['y'], (starfinal-starfinalavg)/starfinalavg)
    ax.set_title(title)
    plt.show(fig)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Perform MCMC to find the coefficients for the LFlat")
    
    # Constants
    parser.add_argument("--chiplen", default=1024,
                        help="The length of the chip: Default = 1048")
    parser.add_argument("--bin", default=20,
                        help="The bin number; used for plotting the meshgrid: Default = 1048")
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
    parser.add_argument("--names", default=['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr'],
                        help="The names for the table")
    parser.add_argument("--types", default=[int, int, int, np.float64, np.float64, np.float64, np.float64],
                        help="The types for the table")
    parser.add_argument("--remnames", default=['filenum', 'chip'],
                        help="The names to remove from the table")
    parser.add_argument("--datafile", 
                        help="The location of the datafile")
                        
    # Filter Table aruments
    parser.add_argument("--mag_constrain", default=[13,19],
                        help="The constraints for the magnitude: Default = [13,19]")
    parser.add_argument("--min_num_obs", type=int, default=4, choices=range(10),
                        help="The minimum number of observations required for each star: Default = 4")
    parser.add_argument("--flux_ratio", type=int, default=5, 
                        help="The minimum flux signal to noise for an observation: Default = 5")
                        
    # MCMC arguments
    parser.add_argument("--n", type=int, choices=[0,1,2,3,4,5,6,7], default=1,
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

    args = parser.parse_args()
        
    #names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr']
    #types = [int, int, int, np.float64, np.float64, np.float64, np.float64]
    #fil = '/Users/dkossakowski/Desktop/Data/sbc_f125lp.phot'
    
    #table = do_table(fil=fil, names=names, types=types, removenames=['filenum', 'chip'])
    #table = do_filter(table)
    #sampler, samples, vaslwerrs, mcmccoeff, f, ndim = do_MCMC(table, nsteps, nwalkers, scale_factor=1e-1, chosenfunc=CHOSENFUNC, n=Norder, burnin=0)


    CHIPXLEN = CHIPYLEN = args.chiplen
    XBIN = YBIN = args.bin
    XPIX = np.linspace(0,     CHIPXLEN,     XBIN)
    YPIX = np.linspace(0,     CHIPYLEN,     YBIN)
    SCALE2ONE = args.scale2one
    if args.usef:
        F = args.usef
        USE_F = True
    else:
        USE_F = False
    
    table = do_table(fil=args.datafile, names=args.names, types=args.types, removenames=args.remnames)
    
    if args.filtertable:
        table = do_filter(table, mag_constrain = args.mag_constrain, min_num_obs = args.min_num_obs, flux_ratio = args.flux_ratio)
    
    if args.mcmc:
        sampler, samples, vaslwerrs, mcmccoeff, f, ndim = do_MCMC(table, nsteps=args.nsteps, nwalkers=args.nwalkers,\
                            scale_factor=args.scale_factor, chosenfunc=args.chosenfunc, n=args.n, burnin=args.burnin)

        plt.show(plotwalkerpaths(samples, len(mcmccoeff)))
        plt.show(plottriangle(samples))
        plt.show(plotmesh(convert2mesh(get_function(args.chosenfunc, args.n)[1], coeff=mcmccoeff, xpixel=XPIX, ypixel=YPIX), title = 'MCMC: ' + str(args.chosenfunc) + ' n = ' + str(args.n)))
    
# Creating constants to keep a "standard"
#CHIPXLEN =  CHIPYLEN = 1024.

#BIN_NUM = 2
#XBIN = YBIN = 10. * BIN_NUM
#XPIX = np.linspace(0,     CHIPXLEN,     XBIN)
#YPIX = np.linspace(0,     CHIPYLEN,     YBIN)

#SCALE2ONE = True
#SPACE_VAL = 'flux'                               # What space we are working in ('flux' or 'mag')
#USE_F = False
#F = 0.05
#Norder = 1
#Nx = Ny = Norder
#CHOSENFUNC = 'cheb'
#funcname = CHOSENFUNC + '2d'
#NWALKERS = 32
#NSTEPS = 100
#path = '/Users/dkossakowski/Desktop/Data/'
#path = '/user/dkossakowski/'
#datafil = 'sbc_f125lp.phot'














############### If we want to plot the before/after of applying the flat (as is as well as the normalized delta)
#final = apply_flat(tab, mcmccoeff, 'flux')
#simple3dplot(tab, tab['flux'], title = 'Before LFlat, just ' + 'flux' + ' values plotted')
#simple3dplot(tab, final, title = 'After LFlat, just ' + 'flux' + ' values plotted')
#simple3dplot(tab, (tab['flux'] - tab['avg'+'flux'])/ tab['avg'+'flux'], title = 'Before LFlat, normalized delta ' + 'flux')
#simple3dplot(tab, (final - tab['avg'+'flux'])/tab['avg'+'flux'], title = 'After LFlat, normalized delta ' + 'flux') # Not QUITE right because there is a new mean
###############

############### If we want to see/plot the mean of each star before and after applying the flat
#for star in np.unique(tab['id']):
#    starrows = tab[np.where(tab['id']==star)[0]]
#    finalstar = apply_flat(starrows, mcmccoeff, 'flux')
#    mean_before = np.mean(starrows['flux'])
#    std_before = np.std(starrows['flux'])
#    mean_after = np.mean(finalstar)
#    std_after = np.std(finalstar)
#    print '***' + str(star)
#    print 'mean, max-min before', mean_before , np.max(starrows['flux']) - np.min(starrows['flux'])
#    print 'std before\t', std_before
#    print 'max-min/mean before', (np.max(starrows['flux']) - np.min(starrows['flux'])) / mean_before
#    print 'mean, max-min after', mean_after, np.max(finalstar) - np.min(finalstar)
#    print 'std after\t', std_after
#    print 'max-min/mean after', (np.max(finalstar) - np.min(finalstar)) / mean_after
#    #simple3dplot(starrows, starrows['flux'], title = 'Original ' + str(star) + ' ' + str(mean_before) + ', ' + str(std_before))
#    #simple3dplot(starrows, finalstar, title = 'Final ' + str(star) + ' ' + str(mean_after) + ', ' + str(std_after))
############### 

############### If we want to see where each star lies on the detector for multiple stars ( plot style )
#def plotnumobsGIF(num, cmap = 'jet', tab = tab):
#    # num : number of stars you want to plot
#    fig = plt.figure()
#    plt.xlim([0,CHIPXLEN])
#    plt.ylim([0,CHIPYLEN])
#    plt.grid(True)
#    plt.title('Observation Locations', fontsize = 18)
#    plt.xticks(np.linspace(0, CHIPXLEN, 9))
#    plt.yticks(np.linspace(0, CHIPYLEN, 9))
#    plt.xlabel('X Pixel', fontsize = 18)
#    plt.ylabel('Y Pixel', fontsize = 18)
#    cmap = plt.get_cmap(cmap)
#    stars = np.unique(tab['id'])
#    colors = [cmap(i) for i in np.linspace(0, 1, len(stars))]
#    for col, star in zip(colors[:num], stars[:num]):
#        tabstar = tab[np.where(tab['id'] == star)[0]]
#        plt.scatter(tabstar['x'], tabstar['y'], s=100, c=col, marker='*', lw=0.2)
#    return fig
#plt.show(plotnumobsGIF(num = 33, cmap = 'jet'))
#
#length = len(np.unique(tab['id']))
#length = 2
#for i in range(length) + np.ones(length):
#    i = int(i)
#    #plt.show(plotnumobsGIF(num = i, cmap = 'jet'))
#    plotnumobsGIF(num=i, cmap='jet').savefig('/Users/dkossakowski/Desktop/FinalPresentation/DitheringGIF/Stars/NumObs' + str(i) + '.png', dpi = 500)
############### 

#if __name__ == '__main__':
#
#### Input argument parsing
#    parser = argparse.ArgumentParser(
#        description='Make catalogs out of HST images using astrodrizzle tools.')
#    parser.add_argument(
#        '--nodriz',help='Turn off drizzle for CR clean production',action='store_true')
#    options = parser.parse_args()



