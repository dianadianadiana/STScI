import numpy as np
import emcee
import matplotlib.pyplot as plt
import scipy.optimize as op
import warnings
import time

# Creating constants to keep a "standard"
CHIPXLEN =  CHIPYLEN = 1024.

BIN_NUM = 2
XBIN = YBIN = 10. * BIN_NUM
XPIX = np.linspace(0,     CHIPXLEN,     XBIN)
YPIX = np.linspace(0,     CHIPYLEN,     YBIN)

SCALE2ONE = True
SPACE_VAL = 'flux'                               # What space we are working in ('flux' or 'mag')
USE_F = False
F = 0.05
Norder = 1
Nx = Ny = Norder
CHOSENFUNC = 'cheb'
funcname = CHOSENFUNC + '2d'
NWALKERS = 16
NSTEPS = 50
path = '/Users/dkossakowski/Desktop/Data/'
path = '/user/dkossakowski/'
datafil = 'sbc_f125lp.phot'

from multiprocessing import Pool

# Functions
from sympy import *
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as leg
from scipy import integrate

# Data
from astropy.table import Table, Column

# Filter
from DataInfoTab import remove_stars_tab, convertmag2flux, convertflux2mag,\
                        make_avgmagandflux, sigmaclip_starmagflux,         \
                        sigmaclip_delmagdelflux                                        # These functions are imported form DataInfoTab.py

# Plotting
import corner
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
def norder2dpoly(n = N):
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
        
def norder2dcheb(nx = Nx, ny = Ny):
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
    
def norder2dlegendre(nx = Nx, ny = Ny):
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
    
def get_function(chosenfunc=CHOSENFUNC, n = Norder):
    nx = ny = n
    if chosenfunc == 'poly':
        func2read, func2fit = norder2dpoly(n) 
    elif chosenfunc == 'cheb':
        func2read, func2fit = norder2dcheb(nx, ny)        # nx th and ny th order 2d Chebyshev Polynomial
    elif chosenfunc == 'leg':
        func2read, func2fit = norder2dlegendre(nx, ny) 
    return [func2read, func2fit]
    
########## Integration
def funcintegrate(x, y, coeff, chosenfunc = CHOSENFUNC):
    func2read, func2fit = get_function(chosenfunc)
    return np.sum(func2fit(x, y) * coeff)

def bounds_y():
    if SCALE2ONE:
        return [-1, 1]
    else:
        return [0, CHIPYLEN]
def bounds_x():
    if SCALE2ONE:
        return [-1, 1]
    else:
        return [0, CHIPXLEN]
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
    
names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr']
types = [int, int, int, np.float64, np.float64, np.float64, np.float64]
fil = '/Users/dkossakowski/Desktop/Data/sbc_f125lp.phot'
table = do_table(fil=fil, names = names, types = types, removenames = ['filenum', 'chip'])

###########################################
###########################################
########################################### 

###########################################
########## Filtering the Table ############
###########################################

def do_filter(tab, mag_constrain = [13,25], min_num_obs = 4, flux_ratio = 5):
    # this is assuming that tab['mag'] and tab['magerr'] exist
    #tab.sort(['id'])                                 # sort the table by starID
    starIDarr = np.unique(tab['id'])                 # collect all the star IDs
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

table = do_filter(table)

###########################################
###########################################
###########################################


##########
def convert2mesh(func2fit, coeff, xpixel = XPIX, ypixel = YPIX):
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
    #print realparams
    if USE_F:
        realparams, lnf = params[:-1], params[-1]
    #a = convert2mesh(func2fit, realparams, xpixel = np.linspace(0,CHIPXLEN,256), ypixel = np.linspace(0,CHIPYLEN,256))[2]
    #diff_maxmin = np.max(a) - np.min(a)
    
    #temprealparams = np.copy(realparams)
    ##temprealparams[0] += 1
    ##print temprealparams
    #integrate_result = integrate.nquad(funcintegrate, [bounds_x(), bounds_y()], args = (temprealparams,))[0]   
    #integrate_result /= area()
    #realparams = realparams / integrate_result
    #print realparams
    #-.12 < realparams[0] < .5 and 
    #if diff_maxmin <= .30: #and -10.0 < lnf < 1.0:
    if -.1 < realparams[0] < .1:
        return 0.0
    return -np.inf
    
########## Doing it with multiprocessing
#def chisqstar(inputs):
#    starrows, p = inputs
########## 
def chisqstar(starrows, p, func2fit):
    starvals = starrows[SPACE_VAL]
    starvalerrs = starrows[SPACE_VAL + 'err']
    func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p)) + 1    # The 'delta' function
    if SCALE2ONE:
        fits = [func(p, (row['x']-CHIPXLEN/2)/(CHIPXLEN/2), (row['y']-CHIPYLEN/2)/(CHIPYLEN/2)) for row in starrows]
    else:
        fits = [func(p, row['x'], row['y']) for row in starrows]
    avgf = np.mean(starvals/fits)                                  # Our 'expected' value for the Flux
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

def get_pos(ndim, NWALKERS, scale_factor, base_coeff):
    ''' Creates the initial tiny gaussian balls '''
    pos = [base_coeff + scale_factor*np.random.randn(ndim) for i in range(NWALKERS)]
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
    while len(pos) - NWALKERS != 0 and len(pos) - NWALKERS < 0:
        num = len(pos) - NWALKERS
        newpos = [base_coeff + 1e-1*np.random.randn(ndim) for i in range(-1*num)]
        pos = np.append(pos, newpos, axis = 0)
        pos = filter_pos(pos)
        if len(pos) - NWALKERS > 0:
            difference = len(pos) - NWALKERS
            pos = pos[:-difference]
        if time.time()-start_time > 45.0:
            warnings.warn("Warning: Finding the intial MCMC walkers is taking too long")
    return pos
    
def do_MCMC(tab, scale_factor = 1e-1, chosenfunc=CHOSENFUNC, n = Norder):

    print '******************************************'
    print '*************** START MCMC ***************'
    print '******************************************'
    func2read, func2fit = get_function(chosenfunc, n)
    print 'Function that is being fit:', func2read
    
    # Reduce the Table so that it doesn't have unused Columns that take up memory/time
    tabreduced = np.copy(tab)               
    tabreduced = Table(tabreduced)
    tabreduced.remove_columns(['avgmag', 'avgmagerr', 'avgflux','avgfluxerr'])
    if SPACE_VAL == 'flux':
        tabreduced.remove_columns(['mag', 'magerr'])
    else:
        tabreduced.remove_columns(['flux','fluxerr'])
    
    # Set up the initial coefficients 
    if USE_F:
        initialcoeff = np.zeros(len(func2read)+1)
        initialcoeff[-1] = np.log(F)
    else:
        initialcoeff = np.zeros(len(func2read))
    
    # Determine the initial locations of the walkers
    ndim = len(initialcoeff)
    start_time = time.time()
    pos = get_pos(ndim, NWALKERS, scale_factor, base_coeff = initialcoeff)
    print 'Time (s) getting the initial positions of the walkers', time.time() - start_time

    start_time = time.time()
    sampler = emcee.EnsembleSampler(NWALKERS, ndim, lnprob, args=(tabreduced, func2fit,))
    #samp = sampler.run_mcmc(pos, NSTEPS)[0]

    print '********'
    writefil = 'chain1.txt'
    f = open(writefil, "w")
    f.write('nsteps: ' + str(NSTEPS) + '\n')
    f.write('nwalkers: ' + str(NWALKERS) + '\n')
    f.write('ndim: ' + str(ndim) + '\n')
    f.close()
    samples1 = np.array([])
    for i, result in enumerate(sampler.sample(pos, iterations=NSTEPS, storechain=True)):
        position = result[0]
        print i
        print position
        f = open(writefil, "a")
        f.write('nstep #' + str(i) + '\n')
        for k in range(position.shape[0]):
            f.write('{0:4d} {1:s}\n'.format(k, " ".join(str(position[k]))))
            samples1 = np.append(samples1, [position[k]])
        f.close()
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    samples1 = samples1.reshape(NSTEPS*NWALKERS, ndim)
    print 'Time to do MCMC', time.time()-start_time
    print 
    print 
    ### Get the true values
    if USE_F:
        samples[:, -1] = np.exp(samples[:, -1])
    valswerrs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(samples, [16, 50, 84],axis=0)))
    valswerrs = np.asarray(valswerrs)
    mcmccoeff = valswerrs.T[0]   
    mcmccoeff[0] += 1 # Because we defined p0 as p0-1 so we are just adding the 1 back in
    print mcmccoeff
    ###
    if USE_F:
        f = mcmccoeff[-1]
        mcmccoeff = mcmccoeff[:-1]
    else:
        f = 0

    valswerrs1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                zip(*np.percentile(samples1, [16, 50, 84],axis=0)))
    valswerrs1 = np.asarray(valswerrs1)
    mcmccoeff1 = valswerrs1.T[0]   
    mcmccoeff1[0] += 1 # Because we defined p0 as p0-1 so we are just adding the 1 back in
    print mcmccoeff1
    
    ### Normalize
    int_resmcmc = integrate.nquad(funcintegrate, [bounds_x(), bounds_y()], args = (mcmccoeff,))[0]  
    int_resmcmc /= area()
    mcmccoeff /= int_resmcmc
    ###
    print '******************************************'
    print '**************** END MCMC ****************'
    print '******************************************\n'
    return [sampler, samples, samples1, valswerrs, mcmccoeff,mcmccoeff1, f, ndim]

sampler, samples,samples1, vaslwerrs, mcmccoeff, mcmccoeff1, f, ndim = do_MCMC(table, scale_factor=1e-1, chosenfunc=CHOSENFUNC, n=Norder)

def get_samplerchain(sampler, start, end, ndim):
    if start < 0:
        start = 0
    totalsamp = sampler.chain[:, :, :].reshape((-1, ndim))
    if end > len(totalsamp):
        end = len(totalsamp)
    samp = sampler.chain[:, start:end, :].reshape((-1, ndim))
    return samp

###########################################
############### Plotting ##################
###########################################

def plotwalkerpaths(samp, num_subplots):
    ''' Plot the walker paths for your choice of num_subplots '''
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)#, figsize=(8, 9))
    for axnum in range(num_subplots):
        #axes[axnum].plot(samp[:,axnum].reshape(NSTEPS, NWALKERS, order='F'), color="k", alpha=.4)
        axes[axnum].plot(sampler.chain[:, :, axnum].T, color="k", alpha=0.4)
        axes[axnum].yaxis.set_major_locator(MaxNLocator(5))
        #axes[axnum].set_ylabel("$0$")
    fig.tight_layout(h_pad=0.0)
    return fig

plt.show(plotwalkerpaths(samples, len(mcmccoeff)))

#plt.show(plotwalkerpaths(samples1, len(mcmccoeff)))
def plottriangle(samp):
    fig = corner.corner(samp)
       
def plotmesh(a, title = ''):
    ''' Returns the figure of the mesh fit plot '''
    X, Y, Z = a                   # a is the xx yy and zz (2d array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red')
    ax.set_title(title)
    #ax.set_zlim([.85,1.15])
    plt.legend()
    return fig

#plt.show(plotmesh(convert2mesh(func2fit, coeff=mcmccoeff), title = 'MCMC: ' + str(funcname) + ' n = ' + str(n)))

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
        ax.scatter(starrows['x'],starrows['y'], starrows[SPACE_VAL], c = col)
    ax.set_title(title)
    plt.show(fig)
    
def simple3dplotindstarsafter(tab, coeff, title = ''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for star in np.unique(tab['id']):
        starrows = tab[np.where(tab['id'] == star)[0]]
        starfinal = apply_flat(starrows, coeff, SPACE_VAL)
        starfinalavg = np.mean(starfinal)
        ax.scatter(starrows['x'], starrows['y'], (starfinal-starfinalavg)/starfinalavg)
    ax.set_title(title)
    plt.show(fig)
    

############### If we want to plot the before/after of applying the flat (as is as well as the normalized delta)
#final = apply_flat(tab, mcmccoeff, SPACE_VAL)
#simple3dplot(tab, tab[SPACE_VAL], title = 'Before LFlat, just ' + SPACE_VAL + ' values plotted')
#simple3dplot(tab, final, title = 'After LFlat, just ' + SPACE_VAL + ' values plotted')
#simple3dplot(tab, (tab[SPACE_VAL] - tab['avg'+SPACE_VAL])/ tab['avg'+SPACE_VAL], title = 'Before LFlat, normalized delta ' + SPACE_VAL)
#simple3dplot(tab, (final - tab['avg'+SPACE_VAL])/tab['avg'+SPACE_VAL], title = 'After LFlat, normalized delta ' + SPACE_VAL) # Not QUITE right because there is a new mean
###############

############### If we want to see/plot the mean of each star before and after applying the flat
#for star in np.unique(tab['id']):
#    starrows = tab[np.where(tab['id']==star)[0]]
#    finalstar = apply_flat(starrows, mcmccoeff, SPACE_VAL)
#    mean_before = np.mean(starrows[SPACE_VAL])
#    std_before = np.std(starrows[SPACE_VAL])
#    mean_after = np.mean(finalstar)
#    std_after = np.std(finalstar)
#    print '***' + str(star)
#    print 'mean, max-min before', mean_before , np.max(starrows[SPACE_VAL]) - np.min(starrows[SPACE_VAL])
#    print 'std before\t', std_before
#    print 'max-min/mean before', (np.max(starrows[SPACE_VAL]) - np.min(starrows[SPACE_VAL])) / mean_before
#    print 'mean, max-min after', mean_after, np.max(finalstar) - np.min(finalstar)
#    print 'std after\t', std_after
#    print 'max-min/mean after', (np.max(finalstar) - np.min(finalstar)) / mean_after
#    #simple3dplot(starrows, starrows[SPACE_VAL], title = 'Original ' + str(star) + ' ' + str(mean_before) + ', ' + str(std_before))
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


#mcmccoeff
#np.savetxt()


