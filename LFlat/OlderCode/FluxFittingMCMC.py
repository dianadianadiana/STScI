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
use_f = False
f = 0.05
n = 5
nx = ny = n
chosenfunc = 'cheb'
funcname = chosenfunc + '2d'
nwalkers = 100
nsteps = 100

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
from sympy import * 
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
    import numpy.polynomial.chebyshev as cheb
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
    import numpy.polynomial.legendre as leg
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
    
###########################################
######### Creating the Table ##############
###########################################

from astropy.table import Table, Column
############
# Read in the data
#path = '/Users/dkossakowski/Desktop/Data/'
#datafil = 'wfc_f606w_r5.lflat'
#data = np.genfromtxt(path + datafil)

# Create an Astropy table
# tab[starID][0]: starID; ...[2]: chip#; ...[3]: x pixel; ...[4]: y pixel; 
# ...[5]: magnitude; ...[6]: magnitude error; rest: dummy

#names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr', 'd1', 'd2', 'd3']
#types = [int, int, int, np.float64, np.float64, np.float64, np.float64,
#         float, float, float]
#tab = Table(data, names=names, dtype=types)
#tab.remove_columns(['filenum','d1','d2','d3'])   # remove the dummy columns  
############

############ Read in NEW data :: comment/uncomment this snippet
path = '/Users/dkossakowski/Desktop/Data/'
path = '/user/dkossakowski/'
datafil = 'flatfielddata.txt'
#datafil = 'flatfielddata_fudged.txt'
#datafil = 'fakeflatdata.txt'
data = np.genfromtxt(path + datafil)
names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr']
types = [int, int, int, np.float64, np.float64, np.float64, np.float64]
tab = Table(data, names=names, dtype=types)
tab.remove_columns(['filenum', 'chip'])
tab = tab[np.where(tab['magerr'] < 1.)[0]]
############
#chosen = np.random.choice(len(tab), 4000, replace = False)
#tab = tab[chosen]
tab.sort(['id'])                                 # sort the table by starID
starIDarr = np.unique(tab['id'])                 # collect all the star IDs

###########################################
###########################################
########################################### 

###########################################
########## Filtering the Table ############
###########################################
from DataInfoTab import remove_stars_tab, convertmag2flux, convertflux2mag,\
                        make_avgmagandflux, sigmaclip_starmagflux,         \
                        sigmaclip_delmagdelflux                                        # These functions are imported form DataInfoTab.py

print 'Initial number of observations:\t', len(tab)                        
tab =  tab[np.where((tab['mag'] <= 25) & (tab['mag'] >= 13))[0]]                       # Constrain magnitudes (13,25)
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)     # Remove rows with less than min num of observations
tab, starIDarr = make_avgmagandflux(tab, starIDarr)                                    # Create columns ('avgmag', 'avgmagerr', 'flux', 'fluxerr', 'avgflux', 'avgfluxerr')
tab, starIDarr = sigmaclip_starmagflux(tab, starIDarr, flux = True, mag = False,  \
                                        low = 3, high = 3)                             # Sigmaclip the fluxes and/or magnitudes for each star
tab, starIDarr = sigmaclip_delmagdelflux(tab, starIDarr, flux = True, mag = False,\
                                        low = 3, high = 3)                             # Sigmaclip the delta magnitudes and/or delta fluxes
tab =  tab[np.where(tab['flux']/tab['fluxerr'] > 5)[0]]                                # S/N ratio for flux is greater than 5
print 'Number of observations after filtering:\t', len(tab)
print 'Number of stars after filtering:\t', len(starIDarr)

###########################################
###########################################
###########################################

if chosenfunc == 'poly':
    func2read, func2fit = norder2dpoly(n) 
elif chosenfunc == 'cheb':
    func2read, func2fit = norder2dcheb(nx, ny)        # nx th and ny th order 2d Chebyshev Polynomial
elif chosenfunc == 'leg':
    func2read, func2fit = norder2dlegendre(nx, ny) 
print 'Function that is being fit:', func2read

##########
def funcintegrate(x, y, coeff):
    return np.sum(func2fit(x, y) * coeff)

from scipy import integrate
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

from multiprocessing import Pool

def lnprior(params):
    realparams = params
    if use_f:
        realparams, lnf = params[:-1], params[-1]
    a = convert2mesh(func2fit, realparams, xpixel = np.linspace(0,CHIPXLEN,256), ypixel = np.linspace(0,CHIPYLEN,256))[2]
    diff_maxmin = np.max(a) - np.min(a)
    #-.12 < realparams[0] < .5 and 
    if diff_maxmin <= .30: #and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf
    
########## Doing it with multiprocessing
#def chisqstar(inputs):
#    starrows, p = inputs
########## 
def chisqstar(starrows, p):
    starvals = starrows[SPACE_VAL]
    starvalerrs = starrows[SPACE_VAL + 'err']
    func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p)) + 1    # The 'delta' function
    if SCALE2ONE:
        fits = [func(p, (row['x']-CHIPXLEN/2)/(CHIPXLEN/2), (row['y']-CHIPYLEN/2)/(CHIPYLEN/2)) for row in starrows]
    else:
        fits = [func(p, row['x'], row['y']) for row in starrows]
    avgf = np.mean(starvals/fits)                                  # Our 'expected' value for the Flux
    
    def get_sigmasq():
        if use_f:
            return np.asarray((starvalerrs/fits))**2 + np.exp(2*lnf)*avgf**2
        return np.asarray((starvalerrs/fits))**2
            
    starsq = (starvals/fits - avgf)**2 / get_sigmasq() + np.log(get_sigmasq()) # ignore the 2pi since that is just a constant for the chisq
    starsq = np.asarray(starsq)
    starsqsum = np.sum(starsq)
    return starsqsum   
       
def lnlike(params, tab, num_cpu=4):
    global count
    if count % 20 == 0: print count 
    count+=1   
    realparams = params
    if use_f:
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
    
    starsqsums = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]], realparams) for star in starIDarr])     # an array of the sq of sums for each star
    totalsqsum = np.sum(starsqsums)
    return -0.5 * totalsqsum

def lnprob(params, tab):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, tab)

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
        newpos = [initialcoeff + 1e-1*np.random.randn(ndim) for i in range(-1*num)]
        pos = np.append(pos, newpos, axis = 0)
        pos = filter_pos(pos)
        if len(pos) - nwalkers > 0:
            difference = len(pos) - nwalkers
            pos = pos[:-difference]
        if time.time()-start_time > 45.0:
            warnings.warn("Warning: Finding the intial MCMC walkers is taking too long")
    return pos
    
# Reduce the Table so that it doesn't have unused Columns that take up memory/time
tabreduced = np.copy(tab)               
tabreduced = Table(tabreduced)
tabreduced.remove_columns(['avgmag', 'avgmagerr', 'avgflux','avgfluxerr'])
count = 0
if use_f:
    initialcoeff = np.zeros(len(func2read)+1)
    initialcoeff[0] = 1
    initialcoeff[-1] = np.log(f)
else:
    initialcoeff = np.zeros(len(func2read))
        
ndim = len(initialcoeff)
some_time = time.time()
pos = get_pos(ndim, nwalkers, scale_factor = 1e-4, base_coeff = initialcoeff)
print 'Getting pos', time.time()-some_time

start_time = time.time()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(tabreduced,))
sampler.run_mcmc(pos, nsteps)
samples = sampler.chain[:, :, :].reshape((-1, ndim))
print 'Time to do MCMC', time.time()-start_time
#import corner
#fig = corner.corner(samples)

### Get the true values
if use_f:
    samples[:, -1] = np.exp(samples[:, -1])
valswerrs = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))
valswerrs = np.asarray(valswerrs)
mcmccoeff = valswerrs.T[0]   
mcmccoeff[0] += 1 # Because we defined p0 as p0-1 so we are just adding the 1 back in
###

### Normalize
if use_f:
    int_resmcmc = integrate.nquad(funcintegrate, [bounds_x(), bounds_y()], args = (mcmccoeff[:-1],))[0]   
else:
    int_resmcmc = integrate.nquad(funcintegrate, [bounds_x(), bounds_y()], args = (mcmccoeff,))[0]  
int_resmcmc /= area()
mcmccoeff /= int_resmcmc
###
if use_f:
    print mcmccoeff[-1]
    mcmccoeff = mcmccoeff[:-1]
    
###########################################
############### Plotting ##################
###########################################

from matplotlib.ticker import MaxNLocator

def plotwalkerpaths(num_subplots):
    ''' Plot the walker paths for your choice of num_subplots '''
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)#, figsize=(8, 9))
    for axnum in range(num_subplots):
        axes[axnum].plot(sampler.chain[:, :, axnum].T, color="k", alpha=0.4)
        axes[axnum].yaxis.set_major_locator(MaxNLocator(5))
        #axes[axnum].set_ylabel("$0$")
    fig.tight_layout(h_pad=0.0)
    return fig

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
       
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