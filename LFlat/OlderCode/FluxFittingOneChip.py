import numpy as np
import time
import scipy.optimize as op

# Creating constants to keep a "standard"
CHIPXLEN =  CHIPYLEN = 1024.

BIN_NUM = 2
XBIN = YBIN = 10. * BIN_NUM
XPIX = np.linspace(0,     CHIPXLEN,     XBIN)
YPIX = np.linspace(0,     CHIPYLEN,     YBIN)

SPACE_VAL = 'flux'                               # What space we are working in ('flux' or 'mag')
SCALE2ONE = False
n = 4
nx = ny = n
chosenfunc = 'cheb'
funcname = chosenfunc + '2d'

###########################################
######### Functions to Optimize ###########
###########################################
'''
Note for all functions: 
   * lambdify needs to take in a list and not array; and for a 2d
     polynomial, the constant term needs to be 1 + 0*x but lambdify
     makes it just 1, which becomes a problem when trying to fit the 
     function and hence why the 0th element turns into np.ones(len(x))
Returns for all the functions:
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
###########################################
###########################################


###########################################
######### Creating the Table ##############
###########################################

from astropy.table import Table, Column
############
# Read in the data
path = '/Users/dkossakowski/Desktop/Data/'
datafil = 'wfc_f606w_r5.lflat'
data = np.genfromtxt(path + datafil)

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
lentab0 = len(tab)
lenstar0 = len(starIDarr)

###########################################
###########################################
###########################################

if chosenfunc == 'poly':
    func2read, func2fit = norder2dpoly(n)             # nth order 2d Polynomial
elif chosenfunc == 'cheb':
    func2read, func2fit = norder2dcheb(nx, ny)        # nx th and ny th order 2d Chebyshev Polynomial
elif chosenfunc == 'leg':
    func2read, func2fit = norder2dlegendre(nx, ny)    # nx th and ny th order 2d Legendre Polynomial

print 'Function that is being fit:', func2read

###########################################
########### Initial Conditions ############
###########################################

def getfit(tab, func2fit):
    '''
    Purpose
    -------
    To get the initial conditions for the later fit
    
    Parameters
    ----------
    tab:        The Astropy table with all the information
    func2fit:   The function that is being optimized    
    
    Returns
    -------
    [x, y, z, zfit, coeff, rsum, resarr]
    x:          X pixel values that were considered in the fit
    y:          Y pixel values that were considered in the fit
    z:          Delta flux values that were considered in the fit corresponding to (x,y)
    zfit:       Values of the delta fluxes at the points (x,y)
    coeff:      Coefficients for the function that was fitted
    rsum:       The chi-squared value of the fit
    resarr:     1D array of the absolute value of the 
    
    How it works:
        1) Set up x to be the x pixel values;
            Set up y to be the y pixel values (and add 2048 if the chipnum is 1);
            Set up z to be the delta flux values
        2) Do the Least Squares Fitting on x,y,z
        3) Fill in zfit 
    '''

    x = np.asarray(tab['x'])
    y = np.asarray(tab['y'])
    z = np.asarray((tab['flux'] - tab['avgflux']) / tab['avgflux'])       # normalized delta flux
    if SCALE2ONE:
        x = (x - CHIPXLEN/2)/(CHIPXLEN/2)
        y = (y - CHIPYLEN/2)/(CHIPYLEN/2)

    f = func2fit(x,y)
    try:                             # The zeroth element is an int 1 and not an array of 1s
        f[0] = np.ones(len(x))
    except TypeError:
        pass
    
    A = np.array(f).T
    B = z
    coeff, rsum, rank, s = np.linalg.lstsq(A, B)

    # Ex: zfit = coeff[0]*x**2 + coeff[1]*y**2 + ... === coeff[0]*f[0] + ...
    zfit = np.zeros(len(x))
    k = 0
    while k < len(coeff):
        zfit += coeff[k]*f[k]
        k+=1
    resarr = zfit - z
    return [x, y, z, zfit, coeff, rsum, resarr]
    
x, y, z, zfit, coeff, rsum, resarr = getfit(tab, func2fit)

initialcoeff = np.zeros(len(func2read))
initialcoeff[0] = 1
print 'Initial Coefficients:'
print initialcoeff
###########################################
###########################################
###########################################


###########################################
######## Fitting by Star Grouping #########
########################################### 
   
from multiprocessing import Pool
def chisqstar(starrows, p):
#def chisqstar(inputs):
        ''' Worker function '''
        # Input is the rows of the table corresponding to a single star so that we don't need to input a whole table
        #starrows, p = inputs
        starvals = starrows['flux']
        starvalerrs = starrows['fluxerr']
        func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))     # The 'delta' function
        if SCALE2ONE:
            fits = [func(p, (row['x']-CHIPXLEN/2)/(CHIPXLEN/2), (row['y']-CHIPYLEN/2)/(CHIPYLEN/2)) for row in starrows]
        else:
            fits = [func(p, row['x'], row['y']) for row in starrows]
        avgf = np.mean(starvals/fits)                                  # Our 'expected' value for the Flux
        starresid = (starvals/fits - avgf)/(starvalerrs/fits)          # Currently an Astropy Column
        return np.asarray(starresid).tolist()                          # Want to return as list so it is possible to flatten totalresid
 
def chisqall(params, tab, num_cpu = 4):
    starIDarr = np.unique(tab['id'])

    global count
    if count % 20 == 0: print count 
    count+=1     
    
    ########## Doing it with multiprocessing
    #runs = [(tab[np.where(tab['id'] == star)[0]], params) for star in stars2consid]
    #pool = Pool(processes=num_cpu)
    #results = pool.map_async(chisqstar, runs)
    #pool.close()
    #pool.join()
    #
    #final = [] 
    #for res in results.get():
    #    final.append(res)
    #final = np.asarray(final)
    #totalresid = reduce(lambda x, y: x + y, final) # flatten totalresid
    #return totalresid
    ########## 
    
    ##########  Doing it the original way
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    totalresid = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]], params) for star in starIDarr])     # an array of different sized lists
    totalresid = reduce(lambda x, y: x + y, totalresid)                                 # flatten totalresid        
    return totalresid
    ########## 
start_time = time.time()  

# Reduce the Table so that it doesn't have unused Columns that take up memory/time
tabreduced = np.copy(tab)               
tabreduced = Table(tabreduced)
tabreduced.remove_columns(['avgmag', 'avgmagerr', 'avgflux','avgfluxerr'])


########## Using op.leastsq
#maxfev = 1400
#count = 0
#print 'Starting Least Square Fit'
#result = op.leastsq(chisqall, initialcoeff, args = (tabreduced), maxfev = maxfev)
#print 'Count: ', count
##########


########## Using op.minimize / set up for MCMC
#f = .1
#def lnlike(params, x, y, z, zerr):
#    #realparams, lnf = params[:-1], params[-1]
#    realparams = params
#    #x = np.asarray(tab['x'])
#    #y = np.asarray(tab['y'])
#    #z = np.asarray((tab['flux'] - tab['avgflux']) / tab['avgflux'])
#    #zerr = get_normdelerr(tab)
#    
#    #if SCALE2ONE:
#    #    x = (x-CHIPXLEN/2)/(CHIPXLEN/2)
#    #    y = (y-CHIPYLEN/2)/(CHIPYLEN/2)  
#    x = np.asarray(x)
#    y = np.asarray(y)
#    z = np.asarray(z)
#    zerr = np.asarray(zerr)
#    
#
#     
#    model = np.array([])
#    for xi, yi in zip(x,y):
#        currfuncvals = func2fit(xi,yi)
#        model = np.append(model, np.sum(currfuncvals * realparams))    
#    inv_sigma2 = 1.0/(zerr**2) # + model**2*np.exp(2*lnf))
#    some = -0.5*(np.sum((z-model)**2*inv_sigma2 - np.log(inv_sigma2)))
#    print '***'
#    #print realparams
#    #print model
#    #print z
#    #print zerr
#    #
#    #print lnf
#    #print inv_sigma2
#    print some
#    return some
#
#import scipy.optimize as op
#initialcoeff = np.zeros(len(func2read))
#initialcoeff[0] = 1
##initialcoeff[-1] = np.log(f)
#nll = lambda *args: -lnlike(*args)
#
#def get_normdelerr(tab):
#    val = np.asarray(tab['flux'])
#    valerr = np.asarray(tab['fluxerr'])
#    avgval = np.asarray(tab['avgflux'])
#    avgvalerr = np.asarray(tab['avgfluxerr'])
#    
#    n = val - valerr
#    d = valerr
#    dn = np.sqrt(valerr**2 + avgvalerr**2)
#    r = np.abs((val-avgval)/avgval)
#    dr = r * np.sqrt((dn/n)**2 + (avgvalerr/avgval)**2)
#    return dr
#
#num = len(tab)
#x = tab['x'][:num]
#y = tab['y'][:num]
#if SCALE2ONE:
#    x = (x-CHIPXLEN/2)/(CHIPXLEN/2)
#    y = (y-CHIPYLEN/2)/(CHIPYLEN/2)
#z = (tab['flux'][:num] - tab['avg'+SPACE_VAL][:num]) / tab['avg'+'flux'][:num]
#zerr = get_normdelerr(tab[:num])
#
##num = 100
##tab = tab[:num]
#result = op.minimize(nll, initialcoeff, args = (x,y,z,zerr))
##########

end_time = time.time()
print "%s seconds for fitting the data going through each star" % (end_time - start_time)
try:
    finalcoeff = result[0]
except KeyError:
    finalcoeff = result.x[:]
    
#finalcoeff = initialcoeff
print 'Final Coefficients:'
print finalcoeff

###########################################
###########################################
###########################################


###########################################
############### Binning ##################
###########################################
import matplotlib.pyplot as plt
def do_bin(tab, xbin = XBIN, ybin = YBIN):  
    ### Taken from Binning.py and FluxFittingScipy.py and modified it ###
    """
    Purpose
    -------
                    To bin all the observations
    Parameters
    ----------
    tab:            The Astropy Table with all the information
    xbin:           number of bins in the x direction
    ybin:           number of bins in the y direction
    
    Returns
    -------
    [zzorig, zznum, zzavg]
    zzorig:         2D array of size xbin * ybin -- the original one -- where if 
                    there is nothing in a bin, 'None' is the element; 
                    and if there are multiple points/fluxes in the bin, 
                    there is an array of the normalized delta fluxes
    zznum:          2D array of size xbin * ybin -- the number one -- where its
                    value in a bin is the number of observations in that bin
                    (helpful to imshow if want to see where the observations lie)
    zzfinal:        2D array of size xbin * ybin -- the final one -- where the 
                    averages of each bin are taken, and if there was nothing in 
                    a bin, the average is set to 0             
    
    Notes/Problems/To Do
    --------------------
    **  For creating zz as a 2D array of size xbin * ybin with all the values 
        being set to 'None' causes some 'FutureWarnings' and I don't really know 
        how to solve the issue and go around it
     
    """
    xall = np.asarray(tab['x'])
    yall = np.asarray(tab['y'])
    delfluxall = np.asarray((tab['flux'] - tab['avgflux']) / tab['avgflux']) # normalized
    
    # Initialize an empty 2D array for the binning;
    # Create xbinarr and ybinarr as the (lengths of xbin and ybin, respectively);
    #     to make up the bins
    # Find dx and dy to help later with binning x+dx and y+dy
    zz = np.array([np.array([None for i in range(np.int(xbin))]) for j in range(np.int(ybin))])
    xbin, ybin = np.double(xbin), np.double(ybin)
    xbinarr = np.linspace(0,     CHIPXLEN,     xbin,     endpoint = False)
    ybinarr = np.linspace(0,     CHIPYLEN,     ybin,     endpoint = False)
    dx, dy = xbinarr[1] - xbinarr[0], ybinarr[1] - ybinarr[0]
    # Each element of xbinarr and ybinarr is the x and y value for a bin. 
    #    We are looking for the index/indicies that follow the 
    #    condition that the xall and yall are in a bin of x+dx and y+dy. 
    # The index/indicies that fall in this condition are stored in an array
    #    called inbin 
    for i, x in enumerate(xbinarr):
        for j, y in enumerate(ybinarr):
            inbin = np.where((xall >= x) & (xall < x + dx) & \
                             (yall >= y) & (yall < y + dy))[0]    # indexes of points in a bin
            if len(inbin):
                zz[i][j] = delfluxall[inbin]
            
            
    # Now deal with zz and take the averages in each bin 
    # Need to also build the x arrays, y arrays, and the delta magnitude arrays
    #     which will be used for the 2D fit 
    zz = zz.T         
    zzorig  = np.copy(zz)
    zznum   = np.copy(zz)
    zzavg   = np.copy(zz)
    for i in range(len(xbinarr)):
        for j in range(len(ybinarr)):
            if zzavg[i][j] == None: # gives a futurewarning : comparison to `None` will result in an elementwise object comparison in the future.
                zzavg[i][j] = 0
                zznum[i][j] = 0
            else:
                delfluxavg = np.mean(zz[i][j])
                zzavg[i][j] = delfluxavg
                zznum[i][j] = len(zz[i][j])
    return [zzorig, zznum, zzavg]
    
zzorig, zznum, zzavg = do_bin(tab)

###########################################
###########################################
###########################################


###########################################
############### Integrate #################
###########################################

def funcintegrate(x, y, coeff):
    return np.sum(func2fit(x, y) * coeff)

from scipy import integrate
def bounds_y():
    if SCALE2ONE:
        return [-1,1]
    else:
        return [0, CHIPYLEN]
def bounds_x():
    if SCALE2ONE:
        return [-1,1]
    else:
        return [0, CHIPXLEN]
def area():
    return np.sum(np.abs(bounds_y() + bounds_x()))
integrate_result = integrate.nquad(funcintegrate, [bounds_x(), bounds_y()], args = (finalcoeff,))[0]   
integrate_result /= area()
finalcoeff /= integrate_result

###########################################
###########################################
###########################################

  
###########################################
############### Plotting ##################
###########################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def convert2mesh(func2fit, coeff, xpixel = XPIX, ypixel = YPIX):
    ''' Creates a mesh using the coefficients and x and y pixel values '''
    if SCALE2ONE:
        xpixel = (xpixel - CHIPXLEN/2)/(CHIPXLEN/2)
        ypixel = (ypixel - CHIPYLEN/2)/(CHIPYLEN/2)
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) 
    fmesh = func2fit(xx, yy)
    coeff = np.asarray(coeff)
    zzfit = [[0 for i in xpixel] for j in ypixel]
    k = 0
    while k < len(coeff):
        zzfit += coeff[k]*fmesh[k]
        k+=1
    # Currently zzfit[0] are the values of varying x and keeping y = 0 (constant
    # Transposing it makes zzfit[0] the values of x = 0 and varying y
    return [xx, yy, zzfit]
       
def plotmesh(a, title = ''):
    ''' Returns the figure of the mesh fit plot '''
    X, Y, Z = a                   # a is the xx yy and zz (2d array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red')
    ax.set_title(title)
    plt.legend()
    return fig

#plt.show(plotmesh(convert2mesh(func2fit, coeff=initialcoeff), title = 'Initial: ' + str(funcname) + ' n = ' + str(n)))
plt.show(plotmesh(convert2mesh(func2fit, coeff=finalcoeff), title = 'Final: ' + str(funcname) + ' n = ' + str(n)))

def plotdelflux(tab):
    x = tab['x']
    y = tab['y']
    delflux = (tab['flux'] - tab['avgflux']) / tab['avgflux'] # normalized
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, delflux, alpha = .1)
    return fig
    
#plt.show(plotdelflux(tab))

def plotall(tab, a, lim, title = ''):
    X, Y, Z = a                    # a is the xx yy and zz (2d array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red')
    
    x = tab['x']
    y = tab['y']
    if SCALE2ONE:
        x = (x-CHIPXLEN/2)/(CHIPXLEN/2)
        y = (y-CHIPYLEN/2)/(CHIPYLEN/2)
    delflux = (tab['flux'] - tab['avgflux']) / tab['avgflux'] # normalized
    ax.scatter(x, y, delflux, s = 3)
    ax.set_zlim([-lim, lim])
    ax.set_title(title)
    plt.legend()
    return fig   
#plt.show(plotall(tab, convert2mesh(func2fit, coeff=initialcoeff), lim = .05, title = 'Initial: ' + str(funcname) + ' n = ' + str(n)))
#plt.show(plotall(tab, convert2mesh(func2fit, coeff=finalcoeff), lim = .05, title = 'Final: ' + str(funcname) + ' n = ' + str(n)))
   
def plotimg(img, title = '', fitplot = False, facecolor='white'):
    fig = plt.figure(facecolor=facecolor)
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
    
plt.show(plotimg(zznum, title = 'Number of Observations in a Bin'))
plt.show(plotimg(zzavg, title = 'Normalized Delta Fluxes Binned'))

#zzfitinitial = convert2mesh(func2fit, coeff=initialcoeff, xpixel = np.double(range(int(CHIPXLEN))), ypixel = np.double(range(int(CHIPYLEN))))[2]      # convert2mesh returns [xx, yy, zzfit]
#imginitial = plotimg(zzfitinitial, title = 'Initial: ' + str(funcname) + ' n = ' + str(n), fitplot = True)
#plt.show(imginitial)

zzfitfinal = convert2mesh(func2fit, coeff=finalcoeff, xpixel = np.double(range(int(CHIPXLEN))), ypixel = np.double(range(int(CHIPYLEN))))[2]      # convert2mesh returns [xx, yy, zzfit]
imgfinal = plotimg(zzfitfinal, title = 'Final: ' + str(funcname) + ' n = ' + str(n), fitplot = True)
plt.show(imgfinal)

finalfunc = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))     # The final flat

def simple3dmesh(coeff):
    fig = plt.figure()
    A = convert2mesh(func2fit, coeff)
    xx,yy,zz = A
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx,yy,zz, rstride=1, cstride=1, color='red')
    plt.show(fig)
    
def simple3dplot(rows, something, title = ''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rows['x'],rows['y'], something)
    ax.set_title(title)
    #ax.scatter(tab['x'], tab['y'], something, color='red')
    plt.show(fig)

def apply_flat(rows, coeff, somethingchar):
    final = np.array([])
    for x,y,something in zip(rows['x'], rows['y'], rows[somethingchar]):
        if SCALE2ONE:
            x = (x - CHIPXLEN/2)/(CHIPXLEN/2)
            y = (y - CHIPYLEN/2)/(CHIPYLEN/2)
        #print '***'
        #print x,y,something
        #print finalfunc(coeff,x,y)
        #print something / finalfunc(coeff,x,y)
        final = np.append(final, something / finalfunc(coeff,x,y))
    return final
    
final = apply_flat(tab, finalcoeff, SPACE_VAL)

############### If we want to plot the before/after of applying the flat (as is as well as the normalized delta)
simple3dplot(tab, tab[SPACE_VAL], title = 'Before LFlat, just ' + SPACE_VAL + ' values plotted')
simple3dplot(tab, final, title = 'After LFlat, just ' + SPACE_VAL + ' values plotted')
simple3dplot(tab, (tab[SPACE_VAL] - tab['avg'+SPACE_VAL])/ tab['avg'+SPACE_VAL], title = 'Before LFlat, normalized delta ' + SPACE_VAL)
simple3dplot(tab, (final - tab['avg'+SPACE_VAL])/tab['avg'+SPACE_VAL], title = 'After LFlat, normalized delta ' + SPACE_VAL)
###############

############### If we want to see/plot the mean of each star before and after applying the flat
#for star in np.unique(tab['id']):
#    starrows = tab[np.where(tab['id']==star)[0]]
#    finalstar = apply_flat(starrows, finalcoeff, SPACE_VAL)
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
    #simple3dplot(starrows, starrows[SPACE_VAL], title = 'Original ' + str(star) + ' ' + str(mean_before) + ', ' + str(std_before))
    #simple3dplot(starrows, finalstar, title = 'Final ' + str(star) + ' ' + str(mean_after) + ', ' + str(std_after))
############### 

###########################################
################ Misc. ####################
###########################################

############### If we want to see where each star lies on the detector ( imshow style )
#for star in np.unique(tab['id']):
#    tabstar = tab[np.where(tab['id'] ==star)[0]]
#    zzorig, zznum, zzavg = do_bin(tabstar, 30, 30)
#    plt.show(plotimg(zznum, title = 'Number of Observations in a Bin for star #' + str(star)))
###############    

############### If we want to see where each star lies on the detector ( plot style )
#for star in np.unique(tab['id'])[:10]:
#    tabstar = tab[np.where(tab['id'] ==star)[0]]
#    fig = plt.figure()
#    plt.scatter(tabstar['x'], tabstar['y'], s=50)
#    plt.xlim([0,CHIPXLEN])
#    plt.ylim([0,CHIPYLEN])
#    plt.grid(True)
#    plt.title('Observation Locations for Star #' + str(star), fontsize = 18)
#    plt.xticks(np.linspace(0, CHIPXLEN, 9))
#    plt.yticks(np.linspace(0, CHIPYLEN, 9))
#    plt.xlabel('X Pixel', fontsize = 18)
#    plt.ylabel('Y Pixel', fontsize = 18)
#    plt.show()
###############   

############### If we want to see where each star lies on the detector for multiple stars ( plot style )
#def plotnumobs(num, cmap = 'viridis', leg = True):
#    # num : number of stars you want to plot
#    fig = plt.figure()
#    plt.xlim([0,CHIPXLEN])
#    plt.ylim([0,CHIPYLEN])
#    plt.grid(True)
#    plt.title('Observation Locations for ' + str(num) + ' Stars', fontsize = 18)
#    plt.xticks(np.linspace(0, CHIPXLEN, 9))
#    plt.yticks(np.linspace(0, CHIPYLEN, 9))
#    plt.xlabel('X Pixel', fontsize = 18)
#    plt.ylabel('Y Pixel', fontsize = 18)
#    cmap = plt.get_cmap(cmap)
#    colors = [cmap(i) for i in np.linspace(0, 1, num)]
#    chosen = np.random.choice(len(np.unique(tab['id'])), num, replace = False)
#    for col, star in zip(colors, chosen):
#        tabstar = tab[np.where(tab['id'] ==star)[0]]
#        plt.scatter(tabstar['x'], tabstar['y'], s=10, c=col, marker='*', label = star)
#    if leg:
#        plt.legend(loc='best')
#    return fig
#plt.show(plotnumobs(num = 5, cmap = 'gnuplot'))
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
##length = 2
#for i in range(length) + np.ones(length):
#    i = int(i)
#    plt.show(plotnumobsGIF(num = i, cmap = 'jet'))
#    #plotnumobsGIF(num=i, cmap='jet').savefig('/Users/dkossakowski/Desktop/FinalPresentation/DitheringGIF/Stars/NumObs' + str(i) + '.png', dpi = 500)
############### 

############### If we want to smooth out the normalized delta flux
#import scipy.signal as signal
#kern = np.ones((2,2))
#zzsmooth = signal.convolve(zzavg, kern)
#zzsmooth = zzsmooth*np.max(zzavg)/ np.max(zzsmooth)
#plt.show(plotimg(zzsmooth, title = 'Smooth'))
###############

###########################################
###########################################
###########################################