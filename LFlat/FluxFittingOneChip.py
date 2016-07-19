import numpy as np
import time
import scipy.optimize as op

# Creating constants to keep a "standard"
CHIPXLEN =  CHIPYLEN = 1024

BIN_NUM = 2
XBIN = YBIN = 10 * BIN_NUM
XPIX = np.linspace(0,     CHIPXLEN,     XBIN)
YPIX = np.linspace(0,     CHIPYLEN,     YBIN)

###########################################
######### Functions to Optimize ###########
###########################################

from sympy import * 
def norder2dpoly(n):
        ''' 
        Purpose
        -------
        Create the 2D nth order polynomial
        
        Returns
        -------
            funclist -- A list of the different components of the poly 
                            (elements are Symbol types) useful for printing
            f        -- A function made from funclist and takes in two
                            parameters, x and y
        How it works:
            a 2nd order can be grouped like: 1; x1y0 x0y1; x2y0 x1y1 x0y2
            (where x0 = x**0, x1 = x**1 and so on)
            So the degree of x starts at a certain number (currnum) and decreases
            by one, while the degree of y starts at 0 and increase by one until currnum
        Note: 
            * lambdify needs to take in a list and not array; and for a 2d
            polynomial, the constant term needs to be 1 + 0*x but lambdify
            makes it just 1, which becomes a problem when trying to fit the 
            function and hence why the 0th element turns into np.ones(len(x))
            * Also just found out (2 weeks later after making this), that there 
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
        
        Returns
        -------
            funclist -- A list of the different components of the poly 
                            (elements are Symbol types) useful for printing
            f        -- A function made from funclist and takes in two
                            parameters, x and y
        Note: 
            * lambdify needs to take in a list and not array; and for a 2d
            polynomial, the constant term needs to be 1 + 0*x but lambdify
            makes it just 1, which becomes a problem when trying to fit the 
            function and hence why the 0th element turns into np.ones(len(x))
        '''
    x = Symbol('x')
    y = Symbol('y')
    funcarr = cheb.chebvander2d(x,y,[nx,ny])
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

# Read in the data
path = '/Users/dkossakowski/Desktop/Data/'
datafil = 'wfc_f606w_r5.lflat'
data = np.genfromtxt(path + datafil)

# Create an Astropy table
# tab[starID][0]: starID; ...[2]: chip#; ...[3]: x pixel; ...[4]: y pixel; 
# ...[5]: magnitude; ...[6]: magnitude error; rest: dummy
############
#names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr', 'd1', 'd2', 'd3']
#types = [int, int, int, np.float64, np.float64, np.float64, np.float64,
#         float, float, float]
#tab = Table(data, names=names, dtype=types)
#tab.remove_columns(['filenum','d1','d2','d3'])   # remove the dummy columns  
############

############ Read in NEW data :: comment/uncomment this snippet
datafil = 'flatfielddata.txt'
data = np.genfromtxt(path + datafil)
names = ['id', 'filenum', 'chip', 'x', 'y', 'mag', 'magerr']
types = [int, int, int, np.float64, np.float64, np.float64, np.float64]
tab = Table(data, names=names, dtype=types)
tab.remove_columns(['filenum'])
tab = tab[np.where(tab['magerr'] < 1.)[0]]
#tab = tab[np.where(tab['chip'] == 2)[0]]
tab = tab[np.where(tab['x'] <= 950)[0]]
tab = tab[np.where(tab['x'] >= 100)[0]]
tab = tab[np.where(tab['y'] <= 950)[0]]
tab = tab[np.where(tab['x'] >= 100)[0]]
############
#chosen = np.random.choice(len(tab), 4000, replace = False)
#tab = tab[chosen]
tab.sort(['id'])                                 # sort the table by starID
starIDarr = np.unique(tab['id'])                 # collect all the star IDs

############ Read in NEW data :: comment/uncomment this snippet
#datafil = 'realistic2.dat'
#data = np.genfromtxt(path+datafil)
#names = ['id', 'chip', 'x', 'y', 'mag', 'magerr']
#types = [int, int, np.float64, np.float64, np.float64, np.float64]
#
#tab = Table(data, names = names, dtype = types)
#tab.remove_row(0)
#chosen = np.random.choice(len(tab), 300, replace = False)
#tab = tab[chosen]
#starIDarr = np.unique(tab['id'])     # collect all the star IDs
############

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

n = 4
func2read, func2fit = norder2dpoly(n)             # nth order 2d Polynomial

nx = ny = 6
#n = nx
#func2read, func2fit = norder2dcheb(nx, ny)        # nx th and ny th order 2d Chebyshev Polynomial
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
    z = np.asarray((tab['flux'] - tab['avgflux']) / tab['avgflux'])          # normalized delta flux
    
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

initialcoeff = coeff
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
        starfluxes = starrows['flux']
        starfluxerrs = starrows['fluxerr']
        func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))     # The 'delta' function
        fits = [func(p, row['x'], row['y']) for row in starrows]
        avgf = np.mean(starfluxes/fits)                                  # Our 'expected' value for the Flux
        starresid = (starfluxes/fits - avgf)/(starfluxerrs/fits)         # Currently an Astropy Column
        return np.asarray(starresid).tolist()                            # Want to return as list so it is possible to flatten totalresid

def chisqstar2(starx,stary,starflux,starfluxerr,p):
        ''' Worker function '''
        # TESTING FUNCITON : inputs are starx,stary,starflux,starfluxerr,params INSTEAD of rows of a tabl
        # Created this function to test speed difference between unraveling the data in the worker function
        #       versus in the bigger function (Found there is a little difference and probably not worth it)
        func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))
        fits = [func(p,i,j) for i,j in zip(starx,stary)]
        avgf = np.mean(starflux/fits)
        starresid = (starflux/fits - avgf)/(starfluxerr/fits) # currently an Astropy Column
        return np.asarray(starresid).tolist()
        
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
    
    ##########  Doing it by unwrapping the information first instead of in the worker function
    #totalresid = np.array([])
    #for star in stars2consid:
    #    starrows = tab[np.where(tab['id'] == star)[0]]
    #    starx = np.asarray(starrows['x'])
    #    stary = np.asarray([row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in starrows])
    #    starflux = np.asarray(starrows['flux'])
    #    starfluxerr = np.asarray(starrows['fluxerr'])
    #    totalresid = np.append(totalresid, chisqstar2(starx,stary,starflux,starfluxerr,params))
    #return totalresid
    ########## 
    
    ##########  Doing it the original way
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    #totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
    totalresid = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]], params) for star in starIDarr])
    totalresid = reduce(lambda x, y: x + y, totalresid) # flatten totalresid        
    return totalresid
    ########## 

start_time = time.time()  

# Reduce the Table so that it doesn't have unused Columns that take up memory/time
tabreduced = np.copy(tab)               
tabreduced = Table(tabreduced)
tabreduced.remove_columns(['mag', 'magerr', 'avgmag', 'avgmagerr', 'avgflux', 'avgfluxerr'])

maxfev = 100
count = 0
print 'Starting Least Square Fit'
result = op.leastsq(chisqall, initialcoeff, args = (tabreduced), maxfev = maxfev)
end_time = time.time()
print "%s seconds for fitting the data going through each star" % (end_time - start_time)
print lentab0/(end_time - start_time), ' = how many observation are being processed per second' 


try:
    finalcoeff = result[0]
except KeyError:
    finalcoeff = result.x
#finalcoeff = initialcoeff
print 'Final Coefficients:'
print finalcoeff
print 'Count: ', count


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
############### Plotting ##################
###########################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def convert2mesh(func2fit, coeff, xpixel = XPIX, ypixel = YPIX):
    ''' Creates a mesh using the coefficients and x and y pixel values '''
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) 
    fmesh = func2fit(xx, yy)
    coeff = np.asarray(coeff)
    zzfit = [[0 for i in xpixel] for j in ypixel]
    k = 0
    while k < len(coeff):
        zzfit += coeff[k]*fmesh[k]
        k+=1
    #zzfit = zzfit.T # Currently zzfit[0] are the values of varying x and keeping y = 0 (constant
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
    
plt.show(plotmesh(convert2mesh(func2fit, coeff=initialcoeff), title = 'Initial'))
plt.show(plotmesh(convert2mesh(func2fit, coeff=finalcoeff), title = 'Final'))

def plotdelflux(tab):
    x = tab['x']
    y = tab['y']
    delflux = (tab['flux'] - tab['avgflux']) / tab['avgflux'] # normalized
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, delflux, alpha = .1)
    return fig
    
plt.show(plotdelflux(tab))

def plotall(tab, a, lim, title = ''):
    X, Y, Z = a                    # a is the xx yy and zz (2d array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='red')
    
    x = tab['x']
    y = tab['y']
    delflux = (tab['flux'] - tab['avgflux']) / tab['avgflux'] # normalized
    ax.scatter(x, y, delflux, s = 3)
    ax.set_zlim([-lim, lim])
    ax.set_title(title)
    plt.legend()
    return fig   
plt.show(plotall(tab, convert2mesh(func2fit, coeff=initialcoeff), lim = .05, title = 'Initial: poly2d n = ' + str(n)))
plt.show(plotall(tab, convert2mesh(func2fit, coeff=finalcoeff), lim = .05, title = 'Final: poly2d n = ' + str(n)))
   
def plotimg(img, vmin, vmax, title = ''):
#def plotimg(img, title = '', *kargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('X Pixel', fontsize = 18);  ax.set_ylabel('Y Pixel', fontsize = 18)
    extent = (0, CHIPXLEN, 0, CHIPYLEN)    
    cax = ax.imshow(np.double(img), cmap = 'viridis', interpolation='nearest', \
                                    origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    return fig
    
#plt.show(plotimg(zznum, vmin = np.min(zznum), vmax = np.max(zznum), title = 'Number of Observations in a Bin'))
#plt.show(plotimg(zzavg, vmin = np.min(zzavg), vmax = np.max(zzavg), title = 'Normalized Delta Flux Binned'))

zzfitinit = convert2mesh(func2fit, coeff=initialcoeff)[2]     # convert2mesh returns [xx, yy, zzfit]
imginitial = plotimg(zzfitinit, vmin = np.min(zzavg), vmax = np.max(zzavg), title = 'Initial: poly2d n = ' + str(n))
plt.show(imginitial)

zzfitfinal = convert2mesh(func2fit, coeff=finalcoeff)[2]      # convert2mesh returns [xx, yy, zzfit]
imgfinal = plotimg(zzfitfinal, vmin = np.min(zzavg), vmax = np.max(zzavg), title = 'Final: poly2d n = ' + str(n))
plt.show(imgfinal)

import scipy.signal as signal
kern = np.ones((2,2))
zzsmooth = signal.convolve(zzavg, kern)
zzsmooth = zzsmooth*np.max(zzavg)/ np.max(zzsmooth)
#plt.show(plotimg(zzsmooth, vmin = np.min(zzsmooth), vmax = np.max(zzsmooth), title = 'Smooth'))

###########################################
###########################################
###########################################