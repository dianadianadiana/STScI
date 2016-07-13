import numpy as np
#from lmfit import minimize, Parameters, report_fit
import time
import scipy.optimize as op

CHIP1XLEN = CHIP2XLEN = 4096 
CHIP1YLEN = CHIP2YLEN = 2048

###########################################
###########################################
###########################################

from sympy import * 
def norder2dpoly(n):
        ''' 
        Purpose: Create the 2D nth order polynomial
        Returns:
            funclist -- A list of the different components of the poly 
                            (elements are Symbol types)
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

from astropy.table import Table, Column
from DataInfoTab import remove_stars_tab, convertmag2flux, convertflux2mag,\
                        make_avgmagandflux, sigmaclip_delmagdelflux

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
tab.remove_columns(['filenum','d1','d2','d3'])   # remove the dummy columns  

#datafil = 'f606w_phot_r5.txt'
#data = np.genfromtxt(path + datafil)
#
#names = ['id', 'image', 'chip', 'x', 'y', 'mag', 'magerr']
#types = [int, int, int, np.float64, np.float64, np.float64, np.float64]
#tab = Table(data, names=names, dtype=types)
#tab.remove_columns(['image'])

chosen = np.random.choice(len(tab), 8000, replace = False)
tab = tab[chosen]
tab.sort(['id'])                                 # sort the table by starID
starIDarr = np.unique(tab['id'])                 # collect all the star IDs

#datafil = 'realistic2.dat'
#data = np.genfromtxt(path+datafil)
#names = ['id', 'chip', 'x', 'y', 'mag', 'magerr']
#types = [int, int, np.float64, np.float64, np.float64, np.float64]
#
#tab = Table(data, names = names, dtype = types)
#tab.remove_row(0)
#chosen = np.random.choice(len(tab), 500, replace = False)
#tab = tab[chosen]
#starIDarr = np.unique(tab['id'])     # collect all the star IDs

tab =  tab[np.where((tab['mag'] <= 25) & (tab['mag'] >= 13))[0]] # (13,25)
print 'Len of tab after constraining the magnitudes'
print len(tab)
print 'Len of tab and starIDarr after 1st min_num_obs'
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)
print len(tab)
print len(starIDarr) 

tab, starIDarr = make_avgmagandflux(tab, starIDarr)
tab, starIDarr = sigmaclip_delmagdelflux(tab, starIDarr, flux = True, mag = False, low = 3, high = 3)
tab =  tab[np.where(tab['flux']/tab['fluxerr'] > 5)[0]] # S/N ratio for flux is greater than 5

###########################################
############ Function to Fit ##############
###########################################
n = 5
func2read, func2fit = norder2dpoly(n)
nx = ny = 5
func2read, func2fit = norder2dcheb(nx,ny)
print 'Function that is being fit:', func2read
func2string = np.copy(func2read)
k = 0
while k < len(func2string):
    func2string[k] = str(func2string[k])
    func2string[k] = func2string[k].replace('*','')
    k+=1

def getfit(tab, xpixel, ypixel, chipnum, n = 5):
    '''
    NEED TO ADD DESCRIPTION
    '''
    starsinboth = np.array([])
    for star in starIDarr:
        starrows = tab[np.where(tab['id']==star)[0]]
        chipavg = np.mean(starrows['chip'])
        if chipavg != 1.0 and chipavg != 2.0:
            starsinboth = np.append(starsinboth, star)
    rows = [row for row in tab if row['chip'] == chipnum or (row['id'] in starsinboth and row['chip'] != chipnum)]

    x = [row['x'] for row in rows]
    y = [row['y'] if  row['chip'] == 2 else row['y'] + CHIP2YLEN for row in rows]
    z = [row['flux'] - row['avgflux'] for row in rows]
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??
    f = func2fit(x,y)
    fmesh = func2fit(xx,yy)
    try:
        f[0] = np.ones(len(x))
        fmesh[0] = np.ones(len(xx))
    except TypeError:
        pass
    
    A = np.array(f).T
    B = z
    coeff, rsum, rank, s = np.linalg.lstsq(A, B)
    zfit = np.zeros(len(x))
    zzfit = [[0 for i in xpixel] for j in ypixel]
    k = 0
    while k < len(coeff):
        zfit += coeff[k]*f[k]
        zzfit += coeff[k]*fmesh[k]
        k+=1
    # Examples:
    # zfit = coeff[0]*x**2 + coeff[1]*y**2 + ... === coeff[0]*f[0] + ...
    # Zfit = coeff[0]*X**2 + coeff[1]*Y**2 + ... === coeff[0]*fmesh[0] + ...
    
    def get_res(z, zfit):
        # returns an array of the error in the fit
        return np.abs(zfit - z)
    resarr = get_res(z, zfit)
    return [x, y, z, zfit, xx, yy, zzfit, coeff, rsum, resarr]
    
xbin = 10
ybin = 5
xpixel1 = np.linspace(0, CHIP1XLEN, xbin)
ypixel1 = np.linspace(CHIP2YLEN, CHIP1YLEN + CHIP2YLEN, ybin)
xpixel2 = np.linspace(0, CHIP2XLEN, xbin)
ypixel2 = np.linspace(0, CHIP2YLEN, ybin)
x1, y1, z1, zfit1, xx1, yy1, zzfit1, coeff1, rsum1, resarr1 = getfit(tab, xpixel1, ypixel1, chipnum = 1, n = n)
x2, y2, z2, zfit2, xx2, yy2, zzfit2, coeff2, rsum2, resarr2 = getfit(tab, xpixel2, ypixel2, chipnum = 2, n = n)

initialvalarra = coeff1
initialvalarrb = coeff2

print 'Length of tab and starIDarr after everything'
lentab0 = len(tab)
lenstar0 = len(starIDarr)
print len(tab)
print len(starIDarr)
print 'Initial:'
print initialvalarra
print initialvalarrb

###########################################
###### Fitting by grouping by star ########
###########################################    
from multiprocessing import Pool
def chisqstar(starrows, p):
#def chisqstar(inputs):
        ''' Worker function '''
        # Input is the rows of the table corresponding to a single star so that we don't need to input a whole table
        #starrows, p = inputs
        starfluxes = starrows['flux']
        starfluxerrs = starrows['fluxerr']
        func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))
        fits = [func(p,row['x'], row['y']) if row['chip'] == 2 else func(p,row['x'] + CHIP2YLEN, row['y']) for row in starrows]
        avgf = np.mean(starfluxes/fits)
        starresid = (starfluxes/fits - avgf)/(starfluxerrs/fits) # currently an Astropy Column
        return np.asarray(starresid).tolist()

def chisqstar2(starx,stary,starflux,starfluxerr,p):
        ''' Worker function '''
        # TESTING fn : inputs are starx,stary,starflux,starfluxerr,params
        func = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))
        fits = [func(p,i,j) for i,j in zip(starx,stary)]
        avgf = np.mean(starflux/fits)
        starresid = (starflux/fits - avgf)/(starfluxerr/fits) # currently an Astropy Column
        return np.asarray(starresid).tolist()
        
def chisqall1(params, tab, chip2fit, num_cpu = 4):
    starIDarr = np.unique(tab['id'])
    stars2consid = starIDarr
    stars2consid = np.array([])
    for star in starIDarr:
        starrows = tab[np.where(tab['id']==star)[0]]
        chipavg = np.mean(starrows['chip'])
        if (chipavg != 1.0 and chipavg != 2.0) or (int(chipavg) == int(chip2fit)):
            stars2consid = np.append(stars2consid, star)
       
    global count
    if count % 20 == 0: print count 
    count+=1     
    
    # Doing it with multiprocessing
    #runs = [(tab[np.where(tab['id'] == star)[0]], params) for star in stars2consid]
#    pool = Pool(processes=num_cpu)
#    results = pool.map_async(chisqstar, runs)
#    pool.close()
#    pool.join()
#    
#    final = [] 
#    for res in results.get():
#        final.append(res)
#    final = np.asarray(final)
#    totalresid = reduce(lambda x, y: x + y, final) # flatten totalresid
#
#    return totalresid
    
    # Doing it by unwrapping the information first instead of in the worker function
    #totalresid = np.array([])
    #for star in stars2consid:
    #    starrows = tab[np.where(tab['id'] == star)[0]]
    #    starx = np.asarray(starrows['x'])
    #    stary = np.asarray([row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in starrows])
    #    starflux = np.asarray(starrows['flux'])
    #    starfluxerr = np.asarray(starrows['fluxerr'])
    #    totalresid = np.append(totalresid, chisqstar2(starx,stary,starflux,starfluxerr,params))
    #return totalresid
    
    # Doing it the original way
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    #totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
    totalresid = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]], params) for star in stars2consid])
    totalresid = reduce(lambda x, y: x + y, totalresid) # flatten totalresid
    return totalresid

start_time = time.time()  

tabreduced = np.copy(tab)
tabreduced = Table(tabreduced)
tabreduced.remove_columns(['mag','magerr','avgmag','avgmagerr','avgflux','avgfluxerr'])

chip2fit = 1
count = 0
print 'Starting chip1'
result1 = op.leastsq(chisqall1, initialvalarra, args = (tabreduced, chip2fit), maxfev = 200)
chip2fit = 2
count = 0
print 'Starting chip2'
result2 = op.leastsq(chisqall1, initialvalarrb, args = (tabreduced, chip2fit), maxfev = 200)
end_time = time.time()
print "%s seconds for fitting the data going through each row" % (end_time - start_time)
print lentab0/(end_time - start_time), ' = how many observation are being processed per second' 

print 'End:'
print result1[0]
print result2[0]

# July 11
# ~ 22 mins for 3000 data points for n=2 for maxfev = 200
# 135 s for 116 data points for n=2 for maxfev = 200
# 10.6s for 342 data points for n=0 for maxfev = 200 no multi (32.26/s) -- it would take 52 mins for 100,000 data points
# 16s for 316 data points for n=0 for maxfev = 200 (multi of 2) (19.7/s)
# 23s for 405 data points for n=0 for maxfev = 200 (multi of 8) (17.6/s)
# 19s for 293 data points for n=0 for maxfev = 200 (multi of 4) (15/s)
# 88s for 3122 data points for n=0 for maxfev = 200 no multi (35.47/s)
# 140s for 3101 data points for n=0 for maxfev = 200 no multi (22.15/s)
# 248s for 164 data points for n=1 for maxfev = 500 (multi of 8) (less than 1 per second) bad fit
# 31s for 192 data points for n=1 for maxfev = 500 no multi bad fit

# July 12
# 1.435s for 88 data points, 22 stars for n=0 maxfev = 200 no multi, using tabreduced (61/s)
# 2.204s for 88 data points, 22 stars for n=0 maxfev = 200 no multi, using tab        (40/s)
# 26.19s for 79 data points, 21 stars for n=2 maxfev = 200 no multi, using tabreduced (3/s)   bad fit (not enough points)
# 41.06s for 79 data points, 21 stars for n=2 maxfev = 200 no multi, using tab        (1.9/s) bad fit (not enough points)
# 74.74s for 70 data points, 19 stars for n=2 maxfev = 200 4  multi, using tabreduced (.93/s) bad fit (not enough points)
# 95.24s for 70 data points, 19 stars for n=2 maxfev = 200 4  multi, using tab        (.73/s) bad fit (not enough points)
# 29.59s for 113 data points, 28 stars for n=2 maxfev = 200 no multi, using tabreduced & breaking apart x and y (3.8/s) bad fit
# 52.1s for 113 data points, 28 stars for n=2 maxfev = 200 no multi, using tab & breaking apart x and y (2.1/s) bad fit


###########################################
############### Plotting ##################
###########################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def convert2mesh(chipnum, resultparams, xbin = 10, ybin = 5, wave = False):
    if chipnum == 2:
        xpixel = np.linspace(0, CHIP2XLEN, xbin)
        ypixel = np.linspace(0, CHIP2YLEN, ybin)
    else: # chipnum = 1
        xpixel = np.linspace(0, CHIP1XLEN, xbin)
        ypixel = np.linspace(CHIP2YLEN, CHIP2YLEN + CHIP1YLEN, ybin) 
        #ypixel = np.linspace(0, CHIP1YLEN, ybin) 
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??
    fmesh = func2fit(xx,yy)
    coeff = np.asarray(resultparams)
    zzfit = [[0 for i in xpixel] for j in ypixel]
    k = 0
    while k < len(coeff):
        zzfit += coeff[k]*fmesh[k]
        k+=1
    return [xx, yy, zzfit]
    
def plotthis(a,b):
    # a, b are the xx yy and zz (2d arrays)
    X1,Y1,Z1 = a
    X2,Y2,Z2 = b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1,Y1,Z1, rstride=1, cstride=1,color='red', label = "CHIP1")
    ax.plot_wireframe(X2,Y2,Z2, rstride=1, cstride=1,color='blue', label = "CHIP2")
    #ax.set_zlim([-2,2])
    return fig
plt.show(plotthis(convert2mesh(chipnum=1, resultparams=initialvalarra), convert2mesh(chipnum=2, resultparams=initialvalarrb)))
plt.show(plotthis(convert2mesh(chipnum=1, resultparams=result1[0]), convert2mesh(chipnum=2, resultparams=result2[0])))

def plotdelflux(tab):
    x = tab['x']
    y = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    delflux = tab['flux'] - tab['avgflux']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,delflux)
    plt.show()
plotdelflux(tab)

def plotall(tab,a,b,lim):
    X1,Y1,Z1 = a
    X2,Y2,Z2 = b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1,Y1,Z1, rstride=1, cstride=1,color='red', label = "CHIP1")
    ax.plot_wireframe(X2,Y2,Z2, rstride=1, cstride=1,color='blue', label = "CHIP2")
    
    x = tab['x']
    y = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    delflux = tab['flux'] - tab['avgflux']
    ax.scatter(x,y,delflux, s = 3)
    ax.set_zlim([-lim, lim])
    plt.legend()
    return fig   
plt.show(plotall(tab,convert2mesh(chipnum=1, resultparams=initialvalarra), convert2mesh(chipnum=2, resultparams=initialvalarrb), lim = 5))
plt.show(plotall(tab,convert2mesh(chipnum=1, resultparams=result1[0]), convert2mesh(chipnum=2, resultparams=result2[0]), lim = 100))

def plotimg(img, title = ''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('X Pixel', fontsize = 18);  ax.set_ylabel('Y Pixel', fontsize = 18)
    extent=(0,4096,0,4096)    
    cax = ax.imshow(np.double(img), cmap = 'gray_r', interpolation='nearest', origin='lower', extent=extent)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    return fig
    
def plot2imgs(img1, img2, title =''):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    #ax1.set_xlabel('X Pixel', fontsize = 18);  ax1.set_ylabel('Y Pixel', fontsize = 18)
    extent1 = (0,4096,2048,4096)
    cax1 = ax1.imshow(np.double(img1), cmap = 'gray_r', interpolation='nearest', origin='lower', extent=extent1)
    ax2 = fig.add_subplot(212)
    extent2 = (0,4096,0,2048)
    cax2 = ax2.imshow(np.double(img2), cmap = 'gray_r', interpolation='nearest', origin='lower', extent=extent2)
    #fig.colorbar(cax1, fraction=0.046, pad=0.04)

    return fig

imgfit = plot2imgs(zzfit1, zzfit2, title = 'Chip 1 on top, Chip 2 on bottom')
plt.show(imgfit)
zzfit1 = convert2mesh(chipnum=1, resultparams=initialvalarra)[2]
zzfit2 = convert2mesh(chipnum=2, resultparams=initialvalarrb)[2]
imgfit = plot2imgs(zzfit1, zzfit2, title = 'Chip 1 on top, Chip 2 on bottom')
plt.show(imgfit)
