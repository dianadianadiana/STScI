import numpy as np
from lmfit import minimize, Parameters, report_fit
import time

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
            lambdify needs to take in a list and not array; and for a 2d
            polynomial, the constant term needs to be 1 + 0*x but lambdify
            makes it just 1, which becomes a problem when trying to fit the 
            function and hence why the 0th element turns into np.ones(len(x))
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
        funclist = funcarr.tolist() # lambdify only takes in lists and not arrays
        f = lambdify((x, y), funclist) # lambdify looks at 1 + 0*x as 1 and makes f[0] = 1
        return funclist, f # return the list in case we want to look at how the function is

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
tab.sort(['id'])                                 # sort the table by starID
tab = tab[10000:10050]
starIDarr = np.unique(tab['id'])                 # collect all the star IDs

datafil = 'one_noise.txt'
data = np.genfromtxt(path+datafil)
names = ['id', 'chip', 'x', 'y', 'mag', 'magerr']
types = [int, int, np.float64, np.float64, np.float64, np.float64]

tab = Table(data, names = names, dtype = types)
tab.remove_row(0)
starIDarr = np.unique(tab['id'])     # collect all the star IDs

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

func2read, func2fit = norder2dpoly(0)
print func2read
func2string = np.copy(func2read)
k = 0
while k < len(func2string):
    func2string[k] = str(func2string[k])
    func2string[k] = func2string[k].replace('*','')
    k+=1

###########################################
####### Setting up the Parameters #########
###########################################      

paramsa = Parameters()
paramsb = Parameters()
'''
#          (Name,  Value, Vary,  Min,  Max,   Expr)
paramsa.add_many(
           ('a1', 1e-20, True,  None,  None,  None),
           ('ax',   0,   True,  None,  None,  None),
           ('ay',   0,   True,  None,  None,  None),
           ('ax2',  0,   True,  None,  None,  None),
           ('axy',  0,   True,  None,  None,  None),
           ('ay2',  0,   True,  None,  None,  None))
           
paramsb.add_many(
           ('b1',  1e-20, True,  None,  None,  None),
           ('bx',    0,   True,  None,  None,  None),
           ('by',    0,   True,  None,  None,  None),
           ('bx2',   0,   True,  None,  None,  None),
           ('bxy',   0,   True,  None,  None,  None),
           ('by2',   0,   True,  None,  None,  None))
       '''    
# To make it more versatile and adaptable... use func2string
#paramsa.add('a' + func2string[0], 1e-8, True, None, None, None)
#paramsb.add('b' + func2string[0], 1e-8, True, None, None, None)


initialvalarra = np.zeros(len(func2string))
initialvalarrb = np.zeros(len(func2string))
initialvalarra[0] = initialvalarrb[0] = 1
initialvalarra2 = [ -4.98347182e+00 ,  3.38102878e-04 ,  3.23045209e-03  , 8.47315131e-08,
  -1.78686557e-07 , -5.05119015e-07]
initialvalarrb2 = [ -1.41263472e-01 ,  1.35223560e-04  , 1.08437470e-04 ,  1.02392897e-07,
  -2.29799058e-07 ,  1.20291872e-08]
initialvalarra5 = np.asarray([ -1.64584435e-11,   4.97169658e-09 , -1.12197506e-08 , -1.22292028e-05,
   1.03423671e-05 , -1.71735331e-06,   6.92467115e-09,  1.70430321e-09,
  -5.79151447e-09  , 1.12096367e-09,  -2.28116739e-12,   6.62550707e-14,
  -2.14688036e-13   ,1.20419983e-12,  -2.41487777e-13,   1.71972376e-16,
   2.51453023e-16  ,-2.62636118e-16,   1.01128111e-16, -9.93518825e-17,
   1.72057803e-17])
initialvalarrb5 = np.asarray([  3.64125186e-06,   1.48570623e-03 , -3.39144333e-04 , -6.15906753e-06,
   1.52775045e-06,  -3.49543054e-06,   9.46819175e-09,  -2.65781172e-09,
   1.53540658e-09,   5.90413613e-09,  -6.03073373e-12,   2.59840035e-12,
  -9.77822378e-13,  -1.58412464e-12,  -3.20367591e-12,   1.35366710e-15,
  -8.74887202e-16,   5.43835810e-16,  -2.02095285e-16,   6.08237419e-16,
   5.45353693e-16])
   
initialvalarra = initialvalarra
initialvalarrb = initialvalarrb

for elem, initialvala, initialvalb in zip(func2string, initialvalarra, initialvalarrb):
    paramsa.add('a' + elem, initialvala, True, None, None, None)
    paramsb.add('b' + elem, initialvalb, True, None, None, None)

paramsa.pretty_print()
paramsb.pretty_print()

###########################################
###### Fitting by grouping by star ########
###########################################      

def chisqstar(starrows, params):
    ''' Worker function '''
    def get_coeff():
        pardict = params.valuesdict()
        return np.asarray(pardict.values())
    def getfitvalue(chipnum, x, y, pointvalue = True):
        if chipnum == 1 and chip2fit == 2: y = y + CHIP2YLEN 
        if chipnum == 2 and chip2fit == 1: y = y - CHIP2YLEN
        funcvalues = func2fit(x,y)
        if not pointvalue:
            funcvalues[0] = np.ones(len(x))
        return np.sum(funcvalues * get_coeff())
    # Input is the rows of the table corresponding to a single star so that we don't need to input a whole table
    starfluxes = starrows['flux']
    starfluxerrs = starrows['fluxerr']
    fits = [getfitvalue(row['chip'], row['x'], row['y']) for row in starrows]
    avgf = np.mean(starfluxes/fits)
    starresid = (starfluxes/fits - avgf)/(starfluxerrs/fits) # currently an Astropy Column
    #print 'starfluxes', starfluxes
    #print 'errors in star flux', starfluxerrs
    #print 'fits', fits
    #print 'avgf', avgf
    return np.asarray(starresid).tolist()
   
def chisqall1(params, func2fit, tab, chip2fit):
    starIDarr = np.unique(tab['id'])
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    #totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
    totalresid = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]], params) for star in starIDarr])
    totalresid = reduce(lambda x, y: x + y, totalresid) # flatten totalresid
    #print np.sum(np.asarray(totalresid)**2)
    #print totalresid
    global count
    count+=1
    return totalresid
count = 0
start_time = time.time()  
chip2fit = 1
resulta = minimize(chisqall1, paramsa, args=(func2fit, tab, chip2fit))
resparamsa = resulta.params
report_fit(resparamsa, show_correl = False)
chip2fit = 2
resultb = minimize(chisqall1, paramsb, args=(func2fit, tab, chip2fit))
resparamsb = resultb.params
report_fit(resparamsb, show_correl = False)
print count
print "%s seconds for fitting the data looking at each star" % (time.time() - start_time)
# 10.6s for tab[10000:10500] 2nd order poly
# 19.17s for tab[10000:10500] 5th order poly
# 110.4s for tab[10000:13000] 5th order poly


###########################################
#### Fitting by going through each row ####
###########################################

def chisqall(params, func2fit, tab, chip2fit):
    def get_coeff():
        pardict = params.valuesdict()
        return np.asarray(pardict.values())
        
    def getfitvalue(chipnum, x, y, pointvalue = True):
        if chipnum == 1 and chip2fit == 2: y = y + CHIP2YLEN 
        if chipnum == 2 and chip2fit == 1: y = y - CHIP2YLEN
        funcvalues = func2fit(x,y)
        if not pointvalue:
            funcvalues[0] = np.ones(len(x))
        return np.sum(funcvalues * get_coeff())
            
    resid = np.array([])
    for row in tab:
        currstar = row['id']
        currstarrows = tab[np.where(tab['id'] == currstar)[0]]
        currx, curry = row['x'], row['y']
        currf, currferr = row['flux'], row['fluxerr']
        currchip = row['chip']
        currfit = getfitvalue(currchip, currx, curry)
        currstarfluxes = currstarrows['flux']
        fits = [getfitvalue(row['chip'], row['x'], row['y']) for row in currstarrows]
        curravgf = np.mean(currstarfluxes/fits)
        resid = np.append(resid, currf/currferr - currfit/currferr * curravgf)
    print np.sum(resid**2)
    return resid

start_time = time.time()  
#chip2fit = 1
#resulta = minimize(chisqall, paramsa, args=(func2fit, tab, chip2fit))
#resparamsa = resulta.params
#report_fit(resparamsa)
#chip2fit = 2
#resultb = minimize(chisqall, paramsb, args=(func2fit, tab, chip2fit))
#resparamsb = resultb.params
#report_fit(resparamsb)
print "%s seconds for fitting the data going through each row" % (time.time() - start_time)
# 53.5s for tab[10000:10500] 2nd order poly
# 99.63s for tab[10000:10500] 5th order poly
# 633.5s for tab[10000:13000] 5th order poly

'''
#          (Name,  Value, Vary,  Min,  Max,   Expr)
paramswavea = Parameters()
paramswaveb = Parameters()
paramswavea.add_many(
           ('aA',   1e20,   True,  None,  None,  None),
           ('au',   -1e-10,   True,  None,  None,  None),
           ('av',   0,   True,  None,  None,  None),
           ('aphi', 0,   True,  None,  None,  None))
paramswaveb.add_many(
           ('bA',   1e20,   True,  None,  None,  None),
           ('bu',   -1e-10,   True,  None,  None,  None),
           ('bv',   0,   True,  None,  None,  None),
           ('bphi', 0,   True,  None,  None,  None))
           
def chisqstarwave(starrows, params):
    """ Worker function """
    def get_coeff():
        pardict = params.valuesdict()
        return np.asarray(pardict.values())
    def getfitvalue(chipnum, x, y, pointvalue = True):
        if chipnum == 1 and chip2fit == 2: y = y + CHIP2YLEN 
        if chipnum == 2 and chip2fit == 1: y = y - CHIP2YLEN
        chipletter = 'a' if chip2fit == 1 else 'b'
        A = params[chipletter + 'A'].value
        u = params[chipletter + 'u'].value
        v = params[chipletter + 'v'].value
        phi = params[chipletter + 'phi'].value
        modelwave = A * np.cos(2*np.pi * (u*x + v*y) + phi)
        return modelwave
    # Input is the rows of the table corresponding to a single star so that we don't need to input a whole table
    starfluxes = starrows['flux']
    starfluxerrs = starrows['fluxerr']
    fits = [getfitvalue(row['chip'], row['x'], row['y']) for row in starrows]
    avgf = np.mean(starfluxes/fits)
    starresid = (starfluxes/fits - avgf)/(starfluxerrs/fits) # currently an Astropy Column
    return np.asarray(starresid).tolist()
   
def chisqallwave(params, func2fit, tab, chip2fit):
    starIDarr = np.unique(tab['id'])
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    #totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
    totalresid = np.asarray([chisqstarwave(tab[np.where(tab['id'] == star)[0]], params) for star in starIDarr])
    return reduce(lambda x, y: x + y, totalresid) # flatten totalresid

start_time = time.time()  
#chip2fit = 1
#resulta = minimize(chisqallwave, paramswavea, args=(func2fit, tab, chip2fit))
#resparamsa = resulta.params
#report_fit(resparamsa, show_correl = False)
#chip2fit = 2
#resultb = minimize(chisqallwave, paramswaveb, args=(func2fit, tab, chip2fit))
#resparamsb = resultb.params
#report_fit(resparamsb, show_correl = False)
print "%s seconds for fitting the data to a sine model going through each star" % (time.time() - start_time)
# 51.5s for tab[10000:10500] 
# 104.68s for tab[10000:13000]

'''

def example():
    # create data to be fitted
    x = np.linspace(0, 15, 301)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2) )
    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data""" 
        amp = params['amp'].value
        shift = params['shift'].value
        omega = params['omega'].value
        decay = params['decay'].value
        model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay) 
        return model - data
    # create a set of Parameters
    params = Parameters()
    params.add('amp',   value= 10,  min=0)
    params.add('decay', value= 0.1)
    params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
    params.add('omega', value= 3.0)
    # do fit, here with leastsq model
    result = minimize(fcn2min, params, args=(x, data))
    # calculate final result
    final = data + result.residual
    plt.plot(x,data)
    #plt.plot(x,final)
    plt.show()
    print result.residual
    resultparams = result.params
    print resultparams
    amp = resultparams['amp'].value
    decay = resultparams['decay'].value
    shift = resultparams['shift'].value
    omega = resultparams['omega'].value
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay) 
    plt.plot(x,model)
    plt.show()
    report_fit(result.params)

###########################################
###########################################
###########################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def convert2mesh(chipnum, resultparams, xbin = 10, ybin = 5, wave = False):
    if chipnum == 2:
        xpixel = np.linspace(0, CHIP2XLEN, xbin)
        ypixel = np.linspace(0, CHIP2YLEN, ybin)
    else: # chipnum = 1
        xpixel = np.linspace(0, CHIP1XLEN, xbin)
        #ypixel = np.linspace(0, CHIP2YLEN + CHIP1YLEN, ybin) 
        ypixel = np.linspace(0, CHIP1YLEN, ybin) 
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??
    if not wave: # if it's a polynomial
        fmesh = func2fit(xx,yy)
        if chipnum == 1:
            yy = yy + CHIP2YLEN
        resultdict = resultparams.valuesdict()
        coeff = np.asarray(resultdict.values())
        zzfit = [[0 for i in xpixel] for j in ypixel]
        k = 0
        while k < len(coeff):
            zzfit += coeff[k]*fmesh[k]
            k+=1
        return [xx, yy, zzfit]
    else: # It's a wave function
        chipletter = 'a' if chipnum == 1 else 'b'
        A = resultparams[chipletter + 'A'].value
        u = resultparams[chipletter + 'u'].value
        v = resultparams[chipletter + 'v'].value
        phi = resultparams[chipletter + 'phi'].value
        modelwave = A * np.cos(2*np.pi * (u*xx + v*yy) + phi)
        return [xx,yy,modelwave]
    
def plotthis(a,b):
    # a, b are the xx yy and zz (2d arrays)
    X1,Y1,Z1 = a
    X2,Y2,Z2 = b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1,Y1,Z1, rstride=1, cstride=1,color='red', label = "CHIP1")
    ax.plot_wireframe(X2,Y2,Z2, rstride=1, cstride=1,color='blue', label = "CHIP2")
    return fig
plt.show(plotthis(convert2mesh(chipnum=1, resultparams=resparamsa), convert2mesh(chipnum=2, resultparams=resparamsb)))

def plotdelflux(tab):
    x = tab['x']
    y = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    delflux = tab['flux'] - tab['avgflux']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,delflux)
    plt.show()
plotdelflux(tab)

def plotimg(title = ''):
    xxa, yya, zza = convert2mesh(chipnum=1, resultparams=resparamsa, wave=False)
    xxb, yyb, zzb = convert2mesh(chipnum=2, resultparams=resparamsb, wave=False)
    imga = zza
    imgb = zzb
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('X Pixel', fontsize = 18);  ax.set_ylabel('Y Pixel', fontsize = 18)
    extent=(0,4096,0,4096)    
    cax = ax.imshow(np.double(imga), cmap = 'gray_r', interpolation='nearest', origin='lower', extent=extent)
    cax = ax.imshow(np.double(imgb), cmap = 'gray_r', interpolation='nearest', origin='lower', extent=extent)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    return fig
imgbin = plotimg(title = 'Average delta flux in bins')
plt.show(imgbin)
def plotall(tab, a,b):
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
    #ax.set_zlim([-5,5])
    plt.legend()
    return fig
plt.show(plotall(tab,convert2mesh(chipnum=1, resultparams=resparamsa), convert2mesh(chipnum=2, resultparams=resparamsb)))
