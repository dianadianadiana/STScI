import numpy as np
from lmfit import minimize, Parameters, report_fit

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

#fitfuncread = func2read * coeff
#print fitfuncread
#fitfunc = fitfuncread.tolist()
#x = Symbol('x')
#y = Symbol('y')
#fitfunc = lambdify((x,y), fitfunc)
#print fitfunc(2,4)

###########################################
###########################################
###########################################

from astropy.table import Table, Column
from DataInfoTab import remove_stars_tab, convertmag2flux, convertflux2mag, make_avgmagandflux
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
tab = tab[10000:10100]
starIDarr = np.unique(tab['id'])     # collect all the star IDs

tab =  tab[np.where((tab['mag'] <= 25) & (tab['mag'] >= 13))[0]] # (13,25)
print 'Len of tab after constraining the magnitudes'
print len(tab)
print 'Len of tab and starIDarr after 1st min_num_obs'
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)
print len(tab)
print len(starIDarr) 

tab, starIDarr = make_avgmagandflux(tab, starIDarr)
tab =  tab[np.where(tab['flux']/tab['fluxerr'] > 5)[0]] # S/N ratio for flux is greater than 5
func2read, func2fit = norder2dpoly(2)

print func2read

###########################################
###########################################
###########################################      

paramsa = Parameters()
paramsb = Parameters()

#          (Name, Value, Vary, Min, Max, Expr)
paramsa.add_many(
           ('a0', 1e-20, True, None, None, None),
           ('ax',   0,  True,  None,  None,  None),
           ('ay',   0,  True,  None, None,  None),
           ('ax2',   0,  True, None, None,  None),
           ('axy',   0,  True,  None,  None,  None),
           ('ay2',  0, True, None, None, None))
           
paramsb.add_many(
           ('b0', 1e-20, True, None, None, None),
           ('bx',   0,  True,  None,  None,  None),
           ('by',   0,  True,  None, None,  None),
           ('bx2',   0,  True, None, None,  None),
           ('bxy',   0,  True,  None,  None,  None),
           ('by2',  0, True, None, None, None))


def get_coeff(chipnum):
    if chipnum == 1: 
        dicta = paramsa.valuesdict() # OrderedDict
        return np.asarray(dicta.values())
    else:
        dictb = paramsb.valuesdict() # OrderedDict
        return np.asarray(dictb.values())

def getdelta(chipnum, xpixel, ypixel):
    #if chipnum == 1: ypixel = ypixel + CHIP2YLEN
    coeff = get_coeff(chipnum)
    #print xpixel, ypixel, chipnum
    delta = func2fit(xpixel, ypixel) * coeff  
    #print delta
    #print np.sum(delta)
    return np.sum(delta)

#def chisqstar(starrows):
#    ''' Worker function '''
#    # Input is the rows of the table corresponding to a single star so that we don't need to input a whole table
#    starfluxes = starrows['flux']
#    starfluxerrs = starrows['fluxerr']
#    deltas = [getdelta(row['chip'], row['x'], row['y']) for row in starrows]
#    print starrows
#    print "STAR FLUXES", starfluxes
#    print starfluxerrs
#    print "DELTAS", deltas
#    avgf = np.mean(starfluxes/deltas)
#    print avgf
#    print starfluxes/deltas
#    print starfluxes/deltas - avgf
#    print starfluxerrs/deltas
#    print (starfluxes/deltas - avgf)/(starfluxerrs/deltas)
#    chisq = np.sum(((starfluxes/deltas - avgf)/(starfluxerrs/deltas)))#**2)
#    return chisq
#   
#def chisqall1(params, x, tab):
#    starIDarr = np.unique(tab['id'])
#    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
#    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
#    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
#    #totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
#    totalsum = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
#    return totalsum
#    

#minsum0 = chisqall(tab)
#params = paramsa +paramsb
#
#result = minimize(chisqall, params, args = tab)

def chisqall(params, x, data):
    chip1rows = tab[np.where(tab['chip'] == 1)[0]]  
    chip2rows = tab[np.where(tab['chip'] == 2)[0]]  
    
    # go through first chip
    #a0 = params['a0'].value
    #ax = params['ax'].value
    #ay = params['ay'].value
    #ax2 = params['ax2'].value
    #axy = params['axy'].value
    #ay2 = params['ay2'].value
    #b0 = params['b0'].value
    #bx = params['bx'].value
    #by = params['by'].value
    #bx2 = params['bx2'].value
    #bxy = params['bxy'].value
    #by2 = params['by2'].value
    
    def get_coeff(chipnum):
        if chipnum == 1: 
            aa = [params[key] for key in params if key[0]=='a']
            paraa = Parameters()
            for param in aa:
                paraa.add(param)
            dicta = paraa.valuesdict() # OrderedDict
            #return np.array([a0,ax,ay,ax2,axy,ay2])
            return np.asarray(dicta.values())
        else:
            bb = [params[key] for key in params if key[0]=='b']
            parbb = Parameters()
            for param in bb:
                parbb.add(param)
            dictb = parbb.valuesdict() # OrderedDict
            #return np.array([b0,bx,by,bx2,bxy,by2])
            return np.asarray(dictb.values())
        
    def getfit(chipnum, x, y, pointvalue = True):
        funcvalues = func2fit(x,y)
        if not pointvalue:
            funcvalues[0] = np.ones(len(x))
        if chipnum == 1:
            #return np.sum(funcvalues * np.array([a0,ax,ay,ax2,axy,ay2]))
            return np.sum(funcvalues * get_coeff(chipnum))
        else:
            #return np.sum(funcvalues * np.array([b0,bx,by,bx2,bxy,by2]))
            return np.sum(funcvalues * get_coeff(chipnum))
            
    resid1 = np.array([])
    currchip = 1
    for row in chip1rows:
        currstar = row['id']
        currstarrows = tab[np.where(tab['id'] == currstar)[0]]
        currx = row['x']
        curry = row['y']
        currf = row['flux']
        currferr = row['fluxerr']
        currfit = getfit(currchip, currx, curry)
        #currfit = a0 + ax*currx + ay*curry + ax2*currx**2 + axy*currx*curry + ay2*curry**2
        currstarfluxes = currstarrows['flux']
        fits = [getfit(row['chip'], row['x'], row['y']) for row in currstarrows]
        curravgf = np.mean(currstarfluxes/fits)
        #print curravgf
        resid1 = np.append(resid1, currf/currferr - currfit/currferr * curravgf)
        
        
    resid2 = np.array([])
    currchip = 2
    for row in chip2rows:
        currstar = row['id']
        currstarrows = tab[np.where(tab['id'] == currstar)[0]]
        currx = row['x']
        curry = row['y']
        currf = row['flux']
        currferr = row['fluxerr']
        currfit = getfit(currchip, currx, curry)
        currstarfluxes = currstarrows['flux']
        fits = [getfit(row['chip'], row['x'], row['y']) for row in currstarrows]
        curravgf = np.mean(currstarfluxes/fits)
        #print curravgf
        
        resid2 = np.append(resid2, currf/currferr - currfit/currferr * curravgf)
    return np.concatenate((resid1, resid2))


params = paramsa + paramsb
#resid = chisqall(tab, params)
x = []
data = []
result = minimize(chisqall, params, args=( x, tab))
report_fit(result.params)
#result1 = minimize(chisqall1, params, args = (x,tab))
#report_fit(result1.params)
#
#paramsa.pretty_print()
#paramsb.pretty_print()
 #a0:    2.1660e-20 +/- 9.42e-16 (4350222.49%) (init= 1e-20)
 #   ax:   -2.7935e-24 +/- 1.22e-19 (4351896.16%) (init= 0)
 #   ay:    3.7290e-24 +/- 1.62e-19 (4349785.31%) (init= 0)
 #   ax2:   4.5335e-28 +/- 1.97e-23 (4351618.47%) (init= 0)
 #   axy:  -1.0902e-27 +/- 4.74e-23 (4349845.64%) (init= 0)
 #   ay2:   2.7740e-28 +/- 1.21e-23 (4350150.96%) (init= 0)
 #   b0:    2.2493e-20 +/- 9.78e-16 (4349554.53%) (init= 1e-20)
 #   bx:   -2.9065e-24 +/- 1.26e-19 (4348903.84%) (init= 0)
 #   by:    1.5195e-24 +/- 6.61e-20 (4349535.37%) (init= 0)
 #   bx2:   4.5022e-28 +/- 1.96e-23 (4348961.11%) (init= 0)
 #   bxy:  -4.0105e-28 +/- 1.74e-23 (4349723.74%) (init= 0)
 #   by2:  -1.0628e-28 +/- 4.62e-24 (4348615.09%) (init= 0)

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
    
    report_fit(result.params)





from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def convert2mesh(chipnum, xbin = 10, ybin = 5):
    if chipnum == 2:
        xpixel = np.linspace(0, CHIP2XLEN, xbin)
        ypixel = np.linspace(0, CHIP2YLEN, ybin)
    else: # chipnum = 1
        xpixel = np.linspace(0, CHIP1XLEN, xbin)
        ypixel = np.linspace(CHIP2YLEN, CHIP1YLEN + CHIP2YLEN, ybin)
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??
    fmesh = func2fit(xx,yy)
    coeff = get_coeff(chipnum)
    if chipnum ==1:
        coeff = np.array(paramsa.valuesdict().values())
    zzfit = [[0 for i in xpixel] for j in ypixel]
    k = 0
    while k < len(coeff):
        zzfit += coeff[k]*fmesh[k]
        k+=1
    return [xx, yy, zzfit]


def plotthis( a,b ):
    X1,Y1,Z1 = a
    X2,Y2,Z2 = b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1,Y1,Z1, rstride=1, cstride=1,color='red', label = "CHIP1")
    ax.plot_wireframe(X2,Y2,Z2, rstride=1, cstride=1,color='blue', label = "CHIP2")
    return fig
plt.show(plotthis(convert2mesh(chipnum=1), convert2mesh(chipnum=2)))

def plotdelflux(tab):
    x = tab['x']
    y = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    delflux = tab['flux'] - tab['avgflux']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,delflux)
    plt.show()
#plotdelflux(tab)

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
    ax.scatter(x,y,delflux)
    ax.set_zlim([-1e-10,1e-10])
    return fig
plt.show(plotall(tab,convert2mesh(chipnum=1), convert2mesh(chipnum=2)))
