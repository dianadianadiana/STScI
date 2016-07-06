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
#def chisqall(tab):
#    starIDarr = np.unique(tab['id'])
#    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
#    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
#    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
#    #totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
#    totalsum = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
#    return totalsum
    

#minsum0 = chisqall(tab)
#params = paramsa +paramsb
#
#result = minimize(chisqall, params, args = tab)

def chisqall(params,x,data):
    chip1rows = tab[np.where(tab['chip'] == 1)[0]]  
    chip2rows = tab[np.where(tab['chip'] == 2)[0]]  
    
    resid1 = np.array([])
    # go through first chip
    a0 = params['a0'].value
    ax = params['ax'].value
    ay = params['ay'].value
    ax2 = params['ax2'].value
    axy = params['axy'].value
    ay2 = params['ay2'].value
    b0 = params['b0'].value
    bx = params['bx'].value
    by = params['by'].value
    bx2 = params['bx2'].value
    bxy = params['bxy'].value
    by2 = params['by2'].value
    
    def getfit(chipnum, x, y, pointvalue = True):
        funcvalues = func2fit(x,y)
        if not pointvalue:
            funcvalues[0] = np.ones(len(x))
        if chipnum == 1:

            a0 = params['a0'].value
            ax = params['ax'].value
            ay = params['ay'].value
            ax2 = params['ax2'].value
            axy = params['axy'].value
            ay2 = params['ay2'].value
            return np.sum(funcvalues * np.array([a0,ax,ay,ax2,axy,ay2]))
        else:
            b0 = params['b0'].value
            bx = params['bx'].value
            by = params['by'].value
            bx2 = params['bx2'].value
            bxy = params['bxy'].value
            by2 = params['by2'].value
            return np.sum(funcvalues * np.array([b0,bx,by,bx2,bxy,by2]))
            
    x = np.asarray(chip1rows['x'])
    y = np.asarray(chip1rows['y'])

    
    ##funcvalues = func2fit(x1,y1)
    ##funcvalues[0] = np.ones(len(x1))
    ##zfit = np.zeros(len(x1))
    ##coeff = paramsa.valuesdict().values() # OrderedDict
    ##k = 0
    ##while k < len(coeff):
    ##    zfit += coeff[k]*funcvalues[k]
    ##    k+=1
    ##print zfit
    #zfit = a0 + ax*x + ay*y + ax2*x**2 + axy*x*y + ay2*y**2
    #print zfit - chip1rows['flux']
    #resid1 = np.asarray((zfit-chip1rows['flux'])/ chip1rows['fluxerr'])
        
    for row in chip1rows:

        
        
        currchip = 1
        currstar = row['id']
        print currstar
        currstarrows = tab[np.where(tab['id'] == currstar)[0]]
        #print currstarrows
        currx = row['x']
        curry = row['y']
        currf = row['flux']
        currferr = row['fluxerr']
        currfit = getfit(1,currx, curry)
        print 'currfit2', currfit
        #currfit = a0 + ax*currx + ay*curry + ax2*currx**2 + axy*currx*curry + ay2*curry**2
        #print 'currfit1', currfit
        currstarfluxes = currstarrows['flux']
        #currstarfluxerrs = currstarrows['fluxerr']
        fits = np.array([])
        for row in currstarrows:
            x = row['x']
            y = row['y']
            chip = row['chip']
            if chip == 1:
                fits = np.append(fits, a0 + ax*x + ay*y + ax2*x**2 + axy*x*y + ay2*y**2)
            else: 
                fits = np.append(fits, b0 + bx*x + by*y + bx2*x**2 + bxy*x*y + by2*y**2)
        print 'fits1', fits
        #fits = [getfit(row['chip'], row['x'], row['y']) for row in currstarrows]
        print 'fits2', fits
        curravgf = np.mean(currstarfluxes/fits)
        #print curravgf
        
        resid1 = np.append(resid1, currf/currferr - currfit/currferr * curravgf)
    print resid1
    return resid1


params = paramsa +paramsb
#resid = chisqall(tab, params)





x = []
data = []
result = minimize(chisqall, params, args=(x,tab))
report_fit(result.params)


# with getfit function
#a0:    1.0000e-20 +/- 0        (0.00%) (init= 1e-20)
#    ax:   -1.3629e-31 +/- 0        (0.00%) (init= 0)
#    ay:   -5.8734e-31 +/- 0        (0.00%) (init= 0)
#    ax2:  -3.8418e-35 +/- 0        (0.00%) (init= 0)
#    axy:  -1.6748e-34 +/- 0        (0.00%) (init= 0)
#    ay2:  -2.2187e-34 +/- 0        (0.00%) (init= 0)
#    b0:    1.0000e-20 +/- 0        (0.00%) (init= 1e-20)
#[  22.33172442   25.30143561   23.00447989   23.55946837   25.03262031
#  271.5         105.55552272  105.61109276    1.29825448   -1.23431281
#    0.57424653   -0.69375945   -1.31360049    8.72517072   -4.22236936
#   -3.92909136   21.22957337   20.39516658   18.1071204    17.98648034
#   20.52419848   19.81163996   19.68130115]

# manually using the params
 #a0:    1.0000e-20 +/- 0        (0.00%) (init= 1e-20)
 #   ax:   -3.8065e-23 +/- 0        (0.00%) (init= 0)
 #   ay:    5.5382e-24 +/- 0        (0.00%) (init= 0)
 #   ax2:  -2.6771e-23 +/- 0        (0.00%) (init= 0)
 #   axy:   5.4831e-24 +/- 0        (0.00%) (init= 0)
 #   ay2:   1.0715e-25 +/- 0        (0.00%) (init= 0)
 #   b0:    1.0000e-20 +/- 0        (0.00%) (init= 1e-20)
 #resid1 = [-30.17095618  21.24620304  14.98954732 -23.41344972 -25.15861616
 #-23.66025465  24.26660409 -50.62007158  -9.8379675    8.24489516
 # -6.670859     5.08113379 -17.89285708  27.02210008 -29.99618636
 #  8.0202657  -23.02615345  -6.88173449 -28.31150748 -28.51337233
 # 21.59236261  -3.19822059]



paramsa.pretty_print()
paramsb.pretty_print()

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
