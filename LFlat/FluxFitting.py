import numpy as np

CHIP1XLEN = CHIP2XLEN = 4096 
CHIP1YLEN = CHIP2YLEN = 2048

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
tab = tab[10000:11000]
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


class Coefficient(dict):
    def __init__(self, **kwargs):
        #for k in kwargs.keys():
        #    #if k in [acceptable_keys_list]:
        #    self.__setattr__(k, kwargs[k])
        dict.__init__(self, **kwargs)
        self.__dict__ = self
    #def __setattr__(self, name, value):
    #    print (name, value)
    #    self.__dict__[name] = value
    #def __getitem__(self, key):
    #    return self.data[key]
    #    #setattr(self, name, value)
a = Coefficient(a0 = 1e-20, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0)
b = Coefficient(b0 = 1e-20, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0)       



def get_coeff(chipnum):
    c = np.array([])
    if chipnum == 1: 
        coeff = a
    else:
        coeff = b
    for key in np.sort(coeff.keys()):
        c = np.append(c, coeff[key])
    return c

def getdelta(chipnum, xpixel, ypixel):
    if chipnum == 1: ypixel = ypixel + CHIP2YLEN
    coeff = get_coeff(chipnum)
    #print xpixel, ypixel, chipnum
    delta = func2fit(xpixel, ypixel) * coeff  
    #print delta
    #print np.sum(delta)
    return delta

def chisqstar(starrows):
    ''' Worker function '''
    # Input is the rows of the table corresponding to a single star so that we don't need to input a whole table
    starfluxes = starrows['flux']
    starfluxerrs = starrows['fluxerr']
    deltas = [np.sum(getdelta(row['chip'], row['x'], row['y'])) for row in starrows]
    #print starfluxes
    #print starfluxerrs
    #print deltas
    avgf = np.mean(starfluxes/deltas)
    #print avgf
    #print starfluxes/deltas
    #print starfluxes/deltas - avgf
    #print starfluxerrs/deltas
    chisq = np.sum(((starfluxes/deltas - avgf)/(starfluxerrs/deltas))**2)
    return chisq
   
def chisqall(tab):
    starIDarr = np.unique(tab['id'])
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
    return totalsum
    

minsum0 = chisqall(tab)

aa = np.linspace(-10e-20, 10e-20, 21)
aa = np.append(np.logspace(-1,-30,10)*-1, np.logspace(-30,-1,10))
bb = np.linspace(-1e-50, 1e-50, 101)
bb = np.append(np.logspace(-1,-50,10)*-1, np.logspace(-50,-1,10))

#c = np.linspace(-1e-13, 1e-13, 5)
#d = np.linspace(-1e-13, 1e-13, 5)
#e = np.linspace(-1e-13, 1e-13, 5)

# in order to first determine the a0 or b0 coefficient since it has more weight
#origco = b.b0
#minco = b.b0
#minsum = chisqall(tab)
#for change in aa:
#    b.b0 = origco + change
#    currsum = chisqall(tab)
#    if currsum < minsum:
#        print 'True******'
#        print minco
#        minsum = currsum
#        minco = b.b0
#b.b0 = minco
#print b

def determine1stcoeff(coeffdict, varyingarr, tab):
    firstkey = np.sort(coeffdict.keys())[0]
    origco = coeffdict[firstkey]
    minco = coeffdict[firstkey]
    minsum = chisqall(tab)
    for change in varyingarr:
        coeffdict[firstkey] = origco + change
        currsum = chisqall(tab)
        if currsum < minsum:
            print 'True******'
            print minco
            minsum = currsum
            minco = coeffdict[firstkey]
    coeffdict[firstkey] = minco
    print minsum
    return coeffdict
a = determine1stcoeff(a, aa, tab)
b = determine1stcoeff(b, aa, tab)

def determinerestcoeff(coeffdict, varyingarr, tab):
    # co = coefficient 
    for co in np.sort(coeffdict.keys())[1:]:
        origco = coeffdict[co]
        minco = coeffdict[co]
        minsum = chisqall(tab)
        for change in varyingarr:
            coeffdict[co] = origco + change
            currsum = chisqall(tab)
            if currsum < minsum:
                print 'True**************'
                print origco, change
                print co, coeffdict[co]
                minsum = currsum
                minco = coeffdict[co]
        coeffdict[co] = minco
    print minsum
    return coeffdict
a = determinerestcoeff(a, bb, tab)
b = determinerestcoeff(b, bb, tab)

print a,b
'''

# co = coefficient 
for co in np.sort(b.keys())[1:]:
    origco = b[co]
    minco = b[co]
    minsum = chisqall(tab)
    for change in bb:
        b[co] = origco + change
        currsum = chisqall(tab)
        if currsum < minsum:
            print 'True**************'
            print co, b[co]
            minsum = currsum
            minco = b[co]
    b[co] = minco
    
print minsum0, minsum
print b



'''


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
plotdelflux(tab)

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