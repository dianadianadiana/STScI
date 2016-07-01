import numpy as np

starIDarr = []
CHIP2YLEN = 2048

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
tab = tab[1000:2000]
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


def get_coeff(chipnum):
    if chipnum == 1: 
        return np.array([1,1,1,1,1,1])
    else:
        return np.array([.000000001,0,0,0,0.000000002,.00000001])

def getdelta(chipnum, xpixel, ypixel):
    if chipnum == 1: ypixel = ypixel + CHIP2YLEN
    coeff = get_coeff(chipnum)
    print xpixel, ypixel, chipnum
    delta = func2fit(xpixel, ypixel) * coeff  
    print delta
    print np.sum(delta)
    return delta

def chisqstar(starrows):
    ''' Worker function '''
    # Input is the rows of the table corresponding to a single star so that we don't need to input a whole table
    starfluxes = starrows['flux']
    starfluxerrs = starrows['fluxerr']
    deltas = [np.sum(getdelta(row['chip'], row['x'], row['y'])) for row in starrows]
    print starfluxes
    print starfluxerrs
    print deltas
    avgf = np.mean(starfluxes/deltas)
    print avgf
    print starfluxes/deltas
    print starfluxes/deltas - avgf
    print starfluxerrs/deltas
    print np.sum(((starfluxes/deltas - avgf)/(starfluxerrs/deltas))**2)
    chisq = np.sum([((flux/delta - avgf) / (fluxerr/delta))**2 for flux, delta, fluxerr in zip(starfluxes, deltas, starfluxerrs)])
    return chisq
    
    
    
    
def chisqall(tab):
    for star in starIDarr:
        starrows = tab[np.where(tab['id'] == star)[0]]
        chisqstar = chisqstar(starrows)
    return

