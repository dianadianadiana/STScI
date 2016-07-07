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
tab = tab[10000:10500]
starIDarr = np.unique(tab['id'])                 # collect all the star IDs

#datafil = 'one_noise.txt'
#data = np.genfromtxt(path+datafil)
#names = ['id', 'chip', 'x', 'y', 'mag', 'magerr']
#types = [int, int, np.float64, np.float64, np.float64, np.float64]
#
#tab = Table(data, names = names, dtype = types)
#tab.remove_row(0)
#starIDarr = np.unique(tab['id'])     # collect all the star IDs
#print tab


tab =  tab[np.where((tab['mag'] <= 25) & (tab['mag'] >= 13))[0]] # (13,25)
print 'Len of tab after constraining the magnitudes'
print len(tab)
print 'Len of tab and starIDarr after 1st min_num_obs'
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)
print len(tab)
print len(starIDarr) 

tab, starIDarr = make_avgmagandflux(tab, starIDarr)
tab, starIDarr = sigmaclip_delmagdelflux(tab, starIDarr, flux = True, mag = False, low = 3, high = 3)

print tab
tab =  tab[np.where(tab['flux']/tab['fluxerr'] > 5)[0]] # S/N ratio for flux is greater than 5
func2read, func2fit = norder2dpoly(2)

print func2read
func2string = np.copy(func2read)
k = 0
while k < len(func2string):
    func2string[k] = str(func2string[k])
    func2string[k] = func2string[k].replace('*','')
    k+=1


###########################################
###########################################
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
# To make it more versatile... use func2string
paramsa.add('a' + func2string[0], 1e-8, True, None, None, None)
paramsb.add('b' + func2string[0], 1e-8, True, None, None, None)

for elem in func2string[1:]:
    paramsa.add('a' + elem, 0, True, None, None, None)
    paramsb.add('b' + elem, 0, True, None, None, None)


paramsa.pretty_print()
paramsb.pretty_print()


def chisqstar(starrows, params):
    ''' Worker function '''
    #nonlocal chip2fit
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
    deltas = [getfitvalue(row['chip'], row['x'], row['y']) for row in starrows]
    #print starrows
    #print "STAR FLUXES", starfluxes
    #print starfluxerrs
    #print "DELTAS", deltas
    avgf = np.mean(starfluxes/deltas)
    #print avgf
    #print starfluxes/deltas
    #print starfluxes/deltas - avgf
    #print starfluxerrs/deltas
    #print (starfluxes/deltas - avgf)/(starfluxerrs/deltas)
    starresid = (starfluxes/deltas - avgf)/(starfluxerrs/deltas)#**2)
    return np.asarray(starresid).tolist()
   
def chisqall1(params, func2fit, tab, chip2fit):
    starIDarr = np.unique(tab['id'])
    # np.where(tab['id'] == star)[0]                -- the indexes in tab where a star is located
    # tab[np.where(tab['id'] == star)[0]]           -- "starrows" = the rows of tab for a certain star
    # chisqstar(tab[np.where(tab['id'] == star)[0]])-- the chi squared for just one star
    #totalsum = np.sum([chisqstar(tab[np.where(tab['id'] == star)[0]]) for star in starIDarr])
    totalresid = np.asarray([chisqstar(tab[np.where(tab['id'] == star)[0]], params) for star in starIDarr])
    return reduce(lambda x,y: x+y,totalresid) # flatten totalresid
    


#chip2fit = 1
#resulta = minimize(chisqall1, paramsa, args=(func2fit, tab, chip2fit))
#resparamsa = resulta.params
#report_fit(resparamsa)
#chip2fit = 2
#resultb = minimize(chisqall1, paramsb, args=(func2fit, tab, chip2fit))
#resparamsb = resultb.params
#report_fit(resparamsb)

#[[Variables]]
    #a1:    1.0004e-08 +/- 0.000359 (3596011.28%) (init= 1e-08)
    #ax:    7.2187e-19 +/- 3.26e-14 (4509327.58%) (init= 0)
    #ay:   -5.2086e-21 +/- 9.07e-16 (17414585.48%) (init= 0)
    #ax2:   3.2464e-19 +/- 1.17e-14 (3597333.77%) (init= 0)
    #axy:  -5.0623e-21 +/- 1.83e-16 (3605713.12%) (init= 0)
    #ay2:   2.6085e-23 +/- 1.88e-18 (7188328.27%) (init= 0)
    #b1:    1.0004e-08 +/- 0.000159 (1598519.08%) (init= 1e-08)
    #bx:    7.6629e-19 +/- 3.84e-14 (5011457.83%) (init= 0)
    #by:   -5.1308e-21 +/- 7.77e-16 (15145977.17%) (init= 0)
    #bx2:   3.4463e-19 +/- 5.51e-15 (1599903.31%) (init= 0)
    #bxy:   3.2341e-21 +/- 5.22e-17 (1612960.81%) (init= 0)
    #by2:   6.2002e-22 +/- 1.07e-17 (1728280.46%) (init= 0)
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
        #print '***'
        #print curravgf
        #print currf, currferr
        #print currf/currferr
        #print currf/currferr - currfit/currferr * curravgf
        resid = np.append(resid, currf/currferr - currfit/currferr * curravgf)
    return resid
        
#chip2fit = 1
#resulta = minimize(chisqall, paramsa, args=(func2fit, tab, chip2fit))
#resparamsa = resulta.params
#report_fit(resparamsa)
#chip2fit = 2
#resultb = minimize(chisqall, paramsb, args=(func2fit, tab, chip2fit))
#resparamsb = resultb.params
#report_fit(resparamsb)


# Take the results parameters and break them up for the different chips
#paramlista = [resultparams[key] for key in resultparams if key[0] == 'a'] # list of Parameter objects
#resparamsa = Parameters()
#for param in paramlista:
#    resparamsa.add(param)
#paramlistb = [resultparams[key] for key in resultparams if key[0] == 'b']
#resparamsb = Parameters()
#for param in paramlistb:
#    resparamsb.add(param)
    
#result1 = minimize(chisqall1, params, args = (func2fit,tab))
#report_fit(result1.params)

#[[Variables]]
    #a1:    9.9981e-09 +/- 0.000343 (3425714.34%) (init= 1e-08)
    #ax:    9.7294e-19 +/- 4.25e-14 (4367911.56%) (init= 0)
    #ay:   -7.0347e-21 +/- 5.42e-16 (7705522.59%) (init= 0)
    #ax2:   4.3754e-19 +/- 1.50e-14 (3427206.49%) (init= 0)
    #axy:  -6.8228e-21 +/- 2.34e-16 (3432324.49%) (init= 0)
    #ay2:   3.5126e-23 +/- 1.60e-18 (4568830.68%) (init= 0)
    #b1:    9.9980e-09 +/- 0.000154 (1537755.05%) (init= 1e-08)
    #bx:    1.0419e-18 +/- 5.30e-14 (5084391.73%) (init= 0)
    #by:   -6.9845e-21 +/- 6.57e-16 (9413613.65%) (init= 0)
    #bx2:   4.6851e-19 +/- 7.21e-15 (1539138.84%) (init= 0)
    #bxy:   4.3965e-21 +/- 6.81e-17 (1548741.05%) (init= 0)
    #by2:   8.4287e-22 +/- 1.36e-17 (1615744.16%) (init= 0)

#paramsa.pretty_print()
#paramsb.pretty_print()



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
    print result.residual
    resultparams = result.params
    print resultparams
    report_fit(result.params)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def convert2mesh(chipnum, resultparams, xbin = 10, ybin = 5):
    if chipnum == 2:
        xpixel = np.linspace(0, CHIP2XLEN, xbin)
        ypixel = np.linspace(0, CHIP2YLEN, ybin)
    else: # chipnum = 1
        xpixel = np.linspace(0, CHIP1XLEN, xbin)
        ypixel = np.linspace(0, CHIP2YLEN, ybin) 
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??

    fmesh = func2fit(xx,yy)
    if chipnum == 1:
        yy += CHIP2YLEN
    
    resultdict = resultparams.valuesdict()
    coeff = np.asarray(resultdict.values())
    zzfit = [[0 for i in xpixel] for j in ypixel]
    k = 0
    while k < len(coeff):
        zzfit += coeff[k]*fmesh[k]
        k+=1
    return [xx, yy, zzfit]

def plotthis(a,b):
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
    #ax.set_zlim([-1e-10,1e-10])
    return fig
plt.show(plotall(tab,convert2mesh(chipnum=1, resultparams=resparamsa), convert2mesh(chipnum=2, resultparams=resparamsb)))
