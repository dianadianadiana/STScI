import numpy as np
from astropy.table import Table

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
start_time = time.time()

# import functions from DataInfoTab (which deals with getting and filtering the data)
from DataInfoTab import remove_stars_tab, sigmaclip, sigmaclip_starmag, make_avgmagandflux, sigmaclip_delmagall, bin_filter, extract_data_dicts

CHIP1XLEN = CHIP2XLEN = 4096 
CHIP1YLEN = CHIP2YLEN = 2048

##########################################################
########## Reading in Data into Astropy Table ############
##########################################################
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
#tab = tab[82384:82410]
#tab = tab[80000:90000]
starIDarr = np.unique(tab['id'])     # collect all the star IDs
starttablen = len(tab)
startstarnum = len(starIDarr)
timeread = time.time()
#mag = tab['mag']
#magerr = tab['magerr']
#plt.plot(mag, magerr, 'o')
#plt.show()
print len(tab)
tab =  tab[np.where(tab['mag'] < 28)[0]] # only contain the values of less than 28
print len(tab)
print "%s seconds for reading in the data" % (timeread - start_time)
##########################################################
##########################################################
##########################################################

##########################################################
################## Filtering the Data ###################
##########################################################

# Remove any stars that don't have enough observations
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)
# Remove any observations of each star whose magnitudes aren't within a certain sigma
low, high = 3, 3
tab, starIDarr = sigmaclip_starmag(tab, starIDarr, low, high)
# Remove any stars that may have less than the min observations after sigmaclipping
tab, starIDarr, removestarlist = remove_stars_tab(tab, starIDarr, min_num_obs = 4)
# Create the absolute magnitude and errors columns
tab, starIDarr = make_avgmagandflux(tab, starIDarr)
# Remove any observations that have too high of a delta magnitude on a large scale
low, high = 3, 3
tab, starIDarr = sigmaclip_delmagall(tab, starIDarr, low, high)
# Bin the data using the delta magnitude and sigma clip each bin
xpixelarr, ypixelarr = np.arange(CHIP1XLEN), np.arange(CHIP1YLEN + CHIP2YLEN)
xbin = ybin = 10
low, high = 3, 3
tab, zz, zzdelm, zztabindex = bin_filter(tab, xpixelarr, ypixelarr, xbin, ybin, low, high)
timefilter = time.time()
print "%s seconds for filtering the data" % (timefilter - timeread)

##########################################################
##########################################################
##########################################################

  
##########################################################
####################### Fitting ##########################
##########################################################
from sympy import *
def func(x, y, n = 5):
    ''' 
    Purpose
    -------
    The 2D nth order polynomial function we want to optimize
    '''
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
    funclist, f = norder2dpoly(n)
    func2optimize = f(x,y)
    func2optimize[0] = np.ones(len(x)) # because f(x,y)[0] = 1 and not [1,1,...,1,1]
    return func2optimize
    #return [1+x*0, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4, x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]
    #return [x**2, y**2, x*y, x, y, 1+x*0] # 3rd order

def getfit(tab, xpixel1, ypixel1, xpixel2, ypixel2, n = 5):
    '''
    NEED TO ADD DESCRIPTION
    '''
    #x = tab['x']
    #y = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    #z = tab['mag'] - tab['avgmag']
    #x = np.asarray(x)
    #y = np.asarray(y)
    #z = np.asarray(z)
    
    x1 = [row['x'] for row in tab if row['chip'] == chip]
    y1 = [row['y'] + CHIP2YLEN for row in tab if row['chip'] == 1]
    x2 = [row['x'] for row in tab if row['chip'] == 2]
    y2 = [row['y'] for row in tab if row['chip'] == 2]
    z1 = [row['mag'] - row['avgmag'] for row in tab if row['chip'] == 1]
    z2 = [row['mag'] - row['avgmag'] for row in tab if row['chip'] == 2]
    
    x1, x2 = np.asarray(x1), np.asarray(x2)
    y1, y2 = np.asarray(y1), np.asarray(y2)
    z1, z2 = np.asarray(z1), np.asarray(z2)
    
    #xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??
    #f = func(x,y,n)
    #fmesh = func(xx,yy,n)
    #A = np.array(f).T
    #B = z
    #coeff, rsum, rank, s = np.linalg.lstsq(A, B)
    
    xx1, yy1 = np.meshgrid(xpixel1, ypixel1, sparse = True, copy = False)
    xx2, yy2 = np.meshgrid(xpixel2, ypixel2, sparse = True, copy = False)
    f1 = func(x1,y1,n)
    f2 = func(x2,y2,n)
    fmesh1 = func(xx1,yy1,n)
    fmesh2 = func(xx2,yy2,n)
    
    A1 = np.array(f1).T
    B1 = z1
    coeff1, rsum1, rank1, s1 = np.linalg.lstsq(A1, B1)
    A2 = np.array(f2).T
    B2 = z2
    coeff2, rsum2, rank2, s2 = np.linalg.lstsq(A2, B2)
    
    
    #zfit = np.zeros(len(x))
    #zzfit = [[0 for i in xpixel] for j in ypixel]
    #k = 0
    #while k < len(coeff):
    #    zfit += coeff[k]*f[k]
    #    zzfit += coeff[k]*fmesh[k]
    #    k+=1
    ## Examples:
    ## zfit = coeff[0]*x**2 + coeff[1]*y**2 + ... === coeff[0]*f[0] + ...
    ## Zfit = coeff[0]*X**2 + coeff[1]*Y**2 + ... === coeff[0]*fmesh[0] + ...
    
    zfit1 = np.zeros(len(x1))
    zzfit1 = [[0 for i in xpixel1] for j in ypixel1]
    k = 0
    while k < len(coeff1):
        zfit1 += coeff1[k]*f1[k]
        zzfit1 += coeff1[k]*fmesh1[k]
        k+=1
        
    zfit2 = np.zeros(len(x2))
    zzfit2 = [[0 for i in xpixel2] for j in ypixel2]
    k = 0
    while k < len(coeff2):
        zfit2 += coeff2[k]*f2[k]
        zzfit2 += coeff2[k]*fmesh2[k]
        k+=1
    def get_res(z, zfit):
        # returns an array of the error in the fit
        return np.abs(zfit - z)
    #resarr = get_res(z, zfit)
    return [x1, y1, z1, zfit1, xx1, yy1, zzfit1, rsum1, x2, y2, z2, zfit2, xx2, yy2, zzfit2, rsum2]



xpixel1=np.linspace(0,CHIP1XLEN,xbin)
ypixel1=np.linspace(CHIP2YLEN,CHIP1YLEN+CHIP2YLEN,ybin)
xpixel2=np.linspace(0,CHIP2XLEN,xbin)
ypixel2=np.linspace(0,CHIP2YLEN,ybin)
a = getfit(tab, xpixel1, ypixel1, xpixel2, ypixel2, n=5)
x1, y1, z1, zfit1, xx1, yy1, zzfit1, rsum1, x2, y2, z2, zfit2, xx2, yy2, zzfit2, rsum2 = a

def plot3dfit(x1,y1,z1,X1,Y1,Z1, x2,y2,z2,X2,Y2,Z2,title = '', scatter = False):    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X1,Y1,Z1, rstride=1, cstride=1,color='red', label = "CHIP1")
    ax.plot_wireframe(X2,Y2,Z2, rstride=1, cstride=1,color='blue', label = "CHIP2")
    ax.set_title(title)
    if scatter:
        ax.scatter(x1,y1,z1, s= 50, alpha = .05,c='red')
        ax.scatter(x2,y2,z2, s= 50, alpha = .05,c='blue')
    #for i in range(len(z)):
    #    zpoint = z[i]
    #    xpoint, ypoint = x[i],y[i]
    #    if np.abs(zpoint) > 1:
    #        ax.scatter(xpoint,ypoint,zpoint, s= 50) # from paper.py
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    plt.legend()
    #ax.set_zlim([-.1,.1])
    return fig

plt.show(plot3dfit(x1,y1,z1,xx1,yy1,zzfit1, x2,y2,z2,xx2,yy2,zzfit2, scatter=True))
















#x1, y1, z1, zfit1, xx1, yy1, zzfit1, rsum1, resarr1 = getfit(tab, xpix1l1, ypixel1, chipnum = 1, n = 5)
#x2, y2, z2, zfit2, xx2, yy2, zzfit2, rsum2, resarr2 = getfit(tab, xpixel2, ypixel2, chipnum = 2,n = 5)



"""
def getfit(tab, xpixel, ypixel, chipnum, n = 5):
    '''
    NEED TO ADD DESCRIPTION
    '''
    x = tab['x']
    y = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    z = tab['mag'] - tab['avgmag']
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??
    f = func(x,y,n)
    fmesh = func(xx,yy,n)
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
    return [x, y, z, zfit, xx, yy, zzfit, rsum, resarr]



x, y, z, zfit, xx, yy, zzfit, rsum, resarr = getfit(tab, xpixel=np.linspace(0,CHIP1XLEN,xbin), ypixel=np.linspace(0,CHIP1YLEN+CHIP2YLEN,ybin))
timefit = time.time()
print "%s seconds for fitting the data" % (timefit - timefilter)
##########################################################
##########################################################
##########################################################
endtablen = len(tab)
endstarnum = len(starIDarr)
print 'Percent of observations removed: ', (1. - np.double(endtablen)/starttablen) * 100, '%'
print 'Percent of stars removed: ', (1. - np.double(endstarnum)/startstarnum) * 100, '%'
print "%s seconds for total time" % (time.time() - start_time) 
##########################################################
##########################################################
##########################################################


##########################################################
##################### Plotting ###########################
##########################################################

def plot3dfit(x,y,z,X,Y,Z, title = '', scatter = False):    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1)
    ax.set_title(title)
    if scatter:
        ax.scatter(x,y,z, s= 50, alpha = .05)
    #for i in range(len(z)):
    #    zpoint = z[i]
    #    xpoint, ypoint = x[i],y[i]
    #    if np.abs(zpoint) > 1:
    #        ax.scatter(xpoint,ypoint,zpoint, s= 50) # from paper.py
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    #ax.set_zlim([-.1,.1])
    return fig
    
def plotimg(img, title = ''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('X Pixel', fontsize = 18);  ax.set_ylabel('Y Pixel', fontsize = 18)
    extent=(0,4096,0,4096)    
    cax = ax.imshow(np.double(img), cmap = 'gray_r', interpolation='nearest', origin='lower', extent=extent)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    return fig

# 3D plotting
plt.show(plot3dfit(x, y, z, xx, yy, zzfit, title = 'Fit of all delta magnitudes in 3D', scatter = True))  # plotting the fit in 3d
plt.show(plot3dfit(x, y, z, xx, yy, zzfit, title = 'Fit of all delta magnitudes in 3D (no scatter)', scatter = False))  # plotting the fit in 3d
plt.show(plot3dfit(x, y, z, xx, yy, zz.T, title = 'Average delta magnitude in bins in 3D')) # plotting the actual binned values in 3d
# Image plotting
imgbin = plotimg(zz.T, title = 'Average delta magnitude in bins')
plt.show(imgbin)
imgfit = plotimg(zzfit, title = 'Fit of all delta magnitudes')
plt.show(imgfit)

##########################################################
##########################################################
##########################################################
"""