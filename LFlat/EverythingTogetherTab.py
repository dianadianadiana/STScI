import numpy as np
from astropy.table import Table

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
start_time = time.time()

# import functions from DataInfoTab (which deals with getting and filtering the data)
from DataInfoTab import remove_stars_tab, sigmaclip, sigmaclip_starmag, make_absmag, sigmaclip_delmagall, bin_filter, extract_data_dicts

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
tab, starIDarr = make_absmag(tab, starIDarr)
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
def func(x,y):
    ''' 
    Purpose
    -------
    The 2D function we want to optimize 
    '''
    return [x**2, y**2, x*y, x, y, 1+x*0]
    
def getfit(tab, xpixel, ypixel):
    '''
    NEED TO ADD DESCRIPTION
    '''
    x = tab['x']
    y = [row['y'] if row['chip'] == 2 else row['y'] + CHIP2YLEN for row in tab]
    z = tab['mag'] - tab['absmag']
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy = False) #why copy false??
    f = func(x,y)
    fmesh = func(xx,yy)
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
        ax.scatter(x,y,z, s= 50)
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