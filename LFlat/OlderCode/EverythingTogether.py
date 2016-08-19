import numpy as np
from astropy.table import Table

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import functions from DataInfo (which deals with getting the data)
from DataInfo import * # extract_data, remove_stars, sigmaclip_dict, extract_out, printnicely

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

# Build up the indicies for each star in the table and store in a dictionary
#     keys: star ID 
#     values: indicies of the rows in tab corresponding to star ID
# Ex: tab[starindexdict[starID] gives the rows corresponding to the star ID
starindexdict = {}
for row in tab[:]:
    try:
        starindexdict[row['id']]
    except KeyError:
        starindexdict[row['id']] = np.where(tab['id'] == row['id'])[0]
##########################################################
##########################################################
##########################################################



##########################################################
################## Organizing the Data ###################
############# into Dictionaries and Arrays ###############
##########################################################
# In this module, we should have by the end:
# stardict                 -- holds the star IDs, 
# xdict, ydict             -- holds the x and y pixel values for a star
# mdict, merrdict          -- holds the magnitudes and associated magnitude errors 
# mabsdict, mabserrdict    -- holds the absolute value for the magnitude and its error
# Note: all dictionaries have array elements except for mabsdict and mabserrdict,
#   which are just floats
# As well as arrays for these values

# Get all the data nicely from the file/table into dictionaries
stardict, xdict, ydict, mdict, merrdict = extract_data(starindexdict, tab)   
# Filter the stars that don't have enough observations 
[stardict, xdict, ydict, mdict, merrdict], removestarlist = remove_stars([stardict, xdict, ydict, mdict, merrdict], 3)
# Look at each star and make sure that observations would not mess up a fit
#   by sigma clipping them
low, high = 3, 3
stardict, xdict, ydict, mdict, merrdict = sigmaclip_dict(stardict, xdict, ydict, mdict, merrdict, low, high)
# Create dictionaries for the absolute magnitude of each star and the error for
#   that absolute magnitude
# Absolute magnitude will just be the mean of all the magnitudes for that star
# The error is the the quadratic error ex. (e1^2 + e2^2 + .. + eN^2)^(1/2) / N
mabsdict = {}
mabserrdict = {}
for star in stardict:
    mabsdict[star] = np.mean(mdict[star])
    mabserrdict[star] = np.sqrt(np.sum(merrdict[star]**2)) / len(merrdict[star])
 
xall, yall, delmall, delmerrall, mall, merrall, mabsall, mabserrall = extract_out(stardict, xdict, ydict, mdict, merrdict, mabsdict, mabserrdict)

##########################################################
##########################################################
##########################################################

xpixel, ypixel = np.arange(CHIP1XLEN), np.arange(CHIP1YLEN + CHIP2YLEN)
xbin, ybin = 10, 10    

##########################################################
##########################################################
def func(x,y):
    ''' 
    Purpose
    -------
    The 2D function we want to optimize 
    '''
    return [x**2, y**2, x*y, x, y, 1+x*0]
    
def getfit(x, y, z, xpixel, ypixel):
    """
    copied from polyfit2d.py
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # sigma clipping: to remove anything that is above/below a certain sigma
    z, remove_arr = sigmaclip(z, low = 3, high = 3)
    x = np.delete(x, remove_arr)
    y = np.delete(y, remove_arr)
    
    xx, yy = np.meshgrid(xpixel, ypixel, sparse = True, copy=False) #why copy false??
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
    
    return [x, y, z, zfit, xx, yy, zzfit, rsum, resarr, remove_arr]
    


def dobin(xpixelarr, ypixelarr, xbin, ybin, stardict, xall, yall, delmall):
    # Initialize an empty 2D array for the binning;
    # Create xbinarr and ybinarr as the (lengths of xbin and ybin, respectively);
    #     to make up the bins
    # Find dx and dy to help later with binning x+dx and y+dy
    # zz is a 2D array that can be used for imshow
    zz = np.array([np.array([None for i in range(np.int(xbin))]) for j in range(np.int(ybin))])
    xbin, ybin = np.double(xbin), np.double(ybin)
    xbinarr = np.linspace(np.min(xpixelarr), np.max(xpixelarr), xbin, endpoint = False)
    ybinarr = np.linspace(np.min(ypixelarr), np.max(ypixelarr), ybin, endpoint = False)
    dx, dy = xbinarr[1] - xbinarr[0], ybinarr[1] - ybinarr[0]
    
    for i, x in enumerate(xbinarr):
        for j, y in enumerate(ybinarr):
            inbin = np.where((xall >= x) & (xall < x + dx) & (yall >= y) & (yall < y + dy))[0]
            if len(inbin): # if inbin exists
                if zz[i][j] == None: # gives a FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
                    zz[i][j] = delmall[inbin]
                else:
                    zz[i][j] = np.append(zz[i][j], delmall[inbin])
    
    # Now deal with zz and take the averages in each bin         
    zzorig = np.copy(zz)
    zzfinal = np.copy(zz)
    for i in range(len(xbinarr)):
        for j in range(len(ybinarr)):
            # if at zz[i][j] there is None, make it 0 otherwise take the average
            zzfinal[i][j] = 0 if zz[i][j] == None else np.mean(zz[i][j])
            # gives a futurewarning : comparison to `None` will result in an elementwise object comparison in the future.
    return [zzorig, zzfinal]
    
def plot3dfit(x,y,z,X,Y,Z):    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1)
    #ax.scatter(x,y,z, s= 50) # from paper.py
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    #ax.set_zlim([-.1,.1])
    return fig

x, y, z, zfit, xx, yy, zzfit, rsum, resarr, remove_arr = getfit(xall, yall, delmall, xpixel=np.linspace(0,CHIP1XLEN,xbin), ypixel=np.linspace(0,CHIP1YLEN+CHIP2YLEN,ybin))
plt.show(plot3dfit(x, y, z, xx, yy, zzfit))  # plotting the fit in 3d
zzorig, zzfinal = dobin(xpixel, ypixel, xbin, ybin, stardict, xall, yall, delmall)
plt.show(plot3dfit(x, y, z, xx, yy, zzfinal.T)) # plotting the actual binned values in 3d

def plotimg(img,xbin=10,ybin=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X Pixel', fontsize = 18);  ax.set_ylabel('Y Pixel', fontsize = 18)
    extent=(0,4096,0,4096)    
    cax = ax.imshow(np.double(img), cmap = 'gray_r', interpolation='nearest', origin='lower', extent=extent)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    return fig
    
imgorig = plotimg(zzfinal.T)
plt.show(imgorig)
imgfit = plotimg(zzfit)
plt.show(imgfit)
