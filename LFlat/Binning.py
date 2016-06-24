import numpy as np
import matplotlib.pyplot as plt


def do_bin(xpixelarr, ypixelarr, xbin, ybin, xarrall, yarrall, marrall, marrallabs):
    """
    Parameters
    ----------
    xpixelarr:      array of the ranging value of pixels in the x direction (1D)
    ypixelarr:      array of the ranging value of pixels in the y direction (1D)
    xbin:           number of bins in the x direction
    ybin:           number of bins in the y direction
    xarrall:        array of arrays: Each element in xarrall corresponds to the 
                    x pixel values of a given star i
    yarrall:        array of arrays: Each element in yarrall corresponds to the 
                    y pixel values of a given star i
    marrall:        array of arrays: Each element in marrall corresponds to the
                    magnitude observed at pixel (x,y) for a given star i
    marrallabs:     array of the absolute certain value of the magnitude for a 
                    given star i (1D)
    
    Returns
    -------
    [xbinarr, ybinarr, zzorig, zzfinal, xarrfinal, yarrfinal, delmarrfinal]
    xbinarr:        (1D for plotting purposes in making a meshgrid if we don't do the fit)
    ybinarr:        (1D for plotting purposes in making a meshgrid if we don't do the fit)
    zzorig:         2D array of size xbin * ybin -- the original one -- where if 
                    there is nothing in a bin, 'None' is the element; 
                    and if there are multiple points/magnitudes in the bin, 
                    there is an array of the magnitudes (see Note 3)
    zzfinal:        2D array of size xbin * ybin -- the final one -- where the 
                    averages of each bin are taken, and if there was nothing in 
                    a bin, the average is set to 0 (see Note 1)
    xarrfinal:      array of the x values (1D for fitting purposes)
    yarrfinal:      array of the y values (1D for fitting purposes)
    delmarrfinal:   array of the delta magnitude values at points (x,y) 
                    (1D for fitting purposes)                
    
    Notes/Problems/To Do
    --------------
    1) for zzfinal, if there is None in a bin, instead of setting the value to 0, 
        maybe we should take the average of the bins around it? Though, this will
        cause more complications with indexing if the bin is on the edge and it
        will also complicate the code by taking more time because it will have 
        to iterate through the 2D array once again
    2) for xbinarr and ybinarr, they may not be necessarily needed in the return
        statement, but they are useful for visually seeing the mag averages in a 
        3D plot (what you need to do is make meshgrid, xx and yy, using the 
        xbinarr and ybinarr and plot that with the zzfinal 2D array)
    3) for zzorig, I'm just keeping it incase we want to see all the different
        delta magnitues at a certain bin
    4) for creating zz as a 2D array of size xbin * ybin with all the values 
        being set to 'None' causes some 'FutureWarnings' and I don't really know 
        how to solve the issue and go around it
     
    """
    zz = np.array([np.array([None for i in range(np.int(xbin))]) for j in range(np.int(ybin))])
    xbin, ybin = np.double(xbin), np.double(ybin)
    xbinarr = np.linspace(np.min(xpixelarr), np.max(xpixelarr), xbin, endpoint = True)
    ybinarr = np.linspace(np.min(ypixelarr), np.max(ypixelarr), ybin, endpoint = True)
    dx, dy = xbinarr[1] - xbinarr[0], ybinarr[1] - ybinarr[0]
    
    for i, (xarr, yarr, marr) in enumerate(zip(xarrall, yarrall, marrall)):
        xarr = np.asarray(xarr)
        yarr = np.asarray(yarr)
        marr = np.asarray(marr)
        mabs = marrallabs[i]
        delmarr = marr - mabs
        #print '***'
        #print '***'
        #print '***'
        #print '***'
        #print '***'
        #print xarr
        #print yarr
        #print marr
        #print mabs
        #print delmarr
        #print '***'
        for i, x in enumerate(xbinarr):
            for j, y in enumerate(ybinarr):
                #print '*********'
                #print i, j
                #print x, x+dx
                #print y, y +dy
                inbin = np.where((xarr >= x) & (xarr < x + dx) & (yarr >= y) & (yarr < y + dy))[0]
                if len(inbin): # if inbin exists
                    if zz[i][j] == None: # gives a FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
                        zz[i][j] = delmarr[inbin]
                    else:
                        zz[i][j] = np.append(zz[i][j], delmarr[inbin])
                
    zzorig = np.copy(zz)
    zzfinal = np.copy(zz)
    xarrfinal = np.array([]) # 1D need for the fit
    yarrfinal = np.array([]) # 1D need for the fit
    delmarrfinal = np.array([]) # 1D need for the fit
    for i in range(len(xbinarr)):
        for j in range(len(ybinarr)):
            if zzfinal[i][j] == None: # gives a futurewarning : comparison to `None` will result in an elementwise object comparison in the future.
                zzfinal[i][j] = 0
            else:
                delmavg = np.mean(zzfinal[i][j])
                zzfinal[i][j] = delmavg
                xarrfinal = np.append(xarrfinal, xbinarr[i])
                yarrfinal = np.append(yarrfinal, ybinarr[j])
                delmarrfinal = np.append(delmarrfinal, delmavg)
    return [xbinarr, ybinarr, zzorig, zzfinal, xarrfinal, yarrfinal, delmarrfinal]
x1 = [ 1598.07,   2042.9 ,   2486.33 ,  1147.49  ,  698.172 , 2043.64  , 2486.17,1142.08 ,   677.55 ]
y1 = [ 2051.4 ,  2464.57 , 2879.63 , 1691.44,  1280.41 , 1627.14,  1157.47 , 2535.5,3028.69]
m1 = [ 24.4 , 24.1 , 24.2  ,24.1,  24.2 , 23.8 , 23.5 , 24.7,  24. ]
xpixel1, ypixel1 = np.arange(4096), np.arange(4096)
xbin1, ybin1 = 10, 10
x2, y2 = x1, y1
m2 = [ 13 , 14 , 13.5  ,14.5,  15.2 , 14.8 , 13.5 , 14.7,  14. ]
ma1 = 24
ma2 = 14
xbinarr, ybinarr, zzorig, zzfinal, xarrfinal, yarrfinal, delmarrfinal = do_bin(xpixel1, ypixel1, xbin1, ybin1, [x1,x2],[y1,y2],[m1,m2],[ma1,ma2])

# NEED TO FIGURE OUT BETTER WAY OF DOING THIS
def plot3d(X,Y,Z):    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1)
    #ax.scatter(x,y,z, s= 50) # from paper.py
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    return fig
xx, yy = np.meshgrid(xbinarr, ybinarr, sparse = True, copy=False)
plt.show(plot3d(xx,yy,zzfinal))

def func(x,y):
    #the function we want to optimize
    return [x**2, y**2,x*y,x,y,1+x*0]
def getfit(x, y, z, pixelx, pixely):
    """
    copied from polyfit2d.py
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    xx, yy = np.meshgrid(pixelx, pixely, sparse = True, copy=False) #why copy false??
    f = func(x,y)
    fmesh = func(xx,yy)
    A = np.array(f).T
    B = z
    coeff, rsum, rank, s = np.linalg.lstsq(A, B)
    
    zfit = np.zeros(len(x))
    Zfit = [[0 for i in pixelx] for j in pixely]
    k = 0
    while k < len(coeff):
        zfit += coeff[k]*f[k]
        Zfit += coeff[k]*fmesh[k]
        k+=1
    #zfit = coeff[0]*x**2 + coeff[1]*y**2 + ... === coeff[0]*f[0] + ...
    #Zfit = coeff[0]*X**2 + coeff[1]*Y**2 + ... === coeff[0]*fmesh[0] + ...
    
    def get_res(z, zfit):
        # returns an array of the error in the fit
        return np.abs(zfit-z)
    resarr = get_res(z,zfit)
    return [zfit, Zfit, x, y, xx, yy, rsum, resarr]
zfit, Zfit, x, y, xx, yy, rsum, resarr = getfit(xarrfinal,yarrfinal,delmarrfinal, np.arange(0,4096,256), range(0,4096,256))

from mpl_toolkits.mplot3d import Axes3D
def plot3d1(x,y,z,X,Y,Z):    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1)
    ax.scatter(x,y,z, s= 50) # from paper.py
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    return fig
#plt.show(plot3d1(x,y,delmarrfinal, xx, yy, Zfit))