import numpy as np
import matplotlib.pyplot as plt


def help(xpixel, ypixel, xbin, ybin, xarr, yarr, marr):
    zz = np.array([np.array([None for i in range(np.int(xbin))]) for j in range(np.int(ybin))])
    print zz
    xbin, ybin = np.double(xbin), np.double(ybin)
    dx = len(xpixel)/xbin
    dy = len(ypixel)/ybin
    print dx, dy
    xbinarr = np.arange(np.min(xpixel), np.max(xpixel), dx)
    ybinarr = np.arange(np.min(ypixel), np.max(ypixel), dy)
    print xbinarr
    print ybinarr
    marr = np.asarray(marr)
    for i, x in enumerate(xbinarr):
        for j, y in enumerate(ybinarr):
            print '*********'
            print i, j
            print x, x+dx
            print y, y +dy
            inbin = np.where((xarr >= x) & (xarr < x + dx) & (yarr >= y) & (yarr < y + dy))[0]
            if len(inbin): # if inbin exists
                print inbin
                if zz[i][j] == None:
                    zz[i][j] = marr[inbin]
                else:
                    zz[i][j] = np.append(zz[i][j], marr[inbin])
                print zz[i][j]
                
    print zz
    return 
xpixel, ypixel = np.arange(5), np.arange(5)
xbin, ybin = 5, 5.0
x0 = [0,1,3,4,2,4,0,1]
y0 = [3,4,0,1,3,4,0,2]
o0 = [25.5,25,24,24,25,25,25,25]

x1 = [ 1598.07,   2042.9 ,   2486.33 ,  1147.49  ,  698.172 , 2043.64  , 2486.17,
  1142.08 ,   677.55 ]
y1 = [ 2051.4 ,  2464.57 , 2879.63 , 1691.44,  1280.41 , 1627.14,  1157.47 , 2535.5,
  3028.69]
m1 = [ 24.4 , 24.1 , 24.2  ,24.1,  24.2 , 23.8 , 23.5 , 24.7,  24. ]
xpixel1, ypixel1 = np.arange(4096), np.arange(4096)
xbin1, ybin1 = 10, 10.0
#help(xpixel, ypixel, xbin, ybin, x0,y0,o0)
#help(xpixel1, ypixel1, xbin1, ybin1, x1,y1,m1)

def help1(xpixel, ypixel, xbin, ybin, xarrall, yarrall, marrall, marrallabs):
    zz = np.array([np.array([None for i in range(np.int(xbin))]) for j in range(np.int(ybin))])
    #print zz
    xbin, ybin = np.double(xbin), np.double(ybin)
    xbinarr = np.linspace(np.min(xpixel), np.max(xpixel), xbin, endpoint = True)
    ybinarr = np.linspace(np.min(ypixel), np.max(ypixel), ybin, endpoint = True)
    dx = xbinarr[1] - xbinarr[0]
    dy = ybinarr[1] - ybinarr[0]
    #print xbinarr
    #print ybinarr
    
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
    xarrfinal = np.array([])
    yarrfinal = np.array([])
    delmarrfinal = np.array([])
    for i in range(len(xbinarr)):
        for j in range(len(ybinarr)):
            if zzfinal[i][j] == None: # gives a futurewarning : comparison to `None` will result in an elementwise object comparison in the future.
                zzfinal[i][j] = 0
            else:
                delmag = np.mean(zzfinal[i][j])
                zzfinal[i][j] = delmag
                xarrfinal = np.append(xarrfinal, xbinarr[i])
                yarrfinal = np.append(yarrfinal, ybinarr[j])
                delmarrfinal = np.append(delmarrfinal, delmag)
    return xbinarr, ybinarr, zzorig, zzfinal, xarrfinal, yarrfinal, delmarrfinal
m2 = [ 13 , 14 , 13.5  ,14.5,  15.2 , 14.8 , 13.5 , 14.7,  14. ]
ma1 = 24
ma2 = 14
xbinarr, ybinarr, zzorig, zzfinal, xarrfinal, yarrfinal, delmarrfinal = help1(xpixel1, ypixel1, xbin1, ybin1, [x1,x1],[y1,y1],[m1,m2],[ma1,ma2])

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
#plt.show(plot3d(xx,yy,zzfinal))

def func(x,y):
    #the function we want to optimize
    return [x**2, y**2,x*y,x,y,1+x*0]
def getfit(x, y, z, pixelx, pixely):
    """
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
plt.show(plot3d1(x,y,delmarrfinal, xx, yy, Zfit))