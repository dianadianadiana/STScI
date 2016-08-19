
###########################################
###########################################
###########################################
# Random functions that may be useful for reference in the future
# CAN'T run as is
###########################################
###########################################
###########################################

############### If we want to read in all the locations of the walkers for each step and just plot chi evolution
#def get_chisqall(fil, nwalkers, nsteps, ndim, chosenfunc, n):
#    #fil = '/Users/dkossakowski/Desktop/trash/testcheb1.txt'
#    data = np.genfromtxt(fil)
#    data = data.reshape(nwalkers, nsteps, ndim, order='F')
#    func2read, func2fit = get_function(chosenfunc, n)
#    chisqall = []
#    for walker in range(nwalkers):
#        print 'walker ', walker
#        chisqwalker = []
#        for step in range(nsteps):
#            currparams = data[walker][step]
#            currchisq = lnlike(currparams, tab, func2fit) * -2.
#            chisqwalker = chisqwalker + [currchisq]
#        chisqall.append(chisqwalker)  
#    return chisqall  
#    
#def plot_chisqall(cmap, fil, nwalkers, nsteps, ndim, chosenfunc, n):
#    chisqall = get_chisqall(fil, nwalkers, nsteps, ndim, chosenfunc, n)
#    fig = plt.figure()
#    cmap = plt.get_cmap(cmap)
#    colors = [cmap(i) for i in np.linspace(0, 1, len(chisqall))]
#    for elem, col in zip(chisqall, colors):
#        plt.scatter(range(len(elem)), elem, c = col, lw = .5)
#    return fig
#cmap = 'jet'
#fil = '/Users/dkossakowski/Desktop/trash/testcheb1.txt'
#nwalkers = 10
#nsteps = 100
#ndim = 4
#chosenfunc = 'cheb'
#n = 1
#plt.show(plot_chisqall(cmap, fil, nwalkers, nsteps, ndim, chosenfunc, n))  
###############

###############
#finalfunc = lambda p, x, y: np.sum(func2fit(x,y) * np.asarray(p))     # The final flat
#def apply_flat(rows, coeff, somethingchar):
#    final = np.array([])
#    for x,y,something in zip(rows['x'], rows['y'], rows[somethingchar]):
#        if SCALE2ONE:
#            x = (x - CHIPXLEN/2)/(CHIPXLEN/2)
#            y = (y - CHIPYLEN/2)/(CHIPYLEN/2)
#        final = np.append(final, something / finalfunc(coeff,x,y))
#    return final
#    
#def simple3dplot(rows, something, title = ''):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(rows['x'],rows['y'], something)
#    ax.set_title(title)
#    plt.show(fig)
#    
#def simple3dplotindstars(tab, title=''):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(tab['id']))))
#    for star, col in zip(np.unique(tab['id']), colors):
#        starrows = tab[np.where(tab['id'] == star)[0]]
#        ax.scatter(starrows['x'],starrows['y'], starrows['flux'], c = col)
#    ax.set_title(title)
#    plt.show(fig)
#    
#def simple3dplotindstarsafter(tab, coeff, title = ''):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    for star in np.unique(tab['id']):
#        starrows = tab[np.where(tab['id'] == star)[0]]
#        starfinal = apply_flat(starrows, coeff, 'flux')
#        starfinalavg = np.mean(starfinal)
#        ax.scatter(starrows['x'], starrows['y'], (starfinal-starfinalavg)/starfinalavg)
#    ax.set_title(title)
#    plt.show(fig)
###############
   
############### If we want to plot the before/after of applying the flat (as is as well as the normalized delta)
#final = apply_flat(tab, mcmccoeff, 'flux')
#simple3dplot(tab, tab['flux'], title = 'Before LFlat, just flux values plotted')
#simple3dplot(tab, final, title = 'After LFlat, just flux values plotted')
#simple3dplot(tab, (tab['flux'] - tab['avgflux'])/ tab['avgflux'], title = 'Before LFlat, normalized delta flux')
#simple3dplot(tab, (final - tab['avgflux'])/tab['avgflux'], title = 'After LFlat, normalized delta flux') # Not QUITE right because there is a new mean
###############

############### If we want to see/plot the mean of each star before and after applying the flat
#for star in np.unique(tab['id']):
#    starrows = tab[np.where(tab['id']==star)[0]]
#    finalstar = apply_flat(starrows, mcmccoeff, 'flux')
#    mean_before = np.mean(starrows['flux'])
#    std_before = np.std(starrows['flux'])
#    mean_after = np.mean(finalstar)
#    std_after = np.std(finalstar)
#    print '***' + str(star)
#    print 'mean, max-min before', mean_before , np.max(starrows['flux']) - np.min(starrows['flux'])
#    print 'std before\t', std_before
#    print 'max-min/mean before', (np.max(starrows['flux']) - np.min(starrows['flux'])) / mean_before
#    print 'mean, max-min after', mean_after, np.max(finalstar) - np.min(finalstar)
#    print 'std after\t', std_after
#    print 'max-min/mean after', (np.max(finalstar) - np.min(finalstar)) / mean_after
#    #simple3dplot(starrows, starrows['flux'], title = 'Original ' + str(star) + ' ' + str(mean_before) + ', ' + str(std_before))
#    #simple3dplot(starrows, finalstar, title = 'Final ' + str(star) + ' ' + str(mean_after) + ', ' + str(std_after))
############### 