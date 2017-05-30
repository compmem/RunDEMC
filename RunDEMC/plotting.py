#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from matplotlib.ticker import MaxNLocator
import pylab as pl
import numpy as np
import os
import sys
from scipy.stats import gaussian_kde,pearsonr

from .density import fast_2d_kde, kdensity

def joint_plot(particles,weights,burnin=50,names=None,legend=False,add_best=True,
               border=.1,sep=0.0,rot=None,fig=None,nxticks=5,nyticks=5,
               take_log=False,grid=False,bold_sig=False,corr_size=(12,18),
               do_scatter=False, num_contours=25, cmap=None):
    # get the fig
    if fig is None:
        fig = pl.gcf()

    if cmap is None:
        cmap = pl.get_cmap('gist_earth_r')
        cmap._init()
        cmap._lut[:5,:] = 1.0
        cmap._lut[:20,-1] = np.linspace(0,1,20)

    # get num grid
    n_p = particles.shape[-1]

    # calc width and height
    width = height = (1.0-(2.*border) - ((n_p-1)*sep))/n_p

    # get best indiv
    best_ind = weights[burnin:].ravel().argmax()
    indiv = [particles[burnin:,:,i].ravel()[best_ind]
             for i in range(particles.shape[-1])]

    # set holder for axes
    ax = np.zeros((n_p,n_p),dtype=np.object)
    for i in range(n_p):
        for j in range(i+1,n_p):
            # create the axis (start at top right)
            left = border + (j*width) + (j*sep)
            #bottom = 1 - (border + ((i+1)*height) + ((i+1)*sep))
            bottom = 1 - (border + (i*height) + (i*sep) + height)
            sharex = sharey = None
            if i > 0:
                if False: #j==0:
                    if i>1:
                        sharex = ax[1,j]
                else:
                    sharex = ax[0,j]
            if j > i+1:
                if False: #i==0:
                    if j>1:
                        sharey = ax[i,1]
                else:
                    sharey = ax[i,j-1]
            ax[i,j] = fig.add_axes((left,bottom,width,height),
                                   sharex=sharex, sharey=sharey)
            a = ax[i,j]

            # clear it
            a.cla()

            # show joint
            if take_log:
                w = np.log(weights[burnin:,:].ravel())
            else:
                w = weights[burnin:,:].ravel()
            if num_contours>0:
                # determine extents
                cx = particles[burnin:,:,i].ravel()
                xdiff = cx.max()-cx.min()
                cy = particles[burnin:,:,j].ravel()
                ydiff = cy.max()-cy.min()
                gap = .025
                extents = (cy.min()-gap*ydiff,cy.max()+gap*ydiff,
                           cx.min()-gap*xdiff,cx.max()+gap*xdiff)

                grd = fast_2d_kde(cy,
                                  cx,
                                  gridsize=(200, 200),
                                  extents=extents,
                                  nocorrelation=False,
                                  weights=None)
                # draw the contour
                a.contourf(np.linspace(extents[0],extents[1],200),
                           np.linspace(extents[2],extents[3],200),
                           np.flipud(grd),num_contours,
                #cmap=pl.get_cmap('GnBu')
                cmap=pl.get_cmap('gist_earth_r')
                    )
                a.axis('tight')

            if do_scatter:
                pts = a.scatter(particles[burnin:,:,j].ravel(),
                                particles[burnin:,:,i].ravel(),
                                c=w,
                                #cmap=pl.get_cmap('gist_earth'),
                                s=20,
                                linewidth=0,
                                alpha=.1)
            if legend:
                cb = pl.colorbar(alpha=1.0)
                cb.set_alpha(1.0)
                cb.set_label('log(Weight)')
                cb.draw_all()
            if add_best:
                a.plot(indiv[j],indiv[i],'rx',markersize=10,markeredgewidth=3)

            # set the n-ticks
            a.xaxis.set_major_locator(MaxNLocator(nxticks))
            a.yaxis.set_major_locator(MaxNLocator(nyticks))

            # set the tick loc
            a.xaxis.set_ticks_position('top')
            a.yaxis.set_ticks_position('right')

            # turn on the grid if wanted
            if grid:
                a.grid('on')

            # clean labels
            if i > 0 or i < 0:
                for label in a.get_xticklabels():
                    label.set_visible(False)
            elif not rot is None:
                for label in a.get_xticklabels():
                    label.set_rotation(rot)
                    #label.set_horizontalalignment('right')
            if j < n_p-1 or j > n_p-1:
                for label in a.get_yticklabels():
                    label.set_visible(False)

    # add text after scatter plot so we don't mess up ranges
    for i in range(n_p):
        # create the axis (start at top right)
        left = border + (i*width) + (i*sep)
        #bottom = 1 - (border + ((i+1)*height) + ((i+1)*sep))
        bottom = 1 - (border + (i*height) + (i*sep) + height)
        sharex = sharey = None
        ax[i,i] = fig.add_axes((left,bottom,width,height),
                               sharex=sharex, sharey=sharey)
        a = ax[i,i]

        # place the labels
        a.text(0.5, 0.5,names[i],
               horizontalalignment='center',
               verticalalignment='center',
               transform = a.transAxes,
               fontsize=24,fontweight='bold')

        # remove all ticks
        a.set_xticks([])
        a.set_yticks([])


    # now the bottom triangle
    for i in range(n_p):
        for j in range(0,i):
                        # create the axis (start at top right)
            left = border + (j*width) + (j*sep)
            #bottom = 1 - (border + ((i+1)*height) + ((i+1)*sep))
            bottom = 1 - (border + (i*height) + (i*sep) + height)
            sharex = sharey = None
            ax[i,j] = fig.add_axes((left,bottom,width,height),
                                   sharex=sharex, sharey=sharey)
            a = ax[i,j]

            # add in correlation plot
            # cc = np.corrcoef(particles[burnin:,:,j].ravel(),
            #                  particles[burnin:,:,i].ravel())[0,1]
            cc,pp = pearsonr(particles[burnin:,:,j].ravel(),
                             particles[burnin:,:,i].ravel())
            a = ax[i,j]
            fs = (corr_size[1]-corr_size[0])*np.abs(cc) + corr_size[0]
            txt = a.text(0.5, 0.5, '%1.2f'%cc,
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = a.transAxes,
                         fontsize=fs)
            # bold if sig
            if bold_sig and pp <= .05:
                txt.set_fontweight('bold')

            # remove all ticks
            a.set_xticks([])
            a.set_yticks([])

    #pl.tight_layout()
    pl.show()

    return ax


def violin_plot(data,positions=None,widths=None,ax=None,
                add_boxplot=False,**fillargs):
    '''
    Create violin plots on an axis.

    fillargs defaults to dict(facecolor='y',alpha=0.3)

    Modified from:
    http://pyinsci.blogspot.com/2009/09/violin-plot-with-matplotlib.html
    '''
    x = data
    # convert x to a list of vectors
    if hasattr(x, 'shape'):
        if len(x.shape) == 1:
            if hasattr(x[0], 'shape'):
                x = list(x)
            else:
                x = [x,]
        elif len(x.shape) == 2:
            nr, nc = x.shape
            if nr == 1:
                x = [x]
            elif nc == 1:
                x = [x.ravel()]
            else:
                x = [x[:,i] for i in range(nc)]
        else:
            raise ValueError
    if not hasattr(x[0], '__len__'):
        x = [x]
    col = len(x)

    # make it called data again
    data = x

    # set the axis
    if ax is None:
        ax = pl.gca()

    # set positions and widths
    if positions is None:
        positions = list(range(1, col + 1))
    if widths is None:
        distance = max(positions) - min(positions)
        widths = min(0.15*max(distance,1.0), 0.5)
    if isinstance(widths, float) or isinstance(widths, int):
        widths = np.ones((col,), float) * widths

    # process the fillargs
    fargs = dict(facecolor='y',alpha=0.3)
    fargs.update(fillargs)

    # loop and add violins
    for d,p,w in zip(data,positions,widths):
        # redo with kdensity
        k = gaussian_kde(d) #calculates the kernel density
        m = k.dataset.min() #lower bound of violin
        M = k.dataset.max() #upper bound of violin
        x = np.arange(m,M,(M-m)/100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v/v.max()*w #scaling the violin to the available space
        ax.fill_betweenx(x,p,v+p,**fargs)
        ax.fill_betweenx(x,p,-v+p,**fargs)

    # see if add boxplots
    if add_boxplot:
        ax.boxplot(data,notch=1,positions=positions,
                   widths=widths,vert=1)

    # show it
    pl.show()


def joint_movie(x,y,weights,burnin,names=('x','y'),fps=10,
                outbase='movie',nfade=10,
                colorbar=False,preview=-1,zoom=False,
                vmin=None,vmax=None,alpha_range=(.1,1.0)):
    ax = pl.gca()
    niter = len(x)
    if vmin is None:
        #vmin = weights[burnin:].min()
        vmin = weights.min()
    if vmax is None:
        #vmax = weights[burnin:].max()
        vmax = weights.max()
    alpha = np.linspace(alpha_range[0],alpha_range[1],nfade)
    for i in range(niter):
        ax.cla()
        sys.stdout.write('%d '%i)
        sys.stdout.flush()
        fname = '/tmp/_tmp%05d.png'%i
        for n,j in enumerate(range(i-nfade+1,i+1)):
            if j<0:
                continue
            ax.scatter(x[j,:].ravel(),
                       y[j,:].ravel(),
                       c=weights[j,:].ravel(),
                       vmin=vmin,
                       vmax=vmax,
                       linewidth=0,
                       s=20,
                       alpha=alpha[n])
        if colorbar:
            cb = ax.colorbar(alpha=1.0)
            cb.set_alpha(1.0)
            cb.set_label('log(Weight)')
            cb.draw_all()

        ax.set_title(str(i))
        if zoom:
            ir = np.arange(i-nfade+1,i+1)
            ir = ir[ir >= 0]
            ax.set_xlim(x[ir,:].min(1).mean(),x[ir,:].max(1).mean())
            ax.set_ylim(y[ir,:].min(1).mean(),y[ir,:].max(1).mean())
        else:
            ax.set_xlim(x.min(),x.max())
            ax.set_ylim(y.min(),y.max())
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        pl.savefig(fname)

        # see if saving a preview image
        if i == preview:
            prev_name = outbase + '_preview_%d.png'%i
            pl.savefig(prev_name)

    # make the movie
    outfile = outbase + '.mp4'
    os.system('rm '+outfile)
    os.system("ffmpeg -r "+str(fps)+" -b 1800 -i /tmp/_tmp%05d.png "+outfile)
    os.system("rm /tmp/_tmp*.png")


# if __name__=="__main__":
#     pos = range(5)
#     data = [normal(size=100) for i in pos]
#     fig=figure()
#     ax = fig.add_subplot(111)
#     violin_plot(ax,data,pos,bp=1)
#     show()
