#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the RunDEMC package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# global imports
import numpy as np
from scipy.stats.distributions import uniform

import rpy2.robjects
from rpy2.robjects.packages import importr
rstats = importr('stats')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def rpdf(dat,x,nbins=512,extrema=None):
    """
    Use R to get the density.
    """
    # set the extrema
    if extrema is None:
        extrema = (x.min(),x.max())
        
    # call R's density function
    rpdf = rstats.density(dat,**{'from':extrema[0],
                                 'to':extrema[1],
                                 'n':nbins})
    # get the x and y
    rx = np.array(rpdf.rx('x'))[0]
    ry = np.array(rpdf.rx('y'))[0]

    # interp to get the points we want
    pdf = np.interp(x,rx,ry)

    return pdf

# make the rlba available
#code = ''.join(file("lba.r").readlines())
code = """
rlba=function(n,b,A,vs,s,t0){
rt=resp=numeric(n)
choices=length(vs)
for(i in 1:n){
d=0
while(all(d<=0)){
d=rnorm(choices,vs,s)
}
k=runif(choices,0,A)
ttf=(b-k)/d
min.rt=min(ttf[ttf>0])
rt[i]=min.rt + t0
resp[i]=c(1:choices)[ttf==min.rt]
}
list(rt=rt,resp=resp)
}"""
_result = rpy2.robjects.r(code)
#rlba = rpy2.robjects.globalenv['rlba']
_rlba_fun = rpy2.robjects.r.rlba
def rlba(n, b, A, vs, s, t0):
    """
    """
    # run the sim in R
    res = _rlba_fun(n,b,A,rpy2.robjects.FloatVector(np.float64(vs)),s,t0)

    # get the rts and responses
    rt = np.array(res.rx('rt'))[0]
    resp = np.array(res.rx('resp'))[0] - 1 # convert to zero

    return rt,resp


def q_ratios(pop, proposal, weights, gamma, epsilon):
    """
    """
    # initialize forward and backward densities
    df = np.zeros(len(pop))
    db = np.zeros(len(pop))

    # get probability from weights
    wp = weights/weights.sum()

    # # loop over all combos
    # for i in xrange(len(pop)):
    #     sys.stdout.write('+')
    #     sys.stdout.flush()
    #     for j in xrange(len(pop)):
    #         for k in xrange(len(pop)):
    #             for p in xrange(len(pop)):
    #                 if len(np.unique([i,j,k])) < 3:
    #                     # we're not at a possible proposal
    #                     continue
    #                 # calc forward and back vals
    #                 #tval = pop[k] + gammas[p]*(pop[i]-pop[j])
    #                 tval = pop[k] + gamma*(pop[i]-pop[j])
    #                 df[p] += np.prod(wp[k]/pop.shape[1]*
    #                                  uniform.pdf(proposal[p],
    #                                              loc=tval-epsilon,
    #                                              scale=2*epsilon))
    #                 db[p] += np.prod(wp[k]/pop.shape[1]*
    #                                  uniform.pdf(pop[p],
    #                                              loc=tval-epsilon,
    #                                              scale=2*epsilon))

    # set up all combos
    i = np.mgrid[:len(pop),:len(pop),:len(pop)][0].flatten()
    j = np.mgrid[:len(pop),:len(pop),:len(pop)][1].flatten()
    k = np.mgrid[:len(pop),:len(pop),:len(pop)][2].flatten()

    # remove where they match
    bad_ind = ((i-j)==0) | ((j-k)==0) | ((i-k)==0)
    i = i[~bad_ind]
    j = j[~bad_ind]
    k = k[~bad_ind]

    # # loop over particles, getting the density
    # for p in xrange(len(pop)):
    #     # calc forward and back vals
    #     #tval = pop[k] + gammas[p]*(pop[i]-pop[j])
    #     tval = pop[k] + gamma*(pop[i]-pop[j])
    #     df[p] = np.prod((wp[k]/pop.shape[1]*
    #                      uniform.pdf(proposal[p][np.newaxis].repeat(len(i),axis=0).flatten(),
    #                                  loc=tval.flatten()-epsilon,
    #                                  scale=2*epsilon).reshape(tval.shape).T),axis=0).sum()
    #     db[p] = np.prod((wp[k]/pop.shape[1]*
    #                      uniform.pdf(pop[p][np.newaxis].repeat(len(i),axis=0).flatten(),
    #                                  loc=tval.flatten()-epsilon,
    #                                  scale=2*epsilon).reshape(tval.shape).T),axis=0).sum()

    # loop over particles, getting the density
    for p in xrange(len(pop)):
        # calc forward and back vals
        #temp=current + gamma*(theta[i,]-theta[j,]) + gamma*(theta[k,]-current)
        tval = pop[p] + gamma*(pop[i]-pop[j]) + gamma*(pop[k]-pop[p])
        df[p] = np.prod((wp[k]/pop.shape[1]*
                         uniform.pdf(proposal[p][np.newaxis].repeat(len(i),axis=0).flatten(),
                                     loc=tval.flatten()-epsilon,
                                     scale=2*epsilon).reshape(tval.shape).T),axis=0).sum()

        #temp=(prop - gamma*(theta[i,]-theta[j,]) - gamma*(theta[k,]))/(1-gamma)
        tval = (proposal[p] - gamma*(pop[i]-pop[j]) - gamma*(pop[k]))/(1-gamma)
        db[p] = np.prod((wp[k]/pop.shape[1]*
                         uniform.pdf(pop[p][np.newaxis].repeat(len(i),axis=0).flatten(),
                                     loc=tval.flatten()-epsilon/(1-gamma),
                                     scale=2*epsilon/(1-gamma)).reshape(tval.shape).T),axis=0).sum()

    
    # calc the ratio of the densities
    #qr = (db + np.finfo(db.dtype).eps)/(df + np.finfo(df.dtype).eps)
    qr = db/df
    return qr

