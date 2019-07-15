def chi2(pars, hexp, hsim, binlow=None, binup=None):
    import numpy as np
    from processor import AddFWHM,  Calibrate,  GetScaleFactor, Scale
    fwhmpars =  pars[: 3]
    calpars =  pars[3:]
    hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  fwhmpars)
    hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
    chisqr = []
    scalefactor =  GetScaleFactor(hexp, hsim_fwhm_ch, binlow=binlow, binup=binup)
    hsim_fwhm_ch_sc = Scale(hsim_fwhm_ch, 'hist_sim_fwhm_cal_sc',  scalefactor )
    if binlow or binup is None:
        binlow = 1; binup = hexp.GetNbinsX()    
    for i in range(binlow, binup + 1):
        if (hexp.GetBinContent(i) == 0): sigma2 = 1
        else: sigma2 = hexp.GetBinContent(i)
        #print (i,  hsim_fwhm_ch.GetBinContent(i))
        chisqr.append( ((hexp.GetBinContent(i) -  hsim_fwhm_ch_sc.GetBinContent(i))**2)/sigma2 ) 
    #print(pars, np.array(chisqr).sum())
    return np.array(chisqr).sum()

    
def minimizeCHI2(initial_guess, hexp, hsim, bounds=None,  binlow=None, binup=None):
    import scipy.optimize as optimize
    #result = optimize.minimize(chi2, initial_guess,args=(hexp, hsim, binlow, binup), method='SLSQP', bounds=bounds)
    result = optimize.minimize(chi2, initial_guess,args=(hexp, hsim, binlow, binup), method='Nelder-Mead', tol=1e-6)
    if result.success:
        fitted_params = result.x
        return fitted_params, result.fun
    else:
        raise ValueError(result.message)


#Log natural of the prior 
def logprior(pars, mflags=[True, True]):
    import numpy as np
    aflag, bflag =  mflags
    l = 1000
    a_min, b_min, c_min, m_min, d_min  = -l,-l,-l,-l,- l
    a_max, b_max, c_max, m_max, d_max  = l,l,l,l,l
    if(aflag and bflag):
        #print("Using a, b and delta")
        a, b, c, m, d = pars
        if ( a_min<a< a_max)and( b_min < b< b_max)and( c_min<c<c_max)and( m_min<m<m_max)and( d_min<d<d_max):
            return 0.0
        return -np.inf
    elif(aflag and (not bflag)):
        #print("Using a and c")
        a, c, m, d = pars
        if ( a_min<a< a_max)and(c_min<c<c_max)and( m_min<m<m_max)and( d_min<d<d_max):
            return 0.0
        return -np.inf
    elif((not aflag) and bflag):
        #print("Using a and b")
        b, c,  m, d = pars
        if (c_min<c<c_max)and( b_min< b< b_max)and( m_min<m<m_max)and( d_min<d<d_max):
            return 0.0
        return -np.inf
    else:
        #print("Using only a")
        c, m, d = pars
        if ( c_min < c < c_max)and( m_min<m<m_max)and( d_min<d<d_max):
            return 0.0
        return -np.inf
##Log natural of the likelihood function. Gaussian.
def loglike(chisq):
    return -0.5*chisq
##Log natural of the posterior
def logpost(pars, hexp, hsim, binlow=None, binup=None, mflags=[True, True]):
    import numpy as np 
    chisq = chi2(pars, hexp, hsim, binlow=binlow, binup=binup )
    lp = logprior(pars, mflags=mflags)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(chisq)

def corner_plot(samples, labels, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('SVA1StyleSheet.mplstyle')
    import corner
    import numpy as np
    burn = 5000
    samples_burned = np.c_[[par[burn:] for par in samples]]
    fig = corner.corner(samples_burned.T, labels=labels,
                        quantiles=[0.16, 0.5, 0.84],  #-1sigma,0sigma,1sigma
                        levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)), #1sigma, 2sigma and 3sigma contours
                        show_titles=True, title_kwargs={"fontsize": 12}, title_fmt= '.3f', 
                        smooth1d=None, plot_contours=True,  
                        no_fill_contours=False, plot_density=True, use_math_text=True, )
    print("Printing file:",  title)
    plt.savefig(title)
    print(title, "Printed")
def MCMC(best_pars, hexp, hsim, binlow=None, binup=None,  nwalkers=50, nsteps=1000,  mflags=[True, True]):
    import emcee
    import itertools
    ndim =  len(list(itertools.compress(xrange(len(mflags)),  mflags))) + 3
    pos = [best_pars + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, threads=1,
                                    args=(hexp, hsim, binlow, binup, mflags))
    print("Runing MCMC ...")
    sampler.run_mcmc(pos, nsteps)
    print("Run finished")

    if(ndim ==4):
        b_chain = sampler.chain[:,:,0]; b_chain_flat = np.reshape(b_chain, (nwalkers*nsteps,))
        c_chain = sampler.chain[:,:,1]; c_chain_flat = np.reshape(c_chain, (nwalkers*nsteps,))
        m_chain = sampler.chain[:,:,2]; m_chain_flat = np.reshape(m_chain, (nwalkers*nsteps,))
        d_chain = sampler.chain[:,:,3]; d_chain_flat = np.reshape(d_chain, (nwalkers*nsteps,))
        samples = np.c_[ b_chain_flat, c_chain_flat, m_chain_flat, d_chain_flat].T
        chains = [ b_chain, c_chain, m_chain, d_chain]
    elif(ndim==5):
        a_chain = sampler.chain[:,:,0]; a_chain_flat = np.reshape(a_chain, (nwalkers*nsteps,))
        b_chain = sampler.chain[:,:,1]; b_chain_flat = np.reshape(b_chain, (nwalkers*nsteps,))
        c_chain = sampler.chain[:,:,2]; c_chain_flat = np.reshape(c_chain, (nwalkers*nsteps,))
        m_chain = sampler.chain[:,:,3]; m_chain_flat = np.reshape(m_chain, (nwalkers*nsteps,))
        d_chain = sampler.chain[:,:,4]; d_chain_flat = np.reshape(d_chain, (nwalkers*nsteps,))
        samples = np.c_[a_chain_flat, b_chain_flat, c_chain_flat, m_chain_flat, d_chain_flat].T
        chains = [a_chain, b_chain, c_chain, m_chain, d_chain]
    sampler.reset()
    return samples, chains
def bestparameters(samples):
    allpars = []
    for i in range (len(samples)):
        par = np.percentile(samples[i], [50]);
        allpars.append(par[0])
    return allpars
            
def percentiles(samples, nsig=1):
    allpars_percent_list = []
    for i in range (0, len(samples)):
        if (nsig==1):
            a_perc = np.percentile(samples[i], [16, 50, 84]); par_perc_list =[a_perc[1], a_perc[0] - a_perc[1], a_perc[2] - a_perc[1]]
            allpars_percent_list.append(par_perc_list)
        elif(nsig==2):
            a_perc = np.percentile(samples[i], [2.3, 50, 97.7] ); par_perc_list =[a_perc[1], a_perc[0] - a_perc[1], a_perc[2] - a_perc[1]]
            allpars_percent_list.append(par_perc_list)
    
    return allpars_percent_list
