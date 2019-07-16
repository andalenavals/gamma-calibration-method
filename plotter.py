import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('SVA1StyleSheet.mplstyle')

def PrettyPlot(x, y, color='black', marker=None, label=None, xlabel='x-axis', ylabel='y-axis', alsize=24, legendsize=24, alpha=1.):
    plt.plot(x, y, color=color, marker=marker,  label=label, alpha=alpha)
    plt.xlim( [min(x),max(x)] )
    plt.ylim(ymin=0)
    plt.xlabel(xlabel, fontsize=alsize)
    plt.ylabel(ylabel, fontsize=alsize)
    plt.legend(loc='best', fontsize=legendsize)
    plt.tight_layout()

def corner_plot(samples, labels, filename, title=None):
    import corner
    import numpy as np
    #burn = 5000
    plt.clf()
    #butning 20% of start data
    samples= np.c_[[par[int(0.2 * len(par)):] for par in samples]].T
    fig = corner.corner(samples, labels=labels,
                        quantiles=[0.16, 0.5, 0.84],  #-1sigma,0sigma,1sigma
                        levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)), #1sigma, 2sigma and 3sigma contours
                        show_titles=True, title_kwargs={"fontsize": 12}, title_fmt= '.4f', 
                        smooth1d=None, plot_contours=True,  
                        no_fill_contours=False, plot_density=True, use_math_text=True, )
    print("Printing file:",  filename)
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(title, "Printed")

def plot_samplesdist(samples, chains, mflags, nwalkers, nsteps,  namemc, namecont):
    import numpy as np
    import emcee
    aflag, bflag, eflag =  mflags
    fig = plt.figure(figsize=(16, 12))
    ndim =  len(samples)        
    
    if(ndim==4):
        axs = fig.subplots(4, 3)
        if(aflag and (not bflag)):
            labels = ['a', 'c', 'm', 'd']
        elif( (not aflag) and bflag):
            labels = ['b', 'c', 'm', 'd']

        axs[2][0].set_xlabel("Ensemble step")
        axs[2][1].set_xlabel("Ensemble step")
        axs[2][2].set_xlabel("Walker Step")
        axs[0][0].set_title("Ensemble dispersion")
        axs[0][1].set_title("Ensemble autocorrelation")
        axs[0][2].set_title("Walker mean and stdev")
        a_chain, b_chain, c_chain, m_chain = chains
        a_chain_mean = np.mean(a_chain, axis=0); a_chain_err = np.std(a_chain, axis=0) / np.sqrt(nwalkers)
        b_chain_mean = np.mean(b_chain, axis=0); b_chain_err = np.std(b_chain, axis=0) / np.sqrt(nwalkers)
        c_chain_mean = np.mean(c_chain, axis=0); c_chain_err = np.std(c_chain, axis=0) / np.sqrt(nwalkers)
        m_chain_mean = np.mean(m_chain, axis=0); m_chain_err = np.std(m_chain, axis=0) / np.sqrt(nwalkers)
        idx = np.arange(len(a_chain_mean))
        axs[0][2].errorbar(x=idx, y=a_chain_mean,
                           yerr=a_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2.,
                           color='k')
        axs[1][2].errorbar(x=idx, y=b_chain_mean,
                           yerr=b_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2., color='k');
        axs[2][2].errorbar(x=idx, y=c_chain_mean,
                           yerr=c_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2., color='k');
        axs[3][2].errorbar(x=idx, y=m_chain_mean,
                           yerr=m_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2., color='k');
    elif(ndim==5):
        axs = fig.subplots(5, 3)
        labels = ['a', 'b', 'c', 'm', 'd']
        axs[4][0].set_xlabel("Ensemble step")
        axs[4][1].set_xlabel("Ensemble step")
        axs[4][2].set_xlabel("Walker Step")
        axs[0][0].set_title("Ensemble dispersion")
        axs[0][1].set_title("Ensemble autocorrelation")
        axs[0][2].set_title("Walker mean and stdev")
        a_chain, b_chain, c_chain, m_chain, d_chain = chains
        a_chain_mean = np.mean(a_chain, axis=0); a_chain_err = np.std(a_chain, axis=0) / np.sqrt(nwalkers)
        b_chain_mean = np.mean(b_chain, axis=0); b_chain_err = np.std(b_chain, axis=0) / np.sqrt(nwalkers)
        c_chain_mean = np.mean(c_chain, axis=0); c_chain_err = np.std(c_chain, axis=0) / np.sqrt(nwalkers)
        m_chain_mean = np.mean(m_chain, axis=0); m_chain_err = np.std(m_chain, axis=0) / np.sqrt(nwalkers)
        d_chain_mean = np.mean(d_chain, axis=0); d_chain_err = np.std(d_chain, axis=0) / np.sqrt(nwalkers)
        idx = np.arange(len(a_chain_mean))
        axs[0][2].errorbar(x=idx, y=a_chain_mean,
                           yerr=a_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2.,
                           color='k')
        axs[1][2].errorbar(x=idx, y=b_chain_mean,
                           yerr=b_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2., color='k');
        axs[2][2].errorbar(x=idx, y=c_chain_mean,
                           yerr=c_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2., color='k');
        axs[3][2].errorbar(x=idx, y=m_chain_mean,
                           yerr=m_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2., color='k');
        axs[4][2].errorbar(x=idx, y=d_chain_mean,
                           yerr=d_chain_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2., color='k');

    for i, par in enumerate(samples):
        axs[i][0].set_ylabel(labels[i])
        idx = np.arange(len(par))
        axs[i][0].scatter(idx, par[idx], marker='o', c='k', s=10.0, alpha=0.1, linewidth=0)
        # Get selfcorrelation using emcee
        ac = emcee.autocorr.function(par)
        idx = np.arange(len(ac),step=1)
        axs[i][1].scatter(idx, ac[idx], marker='o', c='k', s=10.0, alpha=0.1, linewidth=0)
        axs[i][1].axhline(alpha=1., lw=1., color='red')

    print("Printing file:",  namemc)
    plt.tight_layout()
    plt.savefig(namemc)
    plt.close(fig)
    print(namemc, "Printed")
    corner_plot(samples, labels, namecont)
