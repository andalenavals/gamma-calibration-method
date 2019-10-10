import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Finding best resolution and energy calibration parameters comparing Geant4 simulations and experiment. Errors in the esmatimated parameters are calculated using MCMC and marginalization of the posterior distribution.')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_22Na.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--initial_guess_file',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/finalfits/singlesource/finalparsNaI2x2_22Na.yaml',
                        help='.yaml with the initial guess')
    parser.add_argument('--plotspath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method',
                        help='location of the output of the plots')
    parser.add_argument('--nsig', default=1, type=int, 
                        help='How many sigmas for the marginalized confidence interval')
    parser.add_argument('--nsteps', default=1000, type=int, 
                        help='nsteps of MCMC')
    parser.add_argument('--nwalkers', default=100, type=int, 
                        help='nwalkers of MCMC')
    args = parser.parse_args()

    return args

#More proper initial guess to not fall in local minima 
        
def main():
    import numpy as np
    from processor import SaveInTH1, AddFWHM, Calibrate, GetScaleFactor, Scale, FindLowerNoEmptyUpbin,  FindHigherNoEmptyLowbin
    from plotter import PrettyPlot,  plot_samplesdist
    from likelihood import minimizeCHI2,  chi2,  MCMC, minimizeCHI2_list,  chi2_list,  TransfSIM
    from ROOT import TCanvas,  gStyle, kFALSE

    args = parse_args()

    outpath = os.path.expanduser(args.outpath)
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

    plotspath = os.path.expanduser(args.plotspath)
    try:
        if not os.path.exists(plotspath):
            os.makedirs(plotspath)
    except OSError:
        if not os.path.exists(plotspath): raise

  

    print("Fitting using only one source",  args.simfile)
    #Save data in arrays
    ch_sim, counts_sim = np.loadtxt(args.simfile, dtype=int, unpack=True)
    ch_exp, counts_exp = np.loadtxt(args.measurefile, dtype=int, unpack=True)
    nbins_exp, xlow_exp, xup_exp = len(ch_exp), min(ch_exp), max(ch_exp)
    nbins_sim, xlow_sim, xup_sim = len(ch_sim), min(ch_sim), max(ch_sim)
    hexp = SaveInTH1(ch_exp,counts_exp,'hist_exp',nbins_exp,xlow_exp,xup_exp)
    hsim = SaveInTH1(ch_sim,counts_sim,'hist_sim',nbins_sim,xlow_sim,xup_sim)
 
   
    stream = open(args.initial_guess_file, 'r')
    parsfile = yaml.safe_load(stream.read())
    fwhmpars = [parsfile['a'], parsfile['b'], parsfile['c']]
    calpars = [ parsfile['m'], parsfile['d']]
    i_guess =  fwhmpars + calpars
    binlow =  parsfile['binlow']
    binup =  parsfile['binup']
    print('Initial parameters:',  parsfile)
    chisq =  chi2(i_guess, hexp, hsim, binlow=binlow, binup=binup)
    print('Initial Chisq_nu:',  chisq/(binup - binlow))

    print('Initial low bin divergence',  binlow - GetNoEmptyLowbin(hexp))
    print('Initial up bin divergence',  binup - GetNoEmptyUpbin(hexp))
    binlow = GetNoEmptyLowbin(hexp)
    binup = GetNoEmptyUpbin(hexp)
    print('Limit bins used:', binlow, binup)

    #DEPURATING SELECTING PROPER HIGH AND LOW LIMIT
    pars = i_guess; 
    chisqold, chisq =  0, 1.e+10
    binlow_old, binup_old = 0, 0
    n = 0
    if args.history:
        historyfile = os.path.join(outpath, 'pars_history.txt')
        pars_hist =  []
    else:
        historyfile =  None
        pars_hist =  None

    while(binlow!=binlow_old or binup!=binup_old or chisq<chisqold ):
        print('Recursive iteration number',  n)
        binlow_old = binlow; binup_old = binup; chisqold = chisq
        pars, chisq = minimizeCHI2(pars, hexp, hsim, binlow=binlow, binup=binup, pars_hist=pars_hist,  verbose=args.verbose)
       
        fwhmpars = pars[:3]
        calpars =  pars[3:]
        print('FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
        print('Calibration pars mx +d:',  calpars)
        print('Chisq_nu:',  chisq/(binup - binlow))

        #Usually at large energies there is noise in the experiment with almost null counts, and for low energies 
        #binlow, binup = FindNewLimits(hsim, hexp, pars)
        
        print('New binlow and binup:',  binlow, binup)
        n += 1print("Fitting using only one source",  args.simfile)
    #Save data in arrays
    ch_sim, counts_sim = np.loadtxt(args.simfile, dtype=int, unpack=True)
    ch_exp, counts_exp = np.loadtxt(args.measurefile, dtype=int, unpack=True)
    nbins_exp, xlow_exp, xup_exp = len(ch_exp), min(ch_exp), max(ch_exp)
    nbins_sim, xlow_sim, xup_sim = len(ch_sim), min(ch_sim), max(ch_sim)
    hexp = SaveInTH1(ch_exp,counts_exp,'hist_exp',nbins_exp,xlow_exp,xup_exp)
    hsim = SaveInTH1(ch_sim,counts_sim,'hist_sim',nbins_sim,xlow_sim,xup_sim)
 
   
    stream = open(args.initial_guess_file, 'r')
    parsfile = yaml.safe_load(stream.read())
    fwhmpars = [parsfile['a'], parsfile['b'], parsfile['c']]
    calpars = [ parsfile['m'], parsfile['d']]
    i_guess =  fwhmpars + calpars
    binlow =  parsfile['binlow']
    binup =  parsfile['binup']
    print('Initial parameters:',  parsfile)
    chisq =  chi2(i_guess, hexp, hsim, binlow=binlow, binup=binup)
    print('Initial Chisq_nu:',  chisq/(binup - binlow))

    print('Initial low bin divergence',  binlow - GetNoEmptyLowbin(hexp))
    print('Initial up bin divergence',  binup - GetNoEmptyUpbin(hexp))
    binlow = GetNoEmptyLowbin(hexp)
    binup = GetNoEmptyUpbin(hexp)
    print('Limit bins used:', binlow, binup)

    
    ##MCMC withou the first parameter
    print('STARTING MCMC')
    mflags = [False, True]
    samples, chains =  MCMC(i_guess, hexp, hsim, binlow=binlow, binup=binup, nwalkers=args.nwalkers,  nsteps=args.nsteps,  mflags=mflags)
    print('SAMPLES:' , samples)
    print('CHAINS:' , chains)
    mcmcpath = os.path.join(plotspath, 'mcmc_walkers.png')
    parscontoursparth = os.path.join(plotspath, 'par_contours.png')
    if args.plots: plot_samplesdist(samples, chains, mflags, args.nwalkers, args.nsteps, mcmcpath, parscontoursparth )
    print('MCMC finished')


                
          



                             
if __name__ == "__main__":
    main()


''' Note Simulation of the 3x3 detector is missing of the 1200 keV
peak do not include in the analysis.
'''
