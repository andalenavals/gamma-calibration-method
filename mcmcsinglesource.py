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
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/finalfits/singlesource/finalparsNaI2x2_22Na_bc.yaml',
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
    import yaml
    import numpy as np
    from processor import SaveInTH1, GetNoEmptyLowbin, GetNoEmptyUpbin  
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
    fwhmpars = []; mflags = []
    try: fwhmpars.append(parsfile['a']); mflags.append(True)
    except: print('Parameter a does not exist'); mflags.append(False)
    try: fwhmpars.append(parsfile['b']); mflags.append(True)
    except: print('Parameter b does not exist'); mflags.append(False)
    try: fwhmpars.append(parsfile['c']); mflags.append(True)
    except: print('Parameter c does not exist'); mflags.append(False)
    calpars = [ parsfile['m'], parsfile['d']]
    i_guess =  fwhmpars + calpars
    binlow =  parsfile['binlow']
    binup =  parsfile['binup']
    print('Initial parameters:',  parsfile)
    chisq =  chi2(i_guess, hexp, hsim, mflags=mflags, binlow=binlow, binup=binup)
    print('Initial Chisq_nu:',  chisq/(binup - binlow))

    print('Initial low bin divergence',  binlow - GetNoEmptyLowbin(hexp))
    print('Initial up bin divergence',  binup - GetNoEmptyUpbin(hexp))
    binlow = GetNoEmptyLowbin(hexp)
    binup = GetNoEmptyUpbin(hexp)
    print('Limit bins used:', binlow, binup)

    
    ##MCMC withou the first parameter
    print('STARTING MCMC')
    print('model flags', mflags)
    mask = mflags + [True]*2
    print('Initial Guess', i_guess)
    samples, chains =  MCMC(i_guess, hexp, hsim, binlow=binlow, binup=binup, nwalkers=args.nwalkers,  nsteps=args.nsteps,  mflags=mflags)
    print('SAMPLES:' , samples)
    print('CHAINS:' , chains)
    np.savetxt(os.path.join(outpath, 'samples.txt'), samples, fmt='%1.4e')
    np.savetxt(os.path.join(outpath, 'chains.txt'), chains, fmt='%1.4e')
    mcmcpath = os.path.join(plotspath, 'mcmc_walkers.png')
    parscontoursparth = os.path.join(plotspath, 'par_contours.png')
    plot_samplesdist(samples, chains, mflags, args.nwalkers, args.nsteps, mcmcpath, parscontoursparth )
    print('MCMC finished')


                                   
if __name__ == "__main__":
    main()


''' Note Simulation of the 3x3 detector is missing of the 1200 keV
peak do not include in the analysis.
'''
