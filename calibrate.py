import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Alpha beta gamma test solving the fitting problem of system of equatiosn, plotting correlations and final correlation function with bias')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_22Na.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
    parser.add_argument('--nsig', default=1, type=int, 
                        help='How many sigmas for the marginalized confidence interval')
    parser.add_argument('--nsteps', default=1000, type=int, 
                        help='nsteps of MCMC')
    parser.add_argument('--nwalkers', default=100, type=int, 
                        help='nwalkers of MCMC')
    parser.add_argument('--testfit', default=False, action='store_const', const=True, help='Introduce manually the fitting, and compare with the simulation')
    parser.add_argument('--plots', default=False, action='store_const', const=True, help='Do plots during the estimation of the parameters')
    args = parser.parse_args()

    return args

#More proper initial guess to not fall in local minima
def FindInitialGuess(hexp, hsim, binlow=None, binup=None):
    import itertools
    import random
    from ROOT import TRandom
    import numpy as np
    from likelihood import chi2
    #If for any reason the minimum is at a certain limit redefine limits around it
    seed = random.randint(0, 500)
    print("Finding a tentative intial guess. Using seed:", seed  )
    ran = TRandom(seed)
    
    chiaux =  np.inf
    for step in range(10000):
        a = ran.Uniform(0,2)
        b = ran.Uniform(0,2)
        c = ran.Uniform(0,2)
        m = ran.Uniform(0,4)
        d = ran.Uniform(-30,30)
        pars = [a, b, c, m, d]
        if (chi2(pars, hexp, hsim, binlow=binlow, binup=binup) < chiaux):
            chiaux = chi2(pars, hexp, hsim, binlow=binlow, binup=binup)
            print(pars, 'chisq:', chiaux)
    return pars
  
        
def main():
    import numpy as np
    from processor import SaveInTH1, AddFWHM, Calibrate, GetScaleFactor, Scale, FindLowerNoEmptyUpbin,  FindHigherNoEmptyLowbin
    from plotter import PrettyPlot,  plot_samplesdist
    from likelihood import minimizeCHI2,  chi2,  MCMC

    args = parse_args()

    outpath = os.path.expanduser(args.outpath)
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

    
    #Save data in arrays
    ch_sim, counts_sim = np.loadtxt(args.simfile, dtype=int, unpack=True)
    ch_exp, counts_exp = np.loadtxt(args.measurefile, dtype=int, unpack=True)
    nbins_exp, xlow_exp, xup_exp = len(ch_exp), min(ch_exp), max(ch_exp)
    nbins_sim, xlow_sim, xup_sim = len(ch_sim), min(ch_sim), max(ch_sim)
    hexp = SaveInTH1(ch_exp,counts_exp,'hist_exp',nbins_exp,xlow_exp,xup_exp)
    hsim = SaveInTH1(ch_sim,counts_sim,'hist_sim',nbins_sim,xlow_sim,xup_sim)
    

    #i_guess =  FindInitialGuess(hexp, hsim, binlow=binlow, binup=binup)
    
    if args.testfit :
        #fwhmpars  #aE + b*sqrt(E)+c. calpars #mx+d
        #fwhmpars = [ 2.76974933,  -0.2228074,   -0.0468802 ]
        #calpars = [0.87224919,  2.38847193]
        fwhmpars = [2.81443031e-02, 1.24480978e+00, -3.33344549e-04]
        calpars = [1.2814228,  -16.4147601]
        pars = fwhmpars + calpars
        binlow, binup = 1, nbins_exp
        print('chisq:', chi2(pars, hexp, hsim, binlow=binlow, binup=binup))
    else:
        #i_guess = [1, 1, 1, 1, 1]
        i_guess = [0.027, 1.2,  0, 1.3 , -15.6] 
        #bnds =  ((None, None), (None, None), (None, None), (None, None), (None, None))

        ##FIRST GUESS USING THE WHOLE EXPERIMENTAL SPECTRUM
        pars, chisq = minimizeCHI2(i_guess, hexp, hsim)
        print('FIRST GUESS. Chisq_nu:',  chisq/nbins_exp)
        fwhmpars = pars[:3]
        calpars =  pars[3:]
        print('FIRST GUESS. FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
        print('FIRST GUESS. Calibration pars mx +d:',  calpars)

        #DEPURATING SELECTING PROPER HIGH AND LOW LIMIT
        binlow_new, binup_new = 0, 0
        while(binlow!=binlow_new and binup!=binup_new):
            hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  fwhmpars)
            hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
            binlow = FindHigherNoEmptyLowbin(hexp, hsim_fwhm_ch)
            binup = FindLowerNoEmptyUpbin(hexp, hsim_fwhm_ch)
            print('Depurated binlow and binup:',  binlow, binup)
            pars, chisq = minimizeCHI2(pars, hexp, hsim, binlow=binlow, binup=binup)
            print('Depurated. Chisq_nu:',  chisq/(binup - binlow))
            fwhmpars = pars[:3]
            calpars =  pars[3:]
            print('Depurated. FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
            print('Depurated. Calibration pars mx +d:',  calpars)
            binlow = binlow_new; binup = binup_new

        #MCMC (GETTING ERRORS OF ESTIMATES)
        mflags = [True, True]
        samples, chains =  MCMC(pars, hexp, hsim, binlow=binlow, binup=binup, nwalkers=args.nwalkers,  nsteps=nsteps,  mflags=mflags)
        mcmcpath = os.path.join(outpath, 'mcmc_walkers.png')
        parscontoursparth = os.path.join(outpath, 'par_contours.png')
        if args.plots: plot_samplesdist(samples, chains, mflags, args.nwalkers, args.nsteps, mcmcpath, parscontoursparth )
        
        

    
    hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  fwhmpars)
    hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
    scalefactor =  GetScaleFactor(hexp, hsim_fwhm_ch, binlow=binlow, binup=binup)
    print('the scale factor is', scalefactor)
    hsim_fwhm_ch_sc = Scale(hsim_fwhm_ch, 'hist_sim_fwhm_cal_sc',  scalefactor )
    ch_sim_fwhm = [ hsim_fwhm.GetBinCenter(i) for i in range(1, hsim_fwhm.GetNbinsX() + 1)]
    counts_sim_fwhm =  [hsim_fwhm.GetBinContent(i) for i in range(1, hsim_fwhm.GetNbinsX() + 1)]
    ch_sim_fwhm_ch = [ hsim_fwhm_ch.GetBinCenter(i) for i in range(1, hsim_fwhm_ch.GetNbinsX() + 1)]
    counts_sim_fwhm_ch =  [hsim_fwhm_ch.GetBinContent(i) for i in range(1, hsim_fwhm_ch.GetNbinsX() + 1)]
    ch_sim_fwhm_ch_sc = [ hsim_fwhm_ch_sc.GetBinCenter(i) for i in range(1, hsim_fwhm_ch_sc.GetNbinsX() + 1)]
    counts_sim_fwhm_ch_sc =  [hsim_fwhm_ch_sc.GetBinContent(i) for i in range(1, hsim_fwhm_ch_sc.GetNbinsX() + 1)]
    #print(hsim_fwhm_ch.GetNbinsX())
    
    

    if(args.plots):
        #Original Simulation
        plt.clf()
        filepath = os.path.join(outpath, 'original_sim.png')
        PrettyPlot(ch_sim, counts_sim, color='blue', marker=None, label='Simulation', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=24)
        print("Printing file: ", filepath)
        plt.savefig(filepath)
        #Original Experiment
        plt.clf()
        filepath = os.path.join(outpath, 'original_exp.png')
        PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=24)
        print("Printing file: ", filepath)
        plt.savefig(filepath)
        #Sim with FWHM
        plt.clf()
        filepath = os.path.join(outpath, 'sim_fwhm.png')
        PrettyPlot(ch_sim_fwhm, counts_sim_fwhm, color='blue', marker=None, label='Simulation with GEB', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=18)
        print("Printing file: ", filepath)
        plt.savefig(filepath)
        #Sim with FWHM in channels
        plt.clf()
        filepath = os.path.join(outpath, 'sim_fwhm_ch.png')
        PrettyPlot(ch_sim_fwhm_ch, counts_sim_fwhm_ch, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15)
        print("Printing file: ", filepath)
        plt.savefig(filepath)
        #Comparison exp with sim with FWHM and in channels
        plt.clf()
        filepath = os.path.join(outpath, 'exp_vs_sim_fwhm_ch_sc.png')
        PrettyPlot(ch_sim_fwhm_ch_sc, counts_sim_fwhm_ch_sc, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15)
        PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15)
        print("Printing file: ", filepath)
        plt.savefig(filepath, dpi=200)
   
if __name__ == "__main__":
    main()


''' Note Simulation of the 3x3 detector is missing of the 1200 keV
peak.  This is way the global minimum does not match the expected
calibration values
'''
