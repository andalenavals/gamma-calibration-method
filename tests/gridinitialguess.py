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
    parser.add_argument('--filename',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/initialparsNaI2x2_grid.yaml',
                        help='.yaml with the final initial guess')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data',
                        help='location of the output of the plots')
    parser.add_argument('--npoints', default=1000 , type=int,
                        help='Npoints')
    args = parser.parse_args()

    return args

#More proper initial guess to not fall in local minima
def FindInitialGuess(npoints, hexp, hsim, binlow=None, binup=None, filename='grid_initial_guess.yaml'):
    import itertools
    import random
    from ROOT import TRandom
    import numpy as np
    from likelihood import chi2
    import yaml
    data = {} 
    
    #If for any reason the minimum is at a certain limit redefine limits around it
    seed = random.randint(0, 500)
    print("Finding a tentative initial guess. Using seed:", seed  )
    ran = TRandom(seed)
    chiaux =  np.inf
    for step in range(npoints):
        a = ran.Uniform(0,2)
        b = ran.Uniform(0,2)
        c = ran.Uniform(0,2)
        m = ran.Uniform(0,4)
        d = ran.Uniform(-30,30)
        pars = [a, b, c, m, d]
        if (chi2(pars, hexp, hsim, binlow=binlow, binup=binup) < chiaux):
            chiaux = chi2(pars, hexp, hsim, binlow=binlow, binup=binup)
            fpars = pars
            print('step:',  step, 'pars:', pars, 'chisq:', chiaux)
    data['a'] = fpars[0]; data['b'] = fpars[1]; data['c'] = fpars[2]; data['m'] = fpars[3]; data['d'] = fpars[4]
    data['binlow'] = binlow
    data['binup'] =  binup
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        
def main():
    import sys; sys.path.append(".")
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

  
    print("Fitting using only one source",  args.simfile)
    #Save data in arrays
    ch_sim, counts_sim = np.loadtxt(args.simfile, dtype=int, unpack=True)
    ch_exp, counts_exp = np.loadtxt(args.measurefile, dtype=int, unpack=True)
    nbins_exp, xlow_exp, xup_exp = len(ch_exp), min(ch_exp), max(ch_exp)
    nbins_sim, xlow_sim, xup_sim = len(ch_sim), min(ch_sim), max(ch_sim)
    hexp = SaveInTH1(ch_exp,counts_exp,'hist_exp',nbins_exp,xlow_exp,xup_exp)
    hsim = SaveInTH1(ch_sim,counts_sim,'hist_sim',nbins_sim,xlow_sim,xup_sim)

    filename = os.path.join(outpath, args.filename )
    binlow = 1
    binup = nbins_exp
    i_guess =  FindInitialGuess(args.npoints, hexp, hsim, binlow=binlow, binup=binup, filename=filename)
    
   


                             
if __name__ == "__main__":
    main()


''' Note Simulation of the 3x3 detector is missing of the 1200 keV
peak do not include in the analysis.
'''
