import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Alpha beta gamma test solving the fitting problem of system of equatiosn, plotting correlations and final correlation function with bias')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI3x3_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI3x3_22Na.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
    parser.add_argument('--plots', default=False, action='store_const', const=True, help='Do plots during the estimation of the parameters')
    args = parser.parse_args()

    return args


def main():
    import numpy as np
    from processor import SaveInTH1,  AddFWHM, Calibrate,  PrettyPlot
    args = parse_args()

    outpath = os.path.expanduser(args.outpath)
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

    #Save data in arrays
    ch_sim,  counts_sim = np.loadtxt(args.simfile, dtype=int,  unpack=True)
    ch_exp,  counts_exp = np.loadtxt(args.measurefile, dtype=int,  unpack=True)

    nbins = len(ch_exp)
    xlow =  min(ch_sim); xup = max(ch_sim)
    hexp = SaveInTH1(ch_exp, counts_exp, 'hist_exp', nbins, xlow, xup)
    hsim = SaveInTH1(ch_sim, counts_sim, 'hist_sim', nbins, xlow, xup)
    pars = [0.027, 1.2,  0] #aE + b*sqrt(E)+c
    hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  pars)
    ch_sim_fwhm = [ hsim_fwhm.GetBinCenter(i) for i in range(1, hsim_fwhm.GetNbinsX() + 1)]
    counts_sim_fwhm =  [hsim_fwhm.GetBinContent(i) for i in range(1, hsim_fwhm.GetNbinsX() + 1)]
    calpars =  [1.3 , -15.6  ] #mx+d
    hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
    ch_sim_fwhm_ch = [ hsim_fwhm_ch.GetBinCenter(i) for i in range(1, hsim_fwhm_ch.GetNbinsX() + 1)]
    counts_sim_fwhm_ch =  [hsim_fwhm_ch.GetBinContent(i) for i in range(1, hsim_fwhm_ch.GetNbinsX() + 1)]
    print(hsim_fwhm_ch.GetNbinsX())

    

    if(args.plots):
        #Original Simulation
        plt.clf()
        filepath = os.path.join(outpath, 'original_sim.png')
        PrettyPlot(ch_sim, counts_sim, color='blue', marker=None, label='Simulation', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=24, filepath=filepath)
        #Original Experiment
        plt.clf()
        filepath = os.path.join(outpath, 'original_exp.png')
        PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=24, filepath=filepath)
        #Sim with FWHM
        plt.clf()
        filepath = os.path.join(outpath, 'sim_fwhm.png')
        PrettyPlot(ch_sim_fwhm, counts_sim_fwhm, color='blue', marker=None, label='Simulation with GEB', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=18, filepath=filepath)
        #Sim with FWHM in channels
        plt.clf()
        filepath = os.path.join(outpath, 'sim_fwhm_ch.png')
        PrettyPlot(ch_sim_fwhm_ch, counts_sim_fwhm_ch, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=18, filepath=filepath)

    
    
    
    
    
  
    
 
        
        
   
if __name__ == "__main__":
    main()
